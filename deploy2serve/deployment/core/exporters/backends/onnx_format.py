from abc import abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np
import os
import onnx
import onnxslim
import torch
from typing import Any, Dict, Optional, Tuple, List

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig, ExporterFactory
from deploy2serve.deployment.models.export import Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


@ExporterFactory.register(Backend.ONNX)
class ONNXExporter(BaseExporter):
    def __init__(
            self,
            config: ExportConfig
    ) -> None:
        super(ONNXExporter, self).__init__(config)

        self.model: Optional[torch.nn.Module] = None
        self.save_path = Path(self.config.onnx.output_file)
        if not self.save_path.is_absolute():
            self.save_path = Path.cwd().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

    def load_checkpoints(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Need to provide realization in child class.")

    @abstractmethod
    def register_onnx_plugins(self) -> Any:
        raise NotImplementedError(
            "This method doesn't implemented, your should create him in custom class, based on 'ExtendExporter'."
        )

    @abstractmethod
    @contextmanager
    def patch_ops(self) -> None:
        raise NotImplementedError(
            "This method doesn't implemented, your should create him in custom class, based on 'ExtendExporter'."
        )

    @torch.no_grad()
    def benchmark(
            self,
            sess_options: Optional["ort.SessionOptions"] = None,
            providers: Optional[Tuple[str, Dict[str, Any]]] = None
    ) -> None:
        import onnxruntime as ort
        from deploy2serve.deployment.core.executors.backends.onnxrt import ORTExecutor

        self.logger.info(f"Start benchmark of model: {self.save_path}")
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        default_provider = ["CPUExecutionProvider"]
        if providers is None and "cuda" in self.config.device:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()

            provider_options = {
                "device_id": torch.device(self.config.device).index,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
                "enable_skip_layer_norm_strict_mode": True,
                "use_tf32": True,
            }
            providers = ("CUDAExecutionProvider", provider_options)

        if providers:
            default_provider.insert(0, providers)

        session, input_names, output_names = ORTExecutor.load(self.save_path, sess_options, default_provider)

        placeholders = (
            torch.zeros(self.config.input_nodes[node]["shape"],
                        dtype=getattr(torch, self.config.input_nodes[node]["precision"]))
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)

        self.logger.info(f"Benchmark on tensor with shapes:")
        for idx, item in enumerate(placeholders):
            self.logger.info(f"Node '{input_names[idx]}': {tuple(item.shape)}")

        input_feed = {name: placeholders[idx].numpy() for idx, name in enumerate(input_names)}
        self.logger.info(f"Benchmark ONNX model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50, cuda_profiling=False) as t:
            t(lambda: session.run(output_names, input_feed))
        onnx_output = session.run(output_names, input_feed)

        if self.model is None:
            self.model: torch.nn.Module = self.load_checkpoints(
                config_path=self.config.config_path, weights_path=self.config.weights_path
            )
        self.logger.info(f"Benchmark Vanilla PyTorch model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50, cuda_profiling=False) as t:
            t(lambda: self.model(*(item.to(device=self.config.device) for item in placeholders)))
        original_output = self.model(*(item.to(device=self.config.device) for item in placeholders))

        if isinstance(original_output, torch.Tensor):
            original_output = [original_output.detach().cpu().numpy()]
        elif isinstance(original_output, List):
            original_output = [item.detach().cpu().numpy() for item in original_output]
        else:
            TypeError("Type of return value from pytorch model must be 'torch.Tensor' or equal Sequence of them.")

        for i, (torch_out, onnx_out) in enumerate(zip(original_output, onnx_output)):
            if not np.allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5):
                max_diff = np.max(np.abs(torch_out - onnx_out))
                self.logger.warning(f"Output {output_names[i]} is NOT close! Max diff: {max_diff:.4f}.")
            else:
                self.logger.info(f"Output {output_names[i]}  is numerically close to PyTorch output.")

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.onnx.force_rebuild:
            self.logger.info(f"ONNX model already exists at {self.save_path}. Skipping export.")
            return

        if self.model is None:
            self.model: torch.nn.Module = self.load_checkpoints(
                config_path=self.config.config_path, weights_path=self.config.weights_path
            )

        self.logger.info("Try convert PyTorch model to ONNX format")
        placeholders = (
            torch.zeros(self.config.input_nodes[node]["shape"],
                        dtype=getattr(torch, self.config.input_nodes[node]["precision"]), device=self.config.device)
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)

        options = self.config.onnx.specific.model_dump()
        try:
            with self.patch_ops():
                torch.onnx.export(self.model, placeholders, str(self.save_path), **options)
            self.register_onnx_plugins()
            onnx.checker.check_model(self.save_path, full_check=True)

            if self.config.onnx.simplify:
                self.logger.info("Try to simplify ONNX model")
                optimized_onnx_model = onnxslim.slim(
                    str(self.save_path),
                    skip_optimizations=False,
                    skip_fusion_patterns=False
                )
                onnx.checker.check_model(optimized_onnx_model, full_check=True)
                onnx.save_model(optimized_onnx_model, self.save_path)
                self.logger.info("Simplification successfully done")
        except Exception as error:
            self.logger.critical(f"Catch error while apply export: {error}")

        self.logger.info(f"ONNX model successfully stored in: {self.save_path}")
