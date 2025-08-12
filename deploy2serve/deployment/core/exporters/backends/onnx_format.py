from abc import abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os
import onnx
import onnxslim
import torch
from typing import Any, Dict, Optional, List

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig, ExporterFactory
from deploy2serve.deployment.models.export import Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


@ExporterFactory.register(Backend.ONNX)
class ONNXExporter(BaseExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        super(ONNXExporter, self).__init__(config, model)

        self.save_path = Path(self.config.onnx.output_file)
        if not self.save_path.is_absolute():
            self.save_path = Path.cwd().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

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
        provider_options: Optional[Dict[str, Any]] = None,
        providers: Optional[List[str]] = None
    ) -> None:
        import onnxruntime as ort
        from deploy2serve.deployment.core.executors.backends.onnxrt import ORTExecutor

        self.logger.info(f"Start benchmark of model: {self.save_path}")
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if providers is None:
            providers = ["CUDAExecutionProvider"]

        if provider_options is None:
            provider_options = {
                "device_id": torch.device(self.config.device).index,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
                "enable_skip_layer_norm_strict_mode": True,
                "use_tf32": True,
            }

        session, input_names, output_names = ORTExecutor.load(
            self.save_path, sess_options, providers, [provider_options]
        )
        layer_info = next(self.model.parameters())
        placeholder = torch.ones((1, 3, *self.config.input_shape), dtype=layer_info.dtype).numpy()

        self.logger.info(f"Benchmark on tensor with shapes: {tuple(placeholder.shape)}")
        with timer(self.logger, self.config.repeats, warmup_iterations=50, cuda_profiling=False) as t:
            t(lambda: session.run(output_names, {input_names[0]: placeholder}))

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.onnx.force_rebuild:
            return

        self.logger.info("Try convert PyTorch model to ONNX format")
        model = deepcopy(self.model)
        layer_info = next(model.parameters())
        placeholder = torch.zeros((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        options = self.config.onnx.specific.model_dump()
        with self.patch_ops():
            torch.onnx.export(model, (placeholder,), str(self.save_path), **options)
        self.register_onnx_plugins()

        onnx_model = onnx.load(self.save_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        if self.config.onnx.simplify:
            self.logger.info("Try to simplify ONNX model")
            optimized_onnx_model = onnxslim.slim(
                onnx_model,
                model_check=True,
                skip_optimizations=False,
                skip_fusion_patterns=False
            )
            onnx.save_model(optimized_onnx_model, self.save_path)
            self.logger.info("Simplification successfully done")
        self.logger.info(f"ONNX model successfully stored in: {self.save_path}")
