from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import os
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import onnxslim
import shutil
import torch
import tempfile
from typing import Any, Dict, Optional, Tuple, List

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig, ExporterFactory
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
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

        model_optimizations = self.config.onnx.modelopt
        if hasattr(model_optimizations, "quant_mode"):
            self.batcher: BaseBatcher = self.register_batcher()

        self.logger = get_logger(self.__class__.__name__)

    def load_checkpoints(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Need to provide realization in child class.")

    @abstractmethod
    def register_batcher(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "This method doesn't implemented, your should create him in custom class, based on 'ExtendExporter'."
        )

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
        self.logger.info(f"Benchmark for ONNX model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50, cuda_profiling=False) as t:
            t(lambda: session.run(output_names, input_feed))
        onnx_output = session.run(output_names, input_feed)
        del session
        torch.cuda.empty_cache()

        if self.model is None:
            dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
            self.model: torch.nn.Module = self.load_checkpoints(
                config_path=self.config.config_path, weights_path=self.config.weights_path
            )
            self.model.to(device=self.config.device, dtype=dtype)
            self.model.eval()

        self.logger.info(f"Benchmark for PyTorch model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50, cuda_profiling=False) as t, torch.no_grad():
            t(lambda: self.model(*(item.to(device=self.config.device) for item in placeholders)))
        with torch.no_grad():
            original_output = self.model(*(item.to(device=self.config.device) for item in placeholders))
        del self.model
        torch.cuda.empty_cache()

        if isinstance(original_output, torch.Tensor):
            original_output = [original_output.detach().cpu().numpy()]
        elif isinstance(original_output, List):
            original_output = [item.detach().cpu().numpy() for item in original_output]
        else:
            TypeError("Type of return value from pytorch model must be 'torch.Tensor' or equal Sequence of them.")

        self.check_similarity(original_output, onnx_output, output_names)

    def check_similarity(
        self,
        original_output: List[torch.Tensor],
        onnx_output: List[np.ndarray],
        output_nodes: List[str]
    ) -> None:
        for i, (torch_out, onnx_out) in enumerate(zip(original_output, onnx_output)):
            if torch_out.shape != onnx_out.shape:
                self.logger.critical(
                    f"Shape mismatch for output {output_nodes[i]}: "
                    f"PyTorch={torch_out.shape}, ONNX={onnx_out.shape}"
                )
                continue

            abs_diff = np.abs(torch_out - onnx_out)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)

            is_numerically_close = np.allclose(torch_out, onnx_out, rtol=1e-2, atol=1e-4)

            if not is_numerically_close:
                self.logger.warning(
                    f"Output '{output_nodes[i]}' has numerical differences:\n"
                    f"  Max absolute difference: {max_abs_diff:.6f}\n"
                    f"  Mean absolute difference: {mean_abs_diff:.6f}\n"
                )
            else:
                self.logger.info(
                    f"Output '{output_nodes[i]}' is numerically close:\n"
                    f"  Max difference: {max_abs_diff:.6f}"
                )

    def export(self) -> None:
        if not (os.path.exists(self.save_path) and not self.config.onnx.force_rebuild):
            if self.model is None:
                self.model: torch.nn.Module = self.load_checkpoints(
                    config_path=self.config.config_path, weights_path=self.config.weights_path
                )
            self.torch2onnx()
        else:
            if self.config.onnx.simplify:
                self.simplify()
            if self.config.onnx.modelopt.quant_mode and self.config.onnx.modelopt.calib_method:
                self.quantize()

    def torch2onnx(self) -> None:
        self.logger.info("Try convert PyTorch model to ONNX format")
        placeholders = (
            torch.zeros(self.config.input_nodes[node]["shape"],
                        dtype=getattr(torch, self.config.input_nodes[node]["precision"]),
                        device=self.config.device)
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)
        options = self.config.onnx.specific.model_dump()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            temp_onnx_path = Path(tmp_file.name)

        try:
            with self.patch_ops():
                torch.onnx.export(self.model, placeholders, temp_onnx_path.as_posix(), **options)
            self.register_onnx_plugins()
            onnx.checker.check_model(temp_onnx_path.as_posix(), full_check=True)
            onnx_model = onnx.load_model(temp_onnx_path.as_posix())
            convert_model_to_external_data(
                onnx_model, all_tensors_to_one_file=True, location="model.data", size_threshold=0,
                convert_attribute=False
            )
            onnx.save_model(
                onnx_model, self.save_path.as_posix(), save_as_external_data=True, all_tensors_to_one_file=True,
                location="model.data", size_threshold=0,
            )
            onnx.checker.check_model(self.save_path, full_check=True)
            self.logger.info(f"ONNX model successfully stored in: {self.save_path}")
        except Exception as error:
            self.logger.critical(f"Catch error while apply export: {error}")
        finally:
            if temp_onnx_path.exists():
                temp_onnx_path.unlink()

    def simplify(self) -> None:
        try:
            self.logger.info("Try to simplify ONNX model")
            optimized_onnx_model = onnxslim.slim(
                self.save_path.as_posix(),
                skip_optimizations=False,
                skip_fusion_patterns=False
            )
            onnx.checker.check_model(optimized_onnx_model, full_check=True)
            onnx.save_model(optimized_onnx_model, self.save_path)
            self.logger.info(f"Simplification successfully done. ONNX model successfully stored in: {self.save_path}")
        except Exception as error:
            self.logger.critical(f"Catch error while apply optimizations: {error}")

    def quantize(self) -> None:
        from modelopt.onnx.quantization.quantize import quantize as quantize_top_level_api

        self.logger.info("Try to apply Post Training Quantization for ONNX model")
        calibration_data: Optional[Dict[str, List[np.ndarray]]] = None
        if self.batcher:
            calibration_data = {node: [] for node in self.config.input_nodes}
            for i, items in enumerate(self.batcher.dataloader):
                if i == self.config.onnx.modelopt.calib_buffer:
                    break
                for j, item in enumerate(items):
                    calibration_data[list(self.config.input_nodes)[j]].append(item.cpu().numpy())
        else:
            self.logger.warning("The implementation of the calibration data batcher is not defined. Random data will "
                                "be used, and the calibration quality will be significantly degraded.")

        dtype = np.float16 if self.config.enable_mixed_precision else np.float32
        for node in calibration_data:
            calibration_data[node] = np.concatenate(calibration_data[node], axis=0).astype(dtype)

        short_dtype = "fp16" if self.config.enable_mixed_precision else "fp32"
        temp_dir = tempfile.mkdtemp("onnx_quant_cache_")
        try:
            quantize_top_level_api(
                onnx_path=self.save_path.as_posix(),
                quantize_mode=self.config.onnx.modelopt.quant_mode,
                calibration_method=self.config.onnx.modelopt.calib_method,
                calibration_data=calibration_data,
                calibration_cache_path=temp_dir,
                calibration_eps=self.config.onnx.modelopt.calibration_eps,
                use_external_data_format=self.config.onnx.modelopt.use_external_data_format,
                op_types_to_quantize=self.config.onnx.modelopt.op_types_to_quantize,
                nodes_to_exclude=self.config.onnx.modelopt.nodes_to_exclude,
                dq_only=not self.config.onnx.modelopt.qdq_for_weights,
                verbose=True,
                high_precision_dtype=short_dtype,
                mha_accumulation_dtype=short_dtype,
                enable_gemv_detection_for_trt=False,
                enable_shared_constants_duplication=False,
            )
            self.config.onnx.output_file = self.save_path.with_suffix(".quant.onnx").as_posix()
            self.logger.info(f"Quantization successfully done. ONNX model successfully stored "
                             f"in: {self.config.onnx.output_file}")
        except Exception as error:
            shutil.rmtree(temp_dir)
            self.logger.critical(f"Catch error while apply optimizations: {error}")
