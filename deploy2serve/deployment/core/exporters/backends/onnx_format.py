import os
from copy import deepcopy
from pathlib import Path
from statistics import stdev
from typing import List
import numpy as np
import onnx
import onnxslim
import torch

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig
from deploy2serve.utils.logger import get_logger, get_project_root


class ONNXExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(ONNXExporter, self).__init__(config)
        self.onnx_path = Path(self.config.onnx.output_file)
        if not self.onnx_path.is_absolute():
            self.onnx_path = get_project_root().joinpath(self.onnx_path)
        self.onnx_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger("onnx")

    def register_onnx_plugins(self):
        raise NotImplementedError("This method doesn't implemented, your should create him in custom class, "
                                  "based on ExtendExporter")

    def benchmark(self) -> None:
        import onnxruntime as ort
        from deploy2serve.deployment.core.executors.backends.onnxrt import ORTExecutor

        self.logger.info(f"Start benchmark of model: {self.onnx_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider_options = {
            "device_id": torch.device(self.config.device).index,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
            "enable_cuda_graph": True,
            "enable_skip_layer_norm_strict_mode": True,
            "use_tf32": True,
        }

        session, input_names, output_names = ORTExecutor.load(
            self.onnx_path, sess_options, ["CUDAExecutionProvider"], [provider_options]
        )
        layer_info = next(self.model.parameters())
        placeholder = torch.ones((1, 3, *self.config.input_shape), dtype=layer_info.dtype).numpy()

        timings: List[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(50):
            session.run(output_names, {input_names[0]: placeholder})

        for _ in range(self.config.repeats):
            start.record(torch.cuda.current_stream())
            session.run(output_names, {input_names[0]: placeholder})
            end.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

        avg_time = np.mean(timings)
        shape = "x".join(list(map(str, placeholder.shape)))

        self.logger.info(f"[{shape}] Average latency: {avg_time:.2f} ms")
        self.logger.info(f"[{shape}] Min latency: {min(timings):.2f} ms")
        self.logger.info(f"[{shape}] Max latency: {max(timings):.2f} ms")
        self.logger.info(f"[{shape}] Std latency: {stdev(timings):.2f} ms")
        self.logger.info(f"[{shape}] Average throughput: {1000 / avg_time:.2f} FPS")

    def export(self) -> None:
        if os.path.exists(self.onnx_path) and not self.config.onnx.force_rebuild:
            return

        self.logger.info("Try convert PyTorch model to ONNX format")
        model = deepcopy(self.model)
        layer_info = next(model.parameters())
        dummy_input = torch.zeros((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        options = self.config.onnx.specific.model_dump()
        with self.onnx_patch():
            torch.onnx.export(model, (dummy_input,), str(self.onnx_path), **options)
        self.register_onnx_plugins()

        onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        if self.config.onnx.simplify:
            self.logger.info("Try to simplify ONNX model")
            onnxslim.slim(
                onnx_model,
                model_check=True,
                skip_optimizations=False,
                skip_fusion_patterns=False,
                output_model=self.onnx_path,
            )
            self.logger.info("Simplification successfully done")
        self.logger.info(f"ONNX model successfully stored in: {self.onnx_path}")
