import numpy as np
from pathlib import Path
from statistics import stdev
import time
import torch
from typing import List, Tuple, Dict, Optional

from deployment.core.exporters.base import BaseExporter, ExportConfig
from deployment.models.export import Precision
from utils.logger import get_logger


class OpenVINOExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(OpenVINOExporter, self).__init__(config)
        self.openvino_path = Path(self.config.openvino.output_file)
        self.openvino_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger("openvino")

    def export(self) -> None:
        import openvino as ov
        options: Optional[Dict[str, Tuple[Tuple[int, ...],ov.Type]]] = None
        layer_info = next(self.model.parameters())
        dummy_input = torch.zeros((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        for _ in range(10):
            self.model(dummy_input)
        ov_model = ov.convert_model(self.model, input=options, example_input=dummy_input)
        ov.save_model(ov_model, self.openvino_path, compress_to_fp16=self.config.openvino.precision == Precision.FP16)

    def benchmark(self) -> None:
        import openvino as ov
        core = ov.Core()
        model = core.read_model(self.openvino_path)
        compiled_model = core.compile_model(model, self.config.openvino.device)

        placeholder = np.ones((1, 3, *self.config.input_shape))
        for _ in range(50):
            compiled_model(placeholder)

        timings: List[float] = []
        for _ in range(self.config.repeats):
            start_time = time.perf_counter()
            compiled_model(placeholder)
            timings.append((time.perf_counter() - start_time) * 1000 )
        avg_time = np.mean(timings)
        shape = "x".join(list(map(str, placeholder.shape)))

        self.logger.info(f"[{shape}] Average latency: {avg_time:.2f} ms")
        self.logger.info(f"[{shape}] Min latency: {min(timings):.2f} ms")
        self.logger.info(f"[{shape}] Max latency: {max(timings):.2f} ms")
        self.logger.info(f"[{shape}] Std latency: {stdev(timings):.2f} ms")
        self.logger.info(f"[{shape}] Average throughput: {1000 / avg_time:.2f} FPS")
