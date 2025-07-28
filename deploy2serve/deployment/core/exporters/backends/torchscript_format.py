import torch.jit
import os
from copy import deepcopy
import numpy as np
from pathlib import Path
from statistics import stdev
from typing import List, Optional

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig
from deploy2serve.utils.logger import get_logger, get_project_root


class TorchScriptExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(TorchScriptExporter, self).__init__(config)
        self.torchscript_path = Path(self.config.torchscript.output_file)
        if not self.torchscript_path.is_absolute():
            self.torchscript_path = get_project_root().joinpath(self.torchscript_path)
        self.torchscript_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger("torchscript")

        self.traced_model: Optional[torch.jit.ScriptModule] = None

    def benchmark(self) -> None:
        self.logger.info(f"Start benchmark of model: {self.torchscript_path}")

        layer_info = next(self.model.parameters())
        placeholder = torch.ones((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        timings: List[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.traced_model = torch.jit.load(self.torchscript_path, map_location=self.config.device)
        for _ in range(50):
            self.traced_model(placeholder)

        for _ in range(self.config.repeats):
            start.record(torch.cuda.current_stream())
            self.traced_model(placeholder)
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
        if os.path.exists(self.torchscript_path) and not self.config.torchscript.force_rebuild:
            return

        self.logger.info("Try convert PyTorch model to TorchScript format")
        model = deepcopy(self.model)
        layer_info = next(model.parameters())
        placeholder = torch.ones((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        for _ in range(2):
            model(placeholder)
        self.traced_model = torch.jit.trace(model, placeholder, strict=False)
        if self.config.torchscript.optimize:
            try:
                self.logger.info(f"Try optimize traced model")
                self.traced_model = torch.jit.optimize_for_inference(self.traced_model)
                self.logger.info(f"TorchScript model successfully optimized")
            except Exception as error:
                self.logger.critical(error)
        self.traced_model.save(self.torchscript_path)
        self.logger.info(f"TorchScript model successfully stored in: {self.torchscript_path}")
