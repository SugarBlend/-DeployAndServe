from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Optional, Tuple

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig
from deploy2serve.deployment.core.exporters.factory import ExporterFactory
from deploy2serve.deployment.models.common import Precision, Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger, get_project_root


@ExporterFactory.register(Backend.OpenVINO)
class OpenVINOExporter(BaseExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        super(OpenVINOExporter, self).__init__(config, model)
        self.save_path = Path(self.config.openvino.output_file)
        if not self.save_path.is_absolute():
            self.save_path = get_project_root().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

    def benchmark(self) -> None:
        import openvino as ov

        self.logger.info(f"Start benchmark of model: {self.save_path}")
        core = ov.Core()
        model = core.read_model(self.save_path)
        compiled_model = core.compile_model(model, self.config.openvino.device)

        placeholder = np.ones((1, 3, *self.config.input_shape))

        self.logger.info(f"Benchmark on tensor with shapes: {tuple(placeholder.shape)}")
        with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
            t(lambda: compiled_model(placeholder))

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.openvino.force_rebuild:
            return
        import openvino as ov

        self.logger.info("Try convert PyTorch model to OpenVINO model")
        model = deepcopy(self.model)
        layer_info = next(model.parameters())
        dummy_input = torch.zeros((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)
        for _ in range(10):
            model(dummy_input)
        ov_model = ov.convert_model(model, example_input=dummy_input)
        compress_to_fp16 = self.config.openvino.precision == Precision.FP16 or self.config.enable_mixed_precision
        ov.save_model(ov_model, self.save_path, compress_to_fp16)
        self.logger.info(f"OpenVINO model successfully stored in: {self.save_path}")
