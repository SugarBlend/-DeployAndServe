import os
from pathlib import Path
import numpy as np
import torch
from typing import Any, Optional

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig
from deploy2serve.deployment.core.exporters.factory import ExporterFactory
from deploy2serve.deployment.models.common import Precision, Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


@ExporterFactory.register(Backend.OpenVINO)
class OpenVINOExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(OpenVINOExporter, self).__init__(config)

        self.model: Optional[torch.nn.Module] = None
        self.save_path = Path(self.config.openvino.output_file)
        if not self.save_path.is_absolute():
            self.save_path = Path.cwd().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

    def load_checkpoints(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Need to provide realization in child class.")

    def benchmark(self) -> None:
        import openvino as ov

        self.logger.info(f"Start benchmark of model: {self.save_path}")
        core = ov.Core()
        model = core.read_model(self.save_path)
        compiled_model = core.compile_model(model, self.config.openvino.device)

        placeholders = (
            np.ones(self.config.input_nodes[node]["shape"],
                    dtype=getattr(np, self.config.input_nodes[node]["precision"]))
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)

        self.logger.info(f"Benchmark on tensor with shapes:")
        for idx, node in enumerate(self.config.input_nodes):
            self.logger.info(f"Node '{node}': {tuple(self.config.input_nodes[node]['shape'])}")

        self.logger.info(f"Benchmark OpenVINO model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
            t(lambda: compiled_model(*placeholders))

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.openvino.force_rebuild:
            return
        import openvino as ov

        self.logger.info("Try convert PyTorch model to OpenVINO model")
        if self.model is None:
            self.model: torch.nn.Module = self.load_checkpoints(
                config_path=self.config.config_path, weights_path=self.config.weights_path
            )

        placeholders = (
            torch.zeros(self.config.input_nodes[node]["shape"],
                        dtype=getattr(torch, self.config.input_nodes[node]["precision"]), device=self.config.device)
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)

        for _ in range(10):
            self.model(*placeholders)
        ov_model = ov.convert_model(self.model, example_input=placeholders)
        compress_to_fp16 = self.config.openvino.precision == Precision.FP16 or self.config.enable_mixed_precision
        ov.save_model(ov_model, self.save_path, compress_to_fp16)
        self.logger.info(f"OpenVINO model successfully stored in: {self.save_path}")
