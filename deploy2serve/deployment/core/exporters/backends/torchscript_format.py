import os
from copy import deepcopy
from pathlib import Path
from typing import Optional
import torch.jit

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig, timer
from deploy2serve.deployment.core.exporters.factory import ExporterFactory
from deploy2serve.deployment.models.export import Backend
from deploy2serve.utils.logger import get_logger, get_project_root


@ExporterFactory.register(Backend.TorchScript)
class TorchScriptExporter(BaseExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        super(TorchScriptExporter, self).__init__(config, model)

        self.save_path = Path(self.config.torchscript.output_file)
        if not self.save_path.is_absolute():
            self.save_path = get_project_root().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

        self.traced_model: Optional[torch.jit.ScriptModule] = None

    @torch.no_grad()
    def benchmark(self, warmup_iterations: int = 50) -> None:
        self.logger.info(f"Start benchmark of model: {self.save_path}")

        layer_info = next(self.model.parameters())
        placeholder = torch.ones((1, 3, *self.config.input_shape), dtype=layer_info.dtype, device=layer_info.device)

        self.logger.info(f"Benchmark on tensor with shapes: {tuple(placeholder.shape)}")
        with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
            t(lambda: self.traced_model(placeholder))

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.torchscript.force_rebuild:
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
                self.logger.info("Try optimize traced model")
                self.traced_model = torch.jit.optimize_for_inference(self.traced_model)
                self.logger.info("TorchScript model successfully optimized")
            except Exception as error:
                self.logger.critical(error)
        self.traced_model.save(self.save_path)
        self.logger.info(f"TorchScript model successfully stored in: {self.save_path}")
