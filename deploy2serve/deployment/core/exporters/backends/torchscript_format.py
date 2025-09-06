import os
from pathlib import Path
from typing import Optional, Any
import torch.jit

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExportConfig
from deploy2serve.deployment.core.exporters.factory import ExporterFactory
from deploy2serve.deployment.models.export import Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


@ExporterFactory.register(Backend.TorchScript)
class TorchScriptExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(TorchScriptExporter, self).__init__(config)

        self.model: Optional[torch.nn.Module] = None
        self.save_path = Path(self.config.torchscript.output_file)
        if not self.save_path.is_absolute():
            self.save_path = Path.cwd().joinpath(self.save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.__class__.__name__)

        self.traced_model: Optional[torch.jit.ScriptModule] = None

    def load_checkpoints(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Need to provide realization in child class.")

    @torch.no_grad()
    def benchmark(self, warmup_iterations: int = 50) -> None:
        self.logger.info(f"Start benchmark of model: {self.save_path}")

        placeholders = (
            torch.zeros(self.config.input_nodes[node]["shape"],
                        dtype=getattr(torch, self.config.input_nodes[node]["precision"]),
                        device=self.config.device)
            for node in self.config.input_nodes
        )
        placeholders = tuple(placeholders)

        self.logger.info(f"Benchmark on tensor with shapes:")
        for idx, node in enumerate(self.config.input_nodes):
            self.logger.info(f"Node '{node}': {tuple(self.config.input_nodes[node]['shape'])}")

        self.logger.info(f"Benchmark TorchScript model:")
        with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
            t(lambda: self.traced_model(*placeholders))

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.torchscript.force_rebuild:
            return

        self.logger.info("Try convert PyTorch model to TorchScript format")
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

        for _ in range(2):
            self.model(*placeholders)
        self.traced_model = torch.jit.trace(self.model, placeholders, strict=False)
        if self.config.torchscript.optimize:
            try:
                self.logger.info("Try optimize traced model")
                self.traced_model = torch.jit.optimize_for_inference(self.traced_model)
                self.logger.info("TorchScript model successfully optimized")
            except Exception as error:
                self.logger.critical(error)
        self.traced_model.save(self.save_path)
        self.logger.info(f"TorchScript model successfully stored in: {self.save_path}")
