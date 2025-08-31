from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Optional

import torch.nn

from deploy2serve.deployment.core.exporters import ExporterFactory
from deploy2serve.deployment.models.export import Backend, ExportConfig


class Exporter(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config

        self.export_factory = ExporterFactory()
        self.model: Optional[torch.nn.Module] = None
        self.onnx_patch = nullcontext

    @abstractmethod
    def load_checkpoints(self, **kwargs) -> Any:
        pass

    def convert(self, backend: Backend) -> None:
        exporter = self.export_factory.create(backend)(self.config)
        exporter.load_checkpoints = self.load_checkpoints
        exporter.export()

        if self.config.enable_benchmark:
            exporter.benchmark()

        exporter.__dict__.pop("model", None)
        torch.cuda.empty_cache()
