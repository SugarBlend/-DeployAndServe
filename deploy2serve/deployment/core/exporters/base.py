import warnings
from abc import ABC, abstractmethod
import torch
from typing import Dict, Type

from deploy2serve.deployment.models.export import ExportConfig, Backend


class BaseExporter(ABC):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        self.config: ExportConfig = config
        self.model: torch.nn.Module = model

        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    @abstractmethod
    def export(self) -> None:
        pass

    @abstractmethod
    def benchmark(self) -> None:
        pass


class ExporterFactory(object):
    _registry: Dict[Backend, Type[BaseExporter]] = {}

    @classmethod
    def register(cls, backend: Backend):
        def decorator(executor_cls: Type[BaseExporter]):
            cls._registry[backend] = executor_cls
            return executor_cls

        return decorator

    @classmethod
    def create(cls, export_type: Backend) -> Type[BaseExporter]:
        exporter_class = cls._registry.get(export_type)
        if not exporter_class:
            raise ValueError(f"Unsupported exporter type: {export_type}")
        return exporter_class
