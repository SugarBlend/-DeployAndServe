from abc import ABC, abstractmethod
import warnings
import torch
from deployment.models.export import ExportConfig


class BaseExporter(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    @abstractmethod
    def export(self) -> None:
        pass

    @abstractmethod
    def benchmark(self) -> None:
        pass
