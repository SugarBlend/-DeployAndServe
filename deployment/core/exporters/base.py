from abc import ABC, abstractmethod
from deployment.models.export import ExportConfig


class BaseExporter(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config

    @abstractmethod
    def export(self) -> None:
        pass

    @abstractmethod
    def benchmark(self) -> None:
        pass
