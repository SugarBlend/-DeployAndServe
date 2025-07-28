from abc import ABC, abstractmethod
from typing import Any

from deploy2serve.deployment.models.export import ExportConfig


class BaseExecutor(ABC):
    def __init__(self, config: ExportConfig):
        self.config: ExportConfig = config

    @abstractmethod
    def load(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def infer(self, **kwargs) -> Any:
        pass
