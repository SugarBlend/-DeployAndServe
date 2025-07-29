import warnings
from abc import ABC, abstractmethod

import torch

from deploy2serve.deployment.models.export import ExportConfig


class BaseExporter(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def onnx_patch(self) -> None:
        pass

    @abstractmethod
    def export(self) -> None:
        pass

    @abstractmethod
    def benchmark(self) -> None:
        pass
