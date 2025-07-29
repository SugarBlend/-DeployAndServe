from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from deploy2serve.deployment.core.executors import BaseExecutor, ORTExecutor, TensorRTExecutor
from deploy2serve.deployment.core.executors.backends.openvinort import OpenVINORTExecutor
from deploy2serve.deployment.core.executors.backends.torchscript import TorchScriptExecutor
from deploy2serve.deployment.models.export import Backend, ExportConfig
from deploy2serve.utils.logger import get_logger


class ExtendExecutor(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config
        self.backend: Optional[Backend] = None
        self.executor_factory = ExecutorFactory()
        self.logger = get_logger("executor")

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def plotter(self, *args, **kwargs) -> Any:
        pass

    def visualization(self, backend: Backend) -> None:
        self.backend = backend
        executor = self.executor_factory.create(backend)(self.config)

        for name in dir(executor):
            if not name.startswith("__"):
                attr = getattr(executor, name)
                if isinstance(attr, property):
                    setattr(self.__class__, name, attr)
                    continue

                if callable(attr):
                    if isinstance(attr, (classmethod, staticmethod)):
                        continue
                    setattr(self.__class__, name, attr)
        self.plotter()


class ExecutorFactory(object):
    _executors: Dict[Backend, Type[BaseExecutor]] = {
        Backend.TorchScript: TorchScriptExecutor,
        Backend.ONNX: ORTExecutor,
        Backend.TensorRT: TensorRTExecutor,
        Backend.OpenVINO: OpenVINORTExecutor,
    }

    @classmethod
    def create(cls, executor: Backend) -> Type[BaseExecutor]:
        executor_class = cls._executors.get(executor)
        if not executor_class:
            raise ValueError(f"Unsupported executor type: {executor}")
        return executor_class
