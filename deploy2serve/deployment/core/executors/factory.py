from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from deploy2serve.deployment.core.executors.base import ExecutorFactory, BaseExecutor
from deploy2serve.deployment.models.export import Backend, ExportConfig
from deploy2serve.utils.logger import get_logger


class ExtendExecutor(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config

        self.backend: Optional[Backend] = None
        self._executor: Optional[Type[BaseExecutor]] = None
        self.executor_factory = ExecutorFactory()
        self.logger = get_logger(self.__class__.__name__)

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
        self._executor = self.executor_factory.create(backend)(self.config)
        self.plotter()

    def __getattr__(self, name: str) -> Any:
        if self._executor is not None and hasattr(self._executor, name):
            return getattr(self._executor, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")
