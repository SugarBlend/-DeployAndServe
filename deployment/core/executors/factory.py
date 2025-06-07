from abc import abstractmethod, ABC
from typing import Type, Dict, Any

from deployment.models.export import Backend, ExportConfig
from deployment.core.executors import ORTExecutor, TensorRTExecutor, BaseExecutor
from utils.logger import get_logger



class ExtendExecutor(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config
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

        self.plotter(backend)


class ExecutorFactory(object):
    _executors: Dict[Backend, Type[BaseExecutor]] = {
        Backend.ONNX: ORTExecutor,
        Backend.TensorRT: TensorRTExecutor,
    }

    @classmethod
    def create(cls, executor: Backend) -> Type[BaseExecutor]:
        executor_class = cls._executors.get(executor)
        if not executor_class:
            raise ValueError(f"Unsupported executor type: {executor}")
        return executor_class
