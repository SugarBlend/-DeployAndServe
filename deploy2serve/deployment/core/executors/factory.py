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

    def _get_extra_kwargs(self, backend: Backend) -> dict:
        if backend == Backend.TensorRT:
            return {
                "max_batch_size": self.config.tensorrt.specific.profile_shapes[0]["max"][0],
                "log_level": self.config.tensorrt.specific.log_level,
            }
        elif backend == Backend.TorchScript:
            return {
                "enable_mixed_precision": self.config.enable_mixed_precision,
            }
        elif backend == Backend.OpenVINO:
            return {
                "device": self.config.openvino.device,
            }
        return {}

    def visualization(self, backend: Backend) -> None:
        self.backend = backend
        kwargs = {
            "checkpoints_path": getattr(self.config, backend.value).output_file,
            "device": self.config.device,
        }
        kwargs.update(self._get_extra_kwargs(backend))
        executor_cls = self.executor_factory.create(backend)
        self._executor = executor_cls(**kwargs)
        self.plotter()

    def __getattr__(self, name: str) -> Any:
        if self._executor is not None and hasattr(self._executor, name):
            return getattr(self._executor, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")
