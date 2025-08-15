from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from deploy2serve.deployment.models.common import Backend


class BaseExecutor(ABC):
    @abstractmethod
    def load(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def infer(self, **kwargs) -> Any:
        pass


class ExecutorFactory(object):
    _registry: Dict[Backend, Type[BaseExecutor]] = {}

    @classmethod
    def register(cls, backend: Backend):
        def decorator(executor_cls: Type[BaseExecutor]):
            cls._registry[backend] = executor_cls
            return executor_cls

        return decorator

    @classmethod
    def is_registered(cls, backend: Backend) -> bool:
        return backend in cls._registry

    @classmethod
    def create(cls, export_type: Backend) -> Type[BaseExecutor]:
        executor_class = cls._registry.get(export_type)
        if not executor_class:
            raise ValueError(f"Unsupported executor type: '{export_type}'.")
        return executor_class
