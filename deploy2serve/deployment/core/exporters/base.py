import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
import time
import numpy as np
import torch
from statistics import stdev
from typing import List, Dict, Type

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


@contextmanager
def timer(
    logger: logging.Logger,
    estimation_repeats: int,
    warmup_iterations: int,
    cuda_profiling: bool = True
):
    timings: List[float] = []

    def measure(func):
        for _ in range(warmup_iterations):
            func()

        for _ in range(estimation_repeats):
            if cuda_profiling:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

            func()

            if cuda_profiling:
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            else:
                timings.append((time.time() - start_time) * 1000)

        if timings:
            avg_time = np.mean(timings)
            logger.info(f"Average latency: {avg_time:.2f} ms")
            logger.info(f"Min latency: {min(timings):.2f} ms")
            logger.info(f"Max latency: {max(timings):.2f} ms")
            logger.info(f"Std latency: {stdev(timings):.2f} ms")
            logger.info(f"Throughput: {1000 / avg_time:.2f} FPS")

    yield measure
