from contextlib import contextmanager, ContextDecorator
import logging
import numpy as np
from statistics import stdev
import gc
import time
import torch
from typing import List, Any


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


class CudaMemoryManager(ContextDecorator):
    def __init__(self, *objects: Any, cleanup: bool = True):
        self.cleanup = cleanup
        self._objects_to_cleanup = list(objects)

    def __enter__(self) -> "CudaMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cleanup:
            self.clear()

    def add_for_cleanup(self, *obj: Any) -> None:
        self._objects_to_cleanup.extend(obj)

    def clear(self) -> None:
        self._objects_to_cleanup.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
