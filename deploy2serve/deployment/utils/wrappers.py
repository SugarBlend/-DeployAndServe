from contextlib import contextmanager
import logging
import numpy as np
from statistics import stdev
import time
import torch
from typing import List


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
