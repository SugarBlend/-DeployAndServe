import os
from pathlib import Path
from statistics import stdev
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from ultralytics.utils.checks import check_version

from deploy2serve.deployment.core.exporters import BaseExporter
from deploy2serve.deployment.core.exporters.calibration.base_batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.calibration.base_calibrator import EngineCalibrator
from deploy2serve.deployment.models.export import ExportConfig, Precision
from deploy2serve.utils.logger import get_logger, get_project_root


class TensorRTExporter(BaseExporter):
    def __init__(self, config: ExportConfig):
        super(TensorRTExporter, self).__init__(config)
        self.engine_path = Path(self.config.tensorrt.output_file)
        if not self.engine_path.is_absolute():
            self.engine_path = get_project_root().joinpath(self.engine_path)
        cache_path = f"{self.engine_path.parent}/calibration_cache/{self.engine_path.stem}.cache"
        self.calibrator: EngineCalibrator = EngineCalibrator(self.config.tensorrt, cache_path)
        self.batcher: Optional[BaseBatcher] = None
        self.logger = get_logger("tensorrt")

    def register_tensorrt_plugins(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "This method doesn't implemented, your should create him in custom class, " "based on ExtendExporter"
        )

    def benchmark(self) -> None:
        import tensorrt as trt

        from deploy2serve.deployment.core.executors.backends.tensrt import TensorRTExecutor

        self.logger.info(f"Start benchmark of model: {self.engine_path}")
        bindings, binding_address, context = TensorRTExecutor.load(
            self.engine_path,
            self.config.tensorrt.specific.profile_shapes[0]["max"][0],
            self.config.device,
            trt.Logger.ERROR,
        )
        shapes: List[Tuple[int, ...]] = []
        for profile_shapes in self.config.tensorrt.specific.profile_shapes:
            shapes.extend(list(profile_shapes.values()))

        for batch_shape in set(shapes):
            dummy_input = torch.ones(batch_shape, dtype=torch.float16, device=self.config.device)

            if check_version(trt.__version__, "<=8.6.1"):
                context.set_binding_shape(0, dummy_input.shape)
            elif check_version(trt.__version__, ">8.6.1"):
                for name in bindings:
                    if bindings[name].io_mode == "input":
                        context.set_input_shape(name, dummy_input.shape)
                        break

            timings: List[float] = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            for _ in range(50):
                context.execute_v2(list(binding_address.values()))

            for _ in range(self.config.repeats):
                start.record(torch.cuda.current_stream())
                context.execute_v2(list(binding_address.values()))
                end.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))

            avg_time = np.mean(timings)
            shape = "x".join(list(map(str, batch_shape)))

            self.logger.info(f"[{shape}] Average latency: {avg_time:.2f} ms")
            self.logger.info(f"[{shape}] Min latency: {min(timings):.2f} ms")
            self.logger.info(f"[{shape}] Max latency: {max(timings):.2f} ms")
            self.logger.info(f"[{shape}] Std latency: {stdev(timings):.2f} ms")
            self.logger.info(f"[{shape}] Average throughput: {1000 / avg_time:.2f} FPS")

    def export(self) -> None:
        if os.path.exists(self.engine_path) and not self.config.tensorrt.force_rebuild:
            return

        self.logger.info("Try convert ONNX model to TensorRT engine")
        Path(self.engine_path).parent.mkdir(parents=True, exist_ok=True)

        if not Path(self.config.onnx.output_file).is_absolute():
            self.config.onnx.output_file = str(get_project_root().joinpath(self.config.onnx.output_file))

        if not os.path.exists(self.config.onnx.output_file):
            raise FileNotFoundError(f"Onnx model is not found by this way: {self.config.onnx.output_file}")

        import tensorrt as trt

        logger = trt.Logger(self.config.tensorrt.specific.log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        with open(self.config.onnx.output_file, "rb") as file, trt.OnnxParser(network, logger) as parser:
            if not parser.parse(file.read()):
                for error in range(parser.num_errors):
                    logger.log(logger.INTERNAL_ERROR, parser.get_error(error))
                raise RuntimeError("ONNX parsing failed ...")

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        for shapes in self.config.tensorrt.specific.profile_shapes:
            profile.set_shape(input_tensor.name, **shapes)

        if config.add_optimization_profile(profile) < 0:
            logger.log(logger.WARNING, f"Invalid optimization profile {profile}")

        if self.config.tensorrt.specific.precision.name in [trt.BuilderFlag.INT4.name, trt.BuilderFlag.INT8.name]:
            config.set_calibration_profile(profile)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.tensorrt.specific.workspace)
        config.profiling_verbosity = self.config.tensorrt.specific.profiling_verbosity

        if check_version(trt.__version__, ">8.6.1"):
            config.max_aux_streams = self.config.tensorrt.specific.max_aux_streams
        config.avg_timing_iterations = 8

        if check_version(trt.__version__, ">=9.1.0"):
            if self.config.tensorrt.specific.runtime_platform:
                config.runtime_platform = self.config.tensorrt.specific.runtime_platform
            if self.config.tensorrt.specific.compatibility_level:
                config.hardware_compatibility_level = self.config.tensorrt.specific.compatibility_level

        if self.config.tensorrt.specific.tactics and len(self.config.tensorrt.specific.tactics):
            tactics: int = 0
            for tactic in self.config.tensorrt.specific.tactics:
                tactics |= 1 << int(tactic)
            config.set_tactic_sources(tactics)

        for flag in [*self.config.tensorrt.specific.flags, self.config.tensorrt.specific.precision]:
            if flag.name in dir(Precision):
                try:
                    if not getattr(builder, f"platform_has_fast_{flag.name}"):
                        logger.log(
                            trt.Logger.WARNING,
                            "This gpu device doesnt have fast computation " f"on {flag.name} precision",
                        )
                except AttributeError:
                    pass
            config.set_flag(flag)

        if self.config.tensorrt.specific.precision in [trt.BuilderFlag.INT4, trt.BuilderFlag.INT8]:
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = self.calibrator
            config.int8_calibrator.set_image_batcher(self.batcher)

        network = self.register_tensorrt_plugins(network)
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        logger.log(logger.INFO, "Network Description")
        for node in [*inputs, *outputs]:
            logger.log(logger.INFO, f"Node '{node.name}' with shape {node.shape} and dtype {node.dtype}")

        if self.config.tensorrt.enable_timing_cache:
            cache_folder = Path(self.engine_path).parent.joinpath("timing_cache")
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_file = cache_folder.joinpath(f"{self.engine_path.stem}.cache")
            try:
                with open(cache_file, "rb") as file:
                    timing_cache = config.create_timing_cache(file.read())
            except (IOError, TypeError):
                timing_cache = config.create_timing_cache(b"")
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

        with builder.build_serialized_network(network, config) as engine, open(self.engine_path, "wb") as file:
            file.write(engine)
            if self.config.tensorrt.enable_timing_cache:
                with open(cache_file, "wb") as file:
                    file.write(timing_cache.serialize())
        self.logger.info(f"TensorRT engine successfully stored in: {self.engine_path}")
