from abc import ABC
import os
import numpy as np
from pathlib import Path
import torch
from typing import List, Optional, Any
from ultralytics.utils.checks import check_version

from deployment.core.exporters import BaseExporter
from deployment.core.exporters.calibration.base_calibrator import EngineCalibrator
from deployment.core.exporters.calibration.base_batcher import BaseBatcher
from deployment.models.export import ExportConfig, Precision
from utils.logger import get_logger


class TensorRTExporter(BaseExporter, ABC):
    def __init__(self, config: ExportConfig):
        super(TensorRTExporter, self).__init__(config)
        self.engine_path = Path(self.config.tensorrt_opts.output_file)
        cache_path = f"{self.engine_path.parent}/calibration_cache/{self.engine_path.stem}.cache"
        self.calibrator: EngineCalibrator = EngineCalibrator(self.config.tensorrt_opts, cache_path)
        self.batcher: Optional[BaseBatcher] = None
        self.logger = get_logger("tensorrt")

    def register_tensorrt_plugins(self, *args, **kwargs) -> Any:
        raise NotImplemented("This method doesnt implemented in custom class")

    def benchmark(self) -> None:
        pass
        # from utils.loaders import union_engine_loader, trt
        # bindings, binding_address, context = union_engine_loader(self.engine_path, self.config.device)
        #
        # for shapes in self.config.tensorrt_opts.profile_shapes:
        #     for batch_shape in shapes.values():
        #         dummy_input = torch.ones(batch_shape, dtype=torch.float16, device=self.config.device)
        #
        #         if check_version(trt.__version__, "<=8.6.1"):
        #             context.set_binding_shape(0, dummy_input.shape)
        #         elif check_version(trt.__version__, ">8.6.1"):
        #             for name in bindings:
        #                 if bindings[name].io_mode == "input":
        #                     context.set_input_shape(name, dummy_input.shape)
        #                     break
        #
        #         for _ in range(10):
        #             context.execute_v2(list(binding_address.values()))
        #
        #         timings: List[float] = []
        #         start = torch.cuda.Event(enable_timing=True)
        #         end = torch.cuda.Event(enable_timing=True)
        #         for _ in range(self.config.repeats):
        #             start.record(torch.cuda.current_stream())
        #             context.execute_v2(list(binding_address.values()))
        #             end.record(torch.cuda.current_stream())
        #             torch.cuda.synchronize()
        #
        #             timings.append(start.elapsed_time(end))
        #         avg_time = np.mean(timings)
        #         shape = "x".join(list(map(str, batch_shape)))
        #         self.logger.info(f'[{shape}]: Average inference time: {avg_time:.2f} ms ({1000 / avg_time:.2f} FPS)')

    def export(
            self,
    ) -> None:
        if os.path.exists(self.engine_path) and not self.config.tensorrt_opts.force_rebuild:
            return
        Path(self.engine_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"Onnx model is not found by this way: {self.onnx_path}")

        import tensorrt as trt
        logger = trt.Logger(self.config.tensorrt_opts.log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        with open(self.onnx_path, "rb") as file, trt.OnnxParser(network, logger) as parser:
            if not parser.parse(file.read()):
                for error in range(parser.num_errors):
                    logger.log(logger.INTERNAL_ERROR, parser.get_error(error))
                raise RuntimeError("ONNX parsing failed ...")

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        for shapes in self.config.tensorrt_opts.profile_shapes:
            profile.set_shape(input_tensor.name, **shapes)

        if config.add_optimization_profile(profile) < 0:
            logger.log(logger.WARNING, f"Invalid optimization profile {profile}")

        if self.config.tensorrt_opts.precision.name in [trt.BuilderFlag.INT4.name, trt.BuilderFlag.INT8.name]:
            config.set_calibration_profile(profile)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.tensorrt_opts.workspace)
        config.profiling_verbosity = self.config.tensorrt_opts.profiling_verbosity

        if check_version(trt.__version__, ">8.6.1"):
            config.max_aux_streams = self.config.tensorrt_opts.max_aux_streams
        config.avg_timing_iterations = 8

        if check_version(trt.__version__, ">=9.1.0"):
            config.runtime_platform = self.config.tensorrt_opts.runtime_platform
            config.hardware_compatibility_level = self.config.tensorrt_opts.compatibility_level

        if self.config.tensorrt_opts.tactics and len(self.config.tensorrt_opts.tactics):
            tactics: int = 0
            for tactic in self.config.tensorrt_opts.tactics:
                tactics |= (1 << int(tactic))
            config.set_tactic_sources(tactics)

        for flag in [*self.config.tensorrt_opts.flags, self.config.tensorrt_opts.precision]:
            if flag.name in dir(Precision):
                try:
                    if not getattr(builder, f"platform_has_fast_{flag.name}"):
                        logger.log(trt.Logger.WARNING, "This gpu device doesnt have fast computation "
                                                       f"on {flag.name} precision")
                except AttributeError:
                    pass
            config.set_flag(flag)


        if self.config.tensorrt_opts.precision in [trt.BuilderFlag.INT4, trt.BuilderFlag.INT8]:
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = self.calibrator
            config.int8_calibrator.set_image_batcher(self.batcher)

        network = self.register_tensorrt_plugins(network)
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        logger.log(logger.INFO, "Network Description")
        for node in [*inputs, *outputs]:
            logger.log(logger.INFO, f"Node '{node.name}' with shape {node.shape} and dtype {node.dtype}")

        if self.config.tensorrt_opts.enable_timing_cache:
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
            if self.config.tensorrt_opts.enable_timing_cache:
                with open(cache_file, "wb") as file:
                    file.write(timing_cache.serialize())
        logger.log(logger.INFO, f"Weights successfully build by this way: {self.engine_path}")
