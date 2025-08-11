import logging
from abc import abstractmethod
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type
import sys
import torch
import tensorrt as trt
from ultralytics.utils.checks import check_version

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExporterFactory
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.calibration.calibrator import EngineCalibrator
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.models.common import Precision, Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


def get_device_info(logger: logging.Logger) -> None:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA-devices")

        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Usage memory: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            logger.info(f"Reserved memory: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        logger.info("CUDA is not available on this torch compiled package!")


@ExporterFactory.register(Backend.TensorRT)
class TensorRTExporter(BaseExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module):
        super(TensorRTExporter, self).__init__(config, model)

        self.save_path = Path(self.config.tensorrt.output_file)
        if not self.save_path.is_absolute():
            self.save_path = Path.cwd().joinpath(self.save_path)
        cache_path = f"{self.save_path.parent}/calibration_cache/{self.save_path.stem}.cache"

        if self.config.tensorrt.specific.precision in [trt.BuilderFlag.INT4, trt.BuilderFlag.INT8]:
            self.calibrator: EngineCalibrator = EngineCalibrator(self.config.tensorrt, cache_path)
            self.batcher: Optional[Type[BaseBatcher]] = self.register_batcher()
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def register_batcher(self) -> Optional[Type[BaseBatcher]]:
        raise NotImplementedError(
            "This functionality is not implemented in the abstract class and refers to the variant using int8 "
            "precision."
        )

    @abstractmethod
    def register_tensorrt_plugins(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "This method doesn't implemented, your should create him in custom class, " "based on ExtendExporter"
        )

    @torch.no_grad()
    def benchmark(self) -> None:
        from deploy2serve.deployment.core.executors.backends.tensrt import TensorRTExecutor

        self.logger.info(f"Start benchmark of model: {self.save_path}")
        bindings, binding_address, context = TensorRTExecutor.load(
            self.save_path, self.config.tensorrt.specific.profile_shapes[0]["max"][0], self.config.device,
            trt.Logger.ERROR
        )
        shapes: List[Tuple[int, ...]] = []
        for profile_shapes in self.config.tensorrt.specific.profile_shapes:
            shapes.extend(list(profile_shapes.values()))

        for batch_shape in set(shapes):
            placeholder = torch.ones(batch_shape, dtype=torch.float16, device=self.config.device)

            if check_version(trt.__version__, "<=8.6.1"):
                context.set_binding_shape(0, placeholder.shape)
            elif check_version(trt.__version__, ">8.6.1"):
                for name in bindings:
                    if bindings[name].io_mode == "input":
                        context.set_input_shape(name, placeholder.shape)
                        break

            self.logger.info(f"Benchmark on tensor with shapes: {tuple(placeholder.shape)}")
            with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
                t(lambda: context.execute_v2(list(binding_address.values())))

    def _add_optimization_profiles(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition,
        logger: logging.Logger
    ) -> trt.IBuilderConfig:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)

        for shapes in self.config.tensorrt.specific.profile_shapes:
            profile.set_shape(input_tensor.name, **shapes)

        if config.add_optimization_profile(profile) < 0:
            logger.log(logger.WARNING, f"Invalid optimization profile {profile}")

        if self.config.tensorrt.specific.precision.name in [trt.BuilderFlag.INT4.name, trt.BuilderFlag.INT8.name]:
            config.set_calibration_profile(profile)

        return config

    def _apply_builder_flags(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        logger: trt.Logger
    ) -> Tuple[trt.IBuilderConfig, trt.Builder]:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.tensorrt.specific.workspace)
        if self.config.tensorrt.specific.profiling_verbosity:
            config.profiling_verbosity = self.config.tensorrt.specific.profiling_verbosity
        config.avg_timing_iterations = 8

        if check_version(trt.__version__, ">=9.1.0"):
            if self.config.tensorrt.specific.runtime_platform:
                config.runtime_platform = self.config.tensorrt.specific.runtime_platform
            if self.config.tensorrt.specific.compatibility_level:
                config.hardware_compatibility_level = self.config.tensorrt.specific.compatibility_level
            if not sys.stdout.isatty():
                self.logger.warning(
                    "App should be run from an interactive terminal in order to showcase the progress monitor "
                    "correctly.",
                )
            else:
                from deploy2serve.deployment.utils.tensorrt_progress import ProgressMonitor  # noqa: PLC0415
                config.progress_monitor = ProgressMonitor()

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
                            "This gpu device doesn't have fast computation " f"on {flag.name} precision",
                        )
                except (AttributeError,):
                    pass
            config.set_flag(flag)

        if self.config.tensorrt.specific.precision in [trt.BuilderFlag.INT4, trt.BuilderFlag.INT8]:
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = self.calibrator
            config.int8_calibrator.set_image_batcher(self.batcher)

        return config, builder

    @staticmethod
    def log_network_io_info(network: trt.INetworkDefinition, logger: trt.Logger) -> None:
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        logger.log(logger.INFO, "Network Description")
        logger.log(logger.INFO, f"Total inputs: {len(inputs)}")
        logger.log(logger.INFO, f"Total outputs: {len(outputs)}")

        logger.log(logger.INFO, "== Network Inputs ==")
        for i in inputs:
            logger.log(logger.INFO, f"[Input] {i.name}: shape={i.shape}, dtype={i.dtype}")

        logger.log(logger.INFO, "== Network Outputs ==")
        for o in outputs:
            logger.log(logger.INFO, f"[Output] {o.name}: shape={o.shape}, dtype={o.dtype}")

    def _store_files(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition
    ) -> None:
        if self.config.tensorrt.enable_timing_cache:
            cache_folder = Path(self.save_path).parent.joinpath("timing_cache")
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_file = cache_folder.joinpath(f"{self.save_path.stem}.cache")
            try:
                with open(cache_file, "rb") as file:
                    timing_cache = config.create_timing_cache(file.read())
            except (IOError, TypeError):
                timing_cache = config.create_timing_cache(b"")
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

        with builder.build_serialized_network(network, config) as engine, self.save_path.open("wb") as file:
            file.write(engine)

        if self.config.tensorrt.enable_timing_cache:
            with cache_file.open("wb") as file:
                file.write(timing_cache.serialize())
        self.logger.info(f"TensorRT engine successfully stored in: {self.save_path}")

    def export(self) -> None:
        if os.path.exists(self.save_path) and not self.config.tensorrt.force_rebuild:
            return

        self.logger.info("Try convert ONNX model to TensorRT engine")
        self.logger.info(f"TensorRT version: {trt.__version__}")
        get_device_info(self.logger)

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        if not Path(self.config.onnx.output_file).is_absolute():
            self.config.onnx.output_file = str(Path.cwd().joinpath(self.config.onnx.output_file))

        if not os.path.exists(self.config.onnx.output_file):
            raise FileNotFoundError(f"Onnx model is not found by this way: {self.config.onnx.output_file}. "
                                    f"Add to pipeline before tensorrt export")

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

        config = self._add_optimization_profiles(builder, config, network, logger)
        config, builder = self._apply_builder_flags(builder, config, logger)
        network = self.register_tensorrt_plugins(network)
        self.log_network_io_info(network, logger)
        self._store_files(builder, config, network)
