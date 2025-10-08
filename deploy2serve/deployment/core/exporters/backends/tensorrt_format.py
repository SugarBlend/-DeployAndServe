import logging
from abc import abstractmethod
import copy
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Iterable
import sys
import torch
import tensorrt as trt
from packaging import version

from deploy2serve.deployment.core.exporters.base import BaseExporter, ExporterFactory
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.calibration.calibrator import EngineCalibrator
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.models.common import Precision, Backend
from deploy2serve.deployment.utils.wrappers import timer
from deploy2serve.utils.logger import get_logger


def create_bit_mask(flags: Iterable) -> int:
    mask = 0
    for flag in flags:
        mask |= (1 << int(flag))
    return mask


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
    def __init__(self, config: ExportConfig) -> None:
        super(TensorRTExporter, self).__init__(config)

        self.save_path = self.config.tensorrt.output_file
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
        executor = TensorRTExecutor(self.save_path, self.config.device, trt.Logger.ERROR)
        input_feed = {
            node: torch.ones(self.config.input_nodes[node]["shape"],
                             dtype=getattr(torch, self.config.input_nodes[node]["precision"]),
                             device=self.config.device)
            for node in self.config.input_nodes
        }

        with timer(self.logger, self.config.repeats, warmup_iterations=50) as t:
            t(lambda: executor.infer(input_feed, asynchronous=False))

    def _add_optimization_profiles(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition,
        logger: logging.Logger
    ) -> trt.IBuilderConfig:
        profile_shapes = self.config.tensorrt.specific.profile_shapes
        if profile_shapes:
            network_inputs = {network.get_input(idx).name for idx in range(network.num_inputs)}
            missing_inputs = network_inputs - set(profile_shapes.keys())
            if missing_inputs:
                raise Exception(f"Missing shape profiles for inputs: {missing_inputs}. Using default shapes may "
                                "cause issues.")

            # determine the number of profiles based on the maximum from the input nodes
            num_profiles = max([len(profile_shapes[network.get_input(idx).name]) for idx in range(network.num_inputs)])
            # If the number of profiles for the input nodes is not the same, then we copy the last known profile for
            # the given node.
            for node in profile_shapes:
                lack = num_profiles - len(profile_shapes[node])
                for _ in range(lack):
                    profile_shapes[node].append(copy.deepcopy(profile_shapes[node][-1]))

            for profile_idx in range(num_profiles):
                profile = builder.create_optimization_profile()

                for idx in range(network.num_inputs):
                    node = network.get_input(idx)
                    if node.name not in profile_shapes:
                        continue
                    shapes = profile_shapes[node.name][profile_idx]
                    profile.set_shape(node.name, **shapes)

                if config.add_optimization_profile(profile) < 0:
                    logger.log(logger.WARNING, f"Invalid optimization profile {profile}")

                if (not profile_idx and self.config.tensorrt.specific.precision in
                        [trt.BuilderFlag.INT4, trt.BuilderFlag.INT8]):
                    config.set_calibration_profile(profile)

        return config

    def _apply_builder_flags(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        logger: trt.Logger
    ) -> Tuple[trt.IBuilderConfig, trt.Builder]:
        config.builder_optimization_level = self.config.tensorrt.specific.builder_optimization_level
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.tensorrt.specific.workspace)
        config.profiling_verbosity = self.config.tensorrt.specific.profiling_verbosity
        config.avg_timing_iterations = self.config.tensorrt.specific.avg_timing_iterations
        if self.config.tensorrt.specific.tiling_optimization_level:
            config.tiling_optimization_level = self.config.tensorrt.specific.tiling_optimization_level

        if version.parse(trt.__version__) >= version.parse("9.1.0"):
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

        if len(self.config.tensorrt.specific.tactics):
            tactics = create_bit_mask(self.config.tensorrt.specific.tactics)
            config.set_tactic_sources(tactics)

        for flag in [*self.config.tensorrt.specific.flags, self.config.tensorrt.specific.precision]:
            if not flag:
                continue

            if flag.name in dir(Precision):
                try:
                    if not getattr(builder, f"platform_has_fast_{flag.name.lower()}"):
                        logger.log(
                            trt.Logger.WARNING,
                            f"This gpu device doesn't have fast computation on {flag.name} precision",
                        )
                except (AttributeError,):
                    pass
            config.set_flag(flag)

        if self.config.tensorrt.specific.precision in [trt.BuilderFlag.FP4, trt.BuilderFlag.FP8]:
            config.set_flag(trt.BuilderFlag.FP16)

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
        for node in inputs:
            logger.log(logger.INFO, f"[Input] {node.name}: shape={node.shape}, dtype={node.dtype}")

        logger.log(logger.INFO, "== Network Outputs ==")
        for node in outputs:
            logger.log(logger.INFO, f"[Output] {node.name}: shape={node.shape}, dtype={node.dtype}")

    def _store_files(
        self,
        builder: trt.Builder,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition
    ) -> None:
        if self.config.tensorrt.enable_timing_cache:
            cache_folder = self.save_path.parent.joinpath("timing_cache")
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_file = cache_folder.joinpath(f"{self.save_path.stem}.cache")
            try:
                with cache_file.open("rb") as file:
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
        if self.save_path.exists() and not self.config.tensorrt.force_rebuild:
            return

        self.logger.info("Try convert ONNX model to TensorRT engine")
        self.logger.info(f"TensorRT version: {trt.__version__}")
        get_device_info(self.logger)

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        current_folder = os.getcwd()
        # It is necessary so that TensorRT can pull up additional ONNX weight files.
        os.chdir(Path(self.config.onnx.output_file).parent)
        if not os.path.exists(self.config.onnx.output_file):
            raise FileNotFoundError(f"Onnx model is not found by this way: {self.config.onnx.output_file}. "
                                    f"Add to pipeline before tensorrt export")

        logger = trt.Logger(self.config.tensorrt.specific.log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        network_creation_flag = create_bit_mask(self.config.tensorrt.specific.network_creation_flag)
        network = builder.create_network(network_creation_flag)

        with Path(self.config.onnx.output_file).open("rb") as file, trt.OnnxParser(network, logger) as parser:
            if not parser.parse(file.read()):
                for error in range(parser.num_errors):
                    logger.log(logger.INTERNAL_ERROR, str(parser.get_error(error)))
                raise RuntimeError("ONNX parsing failed ...")

        config = self._add_optimization_profiles(builder, config, network, logger)
        config, builder = self._apply_builder_flags(builder, config, logger)
        network = self.register_tensorrt_plugins(network)
        self.log_network_io_info(network, logger)
        self._store_files(builder, config, network)
        os.chdir(current_folder)
