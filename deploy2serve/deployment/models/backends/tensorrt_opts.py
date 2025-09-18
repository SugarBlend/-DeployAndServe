from typing import Dict, List, Optional, Tuple, Union

import tensorrt as trt
from pydantic import BaseModel, Field, field_validator
from ultralytics.utils.checks import check_version

from deploy2serve.deployment.models.common import Plugin, Precision, ModelMeta
from deploy2serve.deployment.models.dataset import Dataset


class SpecificOptions(BaseModel, metaclass=ModelMeta):
    # trt.IBuilderConfig
    tiling_optimization_level: Union[str, trt.TilingOptimizationLevel] = Field(
        default="NONE", description="The optimization level of tiling strategies. A Higher level allows TensorRT to "
                                    "spend more time searching for better optimization strategy. Possible values: "
                                    "'FAST', 'FULL', 'MODERATE', 'NONE'."
    )
    profiling_verbosity: Optional[Union[str, trt.ProfilingVerbosity]] = Field(
        default="LAYER_NAMES_ONLY",
        description="List of verbosity levels of layer information exposed in NVTX "
        "annotations and in IEngineInspector. Possible values: 'DETAILED', 'LAYER_NAMES_ONLY', 'NONE'.",
    )
    compatibility_level: Optional[Union[str, "trt.HardwareCompatibilityLevel"]] = Field(
        default="SAME_COMPUTE_CAPABILITY",
        description="Hardware compatibility allows an engine compatible with GPU architectures other than that of the "
                    "GPU on which the engine was built. "
                    "Possible values: 'AMPERE_PLUS', 'NONE', 'SAME_COMPUTE_CAPABILITY'.",
    )
    flags: Optional[List[Union[str, trt.BuilderFlag]]] = Field(
        default=None, description="The build mode flags to turn on builder options for this network. "
                                  "The flags are listed in the BuilderFlags enum."
    )
    precision: Optional[Union[Precision, trt.BuilderFlag]] = Field(
        default=Precision.FP16, description="Precision of layers weights."
    )
    profile_shapes: Optional[Dict[str, List[Dict[str, Tuple[int, ...]]]]] = Field(
        default=None, description="Inputs shapes for network for optimization."
    )
    workspace: int = Field(default=int(1 << 30) // 4, description="Allowed memory workspace for using in build step.")

    tactics: Optional[List[Union[str, trt.TacticSource]]] = Field(
        default=None, description="List of using tactics for optimizations."
    )
    max_aux_streams: int = Field(
        default=4, description="The maximum number of auxiliary streams that TRT is allowed to use. If the network "
                               "contains operators that can run in parallel, TRT can execute them using auxiliary "
                               "streams in addition to the one provided to the IExecutionContext::enqueueV3() call. "
                               "The default maximum number of auxiliary streams is determined by the heuristics in "
                               "TensorRT on whether enabling multi-stream would improve the performance. This behavior "
                               "can be overridden by calling this API to set the maximum number of auxiliary streams "
                               "explicitly. Set this to 0 to enforce single-stream inference. The resulting engine may "
                               "use fewer auxiliary streams than the maximum if the network does not contain enough "
                               "parallelism or if TensorRT determines that using more auxiliary streams does not help "
                               "improve the performance. Allowing more auxiliary streams does not always give better "
                               "performance since there will be synchronizations overhead between streams. Using CUDA "
                               "graphs at runtime can help reduce the overhead caused by cross-stream synchronizations. "
                               "Using more auxiliary leads to more memory usage at runtime since some activation memory "
                               "blocks will not be able to be reused."
    )
    avg_timing_iterations: int = Field(
        default=4, description="The number of averaging iterations used when timing layers. When timing layers, the "
                               "builder minimizes over a set of average times for layer execution. This parameter "
                               "controls the number of iterations used in averaging. By default the number of "
                               "averaging iterations is 1."
    )
    runtime_platform: Optional[Union[str, "trt.RuntimePlatform"]] = Field(
        default=None,
        description="Describes the intended runtime platform (operating system and CPU "
        "architecture) for the execution of the TensorRT engine. Possible values: 'SAME_AS_BUILD', 'WINDOWS_AMD64'.",
    )

    # trt.Builder
    log_level: Union[trt.Logger.Severity, str] = Field(
        default="WARNING", description="Logging level in build engine step."
    )

    # trt.IInt8Calibrator
    algorithm: Union[str, trt.CalibrationAlgoType] = Field(
        default="ENTROPY_CALIBRATION_2",
        description="Algorithm for calibration layers. Possible values: 'ENTROPY_CALIBRATION', "
                    "'ENTROPY_CALIBRATION_2', 'LEGACY_CALIBRATION', 'MINMAX_CALIBRATION'."
    )

    @field_validator("log_level", mode="before")
    def parse_log_level(cls, level: Union[str, trt.Logger.Severity]) -> trt.Logger.Severity:
        if isinstance(level, str):
            level = getattr(trt.Logger, level.upper())
        return level

    @field_validator("precision", mode="before")
    def parse_precision(cls, precision: Optional[Union[str, trt.BuilderFlag]]) -> Optional[trt.BuilderFlag]:
        if isinstance(precision, str):
            try:
                precision = getattr(trt.BuilderFlag, precision.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                precision = None
        return precision

    @field_validator("profiling_verbosity", mode="before")
    def parse_profiling_verbosity(cls, val: Optional[Union[str, trt.ProfilingVerbosity]]) -> trt.BuilderFlag:
        if isinstance(val, str):
            try:
                val = getattr(trt.ProfilingVerbosity, val.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                val = None
        return val

    @field_validator("tiling_optimization_level", mode="before")
    def parse_tiling_optimization_level(
        cls,
        val: Optional[Union[str, trt.TilingOptimizationLevel]]
    ) -> trt.TilingOptimizationLevel:
        if isinstance(val, str):
            try:
                val = getattr(trt.TilingOptimizationLevel, val.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                val = None
        return val

    @field_validator("flags", mode="before")
    def parse_flags(cls, fields: Optional[List[str]]) -> List[trt.BuilderFlag]:
        if fields is None or not len(fields):
            return []

        flags: List[trt.BuilderFlag] = []
        for field in fields:
            try:
                flags.append(getattr(trt.BuilderFlag, field.upper()))
            except AttributeError as error:
                cls.logger.warning(error)
        return flags

    @field_validator("tactics", mode="before")
    def parse_tactics(cls, tactics: Optional[List[str]]) -> List[trt.TacticSource]:
        if tactics is None or not len(tactics):
            return []

        flags: List[trt.TacticSource] = []
        for tactic in tactics:
            try:
                flags.append(getattr(trt.BuilderFlag, tactic.upper()))
            except AttributeError as error:
                cls.logger.warning(error)
        return flags

    @field_validator("algorithm", mode="before")
    def parse_algorithm(cls, algorithm: Union[str, trt.CalibrationAlgoType]) -> trt.CalibrationAlgoType:
        if isinstance(algorithm, str):
            algorithm = getattr(trt.CalibrationAlgoType, algorithm.upper())
        return algorithm

    @field_validator("compatibility_level", mode="before")
    def parse_compatibility_level(cls, level: Optional[str]) -> Optional["trt.HardwareCompatibilityLevel"]:
        if not level:
            level = None
        elif check_version(trt.__version__, ">=9.1.0"):
            level = getattr(trt.HardwareCompatibilityLevel, level.upper())
        elif check_version(trt.__version__, "<9.1.0"):
            cls.logger.warning(
                f"For such version of tensorrt: {trt.__version__}, property 'trt.HardwareCompatibilityLevel' is not "
                f"supported."
            )
            level = None
        return level

    @field_validator("runtime_platform", mode="before")
    def parse_runtime_platform(cls, platform: Optional[str]) -> Optional["trt.RuntimePlatform"]:
        if not platform:
            platform = None
        elif check_version(trt.__version__, ">=9.1.0"):
            platform = getattr(trt.RuntimePlatform, platform.upper())
        elif check_version(trt.__version__, "<9.1.0"):
            cls.logger.warning(
                f"For such version of tensorrt: {trt.__version__}, property 'trt.RuntimePlatform' is not supported."
            )
            platform = None
        return platform

    class Config:
        arbitrary_types_allowed = True
        validate_default = True


class TensorrtConfig(BaseModel):
    specific: SpecificOptions = Field(description="Specific options for build in tensorrt format.")
    enable_timing_cache: bool = Field(
        default=True, description="Enable cache for faster rebuild in next launch of the same model."
    )
    enable_calibration_cache: bool = Field(
        default=True,
        description="Enable cache for faster rebuild in next launch of the same model with int builder precision.",
    )
    dataset: Optional[Dataset] = Field(default=None, description="")
    plugins: List[Plugin] = Field(default=[], description="List of plugins, which can be connect to model.")
    force_rebuild: bool = Field(default=False, description="Forcefully rebuild the existing model.")
    output_file: str = Field(default="weights/tensorrt/model.plan", description="Path to save converted model.")

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
