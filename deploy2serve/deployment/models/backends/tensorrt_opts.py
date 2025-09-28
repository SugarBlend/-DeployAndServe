from pydantic import BaseModel, Field, field_validator
import tensorrt as trt
from typing import Dict, List, Optional, Tuple, Union, Literal

from deploy2serve.deployment.models.common import Plugin, Precision, ModelMeta
from deploy2serve.deployment.models.dataset import Dataset

# checked for TRT 10.10
CompatibilityLevelType = Literal["AMPERE_PLUS", "NONE", "SAME_COMPUTE_CAPABILITY"]
RuntimePlatformType = Literal["SAME_AS_BUILD", "WINDOWS_AMD64"]


class SpecificOptions(BaseModel, metaclass=ModelMeta):
    # trt.IBuilderConfig
    network_creation_flag: List[Union[str, trt.NetworkDefinitionCreationFlag]] = Field(
        default=["EXPLICIT_BATCH"], description="List of immutable network properties expressed at network creation time. "
                                              "For TRT version >= 10 actual possible values: 'EXPLICIT_BATCH', "
                                              "'PREFER_AOT_PYTHON_PLUGINS', 'PREFER_JIT_PYTHON_PLUGINS', "
                                              "'STRONGLY_TYPED'."
    )
    builder_optimization_level: int = Field(
        default=3, description="The builder optimization level which TensorRT should build the engine at. Setting a "
                               "higher optimization level allows TensorRT to spend longer engine building time "
                               "searching for more optimization options. The resulting engine may have better "
                               "performance compared to an engine built with a lower optimization level. The default "
                               "optimization level is 3. Valid values include integers from 0 to the maximum "
                               "optimization level, which is currently 5. Setting it to be greater than the maximum "
                               "level results in identical behavior to the maximum level."
                               " - Level 0: This enables the fastest compilation by disabling dynamic kernel "
                               "generation and selecting the first tactic that succeeds in execution. This will "
                               "also not respect a timing cache."
                               " - Level 1: Available tactics are sorted by heuristics, but only the top are tested "
                               "to select the best. If a dynamic kernel is generated its compile optimization is low."
                               " - Level 2: Available tactics are sorted by heuristics, but only the fastest tactics "
                               "are tested to select the best."
                               " - Level 3: Apply heuristics to see if a static precompiled kernel is applicable or "
                               "if a new one has to be compiled dynamically."
                               " - Level 4: Always compiles a dynamic kernel."
                               " - Level 5: Always compiles a dynamic kernel and compares it to static kernels."
    )
    tiling_optimization_level: Optional[Union[str, trt.TilingOptimizationLevel]] = Field(
        default="NONE", description="The optimization level of tiling strategies. A Higher level allows TensorRT to "
                                    "spend more time searching for better optimization strategy. Possible values: "
                                    "'FAST', 'FULL', 'MODERATE', 'NONE'."
    )
    profiling_verbosity: Union[str, trt.ProfilingVerbosity] = Field(
        default="LAYER_NAMES_ONLY",
        description="List of verbosity levels of layer information exposed in NVTX "
        "annotations and in IEngineInspector. Possible values: 'DETAILED', 'LAYER_NAMES_ONLY', 'NONE'.",
    )
    compatibility_level: Optional[CompatibilityLevelType] = Field(
        default="SAME_COMPUTE_CAPABILITY",
        description="Hardware compatibility allows an engine compatible with GPU architectures other than that of the "
                    "GPU on which the engine was built. "
                    "Possible values: 'AMPERE_PLUS', 'NONE', 'SAME_COMPUTE_CAPABILITY'.",
    )
    flags: List[Union[str, trt.BuilderFlag]] = Field(
        default=[], description="The build mode flags to turn on builder options for this network. "
                                "The flags are listed in the BuilderFlags enum."
    )
    precision: Optional[Union[Precision, trt.BuilderFlag]] = Field(
        default=Precision.FP16, description="Precision of layers weights."
    )
    profile_shapes: Dict[str, List[Dict[str, Tuple[int, ...]]]] = Field(
        default={}, description="Inputs shapes for network for optimization."
    )
    workspace: int = Field(default=int(1 << 30) // 4, description="Allowed memory workspace for using in build step.")

    tactics: List[Union[str, trt.TacticSource]] = Field(
        default=[], description="List of using tactics for optimizations."
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
    runtime_platform: Optional[RuntimePlatformType] = Field(
        default="SAME_AS_BUILD",
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

    @field_validator("builder_optimization_level", mode="before")
    def validate_optimization_level(cls, val: int) -> int:
        if not 0 <= val <= 5:
            cls.logger.warning(f"Builder optimization level must be between 0 and 5, but you provide: {val}, this "
                                   "value was forcibly converted to default: 3")
            return 3
        return val

    @field_validator("log_level", mode="before")
    def validate_log_level(cls, level: Union[str, trt.Logger.Severity]) -> trt.Logger.Severity:
        if isinstance(level, str):
            level = getattr(trt.Logger, level.upper())
        return level

    @field_validator("profile_shapes", mode="before")
    def validate_profile_shapes(
        cls,
        profile_shapes: Dict[str, List[Dict[str, Tuple[int, ...]]]]
    ) -> Dict[str, List[Dict[str, Tuple[int, ...]]]]:
        for node in profile_shapes:
            if not len(profile_shapes[node]):
                raise Exception("When specifying, each input node must have dimensions specified; the empty list "
                                "state is excluded.")
        return profile_shapes

    @field_validator("network_creation_flag", mode="before")
    def validate_network_creation_flag(
        cls,
        vals: List[Union[str, trt.NetworkDefinitionCreationFlag]]
    ) -> List[trt.NetworkDefinitionCreationFlag]:
        if not len(vals):
            return[trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH]

        for i in range(len(vals) - 1, -1, -1):
            if isinstance(vals[i], str):
                try:
                    vals[i] = getattr(trt.NetworkDefinitionCreationFlag, vals[i].upper())
                except AttributeError as error:
                    cls.logger.warning(f"The following field could not be found in the "
                                       f"structure 'trt.NetworkDefinitionCreationFlag':{error}. Skip this flag.")
        return vals

    @field_validator("precision", mode="before")
    def validate_precision(cls, precision: Optional[Union[str, trt.BuilderFlag]]) -> Optional[trt.BuilderFlag]:
        if isinstance(precision, str):
            try:
                precision = getattr(trt.BuilderFlag, precision.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                cls.logger.warning("After an error recognizing the precision parameter from the export config, this "
                                   "value was forcibly converted to 'fp32'.")
                precision = None
        return precision

    @field_validator("profiling_verbosity", mode="before")
    def validate_profiling_verbosity(
        cls,
        val: Optional[Union[str, trt.ProfilingVerbosity]]
    ) -> Optional[trt.ProfilingVerbosity]:
        if isinstance(val, str):
            try:
                val = getattr(trt.ProfilingVerbosity, val.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                val = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
                cls.logger.warning(f"'profiling_verbosity' parameter was forced to the default value: {val}. Check the "
                                   "possible values in the model field description.")
        return val

    @field_validator("tiling_optimization_level", mode="before")
    def validate_tiling_optimization_level(
        cls,
        val: Optional[Union[str, trt.TilingOptimizationLevel]]
    ) -> Optional[trt.TilingOptimizationLevel]:
        if not hasattr(trt, "TilingOptimizationLevel"):
            cls.logger.warning("TilingOptimizationLevel not available in this TensorRT version. "
                               "This field must be supported in versions >= 10.")
            return None

        if isinstance(val, str):
            try:
                val = getattr(trt.TilingOptimizationLevel, val.upper())
            except AttributeError as error:
                cls.logger.warning(error)
                val = trt.TilingOptimizationLevel.NONE
                cls.logger.warning(f"'tiling_optimization_level' parameter was forced to the default value: {val}. "
                                   "Check the possible values in the model field description.")
        return val

    @field_validator("flags", mode="before")
    def validate_flags(cls, fields: List[Union[str, trt.BuilderFlag]]) -> List[trt.BuilderFlag]:
        for i in range(len(fields) -1, -1, -1):
            if isinstance(fields[i], str):
                try:
                    fields[i] = getattr(trt.BuilderFlag, fields[i].upper())
                except AttributeError as error:
                    cls.logger.warning(f"The following field could not be found in the "
                                       f"structure 'trt.BuilderFlag':{error}. Skip this flag.")
                    fields.pop(i)
        return fields

    @field_validator("tactics", mode="before")
    def validate_tactics(cls, tactics: List[Union[str, trt.TacticSource]]) -> List[trt.TacticSource]:
        for i in range(len(tactics) - 1, -1, -1):
            if isinstance(tactics[i], str):
                try:
                    tactics[i] = getattr(trt.TacticSource, tactics[i].upper())
                except AttributeError as error:
                    cls.logger.warning(f"The following field could not be found in the "
                                       f"structure 'trt.TacticSource':{error}. Skip this tactic.")
                    tactics.pop(i)
        return tactics

    @field_validator("algorithm", mode="before")
    def validate_algorithm(cls, algorithm: Union[str, trt.CalibrationAlgoType]) -> trt.CalibrationAlgoType:
        if isinstance(algorithm, str):
            try:
                algorithm = getattr(trt.CalibrationAlgoType, algorithm.upper())
            except AttributeError as error:
                algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                cls.logger.warning(f"The following field could not be found in the "
                                   f"structure 'trt.CalibrationAlgoType':{error}. "
                                   f"Return to default value: {algorithm}.")
        return algorithm

    @field_validator("compatibility_level", mode="before")
    def validate_compatibility_level(
        cls,
        level: Optional[Union[str]]
    ) -> Optional["trt.HardwareCompatibilityLevel"]:
        if not hasattr(trt, "HardwareCompatibilityLevel"):
            cls.logger.warning("HardwareCompatibilityLevel not available in this TensorRT version. "
                               "This field must be supported in versions >= 9.")
            return None

        if isinstance(level, str):
            try:
                level = getattr(trt.HardwareCompatibilityLevel, level.upper())
            except Exception as error:
                cls.logger.warning(error)
                level = trt.HardwareCompatibilityLevel.NONE
                cls.logger.warning(f"'hardware_compatibility_level' parameter was forced to the default value: {level}. "
                                   "Check the possible values in the model field description.")
        return level

    @field_validator("runtime_platform", mode="before")
    def validate_runtime_platform(cls, platform: Optional[str]) -> Optional["trt.RuntimePlatform"]:
        if not hasattr(trt, "RuntimePlatform"):
            cls.logger.warning("RuntimePlatform not available in this TensorRT version. "
                               "This field must be supported in versions >= 9.")
            return None

        if isinstance(platform, str):
            try:
                platform = getattr(trt.RuntimePlatform, platform.upper())
            except Exception as error:
                cls.logger.warning(error)
                platform = trt.RuntimePlatform.SAME_AS_BUILD
                cls.logger.warning(f"'hardware_compatibility_level' parameter was forced to the default value: {platform}. "
                                   f"Check the possible values in the model field description.")
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
