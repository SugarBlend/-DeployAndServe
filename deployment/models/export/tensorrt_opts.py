from typing import Dict, List, Optional, Tuple, Union
import tensorrt as trt
from pydantic import BaseModel, Field, field_validator
from ultralytics.utils.checks import check_version

from deployment.models.export.common import Precision, Plugin
from utils.logger import get_logger, logging


class SpecificOptions(BaseModel):
    logger: logging.Logger = get_logger("pydantic", logging.INFO)

    log_level: Union[trt.Logger.Severity, str] = Field(
        default="WARNING", description="Logging level in build engine step."
    )
    precision: Optional[Union[Precision, trt.BuilderFlag]] = Field(
        default=Precision.FP16, description="Precision of layers weights."
    )
    profile_shapes: List[Dict[str, Tuple[int, int, int, int]]] = Field(
        description="Inputs shapes for network for optimization."
    )
    workspace: int = Field(default=int(1 << 30) // 4, description="Allowed memory workspace for using in build step.")
    flags: Optional[List[Union[str, trt.BuilderFlag]]] = Field(
        default=None, description="Flag in trt.BuilderFlag format."
    )
    profiling_verbosity: Optional[Union[str, trt.ProfilingVerbosity]] = Field(
        default="LAYER_NAMES_ONLY", description="List of verbosity levels of layer information exposed in NVTX "
                                                "annotations and in IEngineInspector.")
    max_aux_streams: int = Field(default=4, description="")
    runtime_platform: Optional[Union[str, "trt.RuntimePlatform"]] = Field(
        default=None, description="Describes the intended runtime platform (operating system and CPU "
                                  "architecture) for the execution of the TensorRT engine.")
    compatibility_level: Optional[Union[str, "trt.HardwareCompatibilityLevel"]] = Field(
        default="SAME_COMPUTE_CAPABILITY", description="Describes requirements of compatibility with GPU architectures "
                                                       "other than that of the GPU on which the engine was built."
    )
    tactics: Optional[List[Union[str, trt.TacticSource]]] = Field(
        default=None, description="List of using tactics for optimizations."
    )
    algorithm: Union[str, trt.CalibrationAlgoType] = Field(
        default="ENTROPY_CALIBRATION_2", description="Algorithm for calibration layers."
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


class TensorrtConfig(BaseModel):
    specific: SpecificOptions = Field(description="Specific options for build in tensorrt format.")
    enable_timing_cache: bool = Field(
        default=True, description="Enable cache for faster rebuild in next launch of the same model."
    )
    enable_calibration_cache: bool = Field(
        default=True,
        description="Enable cache for faster rebuild in next launch of the same model with int builder precision.",
    )
    plugins: List[Plugin] = Field(default=[], description="List of plugins, which can be connect to model.")
    force_rebuild: bool = Field(default=False, description="Forcefully rebuild the existing model.")
    output_file: str = Field(
        default="weights/tensorrt/model.plan", description="Path to save converted model."
    )

    class Config:
        arbitrary_types_allowed = True
