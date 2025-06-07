from enum import Enum
from pydantic import BaseModel, Field, field_validator
import tensorrt as trt
from typing import Tuple, List, Optional, Union, Dict, Any
from ultralytics.utils.checks import check_version

from utils.logger import get_logger, logging


class Precision(str, Enum):
    FP32 = 'fp32'
    BFP16 = 'bfp16'
    FP16 = 'fp16'
    FP8 = 'fp8'
    FP4 = 'fp4'
    INT8 = 'int8'
    int4 = 'int4'


class Plugin(BaseModel):
    name: str = Field(description="User name of plugin.")
    options: Dict[str, Any] = Field(description="Additional settings for plugin")




class TensorrtConfig(BaseModel):
    logger: logging.Logger = get_logger('pydantic', logging.INFO)

    plugins: List[Plugin] = Field(default=[], description="List of plugins, which can be connect to tensorrt engine.")
    log_level: Union[trt.Logger.Severity, str] = Field(default="WARNING", description="Logging level in build engine step.")
    precision: Union[Precision, trt.BuilderFlag] = Field(default=Precision.FP16, description="Precision of layers weights.")
    profile_shapes: List[Dict[str, Tuple[int, int, int, int]]] = Field(description="Inputs shapes for network for optimization.")
    workspace: int = Field(default=int(1<<30) // 4, description="Allowed memory workspace for using in build step.")
    flags: Optional[List[Union[str, trt.BuilderFlag]]] = Field(default=None, description="Flag in trt.BuilderFlag format.")
    profiling_verbosity: Union[str, trt.ProfilingVerbosity] = Field(default="LAYER_NAMES_ONLY", description="")
    max_aux_streams: int = Field(default=4, description="")
    runtime_platform: Union[str, 'trt.RuntimePlatform'] = Field(default="SAME_AS_BUILD", description="")
    compatibility_level: Union[str, 'trt.HardwareCompatibilityLevel'] = \
        Field(default="SAME_COMPUTE_CAPABILITY", description="")
    tactics: Optional[List[Union[str, trt.TacticSource]]] = \
        Field(default=None, description="List of using tactics for optimizations.")
    enable_timing_cache: bool = Field(default=True,
                                      description="Enable cache for faster rebuild in next launch of the same model.")
    enable_calibration_cache: bool = Field(default=True,
                                      description="Enable cache for faster rebuild in next launch of the same model "
                                                  "with int builder precision.")
    algorithm: Union[str, trt.CalibrationAlgoType] = Field(default="ENTROPY_CALIBRATION_2",
                                               description="Algorithm for calibration layers.")
    force_rebuild: bool = Field(default=False, description="Rebuild engine if already exist.")
    output_file: str = Field(default="deploy_results/tensorrt/model.engine", description="Path to save converted model.")

    @field_validator("log_level", mode="before")
    def parse_log_level(cls, level: str) -> trt.Logger.Severity:
        log_level = getattr(trt.Logger, level.upper())
        return log_level

    @field_validator("precision", mode="before")
    def parse_precision(cls, precision: str) -> trt.BuilderFlag:
        precision = getattr(trt.BuilderFlag, precision.upper())
        return precision

    @field_validator("profiling_verbosity", mode="before")
    def parse_profiling_verbosity(cls, val: str) -> trt.BuilderFlag:
        val = getattr(trt.ProfilingVerbosity, val.upper())
        return val

    @field_validator("flags", mode="before")
    def parse_flags(cls, fields: Optional[List[str]]) -> List[trt.BuilderFlag]:
        if fields is None or not len(fields):
            return []
        return [getattr(trt.BuilderFlag, field.upper()) for field in fields]

    @field_validator("tactics", mode="before")
    def parse_tactics(cls, tactics: Optional[List[str]]) -> List[trt.TacticSource]:
        if tactics is None or not len(tactics):
            return []
        return [getattr(trt.TacticSource, tactic.upper()) for tactic in tactics]

    @field_validator("compatibility_level", mode="before")
    def parse_compatibility_level(cls, level: Optional[str]) -> Optional['trt.HardwareCompatibilityLevel']:
        if check_version(trt.__version__, ">=9.1.0") and level:
            level = getattr(trt.HardwareCompatibilityLevel, level.upper())
        elif check_version(trt.__version__, "<9.1.0"):
            cls.logger.warning(
                f"For such version of tensorrt: {trt.__version__}, property 'trt.HardwareCompatibilityLevel' is not "
                f"supported."
            )
            level = None
        return level

    @field_validator("runtime_platform", mode="before")
    def parse_runtime_platform(cls, platform: Optional[str]) -> Optional['trt.RuntimePlatform']:
        if check_version(trt.__version__, ">=9.1.0") and platform:
            platform = getattr(trt.RuntimePlatform, platform.upper())
        elif check_version(trt.__version__, "<9.1.0"):
            cls.logger.warning(
                f"For such version of tensorrt: {trt.__version__}, property 'trt.RuntimePlatform' is not supported."
            )
            platform = None
        return platform

    class Config:
        arbitrary_types_allowed = True
