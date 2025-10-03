from enum import Enum
import importlib
from typing import Any, Dict, Annotated
from pathlib import Path
from pydantic import AfterValidator
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from urllib.parse import urlparse
from deploy2serve.utils.logger import get_logger, get_project_root


class Precision(str, Enum):
    FP32 = "fp32"
    BFP16 = "bfp16"
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    int4 = "int4"


class Backend(str, Enum):
    Torch = "torch"
    TensorRT = "tensorrt"
    TorchScript = "torchscript"
    OpenVINO = "openvino"
    ONNX = "onnx"


class LoggingMeta(type):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        new_class.logger = get_logger(f"{attrs.get('__module__', '')}.{name}")
        return new_class


class ModelMeta(LoggingMeta, type(BaseModel)):
    pass


def is_url_urllib(string: str) -> bool:
    try:
        result = urlparse(string)
        return bool(result.scheme)
    except (Exception, ):
        return False

def resolve_relative_path(v: Path) -> Path:
    if v.is_absolute():
        return v
    return get_project_root().joinpath(v)

ResolvedPath = Annotated[Path, AfterValidator(resolve_relative_path)]


class Plugin(BaseModel):
    name: str = Field(description="User name of plugin.")
    options: Dict[str, Any] = Field(description="Additional settings for plugin")


class ComponentOverride(BaseModel, metaclass=ModelMeta):
    module: str = Field(description="Dot-path to the module containing the override class.")
    class_name: str = Field(description="Name of the class implementing the override logic.")

    @field_validator("module", mode="before")
    def validate_module_safe(cls, val: str) -> str:
        try:
            importlib.import_module(val)
            return val
        except ImportError:
            cls.logger.warning(f"Module '{val}' may not be available in current environment")
            return val

    @field_validator("class_name", mode="before")
    def validate_class_safe(cls, val: str, info: ValidationInfo) -> str:
        module_path = info.data.get('module')
        if not module_path:
            return val

        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, val):
                cls.logger.warning(f"Class '{val}' not found in module '{module_path}'.")
        except ImportError:
            pass

        return val

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
