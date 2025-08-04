from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


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


class Plugin(BaseModel):
    name: str = Field(description="User name of plugin.")
    options: Dict[str, Any] = Field(description="Additional settings for plugin")


class OverrideClassSpec(BaseModel):
    module_path: str = Field(description="Dot-path to the module containing the override class.")
    class_name: str = Field(description="Name of the class implementing the override logic.")
