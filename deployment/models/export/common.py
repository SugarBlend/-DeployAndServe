from enum import Enum
from typing import Dict, Any
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
    TorchScript = "torch_script"
    OpenVINO = "openvino"
    ONNX = "onnx"


class Plugin(BaseModel):
    name: str = Field(description="User name of plugin.")
    options: Dict[str, Any] = Field(description="Additional settings for plugin")
