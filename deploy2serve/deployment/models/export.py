import json
from typing import List, Tuple, Optional, Dict, Union, OrderedDict

import torch.cuda
import yaml
from pydantic import BaseModel, Field, field_validator

from deploy2serve.deployment.models.calibration import CalibrationConfig
from deploy2serve.deployment.models.common import Backend, ComponentOverride, ModelMeta, ResolvedPath
from deploy2serve.deployment.models.backends.onnx_opts import OnnxConfig
from deploy2serve.deployment.models.backends.openvino_opts import OpenVINOConfig
from deploy2serve.deployment.models.backends.tensorrt_opts import TensorrtConfig
from deploy2serve.deployment.models.backends.torchscript_opts import TorchScriptConfig

Nodes = OrderedDict[str, Dict[str, Union[Tuple[int, ...], str]]]

class ExportConfig(BaseModel, metaclass=ModelMeta):
    weights_path: Optional[ResolvedPath] = Field(
        description="Path to original weights of the model which you want to convert."
    )
    config_path: Optional[ResolvedPath] = Field(
        description="Path to additional configuration file for difficult cases of model initialization."
    )
    formats: List[Backend] = Field(default=["onnx", "tensorrt"], description="Steps for deployment pipeline.")
    enable_mixed_precision: bool = Field(
        default=True, description="Enable convert Pytorch model to fp16 precision " "before launch export steps."
    )
    input_nodes: Nodes = Field(
        description="Shapes for optimization and transfer. The order of node designations must be maintained, "
                    "otherwise unexpected errors may occur."
    )
    output_nodes: Nodes = Field(
        description="Shapes for optimization and transfer. The order of node designations must be maintained, "
                    "otherwise unexpected errors may occur."
    )
    device: str = Field(default="cuda:0", description="Device backend.")
    repeats: int = Field(default=1000, description="Number for repeat iterations for inference estimation.")
    enable_benchmark: bool = Field(default=True, description="Launch benchmarks for every export format.")
    enable_visualization: bool = Field(default=True, description="Launch visualization results after every export.")

    exporter: ComponentOverride = Field(description="A structure describing a 'Exporter' by module and class name.")
    executor: ComponentOverride = Field(description="A structure describing a 'Executor' by module and class name.")

    calibration: Optional[CalibrationConfig] = Field(
        default=None,
        description="Configuration for creating a batcher, a calibration data cache, and fine-tuning the data "
                    "generation stage."
    )

    tensorrt: Optional[TensorrtConfig] = Field(
        default=None, description="Config file which consider parameters for convertation to tensorrt format."
    )
    onnx: Optional[OnnxConfig] = Field(
        default=None, description="Config file which consider parameters for convertation to onnx format."
    )
    torchscript: Optional[TorchScriptConfig] = Field(
        default=None, description="Configuration for TorchScript format convertation."
    )
    openvino: Optional[OpenVINOConfig] = Field(
        default=None, description="Configuration for OpenVINO format convertation."
    )

    @field_validator("device", mode="before")
    def validate_device(cls, val: str) -> str:
        if "cuda" in val and not torch.cuda.is_available():
            cls.logger.warning("Pytorch compiled without CUDA. Force set to CPU device.")
            return "cpu"
        return val

    @field_validator("formats", mode="before")
    def validate_formats(cls, val: List[Backend]) -> List[Backend]:
        return list(dict.fromkeys(val))

    @classmethod
    def from_file(cls, path: str) -> "ExportConfig":
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        elif path.endswith(".yaml"):
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        else:
            raise NotImplementedError("At now support configuration files with such extensions: '.json', '.yml'.")

        return ExportConfig.model_validate(data)

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
