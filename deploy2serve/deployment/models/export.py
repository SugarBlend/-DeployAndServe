import json
from typing import List, Tuple, Optional

import torch.cuda
import yaml
from pydantic import BaseModel, Field, field_validator

from deploy2serve.deployment.models.common import Backend, OverrideFunctionality
from deploy2serve.deployment.models.backends.onnx_opts import OnnxConfig
from deploy2serve.deployment.models.backends.openvino_opts import OpenVINOConfig
from deploy2serve.deployment.models.backends.tensorrt_opts import TensorrtConfig
from deploy2serve.deployment.models.backends.torchscript_opts import TorchScriptConfig


class ExportConfig(BaseModel):
    torch_weights: str = Field(description="Path to original weights of the model which you want to convert.")
    model_configuration: Optional[str] = Field(description="Path to additional configuration file for difficult cases of model initialization.")

    formats: List[Backend] = Field(default=["onnx", "tensorrt"], description="Steps for deployment pipeline.")
    enable_mixed_precision: bool = Field(
        default=True, description="Enable convert Pytorch model to fp16 precision " "before launch export steps."
    )
    input_shape: Tuple[int, int] = Field(description="Shapes for optimization and transfer.")
    device: str = Field(default="cuda:0", description="Device backend.")
    repeats: int = Field(default=1000, description="Number for repeat iterations for inference estimation.")
    enable_benchmark: bool = Field(default=True, description="Launch benchmarks for every export format.")
    enable_visualization: bool = Field(default=True, description="Launch visualization results after every export.")

    exporter: OverrideFunctionality = Field(description="A structure describing a 'Exporter' by module and class name.")
    executor: OverrideFunctionality = Field(description="A structure describing a 'Executor' by module and class name.")

    tensorrt: TensorrtConfig = Field(
        description="Config file which consider parameters for convertation to tensorrt format."
    )
    onnx: OnnxConfig = Field(description="Config file which consider parameters for convertation to onnx format.")
    torchscript: TorchScriptConfig = Field(description="Configuration for TorchScript format convertation.")
    openvino: OpenVINOConfig = Field(description="Configuration for OpenVINO format convertation.")

    @field_validator("device", mode="before")
    def parse_device(cls, val: str) -> str:
        if "cuda" in val:
            assert torch.cuda.is_available(), "Pytorch compiled without CUDA"
        return val

    @classmethod
    def from_file(cls, path: str) -> "ExportConfig":
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        elif path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        else:
            raise NotImplementedError("At now support configuration files with such extensions: '.json', '.yml'.")

        return ExportConfig.model_validate(data)
