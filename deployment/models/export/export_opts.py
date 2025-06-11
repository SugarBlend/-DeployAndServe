import json
from pathlib import Path
from typing import List, Tuple, Union
import torch.cuda
import yaml
from pydantic import BaseModel, Field, field_validator

from deployment.models.export.common import Backend
from deployment.models.export.onnx_opts import OnnxConfig
from deployment.models.export.tensorrt_opts import TensorrtConfig


class ExportConfig(BaseModel):
    formats: List[Backend] = Field(default=["onnx", "tensorrt"], description="Steps for deployment pipeline.")
    input_shape: Tuple[int, int] = Field(description="Shapes for optimization and transfer.")
    device: str = Field(default="cuda:0", description="Device backend.")
    enable_visualization: bool = Field(default=True, description="Launch visualization results after every export.")
    enable_benchmark: bool = Field(default=True, description="Launch benchmarks for every export format.")
    repeats: int = Field(default=1000, description="Number for repeat iterations for inference estimation.")

    tensorrt: TensorrtConfig = Field(
        description="Config file which consider parameters for convertation to tensorrt format."
    )
    onnx: OnnxConfig = Field(description="Config file which consider parameters for convertation to onnx format.")

    @field_validator("device", mode="before")
    def parse_device(cls, val: str) -> str:
        if "cuda" in val:
            assert torch.cuda.is_available(), "Pytorch compiled without CUDA"
        return val

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ExportConfig":
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        elif path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        else:
            raise NotImplementedError("At now support configuration files with such extensions: '.json', '.yml'.")

        return ExportConfig.model_validate(data)
