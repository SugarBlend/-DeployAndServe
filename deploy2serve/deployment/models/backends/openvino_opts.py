from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List

from deploy2serve.deployment.models.common import Precision, ResolvedPath


class OpenVINOConfig(BaseModel):
    precision: Precision = Field(
        default=Precision.FP32,
        description="Perform a set of optimization passes to optimize a model for the " "purposes of inference.",
    )
    device: str = Field(description="Device name for model compile.")
    output_file: ResolvedPath = Field(
        default="checkpoints/openvino/model.xml", description="Path to save converted model."
    )
    input_names: List[str] = Field(description="Names of input nodes.")
    force_rebuild: bool = Field(default=False, description="Forcefully rebuild the existing model.")

    @field_validator("output_file", mode="before")
    def convert_to_path(cls, val: str) -> Path:
        if not val.strip():
            raise ValueError("Path cannot be empty.")
        return Path(val)

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
