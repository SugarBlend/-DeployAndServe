from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from deploy2serve.deployment.models.common import ResolvedPath


class TorchScriptConfig(BaseModel):
    optimize: bool = Field(
        default=True,
        description="Perform a set of optimization passes to optimize a model for the purposes of inference."
    )
    output_file: ResolvedPath = Field(
        default="checkpoints/torchscript/model.pt", description="Path to save converted model."
    )
    force_rebuild: bool = Field(default=False, description="Forcefully rebuild the existing model.")

    @field_validator("output_file", mode="before")
    def convert_to_path(cls, val: str) -> Path:
        if not val.strip():
            raise ValueError("Path cannot be empty.")
        return Path(val)

    class Config:
        arbitrary_types_allowed = True
        validate_default = True
