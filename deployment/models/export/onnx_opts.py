from typing import Any, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator
from torch.onnx import _C_onnx


class OnnxConfig(BaseModel):
    keep_initializers_as_inputs: bool = Field(default=False, description="")
    export_params: bool = Field(default=True, description="")
    verbose: Optional[bool] = Field(default=None, description="")
    input_names: Optional[Sequence[str]] = Field(default=None, description="")
    output_names: Optional[Sequence[str]] = Field(default=None, description="")
    opset_version: Optional[int] = Field(default=13, description="")
    dynamic_axes: Optional[Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]] = Field(
        default=None, description=""
    )
    training: Union[str, _C_onnx.TrainingMode] = Field(default=_C_onnx.TrainingMode.EVAL, description="")
    do_constant_folding: bool = Field(default=True, description="")

    simplify: bool = Field(default=True, description="Enable simplify onnx model structure.")
    output_file: str = Field(default="deploy_results/onnx/model.onnx", description="Path to save converted model.")
    force_rebuild: bool = Field(default=False, description="")

    @field_validator("training", mode="before")
    def parse_training(cls, val: Any) -> _C_onnx.TrainingMode:
        if isinstance(val, str):
            val = getattr(_C_onnx.TrainingMode, val)
        elif isinstance(val, _C_onnx.TrainingMode):
            pass
        else:
            raise Exception

        return val

    class Config:
        arbitrary_types_allowed = True
