from typing import List, Literal

from pydantic import BaseModel, Field

from deploy2serve.deployment.models.common import Precision


class OpenVINOConfig(BaseModel):
    precision: Literal[Precision.FP16, Precision.FP32, Precision.INT8] = Field(
        default=Precision.FP32,
        description="Perform a set of optimization passes to optimize a model for the " "purposes of inference.",
    )
    device: str = Field(description="Device name for model compile.")
    output_file: str = Field(default="weights/openvino/model.xml", description="Path to save converted model.")
    input_names: List[str] = Field(description="Names of input nodes.")
    force_rebuild: bool = Field(description="Forcefully rebuild the existing model.")

    class Config:
        arbitrary_types_allowed = True
