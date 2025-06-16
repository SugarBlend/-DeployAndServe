from pydantic import BaseModel, Field


class TorchScriptConfig(BaseModel):
    optimize: bool = Field(default=True, description="Perform a set of optimization passes to optimize a model for "
                                                     "the purposes of inference.")
    output_file: str = Field(default="weights/torchscript/model.pt",
                             description="Path to save converted model.")
    force_rebuild: bool = Field(description="Forcefully rebuild the existing model.")

    class Config:
        arbitrary_types_allowed = True
