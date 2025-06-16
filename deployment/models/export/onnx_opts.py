from typing import Any, Mapping, Optional, Sequence, Union, List, Tuple, Dict
from pydantic import BaseModel, Field, field_validator
# from pathlib import Path
from torch.onnx import _C_onnx
from deployment.models.export.common import Plugin


class SpecificOptions(BaseModel):
    keep_initializers_as_inputs: bool = Field(
        default=False, description="If True, all the initializers (typically corresponding to model weights) in the "
                                   "exported graph will also be added as inputs to the graph.")
    export_params: bool = Field(default=True, description="If specified, all parameters will be exported.")
    verbose: Optional[bool] = Field(default=None, description="Whether to enable verbose logging.")
    input_names: Optional[Sequence[str]] = Field(
        default=None, description="Names to assign to the input nodes of the graph."
    )
    output_names: Optional[Sequence[str]] = Field(
        default=None, description="Names to assign to the output nodes of the graph."
    )
    opset_version: Optional[int] = Field(default=13, description="The version of the default (ai.onnx) opset to "
                                                                 "target. Must be >= 7.")
    dynamic_axes: Optional[Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]] = Field(
        default=None, description="Describe the dimensional information about input and output."
    )
    training: Union[str, _C_onnx.TrainingMode] = Field(default=_C_onnx.TrainingMode.EVAL, description="Model mode.")
    do_constant_folding: bool = Field(default=True, description="Whether to execute constant folding for optimization.")

    # dynamo export
    # dynamo: bool = Field(default=False, description="Whether to export the model with torch.export ExportedProgram "
    #                                                 "instead of TorchScript.")
    # dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any, ...], List[Any]]] = Field(
    #     default=None, description="A dictionary or a tuple of dynamic shapes for the model inputs. Note that "
    #                               "dynamic_shapes is designed to be used when the model is exported with dynamo=True, "
    #                               "while dynamic_axes is used when dynamo=False.")
    # report: bool = Field(default=False, description="Whether to generate a markdown report for the export process. "
    #                                                 "This option is only valid when dynamo is True.")
    # optimize: bool = Field(default=True, description="Whether to optimize the exported model. This option is only "
    #                                                  "valid when dynamo is True. ")
    # verify: bool = Field(default=True, description="Whether to verify the exported model using ONNX Runtime. This "
    #                                                "option is only valid when dynamo is True.")
    # profile: bool = Field(default=False, description="Whether to profile the export process. This option is only valid "
    #                                                  "when dynamo is True.")
    # artifacts_dir: Union[str, Path] = Field(
    #     default="deploy_results/onnx/artifacts",
    #     description="The directory to save the debugging artifacts like the report and the serialized exported "
    #                 "program. This option is only valid when dynamo is True."
    # )
    # fallback: bool = Field(default=False, description="Whether to fallback to the TorchScript exporter if the dynamo "
    #                                                   "exporter fails. This option is only valid when dynamo is True. ")


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

class OnnxConfig(BaseModel):
    specific: SpecificOptions = Field(description="Specific options for build in onnx format.")
    plugins: List[Plugin] = Field(default=[], description="List of plugins, which can be connect to model.")
    simplify: bool = Field(default=True, description="Enable simplify onnx model structure.")
    output_file: str = Field(default="weights/onnx/model.onnx", description="Path to save converted model.")
    force_rebuild: bool = Field(description="Forcefully rebuild the existing model.")

    class Config:
        arbitrary_types_allowed = True
