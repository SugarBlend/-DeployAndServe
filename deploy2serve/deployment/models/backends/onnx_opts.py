from pydantic import BaseModel, Field, field_validator
from torch.onnx import _C_onnx
from typing import Any, List, Mapping, Optional, Sequence, Union

from deploy2serve.deployment.models.common import Plugin, ModelMeta


class NvidiaModelOpt(BaseModel, metaclass=ModelMeta):
    quant_mode: str = Field(description="Quantization mode. One of 'int8', 'int4' and 'fp8'.")
    calib_method: str = Field(
        description="Calibration method choices. Options are int8/fp8: {'entropy' (default), 'max'} and "
                    "int4: {'awq_clip' (default), 'awq_lite', 'awq_full', 'rtn_dq'}."
    )
    calibration_eps: List[str] = Field(
        default=["cuda"], description="Priority order for the execution providers (EP) to calibrate the model. "
                                      "Any subset of ['NvTensorRtRtx', 'trt', 'cuda', 'dml', 'cpu']."
    )
    op_types_to_quantize: Optional[List[str]] = Field(
        default=["MatMul", "Conv"], description="List of op types to quantize. If None, all supported operators are "
                                                "quantized. This flag does not support regular expression."
    )
    nodes_to_exclude: List[str] = Field(default=[r"/Shape"],
                                        description="List of node names to exclude from quantization.")
    qdq_for_weights: bool = Field(default=False, description="If True, only add DQ nodes to the model. If False, add "
                                                             "Q/DQ nodes to the model.")
    use_external_data_format: bool = Field(default=False, description="If True, separate data path will be used to "
                                                                      "store the weights of the quantized model.")
    calib_buffer: Optional[int] = Field(default=None, description="Frame limit for calibration stage.")
    keep_intermediate_files: bool = Field(default=False,
                                          description="If True, keep all intermediate files generated during the "
                                                      "ONNX model's conversion/calibration.")

    @field_validator("quant_mode", mode="before")
    def parse_quant_mode(cls, val: str) -> str:
        available_modes = ["int8", "int4", "fp8"]
        if val not in available_modes:
            raise Exception(f"Unknown quantization mode: {val}. Available modes: {available_modes}.")
        return val

    @field_validator("calib_method", mode="before")
    def parse_calib_method(cls, val: str) -> str:
        available_calib_methods = ["entropy", "max", "awq_clip", "awq_lite", "awq_full", "rtn_dq"]
        if val not in available_calib_methods:
            raise Exception(f"Unknown calib method: {val}. Available methods: {available_calib_methods}.")
        return val

    @field_validator("calibration_eps", mode="before")
    def parse_calibration_eps(cls, val: List[str]) -> List[str]:
        import onnxruntime as ort
        correspondence = {
            "trt": "TensorrtExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider"
        }
        available_eps = ort.get_available_providers()
        for i in range(len(val) - 1, -1, -1):
            ep = correspondence.get(val[i])
            if not ep or ep not in available_eps:
                cls.logger.warning(f"It is not possible to use this provider: '{val[i]}' because the installed package "
                                   f"does not support it.")
                val.pop(i)
        return val

class SpecificOptions(BaseModel):
    keep_initializers_as_inputs: bool = Field(
        default=False,
        description="If True, all the initializers (typically corresponding to model weights) in the "
        "exported graph will also be added as inputs to the graph.",
    )
    export_params: bool = Field(default=True, description="If specified, all parameters will be exported.")
    verbose: Optional[bool] = Field(default=None, description="Whether to enable verbose logging.")
    input_names: Optional[Sequence[str]] = Field(
        default=None, description="Names to assign to the input nodes of the graph."
    )
    output_names: Optional[Sequence[str]] = Field(
        default=None, description="Names to assign to the output nodes of the graph."
    )
    opset_version: Optional[int] = Field(
        default=13, description="The version of the default (ai.onnx) opset to " "target. Must be >= 7."
    )
    dynamic_axes: Optional[Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]] = Field(
        default=None, description="Describe the dimensional information about input and output."
    )
    training: Union[str, _C_onnx.TrainingMode] = Field(default=_C_onnx.TrainingMode.EVAL, description="Model mode.")
    do_constant_folding: bool = Field(default=True, description="Whether to execute constant folding for optimization.")

    # TODO: Not yet tested on linux platform
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
    modelopt: Optional[NvidiaModelOpt] = Field(
        default=None, description="This ONNX PTQ Toolkit provides a comprehensive suite of tools designed to optimize "
                                  "ONNX (Open Neural Network Exchange) models through quantization."
    )
    plugins: List[Plugin] = Field(default=[], description="List of plugins, which can be connect to model.")
    simplify: bool = Field(default=True, description="Enable simplify onnx model structure.")
    output_file: str = Field(default="weights/onnx/model.onnx", description="Path to save converted model.")
    force_rebuild: bool = Field(description="Forcefully rebuild the existing model.")

    class Config:
        arbitrary_types_allowed = True
