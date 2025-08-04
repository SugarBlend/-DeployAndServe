from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from pydantic import BaseModel, Field
from ultralytics.utils.checks import check_version

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig, ExecutorFactory
from deploy2serve.deployment.models.common import Backend
from deploy2serve.utils.logger import get_project_root


class Binding(BaseModel):
    name: str = Field(description="Node name.")
    dtype: type = Field(description="Type of node tensor.")
    shape: Union[Tuple[int, ...], List[int]] = Field(description="Shape of node tensor.")
    data: torch.Tensor = Field(description="Pytorch tensor pinned for current name of node.")
    ptr: int = Field(description="Address of current named tensor on gpu.")
    io_mode: Literal["output", "input"] = Field(description="Type of node (input / output).")

    def __init__(
        self,
        name: str,
        dtype: type,
        shape: Union[List[int], Tuple[int, ...]],
        data: torch.Tensor,
        ptr: int,
        io_mode: str,
    ) -> None:
        super().__init__(name=name, dtype=dtype, shape=shape, data=data, ptr=ptr, io_mode=io_mode)

    class Config:
        arbitrary_types_allowed = True


@ExecutorFactory.register(Backend.TensorRT)
class TensorRTExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(TensorRTExecutor, self).__init__(config)

        if not Path(self.config.tensorrt.output_file).is_absolute():
            self.config.tensorrt.output_file = str(get_project_root().joinpath(self.config.tensorrt.output_file))

        self.bindings, self.binding_address, self.context = self.load(
            self.config.tensorrt.output_file,
            self.config.tensorrt.specific.profile_shapes[0]["max"][0],
            self.config.device,
            self.config.tensorrt.specific.log_level,
        )
        self.async_stream = torch.cuda.Stream(device=config.device, priority=-1)
        for node in self.bindings:
            if self.bindings[node].io_mode == "input":
                self.input_name = node
                break

    @staticmethod
    def _make_binding(name: str, dtype: type, shape: List[int], io_mode: str, device: str) -> Binding:
        tensor = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(torch.device(device))
        return Binding(name=name, dtype=dtype, shape=shape, data=tensor, ptr=int(tensor.data_ptr()), io_mode=io_mode)

    @staticmethod
    def load(
        weights_path: Union[str, Path], max_batch: int, device: str, log_level: trt.Logger.Severity = trt.Logger.ERROR
    ) -> Tuple[OrderedDict[str, Binding], OrderedDict[str, int], trt.IExecutionContext]:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT model file not found at: '{path}'.")

        logger = trt.Logger(log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(weights_path, "rb") as file, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(file.read())
        bindings = OrderedDict()

        if check_version(trt.__version__, "<=8.6.1") and check_version(trt.__version__, ">=8.2.5.1"):
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = (max_batch, *model.get_binding_shape(index)[1:])
                io_mode = "input" if model.binding_is_input(index) else "output"
                bindings[name] = TensorRTExecutor._make_binding(name, dtype, shape, io_mode, device)
        elif check_version(trt.__version__, ">9.1.0"):
            for index in range(model.num_io_tensors):
                name = model.get_tensor_name(index)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                shape = (max_batch, *model.get_tensor_shape(name)[1:])
                io_mode = "input" if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
                bindings[name] = TensorRTExecutor._make_binding(name, dtype, shape, io_mode, device)
        else:
            raise NotImplementedError(f"Your version of TensorRT: {trt.__version__} is not implemented")

        binding_address = OrderedDict((node, data.ptr) for node, data in bindings.items())
        context = model.create_execution_context()

        return bindings, binding_address, context

    def infer(self, image: torch.Tensor, asynchronous: bool = False, **kwargs) -> List[torch.Tensor]:
        if check_version(trt.__version__, ">9.1.0"):
            self.context.set_input_shape(self.input_name, image.shape)
        else:
            self.context.set_binding_shape(0, image.shape)
        self.binding_address[self.input_name] = int(image.contiguous().data_ptr())

        if asynchronous:
            for node in self.bindings:
                self.context.set_tensor_address(node, self.binding_address[node])
            self.context.execute_async_v3(self.async_stream.cuda_stream)
        else:
            self.context.execute_v2(list(self.binding_address.values()))

        return [self.bindings[node].data for node in self.bindings if self.bindings[node].io_mode == "output"]
