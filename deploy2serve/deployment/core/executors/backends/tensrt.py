from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Tuple, Union, Dict

import numpy as np
import tensorrt as trt
import torch
from packaging import version
from pydantic import BaseModel, Field

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend


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
    def __init__(
        self,
        checkpoints_path: str,
        shapes: Dict[str, Tuple[int, ...]],
        device: str,
        log_level: Union[trt.Logger.Severity, str]
    ) -> None:
        self.checkpoints_path: str = checkpoints_path
        self.device: torch.device = torch.device(device)
        self.shapes: Dict[str, Tuple[int, ...]] = shapes

        if isinstance(log_level, trt.Logger.Severity):
            self.log_level: trt.Logger.Severity = log_level
        else:
            self.log_level: trt.Logger.Severity = getattr(trt.Logger, log_level.upper())

        if not Path(self.checkpoints_path).is_absolute():
            self.checkpoints_path = str(Path.cwd().joinpath(self.checkpoints_path))

        self.bindings, self.binding_address, self.context = self.load(
            self.checkpoints_path,
            self.shapes,
            device,
            self.log_level
        )
        self.async_stream = torch.cuda.Stream(device=self.device, priority=-1)

        self.input_nodes: List[str] = []
        for node in self.bindings:
            if self.bindings[node].io_mode == "input":
                self.input_nodes.append(node)

    @staticmethod
    def _make_binding(name: str, dtype: type, shape: List[int], io_mode: str, device: str) -> Binding:
        tensor = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(torch.device(device))
        return Binding(name=name, dtype=dtype, shape=shape, data=tensor, ptr=int(tensor.data_ptr()), io_mode=io_mode)

    @staticmethod
    def load(
        weights_path: Union[str, Path],
        shapes: Dict[str, Tuple[int, ...]],
        device: str,
        log_level: trt.Logger.Severity = trt.Logger.ERROR
    ) -> Tuple[OrderedDict[str, Binding], OrderedDict[str, int], trt.IExecutionContext]:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT model file not found at: '{path}'.")

        logger = trt.Logger(log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with path.open("rb") as file, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(file.read())
        bindings = OrderedDict()

        if version.parse("8.2.5.1") <= version.parse(trt.__version__) <= version.parse("8.6.1"):
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = shapes.get(name, None)
                if not shape:
                    shape = model.get_binding_shape(index)
                io_mode = "input" if model.binding_is_input(index) else "output"
                bindings[name] = TensorRTExecutor._make_binding(name, dtype, shape, io_mode, device)
        elif version.parse(trt.__version__) > version.parse("9.1.0"):
            for index in range(model.num_io_tensors):
                name = model.get_tensor_name(index)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                shape = shapes.get(name, None)
                if not shape:
                    shape = model.get_tensor_shape(name)
                io_mode = "input" if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
                bindings[name] = TensorRTExecutor._make_binding(name, dtype, shape, io_mode, device)
        else:
            raise NotImplementedError(f"Your version of TensorRT: {trt.__version__} is not implemented")

        binding_address = OrderedDict((node, data.ptr) for node, data in bindings.items())
        context = model.create_execution_context()

        return bindings, binding_address, context

    @staticmethod
    def remove_zero_batches(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if tensor.dim() == 0 or tensor.size(0) == 0:
            return tensor

        if tensor.abs().max() <= eps:
            empty_shape = (0,) + tensor.shape[1:]
            return torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)

        first_non_zero = tensor[0].abs().max() > eps
        last_non_zero = tensor[-1].abs().max() > eps

        if first_non_zero and last_non_zero:
            return tensor

        if tensor.dim() == 1:
            non_zero_mask = tensor.abs() > eps
        else:
            if tensor.dtype == torch.bool:
                flattened = tensor.view(tensor.size(0), -1)
                non_zero_mask = flattened.any(dim=1)
            else:
                flattened = tensor.view(tensor.size(0), -1)
                non_zero_mask = flattened.abs().max(dim=1).values > eps

        non_zero_indices = torch.where(non_zero_mask)[0]

        if len(non_zero_indices) == 0:
            empty_shape = (0,) + tensor.shape[1:]
            return torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)

        return tensor[non_zero_indices]

    def infer(self, input_feed: Dict[str, torch.Tensor], asynchronous: bool = False, **kwargs) -> List[torch.Tensor]:
        for idx, node in enumerate(input_feed):
            input_feed[node] = input_feed[node].to(device=self.device, dtype=self.bindings[node].data.dtype)
            if node in self.input_nodes:
                if version.parse(trt.__version__) > version.parse("9.1.0"):
                    self.context.set_input_shape(node, input_feed[node].shape)
                else:
                    self.context.set_binding_shape(idx, input_feed[node].shape)
                self.binding_address[node] = int(input_feed[node].contiguous().data_ptr())

        if asynchronous:
            for node in self.bindings:
                self.context.set_tensor_address(node, self.binding_address[node])
            self.context.execute_async_v3(self.async_stream.cuda_stream)
        else:
            self.context.execute_v2(list(self.binding_address.values()))

        results: List[torch.Tensor] = []
        for node in self.bindings:
            if self.bindings[node].io_mode == "output":
                cleaned_tensor = self.remove_zero_batches(self.bindings[node].data)
                results.append(cleaned_tensor)
        return results
