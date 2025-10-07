from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Tuple, Union, Dict, Optional

import numpy as np
import tensorrt as trt
import torch
from packaging import version
from pydantic import BaseModel, Field

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend
from deploy2serve.utils.logger import get_logger


class TensorRTAPIAdapter(object):
    def __init__(self, model: trt.ICudaEngine, legacy_mode: bool) -> None:
        self.model: trt.ICudaEngine = model
        self.legacy_mode: bool = legacy_mode

    @property
    def tensor_count(self) -> int:
        return self.model.num_bindings if self.legacy_mode else self.model.num_io_tensors

    def get_name(self, index: int) -> str:
        return self.model.get_binding_name(index) if self.legacy_mode else self.model.get_tensor_name(index)

    def get_dtype(self, index_or_name: Union[str, int]) -> type:
        if self.legacy_mode:
            return trt.nptype(self.model.get_binding_dtype(index_or_name))
        else:
            return trt.nptype(self.model.get_tensor_dtype(index_or_name))

    def get_shape(self, index_or_name: Union[str, int]) -> Tuple[int, ...]:
        if self.legacy_mode:
            return self.model.get_binding_shape(index_or_name)
        else:
            return self.model.get_tensor_shape(index_or_name)

    def is_input(self, index_or_name: Union[str, int]) -> bool:
        if self.legacy_mode:
            return self.model.binding_is_input(index_or_name)
        else:
            return self.model.get_tensor_mode(index_or_name) == trt.TensorIOMode.INPUT


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


class LoggingMixin:
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


@ExecutorFactory.register(Backend.TensorRT)
class TensorRTExecutor(BaseExecutor, LoggingMixin):
    model: trt.ICudaEngine
    context: trt.IExecutionContext
    def __init__(
        self,
        checkpoints_path: str,
        device: str,
        log_level: Union[trt.Logger.Severity, str]
    ) -> None:
        self.checkpoints_path: str = checkpoints_path
        self.device: str = device
        if isinstance(log_level, trt.Logger.Severity):
            self.log_level: trt.Logger.Severity = log_level
        else:
            self.log_level: trt.Logger.Severity = getattr(trt.Logger, log_level.upper())

        if not Path(self.checkpoints_path).is_absolute():
            self.checkpoints_path = str(Path.cwd().joinpath(self.checkpoints_path))

        self.bindings: OrderedDict[str, Binding] = OrderedDict()
        self.binding_address: OrderedDict[str, int] = OrderedDict()
        self.model = self.load(self.checkpoints_path, device, self.log_level)
        self.context = self.get_context()
        self.async_stream = torch.cuda.Stream(device=self.device, priority=-1)

        self.input_nodes: List[str] = []
        self.output_nodes: List[str] = []
        self._initialize_io_nodes()

    def _initialize_io_nodes(self) -> None:
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
        else:
            raise NotImplementedError(f"TensorRT version {trt.__version__} not supported")

        for index in range(adapter.tensor_count):
            name = adapter.get_name(index)
            is_input = adapter.is_input(index if adapter.legacy_mode else name)

            if is_input:
                self.input_nodes.append(name)
            else:
                self.output_nodes.append(name)

    @staticmethod
    def _make_binding(name: str, dtype: type, shape: List[int], io_mode: str, device: str) -> Binding:
        tensor = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(torch.device(device))
        return Binding(name=name, dtype=dtype, shape=shape, data=tensor, ptr=int(tensor.data_ptr()), io_mode=io_mode)

    @staticmethod
    def load(
        weights_path: Union[str, Path],
        device: str,
        log_level: trt.Logger.Severity = trt.Logger.ERROR
    ) -> trt.ICudaEngine:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT model file not found at: '{path}'.")

        logger = trt.Logger(log_level)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with path.open("rb") as file, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(file.read())
        return model

    def get_context(self) -> trt.IExecutionContext:
        return self.model.create_execution_context()

    def update_bindings(self, shapes: Dict[str, Tuple[int, ...]]) -> None:
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
        else:
            raise NotImplementedError(f"Your version of TensorRT: {trt.__version__} is not implemented")

        for index in range(adapter.tensor_count):
            name = adapter.get_name(index)
            if name in shapes:
                if self.bindings.get(name):
                    recreate = shapes[name] != self.bindings[name].shape
                else:
                    recreate = True

                if recreate:
                    dtype = adapter.get_dtype(index if adapter.legacy_mode else name)
                    shape = shapes.get(name)
                    if shape is None:
                        shape = adapter.get_shape(index if adapter.legacy_mode else name)
                    io_mode = "input" if adapter.is_input(index if adapter.legacy_mode else name) else "output"

                    self.bindings[name] = TensorRTExecutor._make_binding(
                        name, dtype, shape, io_mode, self.device
                    )
                    # self.logger.info(f"Successfully installed stub for node: {name}, "
                    #                  f"shape={shape}, dtype={dtype}, io_mode={io_mode}.")
            # else:
            #     self.logger.warning(f"Missing shape for input tensor: '{name}'.")

    def _get_binding_index(self, name: str) -> Optional[int]:
        trt_version = version.parse(trt.__version__)
        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            for idx in range(self.model.num_bindings):
                if self.model.get_binding_name(idx) == name:
                    return idx
        return None

    def _execute_async(self, is_new_api: bool) -> None:
        if is_new_api:
            for node in self.bindings:
                self.context.set_tensor_address(node, self.binding_address[node])
            self.context.execute_async_v3(self.async_stream.cuda_stream)
        else:
            addresses = [self.binding_address.get(self.model.get_binding_name(i), 0)
                         for i in range(self.model.num_bindings)]
            self.context.execute_async_v2(bindings=addresses, stream_handle=self.async_stream.cuda_stream)

    def _get_tensor_dtype(self, name: str, is_new_api: bool) -> type:
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
            idx = self._get_binding_index(name)
            return adapter.get_dtype(idx) if idx is not None else np.float32
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
            return adapter.get_dtype(name)
        else:
            return np.float32

    def _prepare_output_bindings(self, is_new_api: bool, input_feed: Dict[str, torch.Tensor]) -> None:
        for output_node in self.output_nodes:
            if output_node not in self.bindings:
                shape = self._get_output_shape(output_node, is_new_api)
                if shape and all(dim > 0 for dim in shape):
                    dtype = self._get_tensor_dtype(output_node, is_new_api)
                    self.bindings[output_node] = self._make_binding(
                        output_node, dtype, list(shape), "output", self.device
                    )
                    self.binding_address[output_node] = self.bindings[output_node].ptr

    def _get_output_shape(self, name: str, is_new_api: bool) -> Tuple[int, ...]:
        if is_new_api:
            return tuple(self.context.get_tensor_shape(name))
        else:
            idx = self._get_binding_index(name)
            if idx is not None:
                return tuple(self.context.get_binding_shape(idx))
        return tuple()

    def infer(self, input_feed: Dict[str, torch.Tensor], asynchronous: bool = False, **kwargs) -> List[torch.Tensor]:
        input_shapes = {node: input_feed[node].shape for node in input_feed}
        self.update_bindings(input_shapes)

        is_new_api = version.parse(trt.__version__) >= version.parse("9.1.0")
        for node in input_feed:
            input_feed[node] = input_feed[node].to(device=self.device, dtype=self.bindings[node].data.dtype).contiguous()
            self.binding_address[node] = int(input_feed[node].data_ptr())

            if is_new_api:
                self.context.set_input_shape(node, input_feed[node].shape)
            else:
                binding_index = self._get_binding_index(node)
                if binding_index is not None:
                    self.context.set_binding_shape(binding_index, input_feed[node].shape)

        self._prepare_output_bindings(is_new_api, input_feed)

        if asynchronous:
            self._execute_async(is_new_api)
        else:
            self.context.execute_v2(list(self.binding_address.values()))

        return [self.bindings[node].data for node in self.output_nodes]
