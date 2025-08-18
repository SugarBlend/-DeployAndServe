from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import onnxruntime as ort
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend


ONNX_TYPE_TO_NUMPY = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)': np.float64,
    'tensor(int8)': np.int8,
    'tensor(int16)': np.int16,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
    'tensor(uint8)': np.uint8,
    'tensor(uint16)': np.uint16,
    'tensor(uint32)': np.uint32,
    'tensor(uint64)': np.uint64,
    'tensor(bool)': np.bool_,
    'tensor(string)': np.object_,
}


@ExecutorFactory.register(Backend.ONNX)
class ORTExecutor(BaseExecutor):
    def __init__(self, checkpoints_path: str, device: str) -> None:
        self.checkpoints_path: str = checkpoints_path
        self.device: torch.device = torch.device(device)

        sess_options = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]

        if self.device.type == "cuda":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            provider_options = {
                "device_id": self.device.index,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": True,
                "enable_skip_layer_norm_strict_mode": True,
                "use_tf32": True,
            }
            providers.insert(0, ("CUDAExecutionProvider", provider_options))
        elif self.device.type not in ["cpu", "dml"]:
            raise ValueError(f"Doesn't found realization for device backend: {self.device.type}")

        if not Path(self.checkpoints_path).is_absolute():
            self.checkpoints_path = str(Path.cwd().joinpath(self.checkpoints_path))

        self.session, self.input_names, self.output_names = self.load(
            self.checkpoints_path, sess_options, providers
        )
        self.binding: ort.IOBinding = self.session.io_binding()
        self.outputs: List[Union[ort.OrtValue, torch.Tensor]] = []

    @staticmethod
    def load(
        weights_path: Union[str, Path],
        sess_options: Optional[ort.SessionOptions],
        providers: Optional[Sequence[Union[str, Tuple[str, Dict[Any, Any]]]]],
    ) -> Tuple[ort.InferenceSession, List[str], List[str]]:
        available_providers = ort.get_available_providers()

        for provider in providers:
            if isinstance(provider, (List, Tuple)):
                provider_name = provider[0]
            else:
                provider_name = provider
            if provider_name not in available_providers:
                raise RuntimeError(f"Provider: '{provider_name}' is not available in your onnxruntime package, "
                                   f"reinstall wright version. At your package available providers: "
                                   f"{available_providers=}")

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"ONNX model file not found at: '{weights_path}'.")

        inference_session = ort.InferenceSession(weights_path, sess_options=sess_options, providers=providers)
        input_names = [node.name for node in inference_session.get_inputs()]
        output_names = [node.name for node in inference_session.get_outputs()]

        return inference_session, input_names, output_names

    def _get_input_dtypes(self) -> Dict[str, np.dtype]:
        return {
            input.name: ONNX_TYPE_TO_NUMPY.get(input.type)
            for input in self.session.get_inputs()
        }

    def _create_binding(
        self,
        inference_session: ort.InferenceSession,
        inputs: List[Union[np.ndarray, torch.Tensor]]
    ) -> None:
        if not inputs:
            raise ValueError("Length of list with input tensors must be greater than 0.")

        if not isinstance(inputs[0], (np.ndarray, torch.Tensor)):
            raise TypeError("In bind mode, you can create bindings to only two types of data - 'np.ndarray', "
                            f"'torch.Tensor'. But actual type: '{type(inputs[0])}'.")

        device_name, device_id = self.device.type, self.device.index or 0
        batch_size = inputs[0].shape[0]
        output_shapes = [
            (batch_size, *node.shape[1:]) for node in inference_session.get_outputs()
        ]

        is_same_shapes = all(
            (batch_size, *output.shape[1:]) == output_shapes[idx] for idx, output in enumerate(self.outputs)
        )
        if not self.outputs or not is_same_shapes:
            self.binding.clear_binding_outputs()
            self.outputs.clear()

            for node in inference_session.get_outputs():
                raw_placeholder = np.zeros((batch_size, *node.shape[1:]), dtype=ONNX_TYPE_TO_NUMPY[node.type])
                if isinstance(inputs[0], np.ndarray):
                    placeholder = ort.OrtValue.ortvalue_from_numpy(raw_placeholder, device_name, device_id)
                else:
                    placeholder = torch.from_numpy(raw_placeholder).to(self.device).contiguous()

                self.binding.bind_output(
                    name=node.name,
                    device_type=device_name,
                    device_id=device_id,
                    element_type=raw_placeholder.dtype,
                    shape=tuple(raw_placeholder.shape),
                    buffer_ptr=placeholder.data_ptr(),
                )
                self.outputs.append(placeholder)

        self.binding.clear_binding_inputs()
        for idx, node in enumerate(inference_session.get_inputs()):
            if isinstance(inputs[idx], np.ndarray):
                placeholder = ort.OrtValue.ortvalue_from_numpy(inputs[idx], device_name, device_id)
                shape = tuple(placeholder.shape())
            else:
                dtype = getattr(torch, ONNX_TYPE_TO_NUMPY[node.type].__name__)
                placeholder = inputs[idx].to(device=self.device, dtype=dtype).contiguous()
                shape = tuple(placeholder.shape)

            self.binding.bind_input(
                name=node.name,
                device_type=device_name,
                device_id=device_id,
                element_type=ONNX_TYPE_TO_NUMPY[node.type],
                shape=shape,
                buffer_ptr=placeholder.data_ptr(),
            )

    def infer(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor]]],
        with_binding: bool = True,
        **kwargs
    ) -> List[torch.Tensor]:
        input_dtypes = self._get_input_dtypes()
        if not with_binding:
            def convert_input(name, value):
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                expected_dtype = input_dtypes.get(name)
                if expected_dtype is not None and value.dtype != expected_dtype:
                    value = value.astype(expected_dtype)
                return value

            if isinstance(inputs, (torch.Tensor, np.ndarray)):
                name = self.input_names[0]
                input_feed = {
                    name: convert_input(name, inputs)
                }
            elif isinstance(inputs, Dict):
                input_feed = {
                    name: convert_input(name, value) for name, value in inputs.items()
                }
            else:
                raise TypeError(f"Unsupported input type {type(inputs)}")

            outputs = self.session.run(output_names=self.output_names, input_feed=input_feed)
            return [torch.from_numpy(output).to(self.device) for output in outputs]
        else:
            inputs = list(inputs.values()) if isinstance(inputs, Dict) else [inputs]
            self._create_binding(self.session, inputs)
            self.session.run_with_iobinding(self.binding)
            if isinstance(self.outputs[0], ort.OrtValue):
                return [torch.from_numpy(tensor.numpy()).to(device=self.device) for tensor in self.outputs]
            else:
                return self.outputs
