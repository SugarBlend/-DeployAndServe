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

        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        sess_options = ort.SessionOptions()
        provider_options = {}

        if self.device.type == "cuda":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            provider_options = {
                "device_id": self.device.index,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
                "enable_skip_layer_norm_strict_mode": True,
                "use_tf32": True,
            }
            providers = ["CUDAExecutionProvider"]
        elif self.device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Doesn't found realization for device backend: {self.device.type}")

        if not Path(self.checkpoints_path).is_absolute():
            self.checkpoints_path = str(Path.cwd().joinpath(self.checkpoints_path))

        self.session, self.input_names, self.output_names = self.load(
            self.checkpoints_path, sess_options, providers, [provider_options]
        )

    @staticmethod
    def load(
        weights_path: Union[str, Path],
        sess_options: Optional[ort.SessionOptions],
        providers: Optional[Sequence[Union[str, Tuple[str, Dict[Any, Any]]]]],
        provider_options: Optional[Sequence[Dict[Any, Any]]],
    ) -> Tuple[ort.InferenceSession, List[str], List[str]]:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: '{path}'.")

        inference_session = ort.InferenceSession(weights_path, sess_options, providers, provider_options)
        input_names = [inp.name for inp in inference_session.get_inputs()]
        output_names = [out.name for out in inference_session.get_outputs()]
        return inference_session, input_names, output_names

    def get_input_dtypes(self) -> Dict[str, np.dtype]:
        return {
            input.name: ONNX_TYPE_TO_NUMPY.get(input.type)
            for input in self.session.get_inputs()
        }

    def infer(
            self,
            inputs: Union[torch.Tensor, np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor]]],
            **kwargs
    ) -> List[torch.Tensor]:
        input_dtypes = self.get_input_dtypes()

        def convert_input(name, value):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            expected_dtype = input_dtypes.get(name)
            if expected_dtype is not None and value.dtype != expected_dtype:
                value = value.astype(expected_dtype)
            return value

        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            input_feed = {
                self.input_names[0]: convert_input(self.input_names[0], inputs)
            }
        elif isinstance(inputs, dict):
            input_feed = {
                name: convert_input(name, val)
                for name, val in inputs.items()
            }
        else:
            raise TypeError(f"Unsupported input type {type(inputs)}")

        outputs = self.session.run(output_names=self.output_names, input_feed=input_feed)
        return [torch.from_numpy(output).to(self.device) for output in outputs]
