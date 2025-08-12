from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import onnxruntime as ort
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig, ExecutorFactory
from deploy2serve.deployment.models.common import Backend


@ExecutorFactory.register(Backend.ONNX)
class ORTExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(ORTExecutor, self).__init__(config)
        ort.preload_dlls()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider_options = {
            "device_id": torch.device(self.config.device).index,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
            "enable_cuda_graph": False,
            "enable_skip_layer_norm_strict_mode": True,
            "use_tf32": True,
        }

        if not Path(self.config.onnx.output_file).is_absolute():
            self.config.onnx.output_file = str(Path.cwd().joinpath(self.config.onnx.output_file))

        self.session, self.input_names, self.output_names = self.load(
            self.config.onnx.output_file, sess_options, ["CUDAExecutionProvider"], [provider_options]
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

    def infer(
            self,
            inputs: Union[torch.Tensor, np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor]]],
            **kwargs
    ) -> List[torch.Tensor]:
        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            input_feed = {self.input_names[0]: inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs}
        elif isinstance(inputs, Dict):
            input_feed = {name: (val.cpu().numpy() if isinstance(val, torch.Tensor) else val) for name, val in
                          inputs.items()}
        else:
            raise TypeError(f"Unsupported input type {type(inputs)}")

        outputs = self.session.run(output_names=self.output_names, input_feed=input_feed)
        return [torch.from_numpy(output).to(self.config.device) for output in outputs]
