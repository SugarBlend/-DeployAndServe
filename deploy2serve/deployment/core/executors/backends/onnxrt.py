import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
from typing import Any, List, Optional, Sequence, Tuple, Union

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig
from deploy2serve.utils.logger import get_project_root


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
            "enable_cuda_graph": True,
            "enable_skip_layer_norm_strict_mode": True,
            "use_tf32": True,
        }

        if not Path(self.config.onnx.output_file).is_absolute():
            self.config.onnx.output_file = str(get_project_root().joinpath(self.config.onnx.output_file))

        self.session, self.input_names, self.output_names = self.load(
            self.config.onnx.output_file, sess_options, ["CUDAExecutionProvider"], [provider_options]
        )

    @staticmethod
    def load(
        onnx_path: Union[str, Path],
        sess_options: Optional[ort.SessionOptions],
        providers: Optional[Sequence[str | tuple[str, dict[Any, Any]]]],
        provider_options: Optional[Sequence[dict[Any, Any]]],
    ) -> Tuple[ort.InferenceSession, List[str], List[str]]:
        inference_session = ort.InferenceSession(onnx_path, sess_options, providers, provider_options)
        input_names = [inp.name for inp in inference_session.get_inputs()]
        output_names = [out.name for out in inference_session.get_outputs()]
        return inference_session, input_names, output_names

    def infer(self, image: Union[torch.Tensor, np.ndarray], **kwargs) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        outputs = self.session.run(output_names=self.output_names, input_feed={self.input_names[0]: image})
        return [torch.from_numpy(output).to(self.config.device) for output in outputs]
