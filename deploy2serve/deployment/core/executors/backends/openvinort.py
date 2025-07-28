from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig
from deploy2serve.utils.logger import get_project_root


class OpenVINORTExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(OpenVINORTExecutor, self).__init__(config)
        if not Path(self.config.openvino.output_file).is_absolute():
            self.config.openvino.output_file = str(get_project_root().joinpath(self.config.openvino.output_file))

        self.compiled_model = self.load(self.config.openvino.output_file, self.config.openvino.device)

    @staticmethod
    def load(openvino_path: Union[str, Path], device: str) -> "ov.CompiledModel":
        import openvino as ov
        core = ov.Core()
        model = core.read_model(openvino_path)
        compiled_model = core.compile_model(model, device)
        return compiled_model

    def infer(self, image: Union[torch.Tensor, np.ndarray], **kwargs) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        outputs = self.compiled_model(image)
        return [torch.from_numpy(output).to(self.config.device) for output in outputs.values()]
