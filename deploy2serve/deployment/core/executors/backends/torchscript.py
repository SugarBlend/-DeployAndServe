from pathlib import Path
from typing import List, Union

import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig
from deploy2serve.utils.logger import get_project_root


class TorchScriptExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(TorchScriptExecutor, self).__init__(config)
        if not Path(self.config.torchscript.output_file).is_absolute():
            self.config.torchscript.output_file = str(get_project_root().joinpath(self.config.torchscript.output_file))

        self.scripted_model = self.load(
            self.config.torchscript.output_file, self.config.device, self.config.enable_mixed_precision
        )

    @staticmethod
    def load(
        torchscript_path: Union[str, Path], device: str = "cuda:0", enable_mixed_precision: bool = True
    ) -> torch.jit.ScriptModule:
        scripted_model = torch.jit.load(torchscript_path, map_location=device)
        if enable_mixed_precision:
            scripted_model = scripted_model.half()
        return scripted_model

    def infer(self, image: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        outputs = self.scripted_model(image)
        return [outputs]
