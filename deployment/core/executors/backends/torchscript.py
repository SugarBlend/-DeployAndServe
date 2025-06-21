from pathlib import Path
from typing import Union, List
import torch

from deployment.core.executors.base import BaseExecutor, ExportConfig


class TorchScriptExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(TorchScriptExecutor, self).__init__(config)

        self.scripted_model = self.load(self.config.torchscript.output_file, self.config.device,
                                        self.config.enable_mixed_precision)

    @staticmethod
    def load(
        torchscript_path: Union[str, Path],
        device: str = "cuda:0",
        enable_mixed_precision: bool = True
    ) -> torch.jit.ScriptModule:
        scripted_model = torch.jit.load(torchscript_path, map_location=device)
        if enable_mixed_precision:
            scripted_model = scripted_model.half()
        return scripted_model

    def infer(self, image: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        outputs = self.scripted_model(image)
        return [outputs]
