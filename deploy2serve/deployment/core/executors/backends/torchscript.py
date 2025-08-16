from pathlib import Path
from typing import List, Union
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend


@ExecutorFactory.register(Backend.TorchScript)
class TorchScriptExecutor(BaseExecutor):
    def __init__(self, checkpoints_path: str, device: str, enable_mixed_precision: bool) -> None:
        self.checkpoints_path: str = checkpoints_path
        self.device: torch.device = torch.device(device)
        self.enable_mixed_precision: bool = enable_mixed_precision

        if not Path(self.checkpoints_path).is_absolute():
            self.checkpoints_path = str(Path.cwd().joinpath(self.checkpoints_path))

        self.scripted_model = self.load(
            self.checkpoints_path, f"{self.device.type}:{self.device.index}", self.enable_mixed_precision
        )

    @staticmethod
    def load(
        weights_path: Union[str, Path], device: str = "cuda:0", enable_mixed_precision: bool = True
    ) -> torch.jit.ScriptModule:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"TorchScript model file not found at: '{path}'.")

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        scripted_model = torch.jit.load(weights_path, map_location=device)
        if enable_mixed_precision and device != "cpu":
            scripted_model = scripted_model.half()
        return scripted_model

    @torch.no_grad()
    def infer(self, image: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        if self.enable_mixed_precision:
            image = image.half()
        outputs = self.scripted_model(image)
        return [outputs]
