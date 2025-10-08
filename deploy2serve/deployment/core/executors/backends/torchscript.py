from pathlib import Path
from typing import List, Union, Dict
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend


@ExecutorFactory.register(Backend.TorchScript)
class TorchScriptExecutor(BaseExecutor):
    def __init__(self, checkpoints_path: Path, device: str, enable_mixed_precision: bool) -> None:
        self.checkpoints_path: Path = checkpoints_path
        self.device: torch.device = torch.device(device)
        self.enable_mixed_precision: bool = enable_mixed_precision
        self.scripted_model = self.load(
            self.checkpoints_path, device, self.enable_mixed_precision
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
    def infer(self, input_feed: Dict[str, torch.Tensor], **kwargs) -> List[torch.Tensor]:
        if self.enable_mixed_precision:
            for node in input_feed.keys():
                input_feed[node] = input_feed[node].half()
        outputs = self.scripted_model(*list(input_feed.values()))
        return [outputs]
