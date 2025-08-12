from pathlib import Path
from typing import List, Union, TYPE_CHECKING

import numpy as np
import torch

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExportConfig, ExecutorFactory
from deploy2serve.deployment.models.common import Backend

if TYPE_CHECKING:
    from openvino import CompiledModel


@ExecutorFactory.register(Backend.OpenVINO)
class OpenVINORTExecutor(BaseExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(OpenVINORTExecutor, self).__init__(config)
        if not Path(self.config.openvino.output_file).is_absolute():
            self.config.openvino.output_file = str(Path.cwd().joinpath(self.config.openvino.output_file))

        self.compiled_model = self.load(self.config.openvino.output_file, self.config.openvino.device)

    @staticmethod
    def load(weights_path: Union[str, Path], device: str) -> "CompiledModel":
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"OpenVINO model file not found at: '{path}'.")

        try:
            from openvino import Core
        except ImportError:
            raise ImportError("Please install OpenVINO to use 'OpenVINORTExecutor'.")

        core = Core()
        cache_dir = Path("~/.openvino_cache").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})

        model = core.read_model(weights_path)
        compiled_model = core.compile_model(model, device)
        return compiled_model

    def infer(self, image: Union[torch.Tensor, np.ndarray], **kwargs) -> List[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        outputs = self.compiled_model(image)
        return [torch.from_numpy(output).to(self.config.device) for output in outputs.values()]
