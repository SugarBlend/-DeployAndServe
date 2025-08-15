from contextlib import contextmanager
from pathlib import Path
import mmengine.runner.checkpoint
from mmpose.apis import init_model as init_pose_estimator
import tensorrt as trt
import torch
from typing import Optional, Any, Generator, Type

from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.projects.sapiens.batcher import PoseBatcher
from deploy2serve.utils.logger import get_logger, get_project_root


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        super().__init__(config, model)

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module) -> None:
        super().__init__(config, model)

    def register_batcher(self) -> Optional[Type[BaseBatcher]]:
        return PoseBatcher(self.config, "sapiens", self.config.input_shape, self.model.cfg)

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        return network


class SapiensExporter(Exporter):
    def __init__(self, config: ExportConfig) -> None:
        super().__init__(config)
        self.logger = get_logger("onnx")

    def load_checkpoints(self, weights_path: str, config_path: str) -> None:
        def patched_load_checkpoint(filename, map_location=None, logger=None) -> Any:  # noqa: ARG001, ANN001, ANN401
            return torch.load(filename, map_location=map_location, weights_only=False)

        mmengine.runner.checkpoint._load_checkpoint = patched_load_checkpoint  # noqa: SLF001
        if not Path(config_path).is_absolute():
            config_path = str(get_project_root().joinpath(config_path))

        model = init_pose_estimator(config_path, weights_path, device=self.config.device)
        model.eval()
        model.to(self.config.device)
        dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
        model.to(dtype)
        model.test_cfg.flip_test = False
        self.model = model
