from contextlib import contextmanager
from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.projects.kandinsky_2.movq.batcher import MOVQBatcher
from diffusers import KandinskyV22Pipeline
import tensorrt as trt
import torch
from typing import Any, Generator, Optional


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        return MOVQBatcher(self.config, "kandinsky-movq", 1)

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        return MOVQBatcher(self.config, "kandinsky-movq", 1)

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        return network


class MOVQDecodeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: torch.nn.Module = model

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        return self.model.decode(quant)[0]


class MOVQExporter(Exporter):
    def load_checkpoints(self, config_path: str, weights_path: str) -> torch.nn.Module:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=dtype, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        ).movq
        return MOVQDecodeWrapper(model).to(device=self.config.device, dtype=dtype)
