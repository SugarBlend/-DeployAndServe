from contextlib import contextmanager
from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.projects.kandinsky_2.clip.batcher import CLIPBatcher
from diffusers import KandinskyV22PriorPipeline
import tensorrt as trt
import torch
from typing import Any, Generator, Type, Optional


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        pass

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        input_node = list(self.config.input_nodes)[0]
        batch, tokens = self.config.input_nodes[input_node]["shape"]
        return CLIPBatcher(self.config, "kandinsky-clip", (batch, tokens))

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        return network


class CLIPExporter(Exporter):
    def load_checkpoints(self, config_path: str, weights_path: str) -> torch.nn.Module:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        ).text_encoder
        return model.to(device=self.config.device, dtype=dtype)
