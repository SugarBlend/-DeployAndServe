from contextlib import contextmanager
from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.projects.kandinsky_2.prior.batcher import PriorBatcher
from diffusers import KandinskyV22PriorPipeline
import tensorrt as trt
import torch
from typing import Any, Generator, Optional


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        input_node = list(self.config.input_nodes)[0]
        ch, sequence_len = self.config.input_nodes[input_node]["shape"]
        return PriorBatcher(self.config, "kandinsky-prior", (ch, sequence_len))

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        input_node = list(self.config.input_nodes)[0]
        ch, sequence_len = self.config.input_nodes[input_node]["shape"]
        return PriorBatcher(self.config, "kandinsky-prior", (ch, sequence_len))

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        return network


class PriorExporter(Exporter):
    def load_checkpoints(self, config_path: str, weights_path: str) -> torch.nn.Module:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        ).prior
        model.to(device=self.config.device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
