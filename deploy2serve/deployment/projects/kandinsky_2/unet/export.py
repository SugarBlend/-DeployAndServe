from contextlib import contextmanager
from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.projects.kandinsky_2.unet.batcher import UnetBatcher
from diffusers import KandinskyV22Pipeline
import tensorrt as trt
import torch
from typing import Any, Generator, Optional


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        return None

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def register_batcher(self) -> Optional[BaseBatcher]:
        input_node = list(self.config.input_nodes)[0]
        batch, _, h, w = self.config.input_nodes[input_node]["shape"]
        return UnetBatcher(self.config, "kandinsky-unet", (h, w))

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        return network


class UnetWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=torch.float16, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        ).unet

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, t: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        noise_prediction = self.model(
            sample=latent,
            timestep=t,
            encoder_hidden_states=None,
            added_cond_kwargs={"image_embeds": image_embeds},
            return_dict=False,
        )[0]
        return noise_prediction


class UnetExporter(Exporter):
    def load_checkpoints(self, config_path: str, weights_path: str) -> torch.nn.Module:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = UnetWrapper()
        model.to(device=self.config.device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
