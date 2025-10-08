from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from diffusers import KandinskyV22PriorPipeline
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
import torch
from typing import Any


class ImageEncoderBatcher(BaseBatcher):
    image_encoder: CLIPVisionModelWithProjection

    def load_preprocess(self) -> None:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        pipeline = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        )
        self.image_encoder = pipeline.image_encoder

    def transformation(self, prompt: str, *args, **kwargs) -> Any:
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=self.config.device, dtype=self.image_encoder.dtype
        )
        return {
            "image": zero_img
        }
