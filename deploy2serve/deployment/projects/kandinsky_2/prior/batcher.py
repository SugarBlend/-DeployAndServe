from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from diffusers import KandinskyV22PriorPipeline
import random
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import torch
from typing import Any, Optional, List, Union


class PriorBatcher(BaseBatcher):
    tokenizer: CLIPTokenizer
    pipeline: KandinskyV22PriorPipeline

    def load_preprocess(self) -> None:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.pipeline = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype, cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        )
        self.pipeline.to(device=self.config.device, dtype=dtype)
        self.tokenizer = self.pipeline.tokenizer

    def transformation(self, prompt: str, *args, **kwargs) -> Any:
        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        num_images_per_prompt: int = 1
        num_inference_steps: int = 25
        self.pipeline._guidance_scale: float = 4.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
        latents: Optional[torch.Tensor] = None
        negative_prompt: Optional[Union[str, List[str]]] = None

        batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt
        prompt_embeds, text_encoder_hidden_states, text_mask = self.pipeline._encode_prompt(
            prompt, self.config.device, num_images_per_prompt, self.pipeline.do_classifier_free_guidance,
            negative_prompt
        )

        # prior
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.config.device)
        timesteps = self.pipeline.scheduler.timesteps

        embedding_dim = self.pipeline.prior.config.embedding_dim

        latents = self.pipeline.prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            self.config.device,
            generator,
            latents,
            self.pipeline.scheduler,
        )
        self.pipeline._num_timesteps = len(timesteps)
        iterations_restrict = random.choice(timesteps)
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.pipeline.do_classifier_free_guidance else latents

            args = [latent_model_input, torch.tensor([t]), prompt_embeds, text_encoder_hidden_states, text_mask]
            if t == iterations_restrict:
                break
        nodes = list(self.config.input_nodes)
        return {
            nodes[i]: arg for i, arg in enumerate(args)
        }
