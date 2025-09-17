from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.models.export import ExportConfig
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2 import downscale_height_and_width
import torch
from typing import Any, Tuple, Optional, List, Union


class MOVQBatcher(BaseBatcher):
    encoder: KandinskyV22PriorPipeline
    decoder: KandinskyV22Pipeline
    def __init__(self, config: ExportConfig, dataset_name: str, shape: Tuple[int, int]) -> None:
        self.config = config
        self.load_preprocess()
        super().__init__(config, dataset_name, shape)

    def load_preprocess(self) -> None:
        if self.config.enable_mixed_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.encoder = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype,
            cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        )
        self.encoder.to(device=self.config.device, dtype=dtype)

        self.decoder = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=dtype,
            cache_dir="/tmp/kandinsky2",
            low_cpu_mem_usage=True
        )
        self.decoder.to(device=self.config.device, dtype=dtype)

    @torch.no_grad()
    def transformation(self, prompt: str, *args, **kwargs) -> Any:
        num_images_per_prompt: int = 1
        num_inference_steps: int = 25
        guidance_scale: float = 4.0
        height, width = 768, 768
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
        latents: Optional[torch.Tensor] = None

        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        image_embeds, negative_image_embeds = self.encoder(prompt).to_tuple()
        self.decoder._guidance_scale = guidance_scale

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if self.decoder.do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.decoder.unet.dtype, device=self.config.device
            )

        self.decoder.scheduler.set_timesteps(num_inference_steps, device=self.config.device)
        timesteps = self.decoder.scheduler.timesteps

        num_channels_latents = self.decoder.unet.config.in_channels

        height, width = downscale_height_and_width(height, width, self.decoder.movq_scale_factor)

        latents = self.decoder.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            image_embeds.dtype,
            self.config.device,
            generator,
            latents,
            self.decoder.scheduler,
        )

        self.decoder._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.decoder.do_classifier_free_guidance else latents

            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.decoder.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.decoder.do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.decoder.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                    hasattr(self.decoder.scheduler.config, "variance_type")
                    and self.decoder.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.decoder.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        args = [latents]
        return {
            node: args[i] for i, node in enumerate(self.config.input_nodes)
        }
