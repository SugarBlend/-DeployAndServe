from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline, KandinskyPriorPipelineOutput
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_prior import (PriorTransformer, CLIPTokenizer, UnCLIPScheduler,
                                                                          CLIPImageProcessor, CLIPTextModelWithProjection,
                                                                          CLIPVisionModelWithProjection)
import torch
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np

from deploy2serve.deployment.core.executors.backends.tensrt import TensorRTExecutor
from deploy2serve.utils.logger import get_project_root
from deploy2serve.deployment.models.common import LoggingMeta


class PriorPipeline(KandinskyV22PriorPipeline, metaclass=LoggingMeta):
    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: CLIPImageProcessor
    ) -> None:
        device = "cuda:0"
        prior.clip_std.data = prior.clip_std.data.to(device)
        prior.clip_mean.data = prior.clip_mean.data.to(device)
        super().__init__(
            prior=prior,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_processor=image_processor
        )
        root = get_project_root()

        self.text_encoder = TensorRTExecutor(
            f"{root}/checkpoints/kandinsky_2/text_encoder/tensorrt/model.plan",
            device, "ERROR"
        )

        self.prior = TensorRTExecutor(
            f"{root}/checkpoints/kandinsky_2/prior_transformer/tensorrt/model.plan",
            device, "ERROR"
        )
        self.prior.config = prior.config
        self.prior.post_process_latents = prior.post_process_latents

        self.image_encoder = TensorRTExecutor(
            f"{root}/checkpoints/kandinsky_2/image_encoder/tensorrt/model.plan",
            device, "ERROR"
        )
        self.image_encoder.config = image_encoder.config
        self.image_encoder.dtype = image_encoder.dtype

        torch.cuda.empty_cache()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PromptEncoder":
        original_pipe = KandinskyV22PriorPipeline.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        instance = cls(
            prior=original_pipe.prior,
            image_encoder=original_pipe.image_encoder,
            text_encoder=original_pipe.text_encoder,
            tokenizer=original_pipe.tokenizer,
            scheduler=original_pipe.scheduler,
            image_processor=original_pipe.image_processor
        )
        return instance

    def get_zero_embed(self, batch_size: int = 1, device: Optional[str] = None) -> torch.Tensor:
        device = device or self.device
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder.infer({"image": zero_img})[1]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb

    def _encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            self.logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder.infer({"input_ids": text_input_ids.to(device)})

        prompt_embeds = text_encoder_output[1]
        text_encoder_hidden_states = text_encoder_output[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)

            negative_prompt_embeds_text_encoder_output = self.text_encoder.infer({"input_ids": uncond_input.input_ids.to(device)})

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output[1]
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> Union[Tuple[torch.Tensor, np.ndarray], KandinskyPriorPipelineOutput]:
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but "
                f"found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # if the negative prompt is defined we double the batch size to
        # directly retrieve the negative prompt embedding
        if negative_prompt is not None:
            prompt = prompt + negative_prompt
            negative_prompt = 2 * negative_prompt

        device = self.text_encoder.device
        # device = self._execution_device

        batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt

        self._guidance_scale = guidance_scale

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, num_images_per_prompt, self.do_classifier_free_guidance, negative_prompt
        )

        # prior
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            predicted_image_embedding = self.prior.infer({
                "latent": latent_model_input, "timestep": t.reshape(-1), "prompt_embeds": prompt_embeds,
                "encoder_hidden_states": text_encoder_hidden_states, "attention_mask": text_mask
            })[0]

            if self.do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + self.guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                text_encoder_hidden_states = callback_outputs.pop(
                    "text_encoder_hidden_states", text_encoder_hidden_states
                )
                text_mask = callback_outputs.pop("text_mask", text_mask)

        latents = self.prior.post_process_latents(latents)

        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is None:
            zero_embeds = self.get_zero_embed(latents.shape[0], device=latents.device)
        else:
            image_embeddings, zero_embeds = image_embeddings.chunk(2)

        self.maybe_free_model_hooks()

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()
            zero_embeds = zero_embeds.cpu().numpy()

        if not return_dict:
            return image_embeddings, zero_embeds

        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)
