from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from diffusers import KandinskyV22PriorPipeline
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import torch
from typing import Any


class CLIPBatcher(BaseBatcher):
    tokenizer: CLIPTokenizer

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
        self.tokenizer = pipeline.tokenizer

    def transformation(self, prompt: str, *args, **kwargs) -> Any:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
            self.logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        return {
            node: text_input_ids for node in self.config.input_nodes
        }
