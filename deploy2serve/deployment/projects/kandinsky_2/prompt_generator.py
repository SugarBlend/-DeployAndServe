from deploy2serve.deployment.core.exporters.calibration.generators.interface import LabelsGenerator
from pathlib import Path
import ollama
import yaml
from tqdm import trange
from typing import Any, Union, Dict, List


class PromptGenerator(LabelsGenerator):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        super().__init__(dataset_folder)

    def generate_labels(self) -> Dict[str, Any]:
        cache = self.dataset_folder.joinpath("prompts_cache.yml")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            prompts: List[str] = []
            for _ in trange(100):
                prompt = f"""Create 10 unique prompts in txt2img format (Stable Diffusion / SDXL / Kandinsky). Each
                prompt should be a short and concise description of a scene, character or object (cyberpunk, fantasy,
                sci-fi, surrealism, minimalism, retro, realism, portraits, landscapes, architecture). At the end of each
                prompt, add 3-5 visual effects or modifiers separated by commas, for example: photorealism,
                8K resolution, cinematic lighting, depth of field, high contrast, dreamy atmosphere, oil painting,
                watercolor, glitch art, motion blur, tilt-shift. Genre and modifiers should vary between
                prompts. Each prompt should be unique and not similar to the previously generated ones."""
                response = ollama.chat(
                    model="mistral",
                    messages=[{"role": "user", "content": prompt}]
                )
                batch_prompts = response["message"]["content"].splitlines()
                batch_prompts = filter(lambda x: len(x) and len(x) > 100, batch_prompts)
                prompts.extend(batch_prompts)

            with cache.open("w") as file:
                yaml.safe_dump(prompts, file)
        else:
            with cache.open("r") as file:
                prompts = yaml.safe_load(file)

        return {
            "prompts": prompts
        }
