import click
import cv2
from enum import Enum
import numpy as np
import yaml
import time
import torch
from tqdm import tqdm
from typing import List, Type, Tuple
from pathlib import Path
from diffusers.pipelines import DiffusionPipeline
from deploy2serve.utils.logger import get_project_root


class PipelineVersion(str, Enum):
    TRT = "trt"
    HF = "hf"


def get_pipelines(version: str) -> Tuple[Type[DiffusionPipeline], Type[DiffusionPipeline]]:
    if version == PipelineVersion.TRT:
        from deploy2serve.deployment.projects.kandinsky_2.pipelines import PriorPipeline, Pipeline
    else:
        from diffusers.pipelines.kandinsky2_2 import (KandinskyV22Pipeline as Pipeline,
                                                      KandinskyV22PriorPipeline as PriorPipeline)
    return Pipeline, PriorPipeline


def load_prompts(prompts_path: Path) -> List[str]:
    if not prompts_path.exists():
        raise click.FileError(str(prompts_path), hint="Prompts file not found.")

    with prompts_path.open("r") as file:
        prompts = yaml.safe_load(file)

    if not isinstance(prompts, list):
        raise click.BadParameter("Prompts file should contain a list of strings.")
    return prompts


def setup_pipelines(
    prior_model: str,
    decoder_model: str,
    version: str,
    cache_dir: str,
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[DiffusionPipeline, DiffusionPipeline]:
    pipeline_cls, prior_pipeline_cls = get_pipelines(version)

    click.echo(f"Loading {version.upper()} pipelines...")

    prior_pipe = prior_pipeline_cls.from_pretrained(
        prior_model,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    ).to(device="cuda:0")

    pipe = pipeline_cls.from_pretrained(
        decoder_model,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    ).to(device="cuda:0")

    return prior_pipe, pipe


def process_prompts(
    prompts: List[str],
    prior_pipe: DiffusionPipeline,
    pipe: DiffusionPipeline,
    height: int = 768,
    width: int = 768,
    num_inference_steps: int = 50,
    show_preview: bool = True
) -> float:
    images: List[np.ndarray] = []
    with tqdm(prompts, desc="Generating images") as pbar:
        for prompt in pbar:
            click.echo(prompt)
            image_emb, negative_image_emb = prior_pipe(prompt).to_tuple()

            result = pipe(
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                output_type="np"
            ).images

            if show_preview:
                cv2.imshow("Generated Image", result[0])
                cv2.waitKey(30)
            images.extend(images)
    return images


@click.command()
@click.option(
    "--version",
    type=click.Choice([PipelineVersion.TRT, PipelineVersion.HF], case_sensitive=False),
    default=PipelineVersion.TRT,
    show_default=True,
    help="Pipeline version to use."
)
@click.option(
    "--prior-model",
    default="kandinsky-community/kandinsky-2-2-prior",
    show_default=True,
    help="HuggingFace model ID for prior pipeline."
)
@click.option(
    "--decoder-model",
    default="kandinsky-community/kandinsky-2-2-decoder",
    show_default=True,
    help="HuggingFace model ID for decoder pipeline."
)
@click.option(
    "--prompts-file",
    type=click.Path(exists=True, path_type=Path),
    default=lambda: get_project_root().joinpath("checkpoints/calibration_dataset/prompts.yaml"),
    show_default=True,
    help="Path to YAML file with prompts."
)
@click.option(
    "--cache-dir",
    default="/tmp/kandinsky2",
    show_default=True,
    help="Cache directory for models."
)
@click.option(
    "--height",
    default=768,
    show_default=True,
    help="Output image height."
)
@click.option(
    "--width",
    default=768,
    show_default=True,
    help="Output image width."
)
@click.option(
    "--num-inference-steps",
    default=50,
    show_default=True,
    help="Number of denoising steps."
)
@click.option(
    "--torch-dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    show_default=True,
    help="Torch data type."
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Disable image preview."
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Directory to save generated images."
)
def main(
    version: str,
    prior_model: str,
    decoder_model: str,
    prompts_file: Path,
    cache_dir: str,
    height: int,
    width: int,
    num_inference_steps: int,
    torch_dtype: str,
    no_preview: bool,
    output_dir: Path
):
    # Convert torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[torch_dtype]

    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        click.echo("Loading prompts...")
        prompts = load_prompts(prompts_file)
        click.echo(f"Loaded {len(prompts)} prompts")

        prior_pipe, pipe = setup_pipelines(
            prior_model, decoder_model, version, cache_dir, torch_dtype
        )

        click.echo("Starting image generation...")
        start_time = time.perf_counter()

        images = process_prompts(
            prompts=prompts,
            prior_pipe=prior_pipe,
            pipe=pipe,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            show_preview=not no_preview
        )

        elapsed_time = time.perf_counter() - start_time
        images_per_second = len(prompts) / elapsed_time

        if output_dir:
            click.echo(f"Saving images to {output_dir}...")
            for i, (prompt, image) in enumerate(zip(prompts, images)):
                filename = output_dir / f"image_{i:03d}.png"
                cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        click.echo("Generation completed")
        click.echo(f"Total time: {elapsed_time:.2f}s")
        click.echo(f"Performance: {images_per_second:.2f} images/second")
        click.echo(f"Total images: {len(images)}")

    except Exception as error:
        click.echo(f"Error: {error}", err=True)
        raise click.Abort()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
