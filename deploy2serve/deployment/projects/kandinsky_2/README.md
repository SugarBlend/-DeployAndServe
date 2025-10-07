# Kandinsky 2.2 TensorRT Acceleration

High-performance implementation of Kandinsky 2.2 using TensorRT for production deployment with significant speed improvements and memory optimization.

## Quick Start

### Export Models to TensorRT

```powershell
# Text Encoder
python -m deploy2serve.deployment.deploy --deploy_config deploy2serve/deployment/projects/kandinsky_2/text_encoder.yaml

# Image Encoder
python -m deploy2serve.deployment.deploy --deploy_config deploy2serve/deployment/projects/kandinsky_2/image_encoder.yaml

# Prior Transformer
python -m deploy2serve.deployment.deploy --deploy_config deploy2serve/deployment/projects/kandinsky_2/proir_transformer.yaml

# U-Net
python -m deploy2serve.deployment.deploy --deploy_config deploy2serve/deployment/projects/kandinsky_2/unet.yaml

# MOVQ
python -m deploy2serve.deployment.deploy --deploy_config deploy2serve/deployment/projects/kandinsky_2/movq.yaml
```

## Demo

```powershell
python -m deploy2serve.deployment.projects.kandinsky_2.demo --version HF --output_dir ./deploy2serve/deployment/projects/kandinsky_2/generations_hf
```

```powershell
python -m deploy2serve.deployment.projects.kandinsky_2.demo --version TRT --output_dir ./deploy2serve/deployment/projects/kandinsky_2/generations_trt
```

## Performance Comparison

| Metric                       | HuggingFace (Baseline) | TensorRT (Optimized) | Improvement |
|------------------------------|------------------------|----------------------|-------------|
| **Total Pipeline Time**, sec | -                      | -                    | -           |
| **VRAM Usage**, GB           | -                      | -                    | -           |
| **Text Encoder**, sec        | -                      | -                    | -           |
| **Image Encoder**, sec       | -                      | -                    | -           |
| **Prior Transformer**, sec   | -                      | -                    | -           |
| **U-Net**, sec               | -                      | -                    | -           |
| **MOVQ**, sec                | -                      | -                    | -           |

*Testing conducted on NVIDIA RTX 4080 super, 768x768 resolution*

## Performance Benchmarks

### Batch Processing Performance

| Batch Size | HuggingFace, sec | TensorRT, sec | Speedup |
|------------|------------------|---------------|---------|
| 1          | -                | -             | -       |
| 2          | -                | -             | -       |
| 4          | -                | -             | -       |

### Resolution Scaling

| Resolution | VRAM Usage, GB | Time (HuggingFace), sec | Time (TensorRT), sec |
|------------|----------------|-------------------------|----------------------|
| 256x256    | -              | -                       | -                    |
| 512x512    | -              | -                       | -                    |
| 768x768    | -              | -                       | -                    |

### Memory Efficiency

| Metric               | HuggingFace | TensorRT | Improvement |
|----------------------|-------------|----------|-------------|
| Peak VRAM, GB        | -           | -        | -           |
| Memory Footprint, GB | -           | -        | -           |
| Loading Time, sec    | -           | -        | -           |

## Generation Results

| Prompt                                                                                                                                                                       | Diffusers                                              | TensorRT                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------|
| <div style="width: 300px">Realistic Drawing, Detailed Portrait of a Man, Graphite on Paper, A2 size.</div>                                                                   | <img src="./generations/image_000_hf.png" width="300"> | <img src="./generations/image_000_trt.png" width="300"> |
| <div style="width: 300px">Photorealism, Highly detailed and life-like portrait of a pet, Robert Bateman-inspired, Acrylic on Canvas, 16x20 inches.</div>                     | <img src="./generations/image_001_hf.png" width="300"> | <img src="./generations/image_001_trt.png" width="300"> |
| <div style="width: 300px">Pop Art Portraits, Brightly colored and bold images of pop culture icons, Pop art style, Bright colors, Digital art, Iconic feel.</div>            | <img src="./generations/image_002_hf.png" width="300"> | <img src="./generations/image_002_trt.png" width="300"> |
| <div style="width: 300px">Nautical, Sailing Ship on Choppy Sea, Romanticism, Oil on canvas, 24x36 inches.</div>                                                              | <img src="./generations/image_003_hf.png" width="300"> | <img src="./generations/image_003_trt.png" width="300"> |
| <div style="width: 300px">Portraits, Old Woman with Wisdom Lines, Realism, Charcoal, High Detail.</div>                                                                      | <img src="./generations/image_004_hf.png" width="300"> | <img src="./generations/image_004_trt.png" width="300"> |
| <div style="width: 300px">Architecture, Historic Castle, Gothic Revival, Digital Art, Impressive Structure.</div>                                                            | <img src="./generations/image_005_hf.png" width="300"> | <img src="./generations/image_005_trt.png" width="300"> |
| <div style="width: 300px">Portrait, A Wise Old Man, Realism, Charcoal drawing, Detailed facial features, A4 size (21x29.7 cm).</div>                                         | <img src="./generations/image_006_hf.png" width="300"> | <img src="./generations/image_006_trt.png" width="300"> |
| <div style="width: 300px">Fantasy Landscapes, Majestic Mountain Range, Watercolor and Ink, Detailed topography with vibrant colors and flowing water elements.</div>         | <img src="./generations/image_007_hf.png" width="300"> | <img src="./generations/image_007_trt.png" width="300"> |
| <div style="width: 300px">Portraits, Historical Figures, Photorealism, Highly-detailed and lifelike portraits of historical figures throughout history.</div>                | <img src="./generations/image_008_hf.png" width="300"> | <img src="./generations/image_008_trt.png" width="300"> |
| <div style="width: 300px">Portrait, Regal Eagle Overlooking the Mountains, Wildlife Art, Acrylic Painting, Detailed Feathers, Majestic Scenery.</div>                        | <img src="./generations/image_009_hf.png" width="300"> | <img src="./generations/image_009_trt.png" width="300"> |
| <div style="width: 300px">Portrait, Regal Giraffe in Savannah Plains, Wildlife Art, Charcoal Drawing, Detailed Features, Naturalistic Setting.</div>                         | <img src="./generations/image_010_hf.png" width="300"> | <img src="./generations/image_010_trt.png" width="300"> |
| <div style="width: 300px">Mystical Forests, Enchanted Trees in the Moonlight, Fantasy-inspired style, Digital painting with a focus on eerie yet beautiful atmosphere.</div> | <img src="./generations/image_011_hf.png" width="300"> | <img src="./generations/image_011_trt.png" width="300"> |
| <div style="width: 300px">Portrait, Serious Businessman, Photorealism, Digital Art, Sharp Detail.</div>                                                                      | <img src="./generations/image_012_hf.png" width="300"> | <img src="./generations/image_012_trt.png" width="300"> |
| <div style="width: 300px">Cityscape, Busy Street Scene, Urban Realism, Digital art, Highly detailed, realistic rendering, and a fast-paced vibe.</div>                       | <img src="./generations/image_013_hf.png" width="300"> | <img src="./generations/image_013_trt.png" width="300"> |
| <div style="width: 300px">Portrait, A serious-looking woman in Victorian attire, Realistic, Charcoal drawing, 11x14 inches.</div>                                            | <img src="./generations/image_014_hf.png" width="300"> | <img src="./generations/image_014_trt.png" width="300"> |
| <div style="width: 300px">Animal Portrait, Tiger in the Wild, Realism, Photography, Sharply focused with a natural backdrop.</div>                                           | <img src="./generations/image_015_hf.png" width="300"> | <img src="./generations/image_015_trt.png" width="300"> |

## Acknowledgments

- [Kandinsky 2.2](https://github.com/ai-forever/Kandinsky-2) by AI Forever
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
