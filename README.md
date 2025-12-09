# Baseline Generation Notebook - README

## Overview

This notebook implements a baseline image generation pipeline using **Stable Diffusion XL (SDXL)** to generate images of speed bumps and related roadway fixtures. It serves as the foundation for establishing baseline performance before fine-tuning experiments in the project: **"Finding and Solving Hard-to-Generate Examples - Speed Bumps"**.

## Purpose

- Establish baseline performance metrics for speed bump image generation
- Generate a dataset of speed bump images using the pre-trained SDXL model
- Evaluate generation quality and identify areas for improvement
- Provide a reproducible pipeline for comparison with fine-tuned models

## Key Features

- **Model**: Stable Diffusion XL Base 1.0 (stabilityai/stable-diffusion-xl-base-1.0)
- **Framework**: PyTorch + HuggingFace Diffusers
- **Optimizations**: Speed and memory optimizations for efficient generation
- **Evaluation**: Built-in quality assessment and evaluation pipeline
- **Compatibility**: Handles CPU, CUDA, and Apple Silicon (MPS) environments

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU with 10GB+ VRAM (recommended) OR CPU with sufficient RAM
- **Disk Space**: ~7GB free space for model download
- **Memory**: 8GB+ RAM (16GB+ recommended for CPU mode)

### Software Dependencies

The notebook will automatically install required packages, but you can also install them manually:

```bash
pip install torch torchvision
pip install "diffusers[torch]>=0.29.0" "transformers>=4.45.0" "accelerate>=0.34.0"
pip install safetensors pillow tqdm matplotlib
```

## Installation & Setup

1. **Open the notebook** in Jupyter Lab, Jupyter Notebook, or VS Code
2. **Run the first cell** to install dependencies (if needed)
3. **Run the setup cell** to import libraries and configure the environment
4. **Configure parameters** in the configuration section (optional)

## Usage

### Quick Start

1. **Setup and Imports** (Cell 1-2)
   - Installs dependencies and imports required libraries
   - Configures device settings (CPU/CUDA/MPS)

2. **Configuration** (Cell 3)
   - Sets generation parameters (steps, resolution, guidance scale)
   - Configures output directories
   - Sets reproducibility seed

3. **Model Loading** (Cell 4)
   - Downloads SDXL model (~6.9GB) on first run (10-30 minutes)
   - Uses cached model on subsequent runs (2-5 minutes)
   - Applies memory optimizations

4. **Prompt Collection** (Cell 5)
   - Defines positive prompts for speed bump generation
   - Sets negative prompts to avoid artifacts

5. **Generation Pipeline** (Cell 6-7)
   - Implements single image generation function
   - Includes error handling and metadata tracking

6. **Batch Generation** (Cell 8)
   - Generates multiple images from prompt collection
   - Saves images and logs generation metadata

7. **Evaluation Pipeline** (Cell 9)
   - Assesses image quality (brightness, contrast, sharpness)
   - Generates evaluation reports

8. **Results Visualization** (Cell 10)
   - Creates visual summaries of generated images
   - Displays quality metrics

9. **Results Summary** (Cell 11)
   - Comprehensive summary of generation statistics
   - Quality metrics and success rates

### Configuration Parameters

Key parameters you can adjust in the configuration section:

```python
CONFIG = {
    'num_inference_steps': 20,      # Denoising steps (20 = fast, 50 = quality)
    'guidance_scale': 7.5,           # Prompt adherence (1-20)
    'width': 512,                    # Image width (512-1024)
    'height': 512,                   # Image height (512-1024)
    'batch_size': 1,                 # Images per batch
    'device': 'cpu',                 # 'cpu', 'cuda', or auto-detected
}
```

### Speed & Memory Optimizations

The notebook includes several optimizations:

- **Reduced inference steps**: 20 steps (vs. default 50) for ~2.5x speedup
- **Smaller resolution**: 512x512 (vs. 1024x1024) for faster generation
- **Fast scheduler**: EulerDiscreteScheduler for faster generation
- **CPU offload**: Enabled for better memory management
- **VAE slicing**: Additional memory savings
- **Attention slicing**: Reduces memory usage

### Apple Silicon (MPS) Compatibility

⚠️ **Important**: The notebook automatically disables MPS (Apple Silicon GPU) to prevent out-of-memory errors. SDXL is too large for MPS memory limits, so the notebook forces CPU mode on Apple Silicon devices.

## Output Structure

After running the notebook, you'll find:

```
results/
└── baseline/
    ├── image_*.png          # Generated images
    └── metadata_*.json      # Generation metadata

logs/
└── generation_log_*.json    # Generation logs

evaluation_results/
├── evaluation_report_*.json # Evaluation reports
└── quality_*.png            # Quality visualizations
```

## Expected Runtime

- **First Run**:
  - Model download: 10-30 minutes (depends on internet speed)
  - Model loading: 2-5 minutes
  - Image generation: ~30-60 seconds per image (CPU) or ~5-10 seconds (GPU)

- **Subsequent Runs**:
  - Model loading: 2-5 minutes (from cache)
  - Image generation: Same as above

## Troubleshooting

### Model Download Issues

- **Problem**: Download seems stuck
- **Solution**: Check internet connection, look for progress bars, wait patiently (first download can take 30+ minutes)

### Out of Memory Errors

- **Problem**: OOM errors on GPU
- **Solution**: Reduce `batch_size`, `width`, `height`, or enable CPU offload

### MPS (Apple Silicon) Errors

- **Problem**: MPS-related errors
- **Solution**: The notebook automatically handles this by forcing CPU mode. If issues persist, ensure MPS is disabled in the configuration.

### Import Errors

- **Problem**: Missing dependencies
- **Solution**: Run the first cell again to install dependencies, then restart the kernel

## Project Structure

```
cursor/
├── baseline_generation_complete.ipynb  # This notebook
├── notebooks/                          # Other project notebooks
├── results/                            # Generated images and results
├── logs/                               # Generation logs
└── evaluation_results/                 # Evaluation reports
```

## References

- **Paper**: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (2023)
- **Model**: [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Documentation**: [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- **Library**: [HuggingFace Diffusers GitHub](https://github.com/huggingface/diffusers)

## Notes

- The notebook is optimized for speed while maintaining reasonable quality
- For higher quality, increase `num_inference_steps` to 50 and resolution to 1024x1024
- Generation times vary significantly between CPU and GPU modes
- All generated images include metadata for reproducibility

## License

This notebook uses the Stable Diffusion XL model, which is subject to its own license terms. Please refer to the model's license on HuggingFace for usage restrictions.

## Author

Based on HuggingFace Diffusers library and adapted for the Speed Bumps project.

---

**Last Updated**: Based on notebook version with comprehensive evaluation pipeline
