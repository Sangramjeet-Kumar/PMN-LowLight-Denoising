# PMN Low-Light Denoising: Extended Submission

Practical extension of PMN from research-only RAW denoising to a complete project package suitable for demonstration and deployment on laptop-class hardware.

## Table of Contents
- [Project Summary](#project-summary)
- [What Was Added Beyond Baseline PMN](#what-was-added-beyond-baseline-pmn)
- [Validated Results and Evidence](#validated-results-and-evidence)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [How To Run (All Extensions)](#how-to-run-all-extensions)
- [Live Demonstration Flow](#live-demonstration-flow)
- [Submission Notes](#submission-notes)
- [Credits](#credits)

## Project Summary

The original PMN pipeline is strong for RAW denoising but assumes research-style usage. This submission focuses on **practical engineering extensions**:

- Low-resource inference controls (tiling, auto ratio, diagnostic maps)
- Quantitative benchmarking (runtime/memory/quality)
- Desktop GUI workflow
- Temporal video denoising path
- Synthetic low-light domain adaptation toolkit
- RGB-domain fine-tuning path for limited compute
- Export/compression to deployable TorchScript artifacts

## What Was Added Beyond Baseline PMN

### 1) Enhanced inference pipeline
File: `run_inference.py`

- Added tiled inference (`--tile`, `--tile_overlap`) for low VRAM
- Added automatic RAW ratio estimation (`--auto_ratio`, `--target_luma`)
- Added explainability outputs (`--save_maps`)
- Added return path for benchmark integration

### 2) Quantitative benchmark script
File: `benchmark_inference.py`

- Runtime measurement
- CPU RSS and peak traced memory
- GPU peak allocated/reserved memory
- Optional PSNR/SSIM if ground-truth provided

### 3) Desktop GUI
File: `gui_app.py`

- Input/model/output pickers
- Live controls (strength, tile, overlap, auto ratio)
- Background inference execution
- Preview panes for original/output/comparison

### 4) Temporal video denoising
File: `video_denoise.py`

- Frame denoising using PMN path
- Optical-flow-guided temporal smoothing
- Motion-aware blending to reduce flicker

### 5) Synthetic low-light data generator
File: `simulate_low_light.py`

- Generates clean/synthetic low-light pairs from RGB images
- Presets (`mild`, `phone`, `extreme`)
- Saves manifests for reproducibility

### 6) RGB-domain fine-tuning
File: `finetune_rgb.py`

- Synthetic RGB-derived RAW pair creation on the fly
- Low-compute fine-tuning scopes (`head`, `decoder`, `full`)
- Supports laptop-friendly smoke runs

### 7) Export/compression path
File: `export_model.py`

- TorchScript fp32 export
- Optional fp16 export on CUDA
- Export manifest with size and output-diff checks

## Validated Results and Evidence

Measured on a project RAW sample (`test_image/test_image1.ARW`) with `benchmark_inference.py`.

### Runtime and memory

| Configuration | Runtime | CPU RSS | GPU Peak Allocated |
|---|---:|---:|---:|
| Baseline full-frame | 30.025 s | 1770.02 MB | 3417.60 MB |
| Tiled (`--tile 256`) | 24.514 s | 977.73 MB | 59.21 MB |

### Export artifact sizes

| Artifact | Size |
|---|---:|
| TorchScript fp32 | 29.62 MB |
| TorchScript fp16 | 14.82 MB |

### RGB fine-tuning smoke test

| Metric | Value |
|---|---:|
| Train loss | 0.0094 |
| Validation loss | 0.0131 |
| Validation PSNR | 36.23 dB |

## Repository Structure

Core model and pipeline:
- `run_inference.py`
- `archs/`
- `data_process/`
- `checkpoints/`

Extension scripts:
- `benchmark_inference.py`
- `gui_app.py`
- `video_denoise.py`
- `simulate_low_light.py`
- `finetune_rgb.py`
- `export_model.py`

Config and metadata:
- `runfiles/`
- `infos/`
- `resources/`

## Environment Setup

Requirements:
- Python 3.8+
- PyTorch 2.x
- OpenCV
- rawpy
- scikit-image
- numpy

Install:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python rawpy scikit-image numpy
```

Verify:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__)"
```

## How To Run (All Extensions)

Run from project root:

```powershell
cd "C:\Users\sangr\Desktop\Image Processing Project\PMN"
```

### A) Core inference (RAW)

```powershell
python run_inference.py --input "test_image/test_image1.ARW" --auto_ratio --tile 256 --tile_overlap 64 --save_maps --output_dir "results_demo_core"
```

### B) Core inference (RGB)

```powershell
python run_inference.py --input "test_image/RBB_noise.jpg" --strength 0.35 --tile 768 --tile_overlap 64 --save_maps --output_dir "results_demo_rgb"
```

### C) Benchmark baseline vs tiled

```powershell
python benchmark_inference.py --input "test_image/test_image1.ARW" --output_dir "bench_default" --no_save_output
python benchmark_inference.py --input "test_image/test_image1.ARW" --tile 256 --tile_overlap 64 --output_dir "bench_tiled" --no_save_output
```

### D) GUI

```powershell
python gui_app.py
```

Recommended GUI settings (RTX 3050 class):
- RGB strength: `0.3` to `0.4`
- Tile size: `768`
- Tile overlap: `64`

### E) Video denoising

```powershell
python video_denoise.py --input "path/to/input.mp4" --output "results/video_denoised.mp4" --strength 0.35 --tile 768
```

### F) Synthetic low-light generation

```powershell
python simulate_low_light.py --input "test_image/RBB_noise.jpg" --output_dir "results_noise_sim" --preset phone --save_side_by_side
```

### G) RGB fine-tuning (smoke run)

```powershell
python finetune_rgb.py --input_dir "test_image" --output_dir "rgb_finetune_demo" --epochs 1 --samples_per_epoch 2 --batch_size 1 --patch_size 64 --train_scope head
```

### H) Export/compression

```powershell
python export_model.py --output_dir "exports_demo" --name "pmn_unet_export" --height 512 --width 512 --fp16
```

## Live Demonstration Flow

Recommended order (10 to 15 minutes):

1. Core RAW inference + maps
2. Benchmark baseline vs tiled
3. GUI quick run
4. Synthetic low-light generation
5. RGB fine-tuning smoke run
6. Export/compression
7. Optional video command

This order shows both algorithmic extension and engineering evidence.

## Submission Notes

- This README is the single, clean project documentation entry point.
- Generated output folders are excluded by `.gitignore`.
- Repository contains all implemented extension scripts required for evaluation.

## Credits

- Original PMN baseline: Megvii Research
- Extensions, integration, evaluation, and demonstration packaging: this submission work
