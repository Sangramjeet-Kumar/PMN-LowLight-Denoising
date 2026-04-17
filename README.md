# PMN-LowLight-Denoising

Practical extension of PMN for low-light denoising with a complete submission-ready workflow: inference, benchmarking, GUI, temporal video smoothing, synthetic domain adaptation, RGB fine-tuning, and deployment export.

## TL;DR

This repository is not just a baseline PMN run.
It includes end-to-end engineering and evaluation extensions designed for laptop-class hardware (tested with RTX 3050 constraints).

Main outcomes:
- Low-resource inference path for high-resolution RAW images
- Quantitative runtime/memory benchmark tooling
- User-facing desktop GUI
- Video denoising extension
- Synthetic low-light generation for adaptation
- RGB-domain fine-tuning path
- TorchScript export/compression path

## Table of Contents
- [1. Project Goal](#1-project-goal)
- [2. What Is New In This Submission](#2-what-is-new-in-this-submission)
- [3. Verified Evidence](#3-verified-evidence)
- [4. Repository Layout](#4-repository-layout)
- [5. Setup](#5-setup)
- [6. Run Guide](#6-run-guide)
- [7. Live Demonstration Script](#7-live-demonstration-script)
- [8. Output Artifacts You Should Show](#8-output-artifacts-you-should-show)
- [9. Limitations and Future Work](#9-limitations-and-future-work)
- [10. Credits](#10-credits)

## 1. Project Goal

Baseline PMN is powerful for RAW denoising, but in research form it is not ideal for practical submission/demo use.
This project turns PMN into a complete demonstration package with:
- strong usability,
- measurable evidence,
- adaptation workflows,
- and deployment artifacts.

## 2. What Is New In This Submission

### A. Core Inference Enhancements
File: `run_inference.py`

- RAW and RGB processing path in one script
- Tiled inference for low VRAM (`--tile`, `--tile_overlap`)
- Auto RAW ratio (`--auto_ratio`, `--target_luma`)
- Strength-controlled RGB restoration (`--strength`)
- Diagnostic maps (`--save_maps`): residual and noise-reduction views

### B. Quantitative Benchmark Tool
File: `benchmark_inference.py`

- End-to-end runtime
- CPU RSS and peak traced memory
- GPU peak allocated/reserved memory
- Optional PSNR/SSIM against provided ground-truth image

### C. Desktop GUI
File: `gui_app.py`

- File picker for input/model/output
- Live controls for strength/tiling/auto ratio
- Background execution and visual preview

### D. Temporal Video Extension
File: `video_denoise.py`

- Frame-wise PMN denoising
- Optical-flow-guided temporal blending
- Reduced flicker without retraining

### E. Synthetic Low-Light Data Generator
File: `simulate_low_light.py`

- Converts clean RGB images into synthetic low-light samples
- Presets: `mild`, `phone`, `extreme`
- Saves clean/noisy pairs and manifest metadata

### F. RGB Fine-Tuning Path
File: `finetune_rgb.py`

- Synthetic RGB-derived RAW pair generation on the fly
- Low-compute train scopes (`head`, `decoder`, `full`)
- Practical smoke-run mode for constrained GPUs

### G. Export and Compression
File: `export_model.py`

- TorchScript fp32 export
- Optional fp16 export (CUDA path)
- Export manifest with output-diff and size summary

## 3. Verified Evidence

Measured in this project run:

### Runtime and Memory (RAW sample: `test_image/test_image1.ARW`)

| Configuration | Runtime | CPU RSS | GPU Peak Allocated |
|---|---:|---:|---:|
| Full-frame baseline | 30.025 s | 1770.02 MB | 3417.60 MB |
| Tiled (`--tile 256`) | 24.514 s | 977.73 MB | 59.21 MB |

### Export Size

| Artifact | Size |
|---|---:|
| TorchScript fp32 | 29.62 MB |
| TorchScript fp16 | 14.82 MB |

### RGB Fine-Tuning Smoke Run

| Metric | Value |
|---|---:|
| Train loss | 0.0094 |
| Validation loss | 0.0131 |
| Validation PSNR | 36.23 dB |

## 4. Repository Layout

Core:
- `archs/`
- `data_process/`
- `checkpoints/`
- `run_inference.py`

Extensions:
- `benchmark_inference.py`
- `gui_app.py`
- `video_denoise.py`
- `simulate_low_light.py`
- `finetune_rgb.py`
- `export_model.py`

Configs and metadata:
- `runfiles/`
- `infos/`
- `resources/`

## 5. Setup

### Requirements
- Python 3.8+
- PyTorch 2.x
- OpenCV
- rawpy
- scikit-image
- numpy

### Install

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python rawpy scikit-image numpy
```

### Verify

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__)"
```

## 6. Run Guide

Use project root:

```powershell
cd "C:\Users\sangr\Desktop\Image Processing Project\PMN"
```

### 6.1 RAW Inference (Low-resource + maps)

```powershell
python run_inference.py --input "test_image/test_image1.ARW" --auto_ratio --tile 256 --tile_overlap 64 --save_maps --output_dir "results_demo_core"
```

### 6.2 RGB Inference

```powershell
python run_inference.py --input "test_image/RBB_noise.jpg" --strength 0.35 --tile 768 --tile_overlap 64 --save_maps --output_dir "results_demo_rgb"
```

### 6.3 Benchmark Baseline vs Tiled

```powershell
python benchmark_inference.py --input "test_image/test_image1.ARW" --output_dir "bench_default" --no_save_output
python benchmark_inference.py --input "test_image/test_image1.ARW" --tile 256 --tile_overlap 64 --output_dir "bench_tiled" --no_save_output
```

### 6.4 GUI

```powershell
python gui_app.py
```

Recommended GUI settings for RTX 3050 class:
- strength: `0.3` to `0.4`
- tile: `768`
- overlap: `64`

### 6.5 Video Denoising

```powershell
python video_denoise.py --input "path/to/input.mp4" --output "results/video_denoised.mp4" --strength 0.35 --tile 768
```

### 6.6 Synthetic Low-Light Generation

```powershell
python simulate_low_light.py --input "test_image/RBB_noise.jpg" --output_dir "results_noise_sim" --preset phone --save_side_by_side
```

### 6.7 RGB Fine-Tuning (Smoke)

```powershell
python finetune_rgb.py --input_dir "test_image" --output_dir "rgb_finetune_demo" --epochs 1 --samples_per_epoch 2 --batch_size 1 --patch_size 64 --train_scope head
```

### 6.8 Export/Compression

```powershell
python export_model.py --output_dir "exports_demo" --name "pmn_unet_export" --height 512 --width 512 --fp16
```

## 7. Live Demonstration Script

Recommended order for 10 to 15 minutes:

1. RAW inference with auto ratio + tiling + maps
2. Benchmark baseline vs tiled
3. GUI quick run
4. Synthetic low-light generation
5. RGB fine-tune smoke command
6. Export/compression command
7. Optional video command

Why this order works:
- Starts with core visual quality
- Immediately proves measurable engineering value
- Shows usability (GUI)
- Shows research-style adaptation and deployment depth

## 8. Output Artifacts You Should Show

After running commands, highlight:

- `results_demo_core/*_comparison.png`
- `results_demo_core/*_residual_map.png`
- `results_demo_core/*_noise_reduction_map.png`
- benchmark terminal summaries (`bench_default`, `bench_tiled`)
- `results_noise_sim/*_pair.png`
- `rgb_finetune_demo/checkpoints/`
- `exports_demo/*.pt` and export manifest JSON

## 9. Limitations and Future Work

Current limitations:
- Best quality still achieved on RAW-like domain
- Full training from scratch is compute-heavy
- Video extension is not real-time optimized by default

Future work:
- Adaptive strength prediction
- Stronger perceptual loss strategy for RGB
- Real-time optimized video path
- Optional ONNX/TensorRT deployment track

## 10. Credits

- Original PMN baseline: Megvii Research
- Extended engineering, adaptation, benchmarking, and deployment workflow: this submission project
