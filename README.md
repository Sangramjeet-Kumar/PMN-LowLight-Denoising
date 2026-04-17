# PMN Low-Light Denoising (Submission Version)

Clean project submission based on PMN with practical extensions for laptop-friendly inference, benchmarking, GUI demo, video denoising, synthetic domain adaptation, RGB fine-tuning, and export/compression.

## What This Project Adds

Compared to the baseline PMN workflow, this submission adds:

1. Enhanced inference pipeline for RAW and RGB images
- Tiled inference for low VRAM (`--tile`, `--tile_overlap`)
- Auto RAW amplification ratio (`--auto_ratio`, `--target_luma`)
- Diagnostic maps (`--save_maps`)

2. Quantitative benchmark tool
- Runtime, CPU/GPU memory, and optional PSNR/SSIM

3. Desktop GUI demo
- User-friendly app with controls and previews

4. Temporal video denoising
- Inference-only frame denoising with temporal smoothing

5. Synthetic low-light data generation
- Creates clean/degraded pairs for domain adaptation experiments

6. RGB-domain fine-tuning path
- Low-compute adaptation using synthetic RGB-derived RAW pairs

7. Deployment export/compression
- TorchScript export with optional fp16 artifact

## Repository Structure (Important Files)

Core:
- `run_inference.py`
- `archs/`
- `data_process/`
- `checkpoints/`

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

## Requirements

- Python 3.8+
- PyTorch 2.x
- OpenCV
- rawpy
- scikit-image
- numpy

Install example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python rawpy scikit-image numpy
```

## Quick Start

### 1) RAW/RGB inference

```powershell
python run_inference.py --input "test_image/test_image1.ARW" --auto_ratio --tile 256 --tile_overlap 64 --save_maps --output_dir "results_demo"
```

For RGB:

```powershell
python run_inference.py --input "test_image/RBB_noise.jpg" --strength 0.35 --tile 768 --tile_overlap 64 --save_maps --output_dir "results_demo_rgb"
```

### 2) Benchmark (default vs tiled)

```powershell
python benchmark_inference.py --input "test_image/test_image1.ARW" --output_dir "bench_default" --no_save_output
python benchmark_inference.py --input "test_image/test_image1.ARW" --tile 256 --tile_overlap 64 --output_dir "bench_tiled" --no_save_output
```

### 3) GUI demo

```powershell
python gui_app.py
```

Suggested GUI settings for laptop GPU:
- RGB strength: `0.3` to `0.4`
- Tile size: `768`
- Tile overlap: `64`

### 4) Video denoising

```powershell
python video_denoise.py --input "path/to/input.mp4" --output "results/video_denoised.mp4" --strength 0.35 --tile 768
```

### 5) Synthetic low-light data generation

```powershell
python simulate_low_light.py --input "test_image/RBB_noise.jpg" --output_dir "results_noise_sim" --preset phone --save_side_by_side
```

### 6) RGB fine-tuning (smoke run)

```powershell
python finetune_rgb.py --input_dir "test_image" --output_dir "rgb_finetune_demo" --epochs 1 --samples_per_epoch 2 --batch_size 1 --patch_size 64 --train_scope head
```

### 7) Export/compression

```powershell
python export_model.py --output_dir "exports_demo" --name "pmn_unet_export" --height 512 --width 512 --fp16
```

## Notes for Submission

- This repository is cleaned for GitHub submission.
- Generated output folders are excluded via `.gitignore`.
- Keep only this README as the single project documentation entry point.

## Credits

- Original PMN baseline: Megvii Research
- Submission extensions and integration: this project work
