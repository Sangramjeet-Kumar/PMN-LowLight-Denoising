"""
Benchmark PMN inference with runtime, memory, and optional PSNR/SSIM metrics.

This script reuses the existing inference pipeline and reports:
- end-to-end runtime
- CPU memory peak / RSS
- GPU peak allocated / reserved memory when CUDA is available
- PSNR / SSIM when a ground-truth image is provided
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import run_inference as ri


RAW_EXTS = {'.arw', '.dng', '.nef', '.cr2', '.rw2'}
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class BenchmarkResult:
    runtime_sec: float
    cpu_rss_mb: float
    cpu_peak_mb: float
    gpu_peak_allocated_mb: Optional[float]
    gpu_peak_reserved_mb: Optional[float]
    psnr_db: Optional[float]
    ssim: Optional[float]


def _get_process_rss_mb() -> float:
    """Return resident set size in MB using stdlib-only fallbacks."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        pass

    if os.name == 'nt':
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 ** 2)
        return 0.0

    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == 'Darwin':
            return rss / (1024 ** 2)
        return rss / 1024.0
    except Exception:
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 ** 2)


def _load_image_for_metrics(image_path: str) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in RAW_EXTS:
        with ri.rawpy.imread(image_path) as raw:
            rgb16 = raw.postprocess(output_bps=16)
        return cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {image_path}')
    return img


def _load_image_any(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {image_path}')
    return img


def _crop_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _compute_quality_metrics(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> tuple[float, float]:
    pred_bgr, gt_bgr = _crop_to_common(pred_bgr, gt_bgr)
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    psnr = compare_psnr(gt_rgb, pred_rgb, data_range=255)
    ssim = compare_ssim(gt_rgb, pred_rgb, data_range=255, channel_axis=-1)
    return psnr, ssim


def benchmark(
    image_path: str,
    model_path: str,
    output_dir: str,
    ratio: float,
    strength: float,
    auto_ratio: bool,
    target_luma: float,
    tile: int,
    tile_overlap: int,
    gt_path: Optional[str] = None,
    save_output: bool = True,
) -> BenchmarkResult:
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    rss_before = _get_process_rss_mb()
    start = time.perf_counter()

    result = ri.run(
        image_path=image_path,
        model_path=model_path,
        output_dir=output_dir,
        ratio=ratio,
        strength=strength,
        auto_ratio=auto_ratio,
        target_luma=target_luma,
        tile=tile,
        tile_overlap=tile_overlap,
        save_maps=save_output,
        return_results=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _get_process_rss_mb()

    gpu_peak_allocated = None
    gpu_peak_reserved = None
    if torch.cuda.is_available():
        gpu_peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    psnr = None
    ssim = None
    if gt_path:
        gt_bgr = _load_image_any(gt_path)
        psnr, ssim = _compute_quality_metrics(result['output_bgr'], gt_bgr)

    cpu_peak_mb = max(peak / (1024 ** 2), rss_after, rss_before)
    cpu_rss_mb = rss_after

    return BenchmarkResult(
        runtime_sec=runtime,
        cpu_rss_mb=cpu_rss_mb,
        cpu_peak_mb=cpu_peak_mb,
        gpu_peak_allocated_mb=gpu_peak_allocated,
        gpu_peak_reserved_mb=gpu_peak_reserved,
        psnr_db=psnr,
        ssim=ssim,
    )


def _resolve_path(value: str, base_dir: Path = SCRIPT_DIR) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark PMN inference runtime and memory')
    parser.add_argument('--input', '-i', required=True, help='input image path')
    parser.add_argument('--model', '-m', required=False,
                        default=str((SCRIPT_DIR / 'checkpoints' / 'SonyA7S2_Mix_Unet_best_model.pth').resolve()),
                        help='checkpoint path')
    parser.add_argument('--output_dir', '-o', required=False, default='results_benchmark',
                        help='directory for outputs')
    parser.add_argument('--gt', required=False, default=None,
                        help='optional ground-truth image for PSNR/SSIM')
    parser.add_argument('--ratio', '-r', type=float, default=250.0,
                        help='RAW amplification ratio')
    parser.add_argument('--auto_ratio', action='store_true',
                        help='auto-estimate RAW amplification ratio')
    parser.add_argument('--target_luma', type=float, default=0.18,
                        help='target normalized brightness for --auto_ratio')
    parser.add_argument('--strength', '-s', type=float, default=0.4,
                        help='RGB denoising strength')
    parser.add_argument('--tile', type=int, default=0,
                        help='tile size for memory-safe inference')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='tile overlap in pixels')
    parser.add_argument('--no_save_output', action='store_true',
                        help='do not save diagnostic output images')
    args = parser.parse_args()

    metrics = benchmark(
        image_path=_resolve_path(args.input),
        model_path=_resolve_path(args.model),
        output_dir=_resolve_path(args.output_dir),
        ratio=args.ratio,
        strength=args.strength,
        auto_ratio=args.auto_ratio,
        target_luma=args.target_luma,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
        gt_path=_resolve_path(args.gt) if args.gt else None,
        save_output=not args.no_save_output,
    )

    print('\nBenchmark summary')
    print('-----------------')
    print(f'Runtime            : {metrics.runtime_sec:.3f} s')
    print(f'CPU RSS            : {metrics.cpu_rss_mb:.2f} MB')
    print(f'CPU peak trace      : {metrics.cpu_peak_mb:.2f} MB')
    if metrics.gpu_peak_allocated_mb is not None:
        print(f'GPU peak allocated  : {metrics.gpu_peak_allocated_mb:.2f} MB')
        print(f'GPU peak reserved   : {metrics.gpu_peak_reserved_mb:.2f} MB')
    if metrics.psnr_db is not None and metrics.ssim is not None:
        print(f'PSNR                : {metrics.psnr_db:.3f} dB')
        print(f'SSIM                : {metrics.ssim:.4f}')
    else:
        print('PSNR / SSIM         : skipped (no ground truth provided)')


if __name__ == '__main__':
    main()
