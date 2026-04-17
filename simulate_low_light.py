"""Synthetic low-light noise simulator for PMN domain adaptation.

This script generates paired clean/low-light RGB images from normal photos.
It is intentionally lightweight and does not require retraining the model.

Use cases:
- create pseudo-pairs for a small domain-adaptation dataset
- stress-test the denoiser on synthetic phone/camera noise
- generate report figures showing degradation vs restoration
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


PRESETS = {
    'mild': {
        'exposure_scale': 0.35,
        'shot_noise': 0.012,
        'read_noise': 0.0015,
        'row_noise': 0.0,
        'color_jitter': 0.03,
        'wb_jitter': 0.02,
        'jpeg_quality': 95,
    },
    'phone': {
        'exposure_scale': 0.18,
        'shot_noise': 0.025,
        'read_noise': 0.004,
        'row_noise': 0.0025,
        'color_jitter': 0.06,
        'wb_jitter': 0.05,
        'jpeg_quality': 88,
    },
    'extreme': {
        'exposure_scale': 0.08,
        'shot_noise': 0.045,
        'read_noise': 0.008,
        'row_noise': 0.006,
        'color_jitter': 0.12,
        'wb_jitter': 0.08,
        'jpeg_quality': 78,
    },
}


@dataclass
class SimulationParams:
    exposure_scale: float = 0.18
    shot_noise: float = 0.025
    read_noise: float = 0.004
    row_noise: float = 0.0025
    color_jitter: float = 0.06
    wb_jitter: float = 0.05
    jpeg_quality: int = 88
    gamma: float = 2.2


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def _iter_images(path: str) -> Iterable[Path]:
    src = Path(path)
    if src.is_file():
        yield src
        return

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    for file_path in sorted(src.rglob('*')):
        if file_path.suffix.lower() in exts:
            yield file_path


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return image


def _srgb_to_linear(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    return np.power(np.clip(image, 0.0, 1.0), gamma)


def _linear_to_srgb(image_linear: np.ndarray, gamma: float) -> np.ndarray:
    image = np.power(np.clip(image_linear, 0.0, 1.0), 1.0 / gamma)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _apply_row_noise(image_linear: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image_linear
    rows = image_linear.shape[0]
    pattern = np.random.normal(0.0, strength, size=(rows, 1, 1)).astype(np.float32)
    return np.clip(image_linear + pattern, 0.0, 1.0)


def _apply_wb_and_color_shift(image_linear: np.ndarray, params: SimulationParams) -> np.ndarray:
    gains = np.array([
        1.0 + np.random.uniform(-params.wb_jitter, params.wb_jitter),
        1.0 + np.random.uniform(-params.color_jitter, params.color_jitter),
        1.0 + np.random.uniform(-params.wb_jitter, params.wb_jitter),
    ], dtype=np.float32)
    gains = np.clip(gains, 0.6, 1.6)
    return np.clip(image_linear * gains[None, None, :], 0.0, 1.0)


def simulate_low_light(image_bgr: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Convert a clean image into a low-light noisy observation."""
    linear = _srgb_to_linear(image_bgr, gamma=params.gamma)

    # Reduce exposure to mimic a short shutter speed.
    low_light = np.clip(linear * params.exposure_scale, 0.0, 1.0)

    # Apply realistic white-balance/color drift before the noise process.
    low_light = _apply_wb_and_color_shift(low_light, params)

    # Signal-dependent shot noise + constant read noise.
    shot_std = np.sqrt(np.maximum(low_light, 0.0) * params.shot_noise)
    read_std = params.read_noise
    noise = np.random.normal(0.0, 1.0, size=low_light.shape).astype(np.float32)
    noisy = low_light + noise * (shot_std + read_std)

    # Lightweight row noise commonly seen in low-light sensors.
    noisy = _apply_row_noise(noisy, params.row_noise)

    # Re-encode to sRGB to match typical consumer image pipelines.
    return _linear_to_srgb(noisy, gamma=params.gamma)


def _jpeg_reencode(image_bgr: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 30, 100))
    success, encoded = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return image_bgr
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image_bgr if decoded is None else decoded


def _make_side_by_side(clean_bgr: np.ndarray, noisy_bgr: np.ndarray) -> np.ndarray:
    h = max(clean_bgr.shape[0], noisy_bgr.shape[0])
    clean = cv2.copyMakeBorder(clean_bgr, 0, h - clean_bgr.shape[0], 0, 0, cv2.BORDER_REFLECT)
    noisy = cv2.copyMakeBorder(noisy_bgr, 0, h - noisy_bgr.shape[0], 0, 0, cv2.BORDER_REFLECT)
    label_h = 36
    canvas = np.zeros((h + label_h, clean.shape[1] + noisy.shape[1], 3), dtype=np.uint8)
    canvas[label_h:, :clean.shape[1]] = clean
    canvas[label_h:, clean.shape[1]:] = noisy
    cv2.putText(canvas, 'Clean', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, 'Synthetic Low-Light', (clean.shape[1] + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def process_one(image_path: Path, output_dir: str, params: SimulationParams, save_side_by_side: bool) -> dict:
    clean = _load_bgr(image_path)
    noisy = simulate_low_light(clean, params)
    noisy = _jpeg_reencode(noisy, params.jpeg_quality)

    stem = image_path.stem
    clean_path = os.path.join(output_dir, f'{stem}_clean.png')
    noisy_path = os.path.join(output_dir, f'{stem}_synthetic_lowlight.png')
    compare_path = os.path.join(output_dir, f'{stem}_pair.png')

    cv2.imwrite(clean_path, clean)
    cv2.imwrite(noisy_path, noisy)
    if save_side_by_side:
        cv2.imwrite(compare_path, _make_side_by_side(clean, noisy))

    manifest = {
        'input': str(image_path),
        'clean': clean_path,
        'synthetic_lowlight': noisy_path,
        'comparison': compare_path if save_side_by_side else None,
        'params': asdict(params),
    }
    with open(os.path.join(output_dir, f'{stem}_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic low-light noise simulator for domain adaptation')
    parser.add_argument('--input', '-i', required=True, help='input image file or folder')
    parser.add_argument('--output_dir', '-o', required=False, default='results_noise_sim', help='output directory')
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), default='phone', help='noise preset')
    parser.add_argument('--exposure_scale', type=float, default=None, help='override exposure scale')
    parser.add_argument('--shot_noise', type=float, default=None, help='override shot noise strength')
    parser.add_argument('--read_noise', type=float, default=None, help='override read noise strength')
    parser.add_argument('--row_noise', type=float, default=None, help='override row noise strength')
    parser.add_argument('--color_jitter', type=float, default=None, help='override channel color jitter')
    parser.add_argument('--wb_jitter', type=float, default=None, help='override white balance jitter')
    parser.add_argument('--jpeg_quality', type=int, default=None, help='JPEG re-encode quality')
    parser.add_argument('--gamma', type=float, default=2.2, help='gamma curve used for encode/decode')
    parser.add_argument('--save_side_by_side', action='store_true', help='save clean vs noisy comparison images')
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    params = SimulationParams(
        exposure_scale=args.exposure_scale if args.exposure_scale is not None else preset['exposure_scale'],
        shot_noise=args.shot_noise if args.shot_noise is not None else preset['shot_noise'],
        read_noise=args.read_noise if args.read_noise is not None else preset['read_noise'],
        row_noise=args.row_noise if args.row_noise is not None else preset['row_noise'],
        color_jitter=args.color_jitter if args.color_jitter is not None else preset['color_jitter'],
        wb_jitter=args.wb_jitter if args.wb_jitter is not None else preset['wb_jitter'],
        jpeg_quality=args.jpeg_quality if args.jpeg_quality is not None else preset['jpeg_quality'],
        gamma=args.gamma,
    )

    input_path = Path(_resolve_path(args.input))
    output_dir = _ensure_output_dir(_resolve_path(args.output_dir))

    image_files = list(_iter_images(str(input_path)))
    if not image_files:
        raise RuntimeError(f'No images found in {input_path}')

    print(f'Input  : {input_path}')
    print(f'Output : {output_dir}')
    print(f'Preset : {args.preset}')
    print(f'Params : {json.dumps(asdict(params), indent=2)}')

    count = 0
    for image_file in image_files:
        manifest = process_one(image_file, output_dir, params, args.save_side_by_side)
        count += 1
        print(f"Saved  : {manifest['synthetic_lowlight']}")

    print(f'\nDone — generated {count} synthetic low-light sample(s) in {output_dir}')


if __name__ == '__main__':
    main()