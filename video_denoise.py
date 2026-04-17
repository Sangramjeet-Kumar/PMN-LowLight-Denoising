"""Inference-only temporal video denoising for PMN.

This script reuses the existing image denoising pipeline and adds a lightweight
temporal stage:
- each frame is denoised with the PMN + RGB post-processing pipeline
- previous denoised output is warped to the current frame using optical flow
- a motion-aware blend reduces flicker while limiting ghosting

The goal is to extend the original image model without retraining.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import run_inference as ri


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def _resize_if_needed(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    new_size = (max_width, max(2, int(round(frame.shape[0] * scale))))
    new_size = (new_size[0] - new_size[0] % 2, new_size[1] - new_size[1] % 2)
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _process_frame(
    frame_bgr: np.ndarray,
    net,
    device,
    strength: float,
    tile: int,
    tile_overlap: int,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    y_dn = ri.denoise_channel(net, y, device, tile=tile, tile_overlap=tile_overlap)
    alpha = float(np.clip(strength, 0.0, 1.0))
    y_out = np.clip(alpha * y_dn + (1.0 - alpha) * y, 0.0, 1.0)
    y_out = (y_out * 255.0).astype(np.uint8)

    if strength > 0.1:
        cr_dn = cv2.bilateralFilter(cr, 9, int(25 * strength), int(25 * strength))
        cb_dn = cv2.bilateralFilter(cb, 9, int(25 * strength), int(25 * strength))
    else:
        cr_dn, cb_dn = cr, cb

    merged = cv2.cvtColor(np.stack([y_out, cr_dn, cb_dn], axis=-1), cv2.COLOR_YCrCb2BGR)

    nlm_h = max(3, int(10 * strength))
    if strength > 0.05:
        denoised = cv2.fastNlMeansDenoisingColored(merged, None, nlm_h, nlm_h, 7, 21)
    else:
        denoised = merged

    return denoised


def _warp_previous(previous_bgr: np.ndarray, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    h, w = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    return cv2.remap(
        previous_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _motion_to_alpha(current_gray: np.ndarray, warped_prev_gray: np.ndarray, base_alpha: float, max_alpha: float) -> float:
    diff = cv2.absdiff(current_gray, warped_prev_gray)
    motion = float(np.mean(diff) / 255.0)
    alpha = base_alpha + motion * 0.65
    return float(np.clip(alpha, base_alpha, max_alpha))


def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    ratio: float,
    strength: float,
    auto_ratio: bool,
    target_luma: float,
    tile: int,
    tile_overlap: int,
    max_width: int,
    temporal_alpha: float,
    max_temporal_alpha: float,
    save_first_frame_maps: bool,
    use_camera: bool,
) -> None:
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {torch_device}')

    net = ri.load_model(model_path, torch_device)
    print(f'Model  : {model_path}')

    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError('Could not determine video size.')

    first_probe_ok, first_frame = cap.read()
    if not first_probe_ok:
        raise RuntimeError('Video has no frames.')
    first_frame = _resize_if_needed(first_frame, max_width)
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open output writer: {output_path}')

    frame_index = 0
    previous_output = None
    previous_gray = None

    def _frame_iterator():
        yield first_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield _resize_if_needed(frame, max_width)

    for frame in _frame_iterator():
        if auto_ratio:
            # Video frames are RGB/BGR, not RAW; keep the hook for consistency
            # but do not alter the value in this temporal pipeline.
            pass

        current_output = _process_frame(
            frame,
            net=net,
            device=torch_device,
            strength=strength,
            tile=tile,
            tile_overlap=tile_overlap,
        )

        current_gray = cv2.cvtColor(current_output, cv2.COLOR_BGR2GRAY)

        if previous_output is not None and previous_gray is not None:
            warped_prev = _warp_previous(previous_output, previous_gray, current_gray)
            warped_prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
            alpha = _motion_to_alpha(current_gray, warped_prev_gray, temporal_alpha, max_temporal_alpha)
            current_output = cv2.addWeighted(current_output, alpha, warped_prev, 1.0 - alpha, 0.0)
            current_gray = cv2.cvtColor(current_output, cv2.COLOR_BGR2GRAY)

        writer.write(current_output)
        previous_output = current_output
        previous_gray = current_gray
        frame_index += 1

        if frame_index % 10 == 0:
            print(f'Processed {frame_index} frames...')

    cap.release()
    writer.release()
    print(f'\nDone — saved denoised video to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Temporal PMN video denoising')
    parser.add_argument('--input', '-i', required=False, default='0', help='input video path or camera index')
    parser.add_argument('--output', '-o', required=False, default='results/video_denoised.mp4', help='output video path')
    parser.add_argument('--model', '-m', required=False, default='checkpoints/SonyA7S2_Mix_Unet_best_model.pth', help='model checkpoint path')
    parser.add_argument('--ratio', '-r', type=float, default=250.0, help='retained for CLI consistency')
    parser.add_argument('--strength', '-s', type=float, default=0.35, help='per-frame denoising strength')
    parser.add_argument('--auto_ratio', action='store_true', help='retained for CLI consistency')
    parser.add_argument('--target_luma', type=float, default=0.18, help='retained for CLI consistency')
    parser.add_argument('--tile', type=int, default=768, help='tile size for memory-safe inference')
    parser.add_argument('--tile_overlap', type=int, default=64, help='tile overlap in pixels')
    parser.add_argument('--max_width', type=int, default=1280, help='resize video frames above this width')
    parser.add_argument('--temporal_alpha', type=float, default=0.55, help='base blend weight for current frame')
    parser.add_argument('--max_temporal_alpha', type=float, default=0.85, help='upper bound for motion-aware blending')
    parser.add_argument('--save_first_frame_maps', action='store_true', help='save diagnostic maps for the first frame')
    args = parser.parse_args()

    use_camera = args.input.strip() == '0'
    input_source = 0 if use_camera else _resolve_path(args.input)

    process_video(
        input_path=input_source,
        output_path=_resolve_path(args.output),
        model_path=_resolve_path(args.model),
        ratio=args.ratio,
        strength=args.strength,
        auto_ratio=args.auto_ratio,
        target_luma=args.target_luma,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
        max_width=args.max_width,
        temporal_alpha=args.temporal_alpha,
        max_temporal_alpha=args.max_temporal_alpha,
        save_first_frame_maps=args.save_first_frame_maps,
        use_camera=use_camera,
    )


if __name__ == '__main__':
    main()