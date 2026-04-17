"""Export PMN/UNetSeeInDark checkpoints for lightweight deployment.

This script creates TorchScript artifacts that can be shipped more easily than
the original Python checkpoint. It also supports an optional fp16 variant on
CUDA for a smaller model footprint.

Export outputs:
- <name>.pt      TorchScript model in fp32
- <name>_fp16.pt TorchScript model in fp16 (CUDA only)
- <name>.json    Export metadata and size report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from archs.ELD_models import UNetSeeInDark


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    arch = dict(name='UNetSeeInDark', in_nc=4, out_nc=4, nf=32,
                nframes=1, use_dpsv=False, res=False,
                cascade=False, add=False, lock_wb=False)
    model = UNetSeeInDark(arch)
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def _maybe_make_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _tensor_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def _compare_outputs(model: torch.nn.Module, scripted: torch.jit.ScriptModule, example: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        ref = model(example)
        out = scripted(example)
    diff = (ref - out).abs()
    return {
        'mean_abs_diff': float(diff.mean().item()),
        'max_abs_diff': float(diff.max().item()),
    }


def export_model(
    model_path: str,
    output_dir: str,
    name: str,
    height: int,
    width: int,
    fp16: bool,
) -> dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = _maybe_make_dir(output_dir)

    model = _load_model(model_path, device)
    base_name = os.path.join(output_dir, name)

    example = torch.rand(1, 4, height, width, device=device, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example, strict=False)
        scripted = torch.jit.freeze(scripted)
    fp32_path = f'{base_name}.pt'
    scripted.save(fp32_path)

    report: dict[str, Any] = {
        'checkpoint': model_path,
        'export_device': str(device),
        'example_shape': [1, 4, height, width],
        'fp32_torchscript': fp32_path,
        'fp32_size_mb': _tensor_size_mb(fp32_path),
        'fp16_torchscript': None,
        'fp16_size_mb': None,
        'fp16_metrics': None,
    }

    report['fp32_metrics'] = _compare_outputs(model, scripted, example)

    if fp16:
        if device.type != 'cuda':
            raise RuntimeError('fp16 export requires CUDA so the traced ops can run in half precision.')

        model_fp16 = _load_model(model_path, device).half()
        example_fp16 = example.half()
        with torch.no_grad():
            scripted_fp16 = torch.jit.trace(model_fp16, example_fp16, strict=False)
            scripted_fp16 = torch.jit.freeze(scripted_fp16)
        fp16_path = f'{base_name}_fp16.pt'
        scripted_fp16.save(fp16_path)

        report['fp16_torchscript'] = fp16_path
        report['fp16_size_mb'] = _tensor_size_mb(fp16_path)
        report['fp16_metrics'] = _compare_outputs(model_fp16, scripted_fp16, example_fp16)

    manifest_path = f'{base_name}.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    report['manifest'] = manifest_path
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Export PMN checkpoint to TorchScript')
    parser.add_argument('--model', '-m', default='checkpoints/SonyA7S2_Mix_Unet_best_model.pth', help='checkpoint path')
    parser.add_argument('--output_dir', '-o', default='exports', help='output directory')
    parser.add_argument('--name', default='pmn_unetseindark', help='base export name')
    parser.add_argument('--height', type=int, default=512, help='example input height, multiple of 32')
    parser.add_argument('--width', type=int, default=512, help='example input width, multiple of 32')
    parser.add_argument('--fp16', action='store_true', help='also export an fp16 TorchScript model (CUDA only)')
    args = parser.parse_args()

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError('height and width must be multiples of 32 for UNet alignment')

    report = export_model(
        model_path=_resolve_path(args.model),
        output_dir=_resolve_path(args.output_dir),
        name=args.name,
        height=args.height,
        width=args.width,
        fp16=args.fp16,
    )

    print('Export summary')
    print('--------------')
    print(f"FP32 TorchScript : {report['fp32_torchscript']} ({report['fp32_size_mb']:.2f} MB)")
    print(f"FP32 diff        : mean={report['fp32_metrics']['mean_abs_diff']:.6f}, max={report['fp32_metrics']['max_abs_diff']:.6f}")
    if report['fp16_torchscript']:
        print(f"FP16 TorchScript : {report['fp16_torchscript']} ({report['fp16_size_mb']:.2f} MB)")
        print(f"FP16 diff        : mean={report['fp16_metrics']['mean_abs_diff']:.6f}, max={report['fp16_metrics']['max_abs_diff']:.6f}")
    print(f"Manifest         : {report['manifest']}")


if __name__ == '__main__':
    main()