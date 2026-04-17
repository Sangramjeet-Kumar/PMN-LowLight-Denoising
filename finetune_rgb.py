"""Lightweight RGB-domain fine-tuning for PMN.

This script adapts the existing RAW denoiser using clean RGB images.
It synthesizes RAW-like training pairs on the fly:
- clean RGB image -> unprocess -> packed clean RAW target
- apply the repo's own noise model -> noisy RAW input

The default mode fine-tunes only the decoder/head to keep the run practical
on a laptop GPU such as an RTX 3050.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from archs.ELD_models import UNetSeeInDark
from data_process.process import generate_noisy_obs, sample_params
from data_process.unprocess import unprocess


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


@dataclass
class FineTuneConfig:
    input_dir: str
    output_dir: str
    checkpoint: str
    model_name: str
    camera_type: str
    epochs: int
    batch_size: int
    lr: float
    patch_size: int
    samples_per_epoch: int
    val_split: float
    rgb_loss_weight: float
    train_scope: str
    num_workers: int
    seed: int
    amp: bool
    save_every: int


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _random_crop(image: np.ndarray, crop_size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        scale = max(crop_size / max(h, 1), crop_size / max(w, 1))
        new_size = (max(crop_size, int(math.ceil(w * scale))), max(crop_size, int(math.ceil(h * scale))))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]

    top = int(rng.integers(0, h - crop_size + 1))
    left = int(rng.integers(0, w - crop_size + 1))
    return image[top:top + crop_size, left:left + crop_size]


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.integers(2):
        image = np.flip(image, axis=0)
    if rng.integers(2):
        image = np.flip(image, axis=1)
    rotations = int(rng.integers(4))
    if rotations:
        image = np.rot90(image, k=rotations)
    return np.ascontiguousarray(image)


def _unpack4(c4: torch.Tensor) -> np.ndarray:
    c4 = c4.detach().cpu().float().numpy()
    h2, w2 = c4.shape[1], c4.shape[2]
    full = np.zeros((h2 * 2, w2 * 2), dtype=np.float32)
    full[0::2, 0::2] = c4[0]
    full[0::2, 1::2] = c4[1]
    full[1::2, 1::2] = c4[2]
    full[1::2, 0::2] = c4[3]
    return full


def _pack_rgb_like(clean_linear: torch.Tensor) -> torch.Tensor:
    """Pack linear RGB into the RGGB-style 4-channel layout used by PMN."""
    if clean_linear.ndim != 3 or clean_linear.shape[-1] != 3:
        raise ValueError(f'Expected HxWx3 tensor, got {tuple(clean_linear.shape)}')
    chw = clean_linear.permute(2, 0, 1).contiguous()
    red = chw[0, 0::2, 0::2]
    green_red = chw[1, 0::2, 1::2]
    blue = chw[2, 1::2, 1::2]
    green_blue = chw[1, 1::2, 0::2]
    return torch.stack((red, green_red, blue, green_blue), dim=0).contiguous().float()


def _demosaic_to_rgb(packed_raw: torch.Tensor) -> np.ndarray:
    raw = np.clip(_unpack4(packed_raw), 0.0, 1.0)
    raw_u16 = (raw * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(raw_u16, cv2.COLOR_BayerRG2RGB)
    return rgb.astype(np.float32) / 65535.0


def _load_checkpoint(model_path: str, device: torch.device) -> UNetSeeInDark:
    arch = dict(name='UNetSeeInDark', in_nc=4, out_nc=4, nf=32,
                nframes=1, use_dpsv=False, res=False,
                cascade=False, add=False, lock_wb=False)
    model = UNetSeeInDark(arch)
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model.to(device)


def _set_trainable_scope(model: torch.nn.Module, scope: str) -> None:
    encoder_prefixes = ('conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2')
    decoder_prefixes = ('upv6', 'conv6_1', 'conv6_2', 'upv7', 'conv7_1', 'conv7_2', 'upv8', 'conv8_1', 'conv8_2', 'upv9', 'conv9_1', 'conv9_2', 'conv10_1')

    for name, param in model.named_parameters():
        if scope == 'full':
            param.requires_grad = True
        elif scope == 'head':
            param.requires_grad = name.startswith(('conv9_1', 'conv9_2', 'conv10_1'))
        else:  # decoder default
            param.requires_grad = name.startswith(decoder_prefixes)


class SyntheticRGBFineTuneDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        samples_per_epoch: int,
        patch_size: int,
        camera_type: str,
        seed: int,
        augment: bool = True,
    ):
        self.image_paths = list(image_paths)
        self.samples_per_epoch = max(1, int(samples_per_epoch))
        self.patch_size = int(patch_size)
        self.camera_type = camera_type
        self.seed = int(seed)
        self.augment = augment

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        path = self.image_paths[int(rng.integers(0, len(self.image_paths)))]
        rgb = _load_image(path)
        rgb = _random_crop(rgb, self.patch_size * 2, rng)
        if self.augment:
            rgb = _augment(rgb, rng)

        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        clean_linear, _ = unprocess(rgb_t, lock_wb=False, use_gpu=False)
        clean_packed = _pack_rgb_like(clean_linear)

        noise_param = sample_params(camera_type=self.camera_type)
        noisy_packed = generate_noisy_obs(
            clean_packed.detach().cpu().numpy().astype(np.float32),
            camera_type=self.camera_type,
            param=noise_param,
            noise_code='pgrq',
            ori=False,
            clip=False,
        ).astype(np.float32)

        return {
            'lr': torch.from_numpy(noisy_packed),
            'hr': clean_packed,
            'name': path.stem,
        }


def _split_paths(paths: List[Path], val_split: float, seed: int) -> tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    if len(paths) == 1:
        return paths, paths
    val_count = max(1, int(round(len(paths) * val_split)))
    val_count = min(val_count, len(paths) - 1)
    return paths[val_count:], paths[:val_count]


def _collate(batch):
    lr = torch.stack([item['lr'] for item in batch], dim=0).float()
    hr = torch.stack([item['hr'] for item in batch], dim=0).float()
    names = [item['name'] for item in batch]
    return {'lr': lr, 'hr': hr, 'name': names}


def _make_rgb_preview(model: torch.nn.Module, sample: dict, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        lr = sample['lr'].unsqueeze(0).to(device)
        hr = sample['hr'].unsqueeze(0).to(device)
        pred = torch.clamp(model(lr), 0, 1)[0].cpu()
        hr = hr[0].cpu()

    pred_rgb = _demosaic_to_rgb(pred)
    hr_rgb = _demosaic_to_rgb(hr)
    lr_rgb = _demosaic_to_rgb(sample['lr'])

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        pad = np.zeros((img_u8.shape[0] + 36, img_u8.shape[1], 3), dtype=np.uint8)
        pad[36:] = img_u8
        cv2.putText(pad, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return pad

    return np.hstack([
        _label(lr_rgb, 'Noisy synthetic'),
        _label(pred_rgb, 'Fine-tuned output'),
        _label(hr_rgb, 'Synthetic target'),
    ])


def fine_tune(cfg: FineTuneConfig) -> dict:
    _set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    checkpoint = _resolve_path(cfg.checkpoint)
    model = _load_checkpoint(checkpoint, device)
    _set_trainable_scope(model, cfg.train_scope)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f'No trainable parameters found for scope={cfg.train_scope}')

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')

    input_dir = Path(_resolve_path(cfg.input_dir))
    output_dir = Path(_resolve_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    sample_dir = output_dir / 'samples'
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_paths = [p for p in sorted(input_dir.rglob('*')) if p.suffix.lower() in exts]
    if not image_paths:
        raise RuntimeError(f'No RGB images found under {input_dir}')

    train_paths, val_paths = _split_paths(image_paths, cfg.val_split, cfg.seed)
    train_ds = SyntheticRGBFineTuneDataset(train_paths, cfg.samples_per_epoch, cfg.patch_size, cfg.camera_type, cfg.seed, augment=True)
    val_ds = SyntheticRGBFineTuneDataset(val_paths, max(4, min(16, len(val_paths) * 2)), cfg.patch_size, cfg.camera_type, cfg.seed + 999, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=_collate)

    best_val = float('inf')
    history = []

    def _step(batch, train: bool):
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            pred = torch.clamp(model(lr), 0, 1)
            loss_raw = F.l1_loss(pred, hr)
            pred_rgb = torch.stack([torch.from_numpy(_demosaic_to_rgb(p.detach().cpu())).to(device) for p in pred], dim=0)
            hr_rgb = torch.stack([torch.from_numpy(_demosaic_to_rgb(h.detach().cpu())).to(device) for h in hr], dim=0)
            loss_rgb = F.l1_loss(pred_rgb, hr_rgb)
            loss = loss_raw + cfg.rgb_loss_weight * loss_rgb
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return loss.item(), loss_raw.item(), loss_rgb.item(), pred.detach().cpu(), hr.detach().cpu()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_raw = 0.0
        train_rgb = 0.0
        train_psnr = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs}', leave=False)
        for batch in pbar:
            loss, loss_raw, loss_rgb, pred_cpu, hr_cpu = _step(batch, train=True)
            batch_psnr = 0.0
            for pred_item, hr_item in zip(pred_cpu, hr_cpu):
                pred_np = pred_item.numpy()
                hr_np = hr_item.numpy()
                batch_psnr += -10.0 * math.log10(float(np.mean((pred_np - hr_np) ** 2) + 1e-12))
            batch_psnr /= max(1, len(pred_cpu))

            train_loss += loss
            train_raw += loss_raw
            train_rgb += loss_rgb
            train_psnr += batch_psnr
            count += 1
            pbar.set_postfix(loss=f'{loss:.4f}', raw=f'{loss_raw:.4f}', rgb=f'{loss_rgb:.4f}', psnr=f'{batch_psnr:.2f}')

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_count = 0
        preview_batch = None
        with torch.no_grad():
            for batch in val_loader:
                loss, loss_raw, loss_rgb, pred_cpu, hr_cpu = _step(batch, train=False)
                val_loss += loss
                for pred_item, hr_item in zip(pred_cpu, hr_cpu):
                    pred_np = pred_item.numpy()
                    hr_np = hr_item.numpy()
                    val_psnr += -10.0 * math.log10(float(np.mean((pred_np - hr_np) ** 2) + 1e-12))
                val_count += 1
                if preview_batch is None:
                    preview_batch = {k: v[0].cpu() for k, v in batch.items() if torch.is_tensor(v)}

        train_loss /= max(1, count)
        train_raw /= max(1, count)
        train_rgb /= max(1, count)
        train_psnr /= max(1, count)
        val_loss /= max(1, val_count)
        val_psnr /= max(1, val_count)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_raw_loss': train_raw,
            'train_rgb_loss': train_rgb,
            'train_psnr': train_psnr,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
        })

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_psnr={val_psnr:.2f} dB')

        last_path = ckpt_dir / f'{cfg.model_name}_last_model.pth'
        torch.save(model.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / f'{cfg.model_name}_best_model.pth'
            torch.save(model.state_dict(), best_path)

        if preview_batch is not None:
            preview = _make_rgb_preview(model, preview_batch, device)
            preview_path = sample_dir / f'{cfg.model_name}_epoch_{epoch:03d}.png'
            cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        if epoch % cfg.save_every == 0:
            history_path = output_dir / 'history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

    final_report = {
        'config': asdict(cfg),
        'best_val_loss': best_val,
        'history': history,
        'checkpoint_dir': str(ckpt_dir),
    }
    with open(output_dir / 'fine_tune_report.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description='Lightweight RGB-domain fine-tuning for PMN')
    parser.add_argument('--input_dir', '-i', required=True, help='directory containing clean RGB images')
    parser.add_argument('--output_dir', '-o', default='rgb_finetune_runs', help='output directory')
    parser.add_argument('--checkpoint', '-m', default='checkpoints/SonyA7S2_Mix_Unet_best_model.pth', help='starting checkpoint')
    parser.add_argument('--model_name', default='pmn_rgb_ft', help='checkpoint name prefix')
    parser.add_argument('--camera_type', default='SonyA7S2', help='camera noise preset used for synthetic RAW generation')
    parser.add_argument('--epochs', type=int, default=3, help='number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--patch_size', type=int, default=128, help='packed RAW patch size (RGB crop is 2x this size)')
    parser.add_argument('--samples_per_epoch', type=int, default=64, help='number of synthetic samples per epoch')
    parser.add_argument('--val_split', type=float, default=0.1, help='validation split fraction')
    parser.add_argument('--rgb_loss_weight', type=float, default=0.25, help='weight for RGB consistency loss')
    parser.add_argument('--train_scope', choices=['decoder', 'head', 'full'], default='decoder', help='which part of the model to fine-tune')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 is safest on Windows)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    parser.add_argument('--save_every', type=int, default=1, help='save history every N epochs')
    args = parser.parse_args()

    if args.patch_size % 32 != 0:
        raise ValueError('patch_size must be a multiple of 32')

    cfg = FineTuneConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        camera_type=args.camera_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        val_split=args.val_split,
        rgb_loss_weight=args.rgb_loss_weight,
        train_scope=args.train_scope,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        save_every=args.save_every,
    )

    report = fine_tune(cfg)
    print('Fine-tuning summary')
    print('-------------------')
    print(f"Best val loss: {report['best_val_loss']:.6f}")
    print(f"Checkpoint dir: {report['checkpoint_dir']}")


if __name__ == '__main__':
    main()