"""
Denoise low-light RAW/RGB images using the trained UNetSeeInDark model.

Low-resource extensions in this script:
1. Optional tiled inference for memory-constrained GPUs/laptops.
2. Optional automatic RAW amplification ratio estimation.
3. Extra diagnostic maps (residual + noise-reduction estimate).
"""
import os, sys, torch, numpy as np, cv2
import rawpy
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import argparse
from archs.ELD_models import UNetSeeInDark


# ────────────────────────────────────────────────────────────
def load_model(model_path, device):
    arch = dict(name='UNetSeeInDark', in_nc=4, out_nc=4, nf=32,
                nframes=1, use_dpsv=False, res=False,
                cascade=False, add=False, lock_wb=False)
    net = UNetSeeInDark(arch)
    # PyTorch 2.6+ changed the default behavior of torch.load to be
    # `weights_only=True`. For older-style saved objects we must request
    # `weights_only=False`. Try the safe explicit call and fall back
    # to the older signature if the param isn't supported.
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch doesn't accept weights_only kwarg.
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        # Provide a helpful error message while re-raising.
        raise RuntimeError(f"Failed loading checkpoint '{model_path}': {e}\n"
                           "If the file is from an untrusted source do not proceed.") from e

    net.load_state_dict(state, strict=False)
    return net.to(device).eval()


def pack4(ch):
    """HxW → 4 x H/2 x W/2  (Bayer-style packing)."""
    return np.stack([ch[0::2, 0::2], ch[0::2, 1::2],
                     ch[1::2, 0::2], ch[1::2, 1::2]], axis=0)

def unpack4(c4, H, W):
    """4 x H/2 x W/2 → HxW."""
    f = np.zeros((H, W), np.float32)
    f[0::2, 0::2] = c4[0]; f[0::2, 1::2] = c4[1]
    f[1::2, 0::2] = c4[2]; f[1::2, 1::2] = c4[3]
    return f


def estimate_auto_ratio(raw_norm, target_luma=0.18, percentile=50.0,
                        min_ratio=50.0, max_ratio=300.0):
    """Estimate amplification from RAW brightness statistics."""
    p = float(np.percentile(raw_norm, percentile))
    if p < 1e-6:
        return max_ratio
    ratio = target_luma / p
    return float(np.clip(ratio, min_ratio, max_ratio))


def _infer_patch(net, patch, device):
    """Run one padded patch through the model and return full-resolution output."""
    t = torch.from_numpy(pack4(patch)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = net(t).squeeze(0).cpu().numpy()
    return np.clip(unpack4(out, patch.shape[0], patch.shape[1]), 0, 1)


def denoise_channel(net, ch, device, tile=0, tile_overlap=32):
    """Run one greyscale float32 [0-1] channel through the UNet."""
    H0, W0 = ch.shape
    # pad so packed size is divisible by 16
    need_h = ((H0 + 31) // 32) * 32
    need_w = ((W0 + 31) // 32) * 32
    padded = np.pad(ch, ((0, need_h - H0), (0, need_w - W0)), mode='reflect')

    # Single pass (default): fastest on sufficiently large GPUs.
    if tile <= 0 or (need_h <= tile and need_w <= tile):
        den = _infer_patch(net, padded, device)
        return den[:H0, :W0]

    # Tiled pass: enables high-resolution inference on memory-limited GPUs.
    tile = max(32, int(tile))
    tile = (tile // 32) * 32
    overlap = max(0, int(tile_overlap))
    overlap = min(overlap, tile - 32)
    step = tile - overlap
    if step <= 0:
        raise ValueError("Invalid tile/overlap configuration; require tile > overlap.")

    if tile > need_h or tile > need_w:
        den = _infer_patch(net, padded, device)
        return den[:H0, :W0]

    out = np.zeros((need_h, need_w), dtype=np.float32)
    wgt = np.zeros((need_h, need_w), dtype=np.float32)

    win_1d = np.hanning(tile).astype(np.float32)
    if np.max(win_1d) <= 0:
        win_1d = np.ones((tile,), dtype=np.float32)
    win_2d = np.outer(win_1d, win_1d)
    win_2d = np.maximum(win_2d, 1e-3)

    ys = list(range(0, need_h - tile + 1, step))
    xs = list(range(0, need_w - tile + 1, step))
    if ys[-1] != need_h - tile:
        ys.append(need_h - tile)
    if xs[-1] != need_w - tile:
        xs.append(need_w - tile)

    for y0 in ys:
        for x0 in xs:
            patch = padded[y0:y0 + tile, x0:x0 + tile]
            pred = _infer_patch(net, patch, device)
            out[y0:y0 + tile, x0:x0 + tile] += pred * win_2d
            wgt[y0:y0 + tile, x0:x0 + tile] += win_2d

    den = out / np.maximum(wgt, 1e-6)
    return np.clip(den[:H0, :W0], 0, 1)


# ────────────────────────────────────────────────────────────
def run(image_path, model_path, output_dir='results', ratio=250.0, strength=0.4,
    auto_ratio=False, target_luma=0.18, tile=0, tile_overlap=32,
    save_maps=True, return_results=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    net = load_model(model_path, device)
    print(f"Model  : {model_path}")
    # prepare output naming
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    # Support RAW sensor files using rawpy
    if ext in ('.arw', '.dng', '.nef', '.cr2', '.rw2'):
        try:
            with rawpy.imread(image_path) as raw:
                raw_img = raw.raw_image_visible.astype(np.float32)
                # get white/black levels (fall back to data range)
                white = getattr(raw, 'white_level', None) or float(raw_img.max())
                try:
                    bl = raw.black_level_per_channel
                    black = float(min(bl)) if hasattr(bl, '__iter__') else float(bl)
                except Exception:
                    black = 0.0

                # normalize to [0,1]
                raw_norm = np.clip((raw_img - black) / float(white), 0.0, 1.0)

                H, W = raw_norm.shape
                print(f"RAW Image: {image_path}  ({W}x{H})  white={white} black={black}")

                if auto_ratio:
                    ratio = estimate_auto_ratio(raw_norm, target_luma=target_luma)
                    print(f"Auto amplification ratio: {ratio:.2f} (target_luma={target_luma})")
                else:
                    print(f"Amplification ratio: {ratio}")

                # Apply ratio amplification (as done in training)
                raw_amplified = np.clip(raw_norm * ratio, 0.0, 1.0)

                # run model on amplified RAW Bayer channel
                print("Denoising RAW Bayer with trained model …")
                denoised_raw = denoise_channel(net, raw_amplified, device,
                                              tile=tile, tile_overlap=tile_overlap)

                # demosaic denoised RAW -> RGB for display
                demosaic_in = (np.clip(denoised_raw, 0, 1) * 65535.0).astype(np.uint16)
                # assume RGGB ordering (training used RGGB packed input)
                demosaiced = cv2.cvtColor(demosaic_in, cv2.COLOR_BayerRG2BGR)
                denoised_bgr = (demosaiced.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

                # Save intermediate denoised raw visualization
                denoised_raw_vis = (np.clip(denoised_raw, 0, 1) * 255.0).astype(np.uint8)
                # expand to 3 channels for saving
                drv3 = cv2.cvtColor(denoised_raw_vis, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(output_dir, f"{stem}_denoised_raw.png"), drv3)

                # Save demosaiced denoised image
                cv2.imwrite(os.path.join(output_dir, f"{stem}_denoised_demosaic.png"), denoised_bgr)

                # get camera-processed noisy image for comparison
                noisy_rgb16 = raw.postprocess(output_bps=16)
                noisy_bgr = cv2.cvtColor(noisy_rgb16, cv2.COLOR_RGB2BGR)
                noisy_bgr8 = (noisy_bgr.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

                img_bgr = noisy_bgr8  # original noisy reference for labels

                # Compute simple metrics between noisy and denoised
                try:
                    # convert to RGB for skimage
                    noisy_rgb = cv2.cvtColor(noisy_bgr8, cv2.COLOR_BGR2RGB)
                    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
                    psnr_val = compare_psnr(noisy_rgb, denoised_rgb, data_range=255)
                    ssim_val = compare_ssim(noisy_rgb, denoised_rgb, data_range=255, channel_axis=-1)
                    mad = np.mean(np.abs(noisy_rgb.astype(np.float32) - denoised_rgb.astype(np.float32)))
                    print(f"PSNR (noisy->denoised): {psnr_val:.3f} dB | SSIM: {ssim_val:.4f} | mean-abs-diff: {mad:.2f}")

                    # save diff heatmap
                    diff = np.clip(np.abs(noisy_rgb.astype(np.int16) - denoised_rgb.astype(np.int16)), 0, 255).astype(np.uint8)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    diff_col = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(output_dir, f"{stem}_diff_heatmap.png"), diff_col)
                except Exception as e:
                    print(f"Metric computation failed: {e}")

        except Exception as e:
            sys.exit(f"Cannot read RAW {image_path}: {e}")
    else:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            sys.exit(f"Cannot read {image_path}")
        H, W = img_bgr.shape[:2]
        H -= H % 2; W -= W % 2
        img_bgr = img_bgr[:H, :W]
        print(f"Image  : {image_path}  ({W}x{H})")

    # ── YCrCb split ──────────────────────────────────────────
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y  = ycrcb[:, :, 0].astype(np.float32) / 255.0
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    # ── Luminance: model denoise ─────────────────────────────
    print(f"Denoising luminance (Y) with trained model (strength={strength:.2f}) …")
    Y_dn = denoise_channel(net, Y, device, tile=tile, tile_overlap=tile_overlap)
    # Blend: model handles noise, original keeps structure
    # For RGB images from internet, use lower strength to preserve detail
    alpha = strength  # 0 = keep original, 1 = full model output
    Y_out = np.clip(alpha * Y_dn + (1 - alpha) * Y, 0, 1)
    Y_out = (Y_out * 255).astype(np.uint8)

    # ── Chrominance: bilateral filter ────────────────────────
    if strength > 0.1:
        print("Denoising chrominance (Cr, Cb) …")
        Cr_dn = cv2.bilateralFilter(Cr, 9, int(25*strength), int(25*strength))
        Cb_dn = cv2.bilateralFilter(Cb, 9, int(25*strength), int(25*strength))
    else:
        Cr_dn, Cb_dn = Cr, Cb

    merged = cv2.cvtColor(np.stack([Y_out, Cr_dn, Cb_dn], -1),
                          cv2.COLOR_YCrCb2BGR)

    # ── Final NLM cleanup on full colour ─────────────────────
    # Scale NLM strength: lighter for internet RGB images
    nlm_h = max(3, int(10 * strength))  # 3-10 range
    if strength > 0.05:
        print(f"Final NLM cleanup (h={nlm_h}) …")
        denoised = cv2.fastNlMeansDenoisingColored(
            merged, None, nlm_h, nlm_h, 7, 21)
    else:
        denoised = merged

    # ── NLM-only baseline for comparison ─────────────────────
    if strength > 0.05:
        print("NLM-only baseline …")
        nlm_only = cv2.fastNlMeansDenoisingColored(
            img_bgr, None, nlm_h, nlm_h, 7, 21)
    else:
        nlm_only = img_bgr

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(p := os.path.join(output_dir, f"{stem}_denoised.png"), denoised)
    print(f"Saved  : {p}")
    cv2.imwrite(p := os.path.join(output_dir, f"{stem}_nlm_only.png"), nlm_only)
    print(f"Saved  : {p}")

    # labelled side-by-side
    font, lh = cv2.FONT_HERSHEY_SIMPLEX, 30
    def label(img, txt):
        out = np.zeros((img.shape[0]+lh, img.shape[1], 3), np.uint8)
        out[lh:] = img
        cv2.putText(out, txt, (10, 22), font, 0.7, (255,255,255), 2)
        return out

    comp = np.hstack([label(img_bgr,  "Original (Noisy)"),
                      label(denoised, "PMN Model + NLM"),
                      label(nlm_only, "NLM Only")])
    cv2.imwrite(p := os.path.join(output_dir, f"{stem}_comparison.png"), comp)
    print(f"Saved  : {p}")

    if save_maps:
        gray_noisy = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_dn = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY).astype(np.float32)

        residual = np.clip(np.abs(gray_noisy - gray_dn), 0, 255).astype(np.uint8)
        residual_map = cv2.applyColorMap(residual, cv2.COLORMAP_INFERNO)
        cv2.imwrite(p := os.path.join(output_dir, f"{stem}_residual_map.png"), residual_map)
        print(f"Saved  : {p}")

        # Laplacian magnitude approximates local high-frequency noise energy.
        lap_noisy = np.abs(cv2.Laplacian(gray_noisy, cv2.CV_32F, ksize=3))
        lap_dn = np.abs(cv2.Laplacian(gray_dn, cv2.CV_32F, ksize=3))
        noise_reduction = np.clip(lap_noisy - lap_dn, 0, None)
        noise_reduction = noise_reduction / (noise_reduction.max() + 1e-6)
        noise_reduction_u8 = (noise_reduction * 255.0).astype(np.uint8)
        noise_reduction_map = cv2.applyColorMap(noise_reduction_u8, cv2.COLORMAP_TURBO)
        cv2.imwrite(p := os.path.join(output_dir, f"{stem}_noise_reduction_map.png"), noise_reduction_map)
        print(f"Saved  : {p}")

    print(f"\nDone — results in {output_dir}/")

    if return_results:
        return {
            'input_bgr': img_bgr,
            'output_bgr': denoised,
            'nlm_only_bgr': nlm_only,
            'stem': stem,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PMN inference on an image')
    parser.add_argument('--input', '-i', required=False,
                        help='input image path (positional allowed)',
                        default='test image.png')
    parser.add_argument('--model', '-m', required=False,
                        help='checkpoint path',
                        default='checkpoints/SonyA7S2_Mix_Unet_best_model.pth')
    parser.add_argument('--output_dir', '-o', required=False,
                        help='output directory',
                        default='results')
    parser.add_argument('--ratio', '-r', type=float, required=False,
                        help='amplification ratio for RAW (default 250 for SID eval)',
                        default=250.0)
    parser.add_argument('--auto_ratio', action='store_true',
                        help='auto-estimate RAW amplification ratio from image brightness')
    parser.add_argument('--target_luma', type=float, required=False,
                        help='target normalized brightness for --auto_ratio (default 0.18)',
                        default=0.18)
    parser.add_argument('--strength', '-s', type=float, required=False,
                        help='denoising strength for RGB images: 0=original, 1=max denoise (default 0.4)',
                        default=0.4)
    parser.add_argument('--tile', type=int, required=False,
                        help='tile size for memory-safe inference, 0 disables tiling (default 0)',
                        default=0)
    parser.add_argument('--tile_overlap', type=int, required=False,
                        help='tile overlap in pixels when --tile > 0 (default 32)',
                        default=32)
    parser.add_argument('--save_maps', action='store_true',
                        help='save residual/noise-reduction diagnostic maps')
    args, extras = parser.parse_known_args()

    # Support legacy positional invocation: if first CLI arg isn't a flag,
    # allow calling `python run_inference.py image.pgm model.pth outdir`.
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and not sys.argv[1].startswith('/'): 
        # positional fallback
        img = sys.argv[1]
        mdl = sys.argv[2] if len(sys.argv) > 2 else args.model
        out = sys.argv[3] if len(sys.argv) > 3 else args.output_dir
        run(image_path=img, model_path=mdl, output_dir=out,
            ratio=args.ratio, strength=args.strength,
            auto_ratio=args.auto_ratio, target_luma=args.target_luma,
            tile=args.tile, tile_overlap=args.tile_overlap,
            save_maps=args.save_maps)
    else:
        run(image_path=args.input, model_path=args.model, output_dir=args.output_dir,
            ratio=args.ratio, strength=args.strength,
            auto_ratio=args.auto_ratio, target_luma=args.target_luma,
            tile=args.tile, tile_overlap=args.tile_overlap,
            save_maps=args.save_maps)
