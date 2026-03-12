#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm


def _frame_key(path):
    base = os.path.basename(path)
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return base
    return base


def _sorted_paths(pattern):
    return sorted(glob(pattern), key=_frame_key)


def _read_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def _read_mask(path):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.float32)
    # Support masks stored as {0,1} or {0,255}.
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask


def _resolve_bg_mask(mask, mask_mode):
    nonzero_frac = float((mask > 0.5).mean())
    if mask_mode == "auto":
        if nonzero_frac >= 0.5:
            bg_mask = mask > 0.5
            mode_used = "nonzero_is_bg"
        else:
            bg_mask = mask <= 0.5
            mode_used = "nonzero_is_tool"
    elif mask_mode == "nonzero_is_bg":
        bg_mask = mask > 0.5
        mode_used = mask_mode
    else:
        bg_mask = mask <= 0.5
        mode_used = mask_mode
    return bg_mask.astype(bool), mode_used, nonzero_frac


def _psnr(pred, gt, bg_mask):
    mask = bg_mask.astype(np.float32)
    if mask.sum() <= 0:
        return float("nan")
    diff = (pred - gt) ** 2
    mse = (diff * mask[..., None]).sum() / (mask.sum() * pred.shape[-1])
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _ssim(pred, gt, bg_mask, device, ssim_fn):
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(bg_mask.astype(np.bool_)).unsqueeze(0).unsqueeze(0)
    mask_t = mask_t.expand(1, pred_t.shape[1], pred_t.shape[2], pred_t.shape[3]).to(device)
    with torch.no_grad():
        return float(ssim_fn(pred_t, gt_t, mask=mask_t).item())


def _lpips(pred, gt, bg_mask, device, lpips_model):
    if lpips_model is None:
        return float("nan")
    mask = bg_mask.astype(np.float32)[..., None]
    pred_masked = pred * mask + gt * (1.0 - mask)
    pred_t = torch.from_numpy(pred_masked).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    pred_t = pred_t * 2.0 - 1.0
    gt_t = gt_t * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_model(pred_t, gt_t).mean().item())


def _rotation_error_deg(R):
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _pose_errors(T_est, T_gt):
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    trans_err = float(np.linalg.norm(t_est - t_gt))
    rot_err = _rotation_error_deg(R_est @ R_gt.T)
    return trans_err, rot_err


def _relative_pose(T):
    return np.linalg.inv(T[:-1]) @ T[1:]


def _rmse(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt((arr ** 2).mean()))


def compute_pose_metrics(est_path, gt_path):
    if not (os.path.isfile(est_path) and os.path.isfile(gt_path)):
        return {}
    est = np.load(est_path)
    gt = np.load(gt_path)
    n = min(len(est), len(gt))
    est = est[:n]
    gt = gt[:n]

    trans_errs = []
    rot_errs = []
    for i in range(n):
        t_err, r_err = _pose_errors(est[i], gt[i])
        trans_errs.append(t_err)
        rot_errs.append(r_err)

    ate_m = _rmse(trans_errs)

    if n < 2:
        rpe_t = float("nan")
        rpe_r = float("nan")
    else:
        rel_est = _relative_pose(est)
        rel_gt = _relative_pose(gt)
        rel_n = min(len(rel_est), len(rel_gt))
        trans_errs = []
        rot_errs = []
        for i in range(rel_n):
            t_err, r_err = _pose_errors(rel_est[i], rel_gt[i])
            trans_errs.append(t_err)
            rot_errs.append(r_err)
        rpe_t = _rmse(trans_errs)
        rpe_r = _rmse(rot_errs)

    return {
        "ate_m": ate_m,
        "rpet_m": rpe_t,
        "rper_deg": rpe_r,
    }


def compute_recon_metrics(gt_paths, render_paths, mask_paths, mask_mode, device, ssim_fn):
    if len(gt_paths) == 0 or len(render_paths) == 0:
        raise RuntimeError("No GT or render frames found.")
    n = min(len(gt_paths), len(render_paths), len(mask_paths))
    if n == 0:
        raise RuntimeError("No matching GT/render/mask frames found.")
    if not (len(gt_paths) == len(render_paths) == len(mask_paths)):
        print(
            f"[warn] frame count mismatch: gt={len(gt_paths)}, render={len(render_paths)}, mask={len(mask_paths)}; using first {n}."
        )

    lpips_model = None
    try:
        import lpips as lpips_lib

        lpips_model = lpips_lib.LPIPS(net="alex").to(device)
        lpips_model.eval()
    except Exception as exc:
        print(f"[warn] lpips not available ({exc}); LPIPS will be NaN.")

    per_frame = []
    for idx in tqdm(range(n), desc="eval"):
        gt_path = gt_paths[idx]
        render_path = render_paths[idx]
        mask_path = mask_paths[idx]

        gt = _read_rgb(gt_path)
        pred = _read_rgb(render_path)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: gt={gt.shape} pred={pred.shape} ({gt_path} vs {render_path})")

        mask = _read_mask(mask_path)
        if mask.shape[:2] != gt.shape[:2]:
            raise ValueError(f"Mask size mismatch: mask={mask.shape} gt={gt.shape} ({mask_path})")

        bg_mask, mode_used, nonzero_frac = _resolve_bg_mask(mask, mask_mode)
        bg_frac = float(bg_mask.mean())

        psnr = _psnr(pred, gt, bg_mask)
        ssim_val = _ssim(pred, gt, bg_mask, device, ssim_fn)
        lpips_val = _lpips(pred, gt, bg_mask, device, lpips_model)

        per_frame.append(
            {
                "frame_idx": idx,
                "frame_name": os.path.basename(gt_path),
                "psnr": psnr,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "bg_frac": bg_frac,
                "mask_nonzero_frac": nonzero_frac,
                "mask_mode_used": mode_used,
            }
        )

    return per_frame


def _mean_value(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _resolve_output_dir(repo_root, seq, output_dir, render_subdir):
    if output_dir is not None:
        return output_dir
    base = os.path.join(repo_root, "output")
    if not os.path.isdir(base):
        raise RuntimeError("Could not find output directory; pass --output-dir explicitly.")
    def _norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())
    seq_norm = _norm(seq)
    candidates = []
    for name in os.listdir(base):
        cand = os.path.join(base, name)
        if not os.path.isdir(cand):
            continue
        if seq_norm not in _norm(name):
            continue
        if os.path.isdir(os.path.join(cand, render_subdir)):
            candidates.append(cand)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise RuntimeError("No matching output directory found; pass --output-dir explicitly.")
    raise RuntimeError(f"Multiple output dirs match seq '{seq}': {candidates}. Please pass --output-dir.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render-subdir", default="raw_rgb/render")
    parser.add_argument("--gt-subdir", default="video_frames")
    parser.add_argument("--mask-subdir", default="masks")
    parser.add_argument(
        "--mask-mode",
        default="nonzero_is_bg",
        choices=["auto", "nonzero_is_bg", "nonzero_is_tool"],
        help="How to interpret mask pixels.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = os.path.abspath(args.repo_root)

    sys.path.insert(0, repo_root)
    from src.utils.loss_utils import ssim as ssim_fn

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU.")
            device = "cpu"

    input_root = args.input_root
    output_dir = _resolve_output_dir(repo_root, args.seq, args.output_dir, args.render_subdir)
    seq = args.seq

    gt_dir = os.path.join(input_root, seq, args.gt_subdir)
    mask_dir = os.path.join(input_root, seq, args.mask_subdir)
    render_dir = os.path.join(output_dir, args.render_subdir)

    gt_paths = _sorted_paths(os.path.join(gt_dir, "*l.png"))
    mask_paths = _sorted_paths(os.path.join(mask_dir, "*l.png"))
    render_paths = _sorted_paths(os.path.join(render_dir, "*.png"))

    per_frame = compute_recon_metrics(gt_paths, render_paths, mask_paths, args.mask_mode, device, ssim_fn)

    psnr_vals = [row["psnr"] for row in per_frame]
    ssim_vals = [row["ssim"] for row in per_frame]
    lpips_vals = [row["lpips"] for row in per_frame]

    recon_summary = {
        "psnr": _mean_value(psnr_vals),
        "ssim": _mean_value(ssim_vals),
        "lpips": _mean_value(lpips_vals),
    }

    pose_summary = {}
    gt_pose = os.path.join(output_dir, "gt_c2w.npy")
    input_pose = os.path.join(output_dir, "input_c2w.npy")
    opt_pose = os.path.join(output_dir, "optimized_c2w.npy")
    pose_source = None
    pose_path = None
    if os.path.isfile(gt_pose):
        if os.path.isfile(opt_pose):
            pose_source = "optimized"
            pose_path = opt_pose
        elif os.path.isfile(input_pose):
            pose_source = "input"
            pose_path = input_pose
    if pose_source is not None:
        pose_summary = compute_pose_metrics(pose_path, gt_pose)

    if pose_summary:
        ate_m = pose_summary.get("ate_m")
        rper_deg = pose_summary.get("rper_deg")
        rpet_m = pose_summary.get("rpet_m")
    else:
        ate_m = None
        rper_deg = None
        rpet_m = None

    summary = {
        "psnr": recon_summary["psnr"],
        "ssim": recon_summary["ssim"],
        "lpips": recon_summary["lpips"],
        "ate_m": ate_m,
        "rper_deg": rper_deg,
        "rpet_m": rpet_m,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
