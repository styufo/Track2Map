#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import sys
import tempfile
from glob import glob

# Must be set before the first CUDA context is created. The evaluator records and
# enforces this setting again when deterministic execution is configured.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation


METRICS_SCHEMA_VERSION = "track2map.metrics.v2"
EVALUATION_KINDS = (
    "reconstruction_and_pose",
    "independent_verification",
    "pose_only_diagnostic",
)
REQUIRED_POSE_METRICS = ("ate_m", "are_deg", "rpet_m", "rper_deg")
REQUIRED_RECON_METRICS = ("psnr", "ssim", "lpips")
REFERENCE_TOLERANCES = {
    "psnr": {"absolute": 2e-6, "relative": 1e-9},
    "ssim": {"absolute": 2e-6, "relative": 1e-9},
    "lpips": {"absolute": 1e-6, "relative": 1e-9},
    "ate_m": {"absolute": 1e-10, "relative": 1e-9},
    "are_deg": {"absolute": 1e-7, "relative": 1e-9},
    "rpet_m": {"absolute": 1e-10, "relative": 1e-9},
    "rper_deg": {"absolute": 1e-7, "relative": 1e-9},
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _read_bound_bytes(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return data, _sha256_bytes(data)


def _read_bound_json(path):
    data, digest = _read_bound_bytes(path)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc
    return payload, digest


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _array_sha256(array):
    array = np.ascontiguousarray(np.asarray(array))
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(descriptor + b"\0" + array.tobytes(order="C"))


def _atomic_write_json(path, payload):
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path) or os.curdir
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
        try:
            directory_fd = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file replacement itself is already atomic. Some filesystems do
            # not support fsync on a directory descriptor.
            pass
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_determinism(seed=0):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8'.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "random_seed": seed,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(
            getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            getattr(getattr(torch.backends, "cuda", None), "matmul", None).allow_tf32
        ) if hasattr(getattr(torch.backends, "cuda", None), "matmul") else None,
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


def _runtime_provenance(requested_device, resolved_device, determinism):
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "torchvision",
                "imageio",
                "Pillow",
                "lpips",
            )
        },
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "determinism": determinism,
    }
    if str(resolved_device).startswith("cuda"):
        index = torch.cuda.current_device()
        result["cuda_device_index"] = int(index)
        result["cuda_device_name"] = torch.cuda.get_device_name(index)
        result["cuda_device_capability"] = list(torch.cuda.get_device_capability(index))
    return result


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


def _read_image_bytes(path, expected_sha256=None):
    data, digest = _read_bound_bytes(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Artifact changed before metric derivation: {path}; "
            f"expected_sha256={expected_sha256} actual_sha256={digest}"
        )
    return imageio.imread(io.BytesIO(data))


def _read_rgb(path, expected_sha256=None):
    img = _read_image_bytes(path, expected_sha256=expected_sha256)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def _read_mask(path, expected_sha256=None):
    mask = _read_image_bytes(path, expected_sha256=expected_sha256)
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
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


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
    if arr.size == 0:
        return float("nan")
    if not np.all(np.isfinite(arr)):
        raise ValueError("RMSE input contains non-finite values.")
    return float(np.sqrt((arr ** 2).mean()))


def _decode_pose_trajectory(path, data, start=0, stop=None, step=1):
    frame_ids = None
    if path.lower().endswith(".npy"):
        poses = np.load(io.BytesIO(data), allow_pickle=False)
    else:
        rows = []
        frame_ids = []
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Pose file is not valid UTF-8: {path}") from exc
        for line in lines:
            values = line.replace(",", " ").split()
            if not values or values[0].startswith("#"):
                continue
            if len(values) < 8:
                raise ValueError(f"Malformed pose row in {path}: {line.rstrip()}")
            frame_ids.append(values[0])
            rows.append([float(value) for value in values[1:8]])
        if not rows:
            raise ValueError(f"No poses found in {path}")
        rows = np.asarray(rows, dtype=np.float64)
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(rows), axis=0)
        poses[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
        poses[:, :3, 3] = rows[:, :3]
        poses = np.linalg.inv(poses[0])[None, ...] @ poses
    frame_slice = slice(start, stop, step)
    poses = np.asarray(poses)[frame_slice]
    if frame_ids is not None:
        frame_ids = frame_ids[frame_slice]
    return poses, frame_ids


def load_pose_trajectory_with_ids(path, start=0, stop=None, step=1):
    data, _ = _read_bound_bytes(path)
    return _decode_pose_trajectory(path, data, start=start, stop=stop, step=step)


def load_pose_trajectory(path, start=0, stop=None, step=1):
    poses, _ = load_pose_trajectory_with_ids(path, start=start, stop=stop, step=step)
    return poses


def validate_pose_trajectory(poses, label):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{label} must have shape [N,4,4], got {poses.shape}")
    if len(poses) == 0:
        raise ValueError(f"{label} is empty.")
    if not np.all(np.isfinite(poses)):
        bad = np.argwhere(~np.isfinite(poses))[0]
        raise ValueError(f"{label} contains a non-finite value at index {tuple(int(v) for v in bad)}")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(poses[:, 3, :], expected_bottom, atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} contains a non-homogeneous bottom row.")
    rotations = poses[:, :3, :3]
    orthogonality = np.linalg.norm(
        np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3, dtype=np.float64),
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)
    invalid = np.where((orthogonality > 5e-3) | (np.abs(determinants - 1.0) > 5e-3))[0]
    if invalid.size:
        idx = int(invalid[0])
        raise ValueError(
            f"{label} contains a non-rigid rotation at frame {idx}: "
            f"orthogonality_error={orthogonality[idx]} determinant={determinants[idx]}"
        )
    return poses


def compute_pose_metrics_with_rows(est, gt, frame_ids=None):
    if isinstance(est, (str, os.PathLike)):
        est = np.load(est)
    if isinstance(gt, (str, os.PathLike)):
        gt = np.load(gt)
    est = validate_pose_trajectory(est, "estimate trajectory")
    gt = validate_pose_trajectory(gt, "evaluation trajectory")
    if len(est) != len(gt):
        raise ValueError(f"Pose count mismatch: estimate={len(est)} evaluation_gt={len(gt)}")
    n = len(est)
    if frame_ids is not None and len(frame_ids) != n:
        raise ValueError(f"Pose/frame ID count mismatch: ids={len(frame_ids)} poses={n}")

    trans_errs = []
    rot_errs = []
    rows = []
    for i in range(n):
        t_err, r_err = _pose_errors(est[i], gt[i])
        trans_errs.append(t_err)
        rot_errs.append(r_err)
        rows.append(
            {
                "frame_idx": i,
                "pose_frame_id": None if frame_ids is None else frame_ids[i],
                "absolute_translation_error_m": t_err,
                "absolute_rotation_error_deg": r_err,
                "relative_translation_error_m": None,
                "relative_rotation_error_deg": None,
            }
        )

    ate_m = _rmse(trans_errs)
    are_deg = _rmse(rot_errs)

    if n < 2:
        rpe_t = float("nan")
        rpe_r = float("nan")
    else:
        rel_est = _relative_pose(est)
        rel_gt = _relative_pose(gt)
        rel_n = len(rel_est)
        trans_errs = []
        rot_errs = []
        for i in range(rel_n):
            t_err, r_err = _pose_errors(rel_est[i], rel_gt[i])
            trans_errs.append(t_err)
            rot_errs.append(r_err)
            rows[i + 1]["relative_translation_error_m"] = t_err
            rows[i + 1]["relative_rotation_error_deg"] = r_err
        rpe_t = _rmse(trans_errs)
        rpe_r = _rmse(rot_errs)

    summary = {
        "ate_m": ate_m,
        "are_deg": are_deg,
        "rpet_m": rpe_t,
        "rper_deg": rpe_r,
    }
    return summary, rows


def compute_pose_metrics(est, gt):
    summary, _ = compute_pose_metrics_with_rows(est, gt)
    return summary


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model):
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_frame_bindings_current(frame_bindings):
    path_hash_pairs = (
        ("gt_path", "gt_sha256"),
        ("runtime_gt_path", "runtime_gt_sha256"),
        ("render_path", "render_sha256"),
        ("mask_path", "mask_sha256"),
        ("right_rgb_path", "right_rgb_sha256"),
        ("semantic_path", "semantic_sha256"),
    )
    for idx, binding in enumerate(frame_bindings):
        for path_key, hash_key in path_hash_pairs:
            path = binding[path_key]
            actual = _sha256(path)
            expected = binding[hash_key]
            if actual != expected:
                raise ValueError(
                    f"Bound artifact changed during evaluation at frame {idx}: "
                    f"{path_key}={path}; expected_sha256={expected} actual_sha256={actual}"
                )


def _frame_bindings_sha256(frame_bindings):
    return _canonical_json_sha256(frame_bindings)


def bind_reconstruction_frames(gt_paths, runtime_gt_paths, render_paths, mask_paths, pose_frame_ids):
    counts = {
        "gt": len(gt_paths),
        "runtime_gt": len(runtime_gt_paths),
        "render": len(render_paths),
        "mask": len(mask_paths),
    }
    if not counts["gt"] or not counts["render"]:
        raise RuntimeError(f"Missing reconstruction frames: {counts}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Reconstruction frame count mismatch: {counts}")
    if pose_frame_ids is None:
        raise ValueError("Evaluation pose file must expose frame IDs for reconstruction binding.")
    if len(pose_frame_ids) != counts["gt"]:
        raise ValueError(
            f"Pose/image identity count mismatch: pose_ids={len(pose_frame_ids)} images={counts['gt']}"
        )

    gt_names = [os.path.basename(path) for path in gt_paths]
    mask_names = [os.path.basename(path) for path in mask_paths]
    if mask_names != gt_names:
        raise ValueError("GT/mask frame identities do not match exactly.")

    runtime_names = [os.path.basename(path) for path in runtime_gt_paths]
    render_names = [os.path.basename(path) for path in render_paths]
    expected_runtime_names = [f"{idx:05d}.png" for idx in range(counts["gt"])]
    if runtime_names != expected_runtime_names:
        raise ValueError(
            "Runtime GT frame identities are not the expected contiguous local indices: "
            f"expected={expected_runtime_names[:3]}... actual={runtime_names[:3]}..."
        )
    if render_names != expected_runtime_names:
        raise ValueError(
            "Render frame identities are not the expected contiguous local indices: "
            f"expected={expected_runtime_names[:3]}... actual={render_names[:3]}..."
        )

    dataset_frame_ids = [os.path.splitext(name)[0].removesuffix("l") for name in gt_names]
    if dataset_frame_ids != list(pose_frame_ids):
        mismatch = next(
            (
                idx,
                dataset_frame_ids[idx],
                pose_frame_ids[idx],
            )
            for idx in range(len(dataset_frame_ids))
            if dataset_frame_ids[idx] != pose_frame_ids[idx]
        )
        raise ValueError(
            "Pose/image frame identity mismatch at local index "
            f"{mismatch[0]}: image={mismatch[1]} pose={mismatch[2]}"
        )

    bindings = []
    for idx, (gt_path, runtime_gt_path, render_path, mask_path) in enumerate(
        zip(gt_paths, runtime_gt_paths, render_paths, mask_paths)
    ):
        right_rgb_path = gt_path.replace("l.png", "r.png")
        semantic_path = gt_path.replace("video_frames", "semantic_predictions")
        if not os.path.isfile(right_rgb_path):
            raise FileNotFoundError(f"Missing right stereo frame: {right_rgb_path}")
        if not os.path.isfile(semantic_path):
            raise FileNotFoundError(f"Missing semantic input frame: {semantic_path}")
        gt_hash = _sha256(gt_path)
        runtime_gt_hash = _sha256(runtime_gt_path)
        if gt_hash != runtime_gt_hash:
            raise ValueError(
                f"Runtime GT is not an exact copy of dataset frame {gt_names[idx]}: "
                f"dataset_sha256={gt_hash} runtime_sha256={runtime_gt_hash}"
            )
        bindings.append(
            {
                "local_frame_idx": idx,
                "dataset_frame_id": dataset_frame_ids[idx],
                "dataset_frame_name": gt_names[idx],
                "runtime_frame_name": runtime_names[idx],
                "pose_frame_id": pose_frame_ids[idx],
                "gt_path": os.path.abspath(gt_path),
                "runtime_gt_path": os.path.abspath(runtime_gt_path),
                "render_path": os.path.abspath(render_path),
                "mask_path": os.path.abspath(mask_path),
                "right_rgb_path": os.path.abspath(right_rgb_path),
                "semantic_path": os.path.abspath(semantic_path),
                "gt_sha256": gt_hash,
                "runtime_gt_sha256": runtime_gt_hash,
                "render_sha256": _sha256(render_path),
                "mask_sha256": _sha256(mask_path),
                "right_rgb_sha256": _sha256(right_rgb_path),
                "semantic_sha256": _sha256(semantic_path),
            }
        )
    return bindings


def compute_recon_metrics(
    gt_paths,
    render_paths,
    mask_paths,
    mask_mode,
    device,
    ssim_fn,
    frame_bindings=None,
    metric_provenance_out=None,
):
    if len(gt_paths) == 0 or len(render_paths) == 0:
        raise RuntimeError("No GT or render frames found.")
    counts = {
        "gt": len(gt_paths),
        "render": len(render_paths),
        "mask": len(mask_paths),
    }
    if counts["mask"] == 0:
        raise RuntimeError("No matching GT/render/mask frames found.")
    if len(set(counts.values())) != 1:
        raise ValueError(
            "Reconstruction frame count mismatch: "
            f"gt={counts['gt']} render={counts['render']} mask={counts['mask']}."
        )
    n = counts["gt"]

    try:
        import lpips as lpips_lib

        lpips_model = lpips_lib.LPIPS(net="alex").to(device)
        lpips_model.eval()
    except Exception as exc:
        raise RuntimeError(f"LPIPS is required for complete reconstruction evaluation: {exc}") from exc

    if metric_provenance_out is not None:
        metric_provenance_out.update(
            {
                "lpips_distribution_version": _package_version("lpips"),
                "lpips_network": "alex",
                "lpips_model_class": (
                    f"{lpips_model.__class__.__module__}.{lpips_model.__class__.__qualname__}"
                ),
                "lpips_state_dict_sha256": _model_state_sha256(lpips_model),
            }
        )

    per_frame = []
    for idx in tqdm(range(n), desc="eval"):
        gt_path = gt_paths[idx]
        render_path = render_paths[idx]
        mask_path = mask_paths[idx]

        binding = None if frame_bindings is None else frame_bindings[idx]
        gt = _read_rgb(
            gt_path,
            expected_sha256=None if binding is None else binding["gt_sha256"],
        )
        pred = _read_rgb(
            render_path,
            expected_sha256=None if binding is None else binding["render_sha256"],
        )
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: gt={gt.shape} pred={pred.shape} ({gt_path} vs {render_path})")

        mask = _read_mask(
            mask_path,
            expected_sha256=None if binding is None else binding["mask_sha256"],
        )
        if mask.shape[:2] != gt.shape[:2]:
            raise ValueError(f"Mask size mismatch: mask={mask.shape} gt={gt.shape} ({mask_path})")

        bg_mask, mode_used, nonzero_frac = _resolve_bg_mask(mask, mask_mode)
        bg_frac = float(bg_mask.mean())

        psnr = _psnr(pred, gt, bg_mask)
        ssim_val = _ssim(pred, gt, bg_mask, device, ssim_fn)
        lpips_val = _lpips(pred, gt, bg_mask, device, lpips_model)
        metric_values = {"psnr": psnr, "ssim": ssim_val, "lpips": lpips_val}
        invalid = [key for key, value in metric_values.items() if not math.isfinite(float(value))]
        if invalid:
            raise ValueError(f"Non-finite per-frame metrics at index {idx}: {invalid}")

        row = {
            "frame_idx": idx,
            "frame_name": os.path.basename(gt_path),
            "psnr": psnr,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "bg_frac": bg_frac,
            "mask_nonzero_frac": nonzero_frac,
            "mask_mode_used": mode_used,
        }
        if frame_bindings is not None:
            row["frame_binding"] = frame_bindings[idx]
        per_frame.append(row)

    return per_frame


def _mean_value(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Metric mean input contains non-finite values.")
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


def _assert_file_hash(path, expected_sha256, label):
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} changed during evaluation: {path}; "
            f"expected_sha256={expected_sha256} actual_sha256={actual_sha256}"
        )


def _compare_reference_summary(reference_path, computed_summary):
    reference_payload, reference_sha256 = _read_bound_json(reference_path)
    reference_summary = reference_payload.get("summary")
    if not isinstance(reference_summary, dict):
        raise ValueError(f"Reference metrics has no summary object: {reference_path}")

    compared = {}
    for metric in REQUIRED_POSE_METRICS + REQUIRED_RECON_METRICS:
        reference_value = reference_summary.get(metric)
        computed_value = computed_summary.get(metric)
        if reference_value is None or not math.isfinite(float(reference_value)):
            raise ValueError(f"Reference summary metric is missing or non-finite: {metric}")
        if computed_value is None or not math.isfinite(float(computed_value)):
            raise ValueError(f"Computed verification metric is missing or non-finite: {metric}")
        reference_value = float(reference_value)
        computed_value = float(computed_value)
        difference = abs(computed_value - reference_value)
        tolerance = REFERENCE_TOLERANCES[metric]
        maximum_allowed = tolerance["absolute"] + tolerance["relative"] * max(
            abs(computed_value), abs(reference_value)
        )
        compared[metric] = {
            "computed": computed_value,
            "reference": reference_value,
            "absolute_difference": difference,
            "maximum_allowed_difference": maximum_allowed,
            "absolute_tolerance": tolerance["absolute"],
            "relative_tolerance": tolerance["relative"],
            "matches": bool(difference <= maximum_allowed),
        }

    mismatched = [metric for metric, result in compared.items() if not result["matches"]]
    if mismatched:
        raise ValueError(
            "Independent raw-artifact metrics do not match the primary summary: "
            f"{mismatched}"
        )

    context_keys = (
        "pose_units",
        "pose_metric_protocol",
        "reconstruction_metric_protocol",
        "estimate_pose_source",
        "eval_pose_file",
        "eval_pose_file_sha256",
        "num_pose_frames",
        "num_reconstruction_frames",
        "pose_frame_ids_sha256",
    )
    context_matches = {}
    for key in context_keys:
        reference_value = reference_summary.get(key)
        computed_value = computed_summary.get(key)
        context_matches[key] = reference_value == computed_value
    bad_context = [key for key, matches in context_matches.items() if not matches]
    if bad_context:
        raise ValueError(
            "Reference/computed evaluation context mismatch: "
            f"{bad_context}"
        )

    return {
        "status": "match",
        "reference_metrics_path": os.path.abspath(reference_path),
        "reference_metrics_sha256": reference_sha256,
        "reference_summary_only": True,
        "reference_per_frame_rows_read": False,
        "computed_before_reference_read": True,
        "compared_metrics": compared,
        "context_matches": context_matches,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Derive Track2Map pose and reconstruction metrics directly from final artifacts. "
            "Independent verification never uses primary per-frame metric rows."
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-pose-file", required=True)
    parser.add_argument("--estimate-pose-source", default="optimized", choices=["optimized", "input"])
    parser.add_argument("--pose-start", type=int, default=0)
    parser.add_argument("--pose-stop", type=int, default=None)
    parser.add_argument("--pose-step", type=int, default=1)
    parser.add_argument(
        "--evaluation-kind",
        default="reconstruction_and_pose",
        choices=EVALUATION_KINDS,
        help=(
            "Use independent_verification for a second raw-artifact derivation, or "
            "pose_only_diagnostic for an explicitly non-complete diagnostic artifact."
        ),
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Legacy alias for --evaluation-kind pose_only_diagnostic.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--reference-metrics-json",
        default=None,
        help=(
            "Primary metrics to compare after independent derivation. Defaults to "
            "<output-dir>/metrics.json for independent_verification."
        ),
    )
    parser.add_argument("--render-subdir", default="raw_rgb/render")
    parser.add_argument("--runtime-gt-subdir", default="raw_rgb/gt")
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

    evaluation_kind = args.evaluation_kind
    if args.pose_only:
        if evaluation_kind == "independent_verification":
            parser.error("--pose-only cannot be combined with independent_verification.")
        evaluation_kind = "pose_only_diagnostic"
    pose_only = evaluation_kind == "pose_only_diagnostic"
    if evaluation_kind == "independent_verification" and args.estimate_pose_source != "optimized":
        parser.error("independent_verification must derive pose metrics from optimized_c2w.npy.")
    if args.pose_step <= 0:
        parser.error("--pose-step must be positive.")

    if args.repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = os.path.abspath(args.repo_root)

    sys.path.insert(0, repo_root)
    from src.utils.loss_utils import ssim as ssim_fn

    determinism = _configure_determinism(seed=0)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU.")
            device = "cpu"

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(
        _resolve_output_dir(repo_root, args.seq, args.output_dir, args.render_subdir)
    )
    seq = args.seq

    output_json = None if args.output_json is None else os.path.abspath(args.output_json)
    reference_metrics_json = args.reference_metrics_json
    if evaluation_kind == "independent_verification":
        if output_json is None:
            output_json = os.path.join(output_dir, "metrics_verification.json")
        if reference_metrics_json is None:
            reference_metrics_json = os.path.join(output_dir, "metrics.json")
        reference_metrics_json = os.path.abspath(reference_metrics_json)
        if output_json == reference_metrics_json:
            parser.error("Verification output and reference metrics must be different files.")
        if not os.path.isfile(reference_metrics_json):
            raise FileNotFoundError(f"Missing primary metrics reference: {reference_metrics_json}")
    elif reference_metrics_json is not None:
        parser.error("--reference-metrics-json is only valid for independent_verification.")

    provenance_path = os.path.join(output_dir, "pose_provenance.json")
    if not os.path.isfile(provenance_path):
        raise FileNotFoundError(f"Missing pose provenance: {provenance_path}")
    pose_provenance, pose_provenance_sha256 = _read_bound_json(provenance_path)
    if pose_provenance.get("pose_units") != "m":
        raise ValueError(f"Expected metric pose output, got {pose_provenance.get('pose_units')}")
    if pose_provenance.get("dedicated_evaluation_pose_artifact_saved") is not False:
        raise ValueError(
            "Inference pose provenance must state "
            "dedicated_evaluation_pose_artifact_saved=false."
        )

    pose_path = os.path.join(output_dir, f"{args.estimate_pose_source}_c2w.npy")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"Missing estimate pose file: {pose_path}")
    if not os.path.isfile(args.eval_pose_file):
        raise FileNotFoundError(f"Missing evaluation pose file: {args.eval_pose_file}")

    declared_pose_name = pose_provenance.get("files", {}).get(args.estimate_pose_source)
    if declared_pose_name != os.path.basename(pose_path):
        raise ValueError(
            f"Pose provenance does not bind {args.estimate_pose_source} source: "
            f"declared={declared_pose_name} actual={os.path.basename(pose_path)}"
        )
    pose_bytes, pose_file_sha256 = _read_bound_bytes(pose_path)
    estimate_poses = np.load(io.BytesIO(pose_bytes), allow_pickle=False)
    eval_pose_file = os.path.abspath(args.eval_pose_file)
    eval_pose_bytes, eval_pose_file_sha256 = _read_bound_bytes(eval_pose_file)
    eval_poses, eval_pose_frame_ids = _decode_pose_trajectory(
        eval_pose_file,
        eval_pose_bytes,
        start=args.pose_start,
        stop=args.pose_stop,
        step=args.pose_step,
    )
    pose_frame_ids_name = pose_provenance.get("files", {}).get("frame_ids")
    if not pose_frame_ids_name:
        raise ValueError("Pose provenance does not declare a frame-ID sidecar.")
    pose_frame_ids_path = os.path.join(output_dir, pose_frame_ids_name)
    if not os.path.isfile(pose_frame_ids_path):
        raise FileNotFoundError(f"Missing estimate pose frame IDs: {pose_frame_ids_path}")
    estimate_pose_frame_ids, pose_frame_ids_sha256 = _read_bound_json(pose_frame_ids_path)
    if not isinstance(estimate_pose_frame_ids, list) or not all(
        isinstance(frame_id, str) for frame_id in estimate_pose_frame_ids
    ):
        raise ValueError("Estimate pose frame IDs must be a JSON list of strings.")
    if estimate_pose_frame_ids != eval_pose_frame_ids:
        raise ValueError("Estimate/evaluation pose frame IDs do not match exactly.")
    if len(estimate_pose_frame_ids) != len(estimate_poses):
        raise ValueError(
            "Estimate pose/frame ID count mismatch: "
            f"ids={len(estimate_pose_frame_ids)} poses={len(estimate_poses)}"
        )
    provenance_frames = pose_provenance.get("num_frames")
    if provenance_frames != len(estimate_poses):
        raise ValueError(
            "Pose provenance count mismatch: "
            f"provenance={provenance_frames} estimate={len(estimate_poses)}."
        )
    pose_summary, per_frame_pose = compute_pose_metrics_with_rows(
        estimate_poses,
        eval_poses,
        estimate_pose_frame_ids,
    )

    per_frame = []
    recon_summary = {"psnr": None, "ssim": None, "lpips": None}
    recon_source_counts = None
    frame_bindings = []
    reconstruction_metric_provenance = {}
    if not pose_only:
        gt_dir = os.path.join(input_root, seq, args.gt_subdir)
        mask_dir = os.path.join(input_root, seq, args.mask_subdir)
        render_dir = os.path.join(output_dir, args.render_subdir)
        runtime_gt_dir = os.path.join(output_dir, args.runtime_gt_subdir)

        frame_slice = slice(args.pose_start, args.pose_stop, args.pose_step)
        gt_paths = _sorted_paths(os.path.join(gt_dir, "*l.png"))[frame_slice]
        mask_paths = _sorted_paths(os.path.join(mask_dir, "*l.png"))[frame_slice]
        render_paths = _sorted_paths(os.path.join(render_dir, "*.png"))
        runtime_gt_paths = _sorted_paths(os.path.join(runtime_gt_dir, "*.png"))
        recon_source_counts = {
            "gt": len(gt_paths),
            "runtime_gt": len(runtime_gt_paths),
            "render": len(render_paths),
            "mask": len(mask_paths),
        }
        frame_bindings = bind_reconstruction_frames(
            gt_paths,
            runtime_gt_paths,
            render_paths,
            mask_paths,
            estimate_pose_frame_ids,
        )
        per_frame = compute_recon_metrics(
            gt_paths,
            render_paths,
            mask_paths,
            args.mask_mode,
            device,
            ssim_fn,
            frame_bindings=frame_bindings,
            metric_provenance_out=reconstruction_metric_provenance,
        )
        _validate_frame_bindings_current(frame_bindings)
        recon_summary = {
            "psnr": _mean_value([row["psnr"] for row in per_frame]),
            "ssim": _mean_value([row["ssim"] for row in per_frame]),
            "lpips": _mean_value([row["lpips"] for row in per_frame]),
        }

    if not pose_only and len(per_frame) != len(estimate_poses):
        raise ValueError(
            "Reconstruction/pose frame count mismatch: "
            f"reconstruction={len(per_frame)} pose={len(estimate_poses)}."
        )

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
        "are_deg": pose_summary.get("are_deg") if pose_summary else None,
        "rper_deg": rper_deg,
        "rpet_m": rpet_m,
        "pose_units": "m",
        "pose_metric_protocol": "first_frame_normalized_unaligned_delta1",
        "reconstruction_metric_protocol": "training_view_background_masked_png8",
        "estimate_pose_source": args.estimate_pose_source,
        "eval_pose_file": eval_pose_file,
        "eval_pose_file_sha256": eval_pose_file_sha256,
        "num_pose_frames": int(len(estimate_poses)),
        "num_reconstruction_frames": None if pose_only else int(len(per_frame)),
        "pose_frame_ids_sha256": pose_frame_ids_sha256,
    }

    required_metrics = list(REQUIRED_POSE_METRICS)
    if not pose_only:
        required_metrics.extend(REQUIRED_RECON_METRICS)
    invalid_metrics = [
        key
        for key in required_metrics
        if summary.get(key) is None or not math.isfinite(float(summary[key]))
    ]
    if invalid_metrics:
        raise ValueError(f"Required metrics are missing or non-finite: {invalid_metrics}")

    evaluator_path = os.path.abspath(__file__)
    loss_utils_path = os.path.join(repo_root, "src", "utils", "loss_utils.py")
    evaluator_sha256 = _sha256(evaluator_path)
    loss_utils_sha256 = _sha256(loss_utils_path)

    # Rehash every pathname-backed source immediately before sealing. Image
    # bytes used in the calculation were also checked against these bindings at
    # read time, so the artifact binds the exact bytes that produced each row.
    _assert_file_hash(pose_path, pose_file_sha256, "Estimate pose artifact")
    _assert_file_hash(eval_pose_file, eval_pose_file_sha256, "Evaluation pose artifact")
    _assert_file_hash(pose_frame_ids_path, pose_frame_ids_sha256, "Pose frame-ID sidecar")
    _assert_file_hash(provenance_path, pose_provenance_sha256, "Pose provenance")
    if frame_bindings:
        _validate_frame_bindings_current(frame_bindings)
    _assert_file_hash(evaluator_path, evaluator_sha256, "Evaluator source")
    _assert_file_hash(loss_utils_path, loss_utils_sha256, "SSIM implementation source")

    artifact_kind = {
        "reconstruction_and_pose": "primary_metrics",
        "independent_verification": "independent_metrics_verification",
        "pose_only_diagnostic": "pose_only_diagnostic",
    }[evaluation_kind]
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "evaluation_kind": evaluation_kind,
        "complete_seven_metric_set": not pose_only,
        "completion_eligible": False if pose_only else None,
        "completion_eligibility_authority": "matrix_runner_run_kind_contract",
        "summary": summary,
        "pose_provenance": pose_provenance,
        "pose_frame_ids": estimate_pose_frame_ids,
        "reconstruction_source_counts": recon_source_counts,
        "per_frame_pose": per_frame_pose,
        "per_frame_reconstruction": per_frame,
        "derivation_provenance": {
            "calculation_source": "raw_final_artifacts",
            "primary_metric_rows_used_for_derivation": False,
            "evaluation_reference_classification": {
                "pose": "dataset_ground_truth",
                "reconstruction_rgb": None if pose_only else "dataset_ground_truth_training_view",
                "reconstruction_mask": None if pose_only else "dataset_provided_binary_mask",
                "oracle_pose_used_as_inference_input": bool(
                    pose_provenance.get("oracle_pose_input", False)
                ),
            },
            "artifact_roles": {
                "optimized_c2w": "pose_metric_estimate",
                "evaluation_pose_file": "pose_metric_reference_only",
                "dataset_left_rgb": None if pose_only else "reconstruction_metric_reference",
                "rendered_rgb": None if pose_only else "reconstruction_metric_prediction",
                "dataset_mask": None if pose_only else "reconstruction_metric_inclusion_mask",
                "runtime_gt_copy": "identity_attestation_only",
                "right_rgb": "inference_input_lineage_only",
                "semantic_prediction": "inference_input_lineage_only",
            },
            "sequence": seq,
            "slice": {
                "start": args.pose_start,
                "stop": args.pose_stop,
                "step": args.pose_step,
            },
            "mask_mode_requested": args.mask_mode,
            "estimate_pose": {
                "path": os.path.abspath(pose_path),
                "sha256": pose_file_sha256,
                "numpy_dtype": str(np.asarray(estimate_poses).dtype),
                "numpy_shape": list(np.asarray(estimate_poses).shape),
                "decoded_array_sha256": _array_sha256(estimate_poses),
            },
            "evaluation_pose": {
                "path": eval_pose_file,
                "sha256": eval_pose_file_sha256,
                "selected_numpy_dtype": str(np.asarray(eval_poses).dtype),
                "selected_numpy_shape": list(np.asarray(eval_poses).shape),
                "selected_array_sha256": _array_sha256(eval_poses),
            },
            "pose_frame_ids": {
                "path": os.path.abspath(pose_frame_ids_path),
                "sha256": pose_frame_ids_sha256,
                "ordered_ids_sha256": _canonical_json_sha256(estimate_pose_frame_ids),
            },
            "pose_provenance_artifact": {
                "path": os.path.abspath(provenance_path),
                "sha256": pose_provenance_sha256,
            },
            "reconstruction": None if pose_only else {
                "frame_count": len(frame_bindings),
                "ordered_frame_bindings_sha256": _frame_bindings_sha256(frame_bindings),
                "binding_location": "per_frame_reconstruction[*].frame_binding",
            },
            "metric_implementation": {
                "evaluator_source_path": evaluator_path,
                "evaluator_source_sha256": evaluator_sha256,
                "ssim_source_path": os.path.abspath(loss_utils_path),
                "ssim_source_sha256": loss_utils_sha256,
                "pose": "first-frame-normalized, unaligned, delta-1 RMSE",
                "psnr": "background-mask-normalized RGB MSE in PNG8 domain",
                "ssim": "masked Track2Map loss_utils.ssim",
                "lpips": "excluded prediction pixels replaced by GT before LPIPS-Alex",
                **reconstruction_metric_provenance,
            },
            "runtime": _runtime_provenance(args.device, device, determinism),
            "resolved_inputs": {
                "repo_root": repo_root,
                "input_root": input_root,
                "output_dir": output_dir,
                "render_subdir": args.render_subdir,
                "runtime_gt_subdir": args.runtime_gt_subdir,
                "gt_subdir": args.gt_subdir,
                "mask_subdir": args.mask_subdir,
            },
        },
    }

    if evaluation_kind == "independent_verification":
        # This is intentionally performed only after the raw-artifact payload is
        # fully derived. The comparator reads the primary summary, never its rows.
        payload["reference_comparison"] = _compare_reference_summary(
            reference_metrics_json,
            summary,
        )

    payload["attestation"] = {
        "canonical_payload_sha256": _canonical_json_sha256(payload),
        "hash_scope": "canonical JSON payload excluding the attestation object",
        "atomic_write": True,
    }

    if output_json is not None:
        _atomic_write_json(output_json, payload)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
