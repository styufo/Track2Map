#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Quaternion = Tuple[float, float, float, float]
Translation = Tuple[float, float, float]


def quat_normalize(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    inv = 1.0 / norm
    return x * inv, y * inv, z * inv, w * inv


def quat_mul(lhs: Quaternion, rhs: Quaternion) -> Quaternion:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_inverse(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quat_normalize(quaternion)
    return -x, -y, -z, w


def quaternion_angular_distance_deg(a: Quaternion, b: Quaternion) -> float:
    delta = quat_mul(quat_normalize(a), quat_inverse(quat_normalize(b)))
    w = max(-1.0, min(1.0, abs(delta[3])))
    return math.degrees(2.0 * math.acos(w))


def quat_from_axis_angle(axis: Translation, angle_rad: float) -> Quaternion:
    ax, ay, az = axis
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return quat_normalize((ax * sin_half, ay * sin_half, az * sin_half, math.cos(half)))


def random_unit_axis(rng: random.Random) -> Translation:
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return radius * math.cos(theta), radius * math.sin(theta), z


def sample_scalar_noise(rng: random.Random, scale: float, distribution: str) -> float:
    if scale <= 0.0:
        return 0.0
    if distribution == "uniform":
        return rng.uniform(-scale, scale)
    return rng.gauss(0.0, scale)


def sample_small_rotation(rng: random.Random, scale_deg: float, distribution: str) -> Quaternion:
    if scale_deg <= 0.0:
        return 0.0, 0.0, 0.0, 1.0
    angle_deg = sample_scalar_noise(rng, scale_deg, distribution)
    if angle_deg == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    angle = math.radians(angle_deg)
    axis = random_unit_axis(rng)
    return quat_from_axis_angle(axis, angle)


def parse_groundtruth(path: Path) -> List[Dict[str, object]]:
    poses: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                tx = float(parts[1])
                ty = float(parts[2])
                tz = float(parts[3])
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
            except ValueError as error:
                raise ValueError(f"Invalid pose at {path}:{line_number}") from error
            poses.append(
                {
                    "frame": parts[0],
                    "t": (tx, ty, tz),
                    "q": quat_normalize((qx, qy, qz, qw)),
                }
            )
    if not poses:
        raise ValueError(f"No valid pose found in {path}")
    return poses


def format_float(value: float) -> str:
    return f"{value:.17g}"


def write_groundtruth(path: Path, poses: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pose in poses:
            tx, ty, tz = pose["t"]  # type: ignore[index]
            qx, qy, qz, qw = pose["q"]  # type: ignore[index]
            row = [
                str(pose["frame"]),
                format_float(tx),
                format_float(ty),
                format_float(tz),
                format_float(qx),
                format_float(qy),
                format_float(qz),
                format_float(qw),
            ]
            handle.write(" ".join(row) + "\n")


def sequence_seed(base_seed: int, name: str) -> int:
    return base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def add_pose_noise(
    poses: Sequence[Dict[str, object]],
    rng: random.Random,
    trans_sigma: float,
    rot_sigma_deg: float,
    trans_drift_sigma: float,
    rot_drift_sigma_deg: float,
    motion_iid_trans_scale: float,
    motion_iid_rot_scale: float,
    noise_distribution: str,
    noise_first_pose: bool,
    freeze_on_stationary: bool,
    stationary_trans_thresh: float,
    stationary_rot_thresh_deg: float,
    lock_translation_on_pure_rotation: bool,
    pure_rotation_trans_thresh: float,
    pure_rotation_rot_min_deg: float,
) -> List[Dict[str, object]]:
    noisy: List[Dict[str, object]] = []
    drift_tx, drift_ty, drift_tz = 0.0, 0.0, 0.0
    drift_quaternion: Quaternion = (0.0, 0.0, 0.0, 1.0)
    current_noise_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)

    for idx, pose in enumerate(poses):
        tx, ty, tz = pose["t"]  # type: ignore[index]
        qx, qy, qz, qw = pose["q"]  # type: ignore[index]

        if idx == 0 and not noise_first_pose:
            noisy.append({"frame": pose["frame"], "t": (tx, ty, tz), "q": (qx, qy, qz, qw)})
            continue

        is_stationary = False
        is_pure_rotation = False
        if idx > 0:
            prev_tx, prev_ty, prev_tz = poses[idx - 1]["t"]  # type: ignore[index]
            prev_q = poses[idx - 1]["q"]  # type: ignore[index]
            trans_delta = math.sqrt((tx - prev_tx) ** 2 + (ty - prev_ty) ** 2 + (tz - prev_tz) ** 2)
            rot_delta = quaternion_angular_distance_deg((qx, qy, qz, qw), prev_q)  # type: ignore[arg-type]
            if freeze_on_stationary:
                is_stationary = trans_delta <= stationary_trans_thresh and rot_delta <= stationary_rot_thresh_deg
            if lock_translation_on_pure_rotation:
                is_pure_rotation = trans_delta <= pure_rotation_trans_thresh and rot_delta > pure_rotation_rot_min_deg

        if is_stationary and idx > 0:
            prev_noisy = noisy[-1]
            noisy.append({"frame": pose["frame"], "t": prev_noisy["t"], "q": prev_noisy["q"]})
            continue

        if trans_drift_sigma > 0.0:
            drift_tx += sample_scalar_noise(rng, trans_drift_sigma, noise_distribution)
            drift_ty += sample_scalar_noise(rng, trans_drift_sigma, noise_distribution)
            drift_tz += sample_scalar_noise(rng, trans_drift_sigma, noise_distribution)

        if not is_pure_rotation:
            frame_trans_sigma = trans_sigma * max(0.0, motion_iid_trans_scale)
            iid_tx = sample_scalar_noise(rng, frame_trans_sigma, noise_distribution)
            iid_ty = sample_scalar_noise(rng, frame_trans_sigma, noise_distribution)
            iid_tz = sample_scalar_noise(rng, frame_trans_sigma, noise_distribution)
            noisy_translation = (tx + drift_tx + iid_tx, ty + drift_ty + iid_ty, tz + drift_tz + iid_tz)
        else:
            noisy_translation = noisy[-1]["t"]  # type: ignore[assignment]

        if rot_drift_sigma_deg > 0.0:
            drift_step = sample_small_rotation(rng, rot_drift_sigma_deg, noise_distribution)
            drift_quaternion = quat_normalize(quat_mul(drift_step, drift_quaternion))

        frame_rot_sigma_deg = rot_sigma_deg * max(0.0, motion_iid_rot_scale)
        iid_rotation = sample_small_rotation(rng, frame_rot_sigma_deg, noise_distribution)
        current_noise_rotation = quat_normalize(quat_mul(iid_rotation, drift_quaternion))
        noisy_quaternion = quat_normalize(quat_mul(current_noise_rotation, (qx, qy, qz, qw)))

        noisy.append({"frame": pose["frame"], "t": noisy_translation, "q": noisy_quaternion})

    return noisy


def compute_noise_metrics(original: Sequence[Dict[str, object]], noisy: Sequence[Dict[str, object]]) -> Dict[str, float]:
    n = min(len(original), len(noisy))
    trans_err = []
    rot_err = []
    for idx in range(n):
        tx, ty, tz = original[idx]["t"]  # type: ignore[index]
        ntx, nty, ntz = noisy[idx]["t"]  # type: ignore[index]
        trans_err.append(math.sqrt((tx - ntx) ** 2 + (ty - nty) ** 2 + (tz - ntz) ** 2))
        q = original[idx]["q"]  # type: ignore[index]
        nq = noisy[idx]["q"]  # type: ignore[index]
        rot_err.append(quaternion_angular_distance_deg(q, nq))  # type: ignore[arg-type]
    if not trans_err:
        return {"mean_translation_error": 0.0, "mean_rotation_error_deg": 0.0}
    return {
        "mean_translation_error": float(sum(trans_err) / len(trans_err)),
        "mean_rotation_error_deg": float(sum(rot_err) / len(rot_err)),
    }


def discover_sequences(input_root: Path, selected: Sequence[str] | None) -> List[Tuple[str, Path]]:
    if (input_root / "groundtruth.txt").is_file():
        return [(input_root.name, input_root)]

    if selected:
        sequence_names = list(selected)
    else:
        sequence_names = sorted(
            [child.name for child in input_root.iterdir() if child.is_dir() and (child / "groundtruth.txt").is_file()]
        )

    results: List[Tuple[str, Path]] = []
    for name in sequence_names:
        seq_dir = input_root / name
        gt = seq_dir / "groundtruth.txt"
        if not gt.is_file():
            raise FileNotFoundError(f"Missing groundtruth.txt for sequence '{name}': {gt}")
        results.append((name, seq_dir))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noisy StereoMIS poses from groundtruth.")
    parser.add_argument("--input-root", type=Path, required=True, help="StereoMIS root or one sequence directory.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root; saves <out-root>/<seq>/groundtruth_noisy.txt.")
    parser.add_argument("--seq", nargs="+", default=None, help="Optional sequence list, e.g. P1_1 P2_0 P2_1 P3_1 P3_2.")

    parser.add_argument("--trans-sigma", type=float, default=0.0006)
    parser.add_argument("--rot-sigma-deg", type=float, default=0.6)
    parser.add_argument("--noise-distribution", choices=("gaussian", "uniform"), default="uniform")
    parser.add_argument("--motion-iid-trans-scale", type=float, default=2.0)
    parser.add_argument("--motion-iid-rot-scale", type=float, default=1.8)
    parser.add_argument("--trans-drift-sigma", type=float, default=0.0)
    parser.add_argument("--rot-drift-sigma-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-first-pose", action="store_true")

    parser.add_argument("--no-freeze-on-stationary", action="store_true")
    parser.add_argument("--stationary-trans-thresh", type=float, default=5e-5)
    parser.add_argument("--stationary-rot-thresh-deg", type=float, default=0.02)
    parser.add_argument("--no-lock-translation-on-pure-rotation", action="store_true")
    parser.add_argument("--pure-rotation-trans-thresh", type=float, default=5e-5)
    parser.add_argument("--pure-rotation-rot-min-deg", type=float, default=0.02)

    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(input_root, args.seq)
    summary = {
        "input_root": str(input_root),
        "out_root": str(out_root),
        "noise_parameters": {
            "noise_distribution": args.noise_distribution,
            "trans_sigma": args.trans_sigma,
            "rot_sigma_deg": args.rot_sigma_deg,
            "motion_iid_trans_scale": args.motion_iid_trans_scale,
            "motion_iid_rot_scale": args.motion_iid_rot_scale,
            "trans_drift_sigma": args.trans_drift_sigma,
            "rot_drift_sigma_deg": args.rot_drift_sigma_deg,
            "noise_first_pose": args.noise_first_pose,
            "freeze_on_stationary": not args.no_freeze_on_stationary,
            "stationary_trans_thresh": args.stationary_trans_thresh,
            "stationary_rot_thresh_deg": args.stationary_rot_thresh_deg,
            "lock_translation_on_pure_rotation": not args.no_lock_translation_on_pure_rotation,
            "pure_rotation_trans_thresh": args.pure_rotation_trans_thresh,
            "pure_rotation_rot_min_deg": args.pure_rotation_rot_min_deg,
            "seed": args.seed,
        },
        "generated_sequences": [],
    }

    for name, seq_dir in sequences:
        gt_path = seq_dir / "groundtruth.txt"
        original = parse_groundtruth(gt_path)
        rng = random.Random(sequence_seed(args.seed, name))
        noisy = add_pose_noise(
            poses=original,
            rng=rng,
            trans_sigma=args.trans_sigma,
            rot_sigma_deg=args.rot_sigma_deg,
            trans_drift_sigma=args.trans_drift_sigma,
            rot_drift_sigma_deg=args.rot_drift_sigma_deg,
            motion_iid_trans_scale=args.motion_iid_trans_scale,
            motion_iid_rot_scale=args.motion_iid_rot_scale,
            noise_distribution=args.noise_distribution,
            noise_first_pose=args.noise_first_pose,
            freeze_on_stationary=not args.no_freeze_on_stationary,
            stationary_trans_thresh=args.stationary_trans_thresh,
            stationary_rot_thresh_deg=args.stationary_rot_thresh_deg,
            lock_translation_on_pure_rotation=not args.no_lock_translation_on_pure_rotation,
            pure_rotation_trans_thresh=args.pure_rotation_trans_thresh,
            pure_rotation_rot_min_deg=args.pure_rotation_rot_min_deg,
        )

        out_seq = out_root / name
        out_seq.mkdir(parents=True, exist_ok=True)
        out_pose = out_seq / "groundtruth_noisy.txt"
        report_path = out_seq / "noise_report.json"

        if out_pose.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_pose}. Use --overwrite to replace.")
        if report_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {report_path}. Use --overwrite to replace.")

        write_groundtruth(out_pose, noisy)
        metrics = compute_noise_metrics(original, noisy)
        report = {
            "sequence": name,
            "source_groundtruth": str(gt_path),
            "output_groundtruth_noisy": str(out_pose),
            "metrics": metrics,
            "num_frames": len(original),
            "seed": sequence_seed(args.seed, name),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary["generated_sequences"].append(report)
        print(
            f"[OK] {name}: frames={len(original)}, "
            f"mean_trans={metrics['mean_translation_error']:.6g}, "
            f"mean_rot_deg={metrics['mean_rotation_error_deg']:.6g}"
        )
        print(f"     noisy poses: {out_pose}")

    summary_path = out_root / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {summary_path}. Use --overwrite to replace.")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
