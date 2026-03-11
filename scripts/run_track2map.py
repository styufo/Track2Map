#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import yaml


SEQS = ["P1_1", "P2_0", "P2_1", "P3_1", "P3_2"]
MODES = ["clean_pose", "noisy_auto_gate", "no_pose"]
FLOW_INIT_SOURCES = ["raft", "cotracker3", "hybrid", "foundation", "hybrid_foundation"]
GATE_PROFILES = ["auto", "1x", "10x"]

DEFAULT_FOUNDATION_ROOT = os.environ.get("FOUNDATION_ROOT", "/path/to/FoundationStereo")
DEFAULT_FOUNDATION_CKPT = os.environ.get(
    "FOUNDATION_CKPT",
    "/path/to/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
)
DEFAULT_FOUNDATION_CFG = os.environ.get(
    "FOUNDATION_CFG",
    "/path/to/FoundationStereo/pretrained_models/23-51-11/cfg.yaml",
)
DEFAULT_FOUNDATION_INTRINSIC = os.environ.get(
    "FOUNDATION_INTRINSIC_FILE",
    "/path/to/FoundationStereo/assets/K.txt",
)
DEFAULT_COTRACKER_REPO = os.environ.get("COTRACKER_REPO", "/path/to/co-tracker")

GATE_THRESHOLDS = {
    "1x": {
        "first_frame_pose_trust_gate_min_psnr": 0.0,
        "first_frame_pose_trust_gate_min_ssim": 0.0,
        "first_frame_pose_trust_gate_max_psnr_drop": 100.0,
        "first_frame_pose_trust_gate_max_ssim_drop": 1.0,
    },
    "10x": {
        "first_frame_pose_trust_gate_min_psnr": 45.0,
        "first_frame_pose_trust_gate_min_ssim": 0.90,
        "first_frame_pose_trust_gate_max_psnr_drop": 0.20,
        "first_frame_pose_trust_gate_max_ssim_drop": 0.005,
    },
}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def infer_gate_profile_from_pose_file(pose_file: str) -> str:
    if not pose_file:
        return "1x"
    lower = pose_file.lower()
    if any(tag in lower for tag in ("x10", "transx10", "noisyx10")):
        return "10x"
    return "1x"


def build_config(args: argparse.Namespace, repo_root: Path) -> Dict[str, Any]:
    base_cfg_path = repo_root / "configs" / "StereoMIS" / f"{args.seq}.yaml"
    if not base_cfg_path.is_file():
        raise FileNotFoundError(f"Missing base config: {base_cfg_path}")
    cfg = load_yaml(base_cfg_path)

    common_overrides = {
        "data": {
            "depth_input_source": "foundation_stereo",
            "foundation_root": args.foundation_root,
            "foundation_ckpt": args.foundation_ckpt,
            "foundation_cfg": args.foundation_cfg,
            "foundation_intrinsic_file": args.foundation_intrinsic_file,
            "foundation_valid_iters": args.foundation_valid_iters,
            "foundation_depth_formula": "raft_compatible",
            "foundation_use_sequence_camera_params": True,
            "output": args.output,
        },
        "training": {
            "pt_tracker_backend": "cotracker3_online",
            "pt_cotracker_repo": args.cotracker_repo,
            "pt_cotracker_model": "cotracker3_online",
            "optical_flow_init_source": args.flow_init_source,
        },
    }
    deep_update(cfg, common_overrides)

    if args.start is not None:
        cfg.setdefault("data", {})["start"] = int(args.start)
    if args.stop is not None:
        cfg.setdefault("data", {})["stop"] = int(args.stop)
    if args.step is not None:
        cfg.setdefault("data", {})["step"] = int(args.step)

    if args.mode == "clean_pose":
        mode_overrides = {
            "training": {
                "pose_optimization": {
                    "enabled": False,
                    "optimize_first_frame": False,
                    "first_frame_pose_trust_gate_enabled": False,
                }
            }
        }
        deep_update(cfg, mode_overrides)

    elif args.mode == "noisy_auto_gate":
        gate_profile = args.gate_profile
        if gate_profile == "auto":
            gate_profile = infer_gate_profile_from_pose_file(args.pose_file)
        if gate_profile not in GATE_THRESHOLDS:
            raise ValueError(f"Unknown gate profile: {gate_profile}")

        gate_cfg = GATE_THRESHOLDS[gate_profile]
        mode_overrides = {
            "training": {
                "pose_optimization": {
                    "enabled": True,
                    "pose_init_mode": "dataset",
                    "pose_no_prior_vo_enabled": True,
                    "pose_no_prior_vo_chain_source": "input",
                    "first_frame_pose_trust_gate_enabled": True,
                    "first_frame_pose_trust_gate_fallback_mode": "no_prior",
                    "first_frame_pose_trust_gate_enable_no_prior_vo": True,
                    "first_frame_pose_trust_gate_chain_source": "input",
                    "first_frame_pose_trust_gate_fallback_w_pose_prior": 0.0,
                    **gate_cfg,
                }
            }
        }
        deep_update(cfg, mode_overrides)
        cfg.setdefault("track2map_runtime", {})["gate_profile_resolved"] = gate_profile

    elif args.mode == "no_pose":
        mode_overrides = {
            "data": {
                "pose_file": "__missing_pose_no_prior__.txt",
            },
            "training": {
                "pose_optimization": {
                    "enabled": True,
                    "pose_init_mode": "no_prior",
                    "pose_no_prior_vo_enabled": True,
                    "pose_no_prior_vo_chain_source": "input",
                    "w_pose_prior": 0.0,
                    "first_frame_pose_trust_gate_enabled": False,
                }
            },
        }
        deep_update(cfg, mode_overrides)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track2Map unified launcher")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seq", required=True, choices=SEQS)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose-file", default=None)
    parser.add_argument("--flow-init-source", default="hybrid", choices=FLOW_INIT_SOURCES)
    parser.add_argument("--gate-profile", default="auto", choices=GATE_PROFILES)
    parser.add_argument("--foundation-root", default=DEFAULT_FOUNDATION_ROOT)
    parser.add_argument("--foundation-ckpt", default=DEFAULT_FOUNDATION_CKPT)
    parser.add_argument("--foundation-cfg", default=DEFAULT_FOUNDATION_CFG)
    parser.add_argument("--foundation-intrinsic-file", default=DEFAULT_FOUNDATION_INTRINSIC)
    parser.add_argument("--foundation-valid-iters", type=int, default=32)
    parser.add_argument("--cotracker-repo", default=DEFAULT_COTRACKER_REPO)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-config", default=None, help="Optional path to save generated config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode in ("clean_pose", "noisy_auto_gate") and not args.pose_file:
        parser.error(f"--pose-file is required for mode '{args.mode}'")
    return args


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    for path_like in (args.foundation_ckpt, args.foundation_cfg, args.foundation_intrinsic_file):
        if not Path(path_like).is_file():
            raise FileNotFoundError(f"Missing file: {path_like}")
    if not Path(args.foundation_root).is_dir():
        raise FileNotFoundError(f"Missing directory: {args.foundation_root}")

    cfg = build_config(args, repo_root)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.save_config:
        cfg_path = Path(args.save_config).resolve()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        tmp_dir = repo_root / "tmp" / "generated_configs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = tmp_dir / f"{args.seq}_{args.mode}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    cmd = [sys.executable, "run.py", str(cfg_path), "--input_folder", args.input_folder, "--output", args.output]

    if args.mode != "no_pose" and args.pose_file:
        cmd.extend(["--pose_file", args.pose_file])
    if args.visualize:
        cmd.append("--visualize")
    if args.debug:
        cmd.append("--debug")

    print("[Track2Map] Generated config:", cfg_path)
    print("[Track2Map] Mode:", args.mode)
    print("[Track2Map] Sequence:", args.seq)
    if args.mode == "noisy_auto_gate":
        profile = cfg.get("track2map_runtime", {}).get("gate_profile_resolved", args.gate_profile)
        print("[Track2Map] Gate profile:", profile)
    print("[Track2Map] Command:", " ".join(cmd))

    if args.dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(repo_root))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
