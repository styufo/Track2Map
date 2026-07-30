#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.inference_policy import (
    STRICT_GT_FREE,
    enforce_inference_policy,
    pose_condition_contract,
)


SEQS = ["P1_1", "P2_0", "P2_1", "P3_1", "P3_2"]
MODES = ["clean_pose", "light_noise", "heavy_noise", "noisy", "noisy_auto_gate", "no_pose"]
FLOW_INIT_SOURCES = ["raft", "cotracker3", "hybrid", "foundation", "hybrid_foundation"]
GATE_PROFILES = ["auto", "1x", "10x"]
COTRACKER_RUNTIME_MODES = ["precompute", "online_prefix"]
RUN_KINDS = ["adhoc", "full_evaluation", "smoke", "pose_only_diagnostic"]
DEPTH_INPUT_SOURCES = ["raft_stereo", "foundation_stereo"]

DEFAULT_FOUNDATION_ROOT = os.environ.get("FOUNDATION_ROOT", "foundationstereo")
DEFAULT_FOUNDATION_CKPT = os.environ.get(
    "FOUNDATION_CKPT",
    "foundationstereo/pretrained_models/23-51-11/model_best_bp2.pth",
)
DEFAULT_FOUNDATION_CFG = os.environ.get(
    "FOUNDATION_CFG",
    "foundationstereo/pretrained_models/23-51-11/cfg.yaml",
)
DEFAULT_FOUNDATION_INTRINSIC = os.environ.get(
    "FOUNDATION_INTRINSIC_FILE",
    "foundationstereo/assets/K.txt",
)
DEFAULT_COTRACKER_REPO = os.environ.get("COTRACKER_REPO", "cotracker")

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

FORMAL_MODE_POSE_ROOTS = {
    "light_noise": "stereomis_noisy_light",
    "heavy_noise": "stereomis_noisy_light_transx10",
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


def resolve_path(path_like: str, repo_root: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / path).resolve()


def verify_condition_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Bind a declared StereoMIS condition to concrete pose files and hashes."""
    input_folder = Path(args.input_folder).resolve()
    eval_pose = Path(args.eval_pose_file).resolve()
    pose_file = Path(args.pose_file).resolve() if args.pose_file else None
    expected_eval = (input_folder / "groundtruth.txt").resolve()

    if input_folder.name != args.seq:
        raise ValueError(
            f"Input folder/sequence mismatch: folder={input_folder.name} sequence={args.seq}"
        )
    if eval_pose != expected_eval:
        raise ValueError(
            f"Evaluation pose must be the selected sequence groundtruth: "
            f"expected={expected_eval} actual={eval_pose}"
        )

    eval_hash = _sha256(eval_pose)
    pose_hash = _sha256(pose_file) if pose_file is not None else None
    verification = {
        "status": "verified",
        "declared_mode": args.mode,
        "sequence": args.seq,
        "evaluation_pose_file": str(eval_pose),
        "evaluation_pose_sha256": eval_hash,
        "input_pose_file": str(pose_file) if pose_file is not None else None,
        "input_pose_sha256": pose_hash,
        "evaluation_pose_is_canonical_sequence_groundtruth": True,
        "input_pose_bytes_equal_evaluation_pose": pose_hash == eval_hash if pose_hash else False,
    }

    if args.mode == "no_pose":
        if pose_file is not None:
            raise ValueError("no_pose forbids --pose-file; initialization must be identity/causal VO.")
        verification["condition_role"] = "identity_initialization_without_pose_file"
        verification["condition_label_derived_from"] = "absence_of_input_pose_file"
        return verification

    if pose_file is None:
        raise ValueError(f"{args.mode} requires an input pose file.")
    if pose_file.parent.name != args.seq:
        raise ValueError(
            f"Input pose/sequence mismatch: pose_parent={pose_file.parent.name} sequence={args.seq}"
        )

    if args.mode == "clean_pose":
        if pose_file != eval_pose or pose_hash != eval_hash:
            raise ValueError("clean_pose requires the canonical evaluation pose as oracle input.")
        verification["condition_role"] = "oracle_pose_mapping_control"
        verification["condition_label_derived_from"] = "identical_canonical_path_and_hash"
        return verification

    if pose_hash == eval_hash:
        raise ValueError(f"{args.mode} input pose unexpectedly equals the evaluation pose.")

    if args.mode in FORMAL_MODE_POSE_ROOTS:
        expected_root = FORMAL_MODE_POSE_ROOTS[args.mode]
        if expected_root not in pose_file.parts:
            raise ValueError(
                f"{args.mode} requires a pose under canonical root '{expected_root}': {pose_file}"
            )
        if args.mode == "light_noise" and "stereomis_noisy_light_transx10" in pose_file.parts:
            raise ValueError("light_noise cannot use the heavy-noise pose root.")
        if pose_file.name != "groundtruth_noisy.txt":
            raise ValueError(f"Unexpected noisy-pose filename for {args.mode}: {pose_file.name}")
        verification["condition_role"] = {
            "light_noise": "light_noise_plus_1x_gate_policy_bundle",
            "heavy_noise": "heavy_translation_noise_plus_10x_gate_policy_bundle",
        }[args.mode]
        verification["condition_label_derived_from"] = "canonical_root_path_and_nonoracle_hash"
        verification["canonical_input_pose_root"] = expected_root
        return verification

    verification["condition_role"] = "user_declared_nonoracle_noisy_pose"
    verification["condition_label_derived_from"] = "nonoracle_hash_only"
    return verification


def inference_condition_view(verification: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Return only condition facts that the inference child is allowed to receive."""
    if mode == "clean_pose":
        return dict(verification)
    forbidden = {
        "evaluation_pose_file",
        "evaluation_pose_sha256",
        "evaluation_pose_is_canonical_sequence_groundtruth",
        "input_pose_bytes_equal_evaluation_pose",
    }
    return {key: value for key, value in verification.items() if key not in forbidden}


def inference_visible_launch_provenance(provenance: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Build the only launcher provenance artifact present while inference runs."""
    visible = dict(provenance)
    contract = pose_condition_contract(mode)
    if contract["exact_evaluation_pose_available_to_inference"]:
        return visible
    visible.pop("eval_pose_file", None)
    visible.pop("eval_pose_file_sha256", None)
    visible["condition_input_verification"] = inference_condition_view(
        provenance["condition_input_verification"],
        mode,
    )
    visible["evaluation_reference_provenance"] = "deferred_until_post_inference"
    return visible


def build_config(args: argparse.Namespace, repo_root: Path) -> Dict[str, Any]:
    base_cfg_path = repo_root / "configs" / "StereoMIS" / f"{args.seq}.yaml"
    if not base_cfg_path.is_file():
        raise FileNotFoundError(f"Missing base config: {base_cfg_path}")
    cfg = load_yaml(base_cfg_path)

    data_cfg = cfg.setdefault("data", {})
    if "deformation_time_denominator" not in data_cfg:
        start = int(data_cfg.get("start", 0))
        stop = int(data_cfg.get("stop", start + 1))
        step = int(data_cfg.get("step", 1))
        nominal_frames = len(range(start, stop, step))
        data_cfg["deformation_time_denominator"] = max(nominal_frames - 1, 1)

    common_overrides = {
        "data": {
            "depth_input_source": args.depth_input_source,
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
            "pt_cotracker_runtime_mode": args.cotracker_runtime_mode,
            "optical_flow_init_source": args.flow_init_source,
        },
        "track2map_runtime": {
            "inference_policy": STRICT_GT_FREE,
            "experiment_mode": args.mode,
            "run_kind": getattr(args, "run_kind", "adhoc"),
            "pose_condition_contract": pose_condition_contract(args.mode),
            "condition_input_verification": inference_condition_view(
                getattr(
                    args,
                    "condition_input_verification",
                    {"status": "not_executed_config_only"},
                ),
                args.mode,
            ),
        },
    }
    deep_update(cfg, common_overrides)

    if args.start is not None:
        cfg.setdefault("data", {})["start"] = int(args.start)
    if args.stop is not None:
        cfg.setdefault("data", {})["stop"] = int(args.stop)
    if args.step is not None:
        cfg.setdefault("data", {})["step"] = int(args.step)
    if args.iters is not None:
        cfg.setdefault("training", {})["iters"] = int(args.iters)
        cfg.setdefault("training", {}).setdefault("pose_optimization", {})["iters"] = int(args.iters)
    if args.iters_first is not None:
        cfg.setdefault("training", {})["iters_first"] = int(args.iters_first)

    if args.mode == "clean_pose":
        mode_overrides = {
            "data": {"pose_source": "file"},
            "training": {
                "pose_optimization": {
                    "enabled": False,
                    "optimize_first_frame": False,
                    "pose_init_mode": "dataset",
                    "pose_no_prior_vo_enabled": False,
                    "tool_motion_gate_enabled": False,
                    "tool_motion_soft_gate_enabled": False,
                    "pose_step_limit_enabled": False,
                    "pose_step_adaptive_enabled": False,
                    "first_frame_pose_trust_gate_enabled": False,
                    "save_poses": True,
                }
            }
        }
        deep_update(cfg, mode_overrides)

    elif args.mode in ("light_noise", "heavy_noise", "noisy", "noisy_auto_gate"):
        if args.mode == "light_noise":
            gate_profile = "1x"
        elif args.mode == "heavy_noise":
            gate_profile = "10x"
        else:
            gate_profile = args.gate_profile
        if gate_profile == "auto":
            gate_profile = infer_gate_profile_from_pose_file(args.pose_file)
        if gate_profile not in GATE_THRESHOLDS:
            raise ValueError(f"Unknown gate profile: {gate_profile}")

        gate_cfg = GATE_THRESHOLDS[gate_profile]
        mode_overrides = {
            "data": {"pose_source": "file"},
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
                    "save_poses": True,
                    **gate_cfg,
                }
            }
        }
        deep_update(cfg, mode_overrides)
        cfg.setdefault("track2map_runtime", {})["gate_profile_resolved"] = gate_profile

    elif args.mode == "no_pose":
        mode_overrides = {
            "data": {
                "pose_source": "identity",
            },
            "training": {
                "pose_optimization": {
                    "enabled": True,
                    "pose_init_mode": "no_prior",
                    "pose_no_prior_vo_enabled": True,
                    "pose_no_prior_vo_chain_source": "input",
                    "w_pose_prior": 0.0,
                    "first_frame_pose_trust_gate_enabled": False,
                    "save_poses": True,
                }
            },
        }
        deep_update(cfg, mode_overrides)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return enforce_inference_policy(cfg)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_launch_provenance(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track2Map unified launcher")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seq", required=True, choices=SEQS)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose-file", default=None)
    parser.add_argument("--eval-pose-file", required=True)
    parser.add_argument("--depth-input-source", default="raft_stereo", choices=DEPTH_INPUT_SOURCES)
    parser.add_argument("--flow-init-source", default="raft", choices=FLOW_INIT_SOURCES)
    parser.add_argument("--gate-profile", default="auto", choices=GATE_PROFILES)
    parser.add_argument("--foundation-root", default=DEFAULT_FOUNDATION_ROOT)
    parser.add_argument("--foundation-ckpt", default=DEFAULT_FOUNDATION_CKPT)
    parser.add_argument("--foundation-cfg", default=DEFAULT_FOUNDATION_CFG)
    parser.add_argument("--foundation-intrinsic-file", default=DEFAULT_FOUNDATION_INTRINSIC)
    parser.add_argument("--foundation-valid-iters", type=int, default=32)
    parser.add_argument("--cotracker-repo", default=DEFAULT_COTRACKER_REPO)
    parser.add_argument("--cotracker-runtime-mode", default="online_prefix", choices=COTRACKER_RUNTIME_MODES)
    parser.add_argument("--run-kind", default="adhoc", choices=RUN_KINDS)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--iters-first", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-config", default=None, help="Optional path to save generated config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode != "no_pose" and not args.pose_file:
        parser.error(f"--pose-file is required for mode '{args.mode}'")
    return args


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT

    args.foundation_root = str(resolve_path(args.foundation_root, repo_root))
    args.foundation_ckpt = str(resolve_path(args.foundation_ckpt, repo_root))
    args.foundation_cfg = str(resolve_path(args.foundation_cfg, repo_root))
    args.foundation_intrinsic_file = str(resolve_path(args.foundation_intrinsic_file, repo_root))
    args.cotracker_repo = str(resolve_path(args.cotracker_repo, repo_root))

    args.input_folder = str(Path(args.input_folder).expanduser().resolve())
    args.output = str(Path(args.output).expanduser().resolve())
    args.eval_pose_file = str(Path(args.eval_pose_file).expanduser().resolve())
    if args.pose_file:
        args.pose_file = str(Path(args.pose_file).expanduser().resolve())

    if not Path(args.input_folder).is_dir():
        raise FileNotFoundError(f"Missing input directory: {args.input_folder}")
    if not Path(args.eval_pose_file).is_file():
        raise FileNotFoundError(f"Missing evaluation pose file: {args.eval_pose_file}")
    if args.pose_file and not Path(args.pose_file).is_file():
        raise FileNotFoundError(f"Missing input pose file: {args.pose_file}")
    args.condition_input_verification = verify_condition_inputs(args)
    if args.depth_input_source == "foundation_stereo":
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
    condition_contract = pose_condition_contract(args.mode)
    if condition_contract["exact_evaluation_pose_available_to_inference"]:
        print("[Track2Map] Evaluation reference is the declared oracle pose input:", args.eval_pose_file)
    else:
        print("[Track2Map] Exact evaluation-reference provenance is deferred until inference exits.")
    if args.mode in ("light_noise", "heavy_noise", "noisy", "noisy_auto_gate"):
        profile = cfg.get("track2map_runtime", {}).get("gate_profile_resolved", args.gate_profile)
        print("[Track2Map] Gate profile:", profile)
    print("[Track2Map] Command:", " ".join(cmd))

    provenance_path = Path(args.output) / "launch_provenance.json"
    pre_inference_provenance_path = Path(args.output) / "pre_inference_provenance.json"
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "sequence": args.seq,
        "run_kind": args.run_kind,
        "inference_policy": STRICT_GT_FREE,
        "inference_policy_scope": (
            "forbid_track_annotations_and_dedicated_evaluation_reference_reads; "
            "declared pose-prior access is governed by pose_condition_contract"
        ),
        "pose_condition_contract": condition_contract,
        "input_folder": args.input_folder,
        "input_pose_file": args.pose_file,
        "input_pose_file_sha256": (
            _sha256(Path(args.pose_file)) if args.pose_file is not None else None
        ),
        "eval_pose_file": args.eval_pose_file,
        "eval_pose_file_sha256": _sha256(Path(args.eval_pose_file)),
        "dedicated_eval_reference_argument_passed_to_inference": False,
        "evaluation_pose_bytes_used_as_oracle_input": args.mode == "clean_pose",
        "condition_input_verification": args.condition_input_verification,
        "config_file": str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "command": cmd,
        "status": "dry_run" if args.dry_run else "launching",
    }
    pre_inference_provenance = inference_visible_launch_provenance(provenance, args.mode)
    _write_launch_provenance(pre_inference_provenance_path, pre_inference_provenance)

    if args.dry_run:
        provenance["evaluation_reference_provenance"] = "dry_run_only_no_inference_executed"
        _write_launch_provenance(provenance_path, provenance)
        return 0
    result = subprocess.run(cmd, cwd=str(repo_root))
    provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
    provenance["evaluation_reference_provenance"] = "attached_post_inference"
    provenance["returncode"] = int(result.returncode)
    post_condition_verification = verify_condition_inputs(args)
    provenance["post_inference_condition_input_verification"] = post_condition_verification
    if result.returncode == 0 and post_condition_verification != args.condition_input_verification:
        provenance["returncode"] = 1
        provenance["status"] = "failed_input_lineage"
    else:
        provenance["status"] = "inference_completed" if result.returncode == 0 else "failed"
    _write_launch_provenance(provenance_path, provenance)
    return int(provenance["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
