#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
import time
import zipfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import yaml


SEQUENCES = ("P1_1", "P2_0", "P2_1", "P3_1", "P3_2")
MODES = ("clean_pose", "light_noise", "heavy_noise", "no_pose")
RUN_KINDS = ("full_evaluation", "smoke", "pose_only_diagnostic")
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CONTRACTS = {
    "full_evaluation": {
        "completed_status": "completed",
        "pose_only": False,
        "exact_frame_slice": (0, 200, 1),
    },
    "smoke": {
        "completed_status": "completed_smoke",
        "pose_only": False,
        "exact_frame_slice": None,
    },
    "pose_only_diagnostic": {
        "completed_status": "completed_pose_diagnostic",
        "pose_only": True,
        "exact_frame_slice": None,
    },
}
SEALED_STATUSES = frozenset(
    contract["completed_status"] for contract in RUN_CONTRACTS.values()
)
DATA_ROOT = Path(os.environ.get("TRACK2MAP_DATA_ROOT", REPO_ROOT / "steremis_tracking"))
LIGHT_ROOT = Path(
    os.environ.get("TRACK2MAP_LIGHT_POSE_ROOT", REPO_ROOT / "stereomis_noisy_light")
)
HEAVY_ROOT = Path(
    os.environ.get(
        "TRACK2MAP_HEAVY_POSE_ROOT",
        REPO_ROOT / "stereomis_noisy_light_transx10",
    )
)
FOUNDATION_ROOT = Path(
    os.environ.get("FOUNDATION_STEREO_ROOT", REPO_ROOT / "foundationstereo")
)
FOUNDATION_CKPT = FOUNDATION_ROOT / "pretrained_models/23-51-11/model_best_bp2.pth"
FOUNDATION_CFG = FOUNDATION_ROOT / "pretrained_models/23-51-11/cfg.yaml"
FOUNDATION_INTRINSIC = FOUNDATION_ROOT / "assets/K.txt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.inference_policy import pose_condition_contract

DEFAULT_START = 0
DEFAULT_STOP = 200
DEFAULT_STEP = 1
CODE_ARTIFACTS = (
    "run.py",
    "scripts/run_track2map.py",
    "scripts/run_phase0_matrix.py",
    "scripts/eval_track2map_metrics.py",
    "src/utils/datasets.py",
    "src/utils/inference_policy.py",
)


def model_weight_artifacts():
    return (
        FOUNDATION_CKPT,
        FOUNDATION_CFG,
        FOUNDATION_INTRINSIC,
        Path.home() / ".cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
        Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/lpips/weights/v0.1/alex.pth",
    )


def configure_paths(args):
    global DATA_ROOT, LIGHT_ROOT, HEAVY_ROOT
    global FOUNDATION_ROOT, FOUNDATION_CKPT, FOUNDATION_CFG, FOUNDATION_INTRINSIC

    DATA_ROOT = args.data_root.expanduser().resolve()
    LIGHT_ROOT = args.light_pose_root.expanduser().resolve()
    HEAVY_ROOT = args.heavy_pose_root.expanduser().resolve()
    FOUNDATION_ROOT = args.foundation_root.expanduser().resolve()
    FOUNDATION_CKPT = (
        args.foundation_ckpt.expanduser().resolve()
        if args.foundation_ckpt is not None
        else FOUNDATION_ROOT / "pretrained_models/23-51-11/model_best_bp2.pth"
    )
    FOUNDATION_CFG = (
        args.foundation_cfg.expanduser().resolve()
        if args.foundation_cfg is not None
        else FOUNDATION_ROOT / "pretrained_models/23-51-11/cfg.yaml"
    )
    FOUNDATION_INTRINSIC = (
        args.foundation_intrinsic_file.expanduser().resolve()
        if args.foundation_intrinsic_file is not None
        else FOUNDATION_ROOT / "assets/K.txt"
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_json_constant(value):
    raise ValueError(f"Non-standard JSON numeric constant is forbidden: {value}")


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def code_hashes():
    return {relative: sha256(REPO_ROOT / relative) for relative in CODE_ARTIFACTS}


def runtime_source_paths():
    source_paths = [REPO_ROOT / "run.py"]
    for source_root in (REPO_ROOT / "scripts", REPO_ROOT / "src", REPO_ROOT / "configs"):
        if source_root.is_dir():
            source_paths.extend(source_root.rglob("*.py"))
            source_paths.extend(source_root.rglob("*.yaml"))
            source_paths.extend(source_root.rglob("*.yml"))
    native_root = REPO_ROOT / "src/submodules/gaussian-rasterization"
    native_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    if native_root.is_dir():
        source_paths.extend(
            path
            for path in native_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and ".git" not in path.parts
            and (
                path.suffix.lower() in native_suffixes
                or path.name in {"CMakeLists.txt", "setup.py"}
            )
        )
    source_paths.append(native_extension_path())
    return sorted(
        (
            path
            for path in set(source_paths)
            if path.is_file() and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )


def native_extension_path():
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python EXT_SUFFIX is unavailable; cannot bind the rasterizer binary.")
    path = (
        REPO_ROOT
        / "src/submodules/gaussian-rasterization/diff_gaussian_rasterization"
        / f"_C{extension_suffix}"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Active Gaussian rasterizer extension is missing: {path}")
    return path


def _optional_command_output(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def native_extension_lineage():
    binary = native_extension_path()
    native_root = REPO_ROOT / "src/submodules/gaussian-rasterization"
    source_paths = [
        path
        for path in runtime_source_paths()
        if native_root in path.parents and path != binary
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")

    build_id = None
    readelf_output = _optional_command_output(["readelf", "-n", str(binary)])
    if readelf_output:
        for line in readelf_output.splitlines():
            if "Build ID:" in line:
                build_id = line.split("Build ID:", 1)[1].strip()
                break
    return [
        {
            "backend": "diff_gaussian_rasterization",
            "module": "diff_gaussian_rasterization._C",
            "binary_path": str(binary),
            "binary_relative_path": binary.relative_to(REPO_ROOT).as_posix(),
            "binary_sha256": sha256(binary),
            "elf_build_id": build_id,
            "native_source_file_count": len(source_paths),
            "native_source_tree_sha256": digest.hexdigest(),
            "nvcc_version": _optional_command_output(["nvcc", "--version"]),
            "cuda_elf_inventory": _optional_command_output(
                ["cuobjdump", "--list-elf", str(binary)]
            ),
        }
    ]


def environment_fingerprint():
    packages = {}
    for distribution in ("torch", "torchvision", "numpy", "scipy", "imageio", "lpips", "PyYAML"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        gpu_inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception:
        gpu_inventory = []
    try:
        import torch

        torch_build = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception:
        torch_build = {"torch_cuda_version": None, "cudnn_version": None}
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "gpu_inventory": gpu_inventory,
        "torch_build": torch_build,
        "environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "determinism_policy": {
            "numpy_seed": 0,
            "python_random_seed": 0,
            "torch_seed": 0,
            "torch_cuda_seed_all": 0,
            "torch_deterministic_algorithms": "enabled_warn_only",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
    }


def runtime_lineage():
    digest = hashlib.sha256()
    for path in runtime_source_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    weights = {
        str(path): sha256(path)
        for path in model_weight_artifacts()
        if path.is_file()
    }
    return {
        "source_tree_sha256": digest.hexdigest(),
        "model_weight_sha256": weights,
        "native_extensions": native_extension_lineage(),
        "python_executable": sys.executable,
        "random_seed": 0,
        "environment": environment_fingerprint(),
    }


def write_source_snapshot(path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source_path in runtime_source_paths():
            relative = source_path.relative_to(REPO_ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source_path.read_bytes())
    os.replace(tmp, path)
    return sha256(path)


def validate_source_snapshot(path, expected_runtime_lineage):
    expected = {
        source_path.relative_to(REPO_ROOT).as_posix(): sha256(source_path)
        for source_path in runtime_source_paths()
    }
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Source snapshot entries do not match the runtime source tree.")
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    if actual != expected:
        raise ValueError("Source snapshot content does not match the runtime source tree.")
    digest = hashlib.sha256()
    for relative in sorted(actual):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(actual[relative].encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != expected_runtime_lineage["source_tree_sha256"]:
        raise ValueError("Source snapshot tree hash does not match launch runtime lineage.")


def requested_frame_slice(args):
    start = DEFAULT_START if args.start is None else int(args.start)
    stop = DEFAULT_STOP if args.stop is None else int(args.stop)
    step = DEFAULT_STEP if args.step is None else int(args.step)
    if start < 0 or stop <= start or step <= 0:
        raise ValueError(f"Invalid Phase 0 frame slice: start={start} stop={stop} step={step}")
    return start, stop, step


def run_contract(run_kind):
    try:
        return RUN_CONTRACTS[run_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown Phase 0 run kind: {run_kind}") from exc


def completed_status(run_kind):
    return run_contract(run_kind)["completed_status"]


def is_pose_only(run_kind):
    return bool(run_contract(run_kind)["pose_only"])


def expected_frame_count(args):
    start, stop, step = requested_frame_slice(args)
    return len(range(start, stop, step))


def validate_run_contract(args):
    contract = run_contract(args.run_kind)
    frame_slice = requested_frame_slice(args)
    if expected_frame_count(args) < 2:
        raise ValueError("Phase 0 evaluation requires at least two selected frames.")
    if args.pose_only_eval and args.run_kind != "pose_only_diagnostic":
        raise ValueError("--pose-only-eval is valid only with --run-kind pose_only_diagnostic.")
    args.pose_only_eval = bool(contract["pose_only"])
    if args.run_kind == "full_evaluation":
        if frame_slice != contract["exact_frame_slice"]:
            raise ValueError(
                "full_evaluation requires the exact frame slice 0:200:1; "
                f"got {frame_slice[0]}:{frame_slice[1]}:{frame_slice[2]}."
            )
        if args.iters is not None or args.iters_first is not None:
            raise ValueError(
                "full_evaluation forbids iteration overrides; use the audited 100/1000 config."
            )
        if args.debug:
            raise ValueError("full_evaluation forbids --debug.")
    return {
        "schema_version": 1,
        "run_kind": args.run_kind,
        "completed_status": contract["completed_status"],
        "pose_only": bool(contract["pose_only"]),
        "frame_slice": {"start": frame_slice[0], "stop": frame_slice[1], "step": frame_slice[2]},
        "expected_num_frames": expected_frame_count(args),
        "required_metrics": (
            ["ate_m", "are_deg", "rpet_m", "rper_deg"]
            if contract["pose_only"]
            else ["psnr", "ssim", "lpips", "ate_m", "are_deg", "rpet_m", "rper_deg"]
        ),
    }


@contextmanager
def exclusive_run_lock(run_dir):
    lock_path = Path(run_dir) / ".phase0.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Run directory is already locked: {run_dir}") from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def input_data_lineage(seq, args):
    start, stop, step = requested_frame_slice(args)
    seq_dir = DATA_ROOT / seq
    left_paths = sorted((seq_dir / "video_frames").glob("*l.png"))[slice(start, stop, step)]
    if not left_paths:
        raise ValueError(f"No selected input frames for {seq}: {start}:{stop}:{step}")

    digest = hashlib.sha256()
    frame_ids = []
    artifact_count = 0
    for left_path in left_paths:
        frame_id = left_path.stem.removesuffix("l")
        frame_ids.append(frame_id)
        artifacts = {
            "left_rgb": left_path,
            "right_rgb": Path(str(left_path).replace("l.png", "r.png")),
            "mask": Path(str(left_path).replace("video_frames", "masks")),
            "semantic": Path(str(left_path).replace("video_frames", "semantic_predictions")),
        }
        for role, artifact in artifacts.items():
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing {role} input for frame {frame_id}: {artifact}")
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
            digest.update(frame_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256(artifact).encode("ascii"))
            digest.update(b"\n")
            artifact_count += 1
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"Input frame IDs are not unique for {seq}.")
    return {
        "frame_slice": {"start": start, "stop": stop, "step": step},
        "num_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "artifact_count": artifact_count,
        "combined_sha256": digest.hexdigest(),
    }


def pose_file_lineage(input_pose, eval_pose):
    return {
        "input_pose_file": str(Path(input_pose).resolve()) if input_pose is not None else None,
        "input_pose_sha256": sha256(input_pose) if input_pose is not None else None,
        "evaluation_pose_file": str(Path(eval_pose).resolve()),
        "evaluation_pose_sha256": sha256(eval_pose),
    }


def inference_visible_pose_lineage(pose_lineage, mode):
    if mode == "clean_pose":
        return dict(pose_lineage)
    return {
        "input_pose_file": pose_lineage["input_pose_file"],
        "input_pose_sha256": pose_lineage["input_pose_sha256"],
        "evaluation_reference_provenance": "deferred_until_post_inference",
    }


def redact_evaluation_reference_from_command(command, mode):
    redacted = list(command)
    if mode == "clean_pose":
        return redacted
    try:
        value_index = redacted.index("--eval-pose-file") + 1
    except ValueError:
        return redacted
    if value_index >= len(redacted):
        raise ValueError("Malformed launch command: --eval-pose-file has no value.")
    redacted[value_index] = "<deferred-until-post-inference>"
    return redacted


def _finite_metric(value, label):
    if value is None:
        raise ValueError(f"Missing per-frame metric: {label}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite per-frame metric: {label}={value}")
    return parsed


def _assert_summary_matches(summary, key, computed):
    reported = _finite_metric(summary.get(key), f"summary.{key}")
    if not math.isclose(reported, computed, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"Summary/per-frame mismatch for {key}: reported={reported} recomputed={computed}"
        )


def validate_metrics_file(
    path,
    run_kind,
    eval_pose=None,
    expected_frames=None,
    expected_evaluation_kind=None,
):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_json_constant)
    pose_only = is_pose_only(run_kind)
    if payload.get("schema_version") != "track2map.metrics.v2":
        raise ValueError(f"Unexpected metrics schema: {payload.get('schema_version')}")
    actual_evaluation_kind = payload.get("evaluation_kind")
    if expected_evaluation_kind is None:
        expected_evaluation_kind = (
            "pose_only_diagnostic" if pose_only else "reconstruction_and_pose"
        )
    if actual_evaluation_kind != expected_evaluation_kind:
        raise ValueError(
            f"Evaluation-kind mismatch: expected={expected_evaluation_kind} "
            f"actual={actual_evaluation_kind}"
        )
    if payload.get("complete_seven_metric_set") is not (not pose_only):
        raise ValueError("Metrics completeness flag does not match the run contract.")
    if pose_only and payload.get("completion_eligible") is not False:
        raise ValueError("Pose-only diagnostic must declare completion_eligible=false.")
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        raise ValueError("Missing evaluator payload attestation.")
    unattested_payload = dict(payload)
    unattested_payload.pop("attestation", None)
    if attestation.get("canonical_payload_sha256") != canonical_json_sha256(unattested_payload):
        raise ValueError("Evaluator payload attestation hash mismatch.")
    summary = payload.get("summary", {})
    required = ["ate_m", "are_deg", "rpet_m", "rper_deg"]
    if not pose_only:
        required.extend(["psnr", "ssim", "lpips"])
    invalid = [
        key
        for key in required
        if summary.get(key) is None or not math.isfinite(float(summary[key]))
    ]
    if invalid:
        raise ValueError(f"Required metrics are missing or non-finite: {invalid}")

    pose_frames = summary.get("num_pose_frames")
    if not isinstance(pose_frames, int) or pose_frames < 2:
        raise ValueError(f"Invalid pose frame count: {pose_frames}")
    if expected_frames is not None and pose_frames != int(expected_frames):
        raise ValueError(
            f"Run-contract frame count mismatch: expected={expected_frames} actual={pose_frames}"
        )
    pose_frame_ids = payload.get("pose_frame_ids")
    if not isinstance(pose_frame_ids, list) or len(pose_frame_ids) != pose_frames:
        raise ValueError(
            f"Pose frame-ID count mismatch: ids={len(pose_frame_ids) if isinstance(pose_frame_ids, list) else None} "
            f"poses={pose_frames}"
        )
    if len(set(pose_frame_ids)) != pose_frames:
        raise ValueError("Pose frame IDs must be unique.")
    per_frame_pose = payload.get("per_frame_pose")
    if not isinstance(per_frame_pose, list) or len(per_frame_pose) != pose_frames:
        raise ValueError(
            "Per-frame pose count mismatch: "
            f"rows={len(per_frame_pose) if isinstance(per_frame_pose, list) else None} "
            f"summary={pose_frames}"
        )
    abs_trans = []
    abs_rot = []
    rel_trans = []
    rel_rot = []
    for idx, row in enumerate(per_frame_pose):
        if not isinstance(row, dict) or row.get("frame_idx") != idx:
            raise ValueError(f"Invalid per-frame pose row at index {idx}")
        if row.get("pose_frame_id") != pose_frame_ids[idx]:
            raise ValueError(f"Per-frame pose identity mismatch at index {idx}")
        abs_trans.append(
            _finite_metric(row.get("absolute_translation_error_m"), f"pose[{idx}].absolute_translation")
        )
        abs_rot.append(
            _finite_metric(row.get("absolute_rotation_error_deg"), f"pose[{idx}].absolute_rotation")
        )
        if idx == 0:
            if row.get("relative_translation_error_m") is not None or row.get(
                "relative_rotation_error_deg"
            ) is not None:
                raise ValueError("The first pose row must not declare a delta-1 RPE value.")
        else:
            rel_trans.append(
                _finite_metric(row.get("relative_translation_error_m"), f"pose[{idx}].relative_translation")
            )
            rel_rot.append(
                _finite_metric(row.get("relative_rotation_error_deg"), f"pose[{idx}].relative_rotation")
            )
    _assert_summary_matches(
        summary,
        "ate_m",
        math.sqrt(math.fsum(value * value for value in abs_trans) / len(abs_trans)),
    )
    _assert_summary_matches(
        summary,
        "are_deg",
        math.sqrt(math.fsum(value * value for value in abs_rot) / len(abs_rot)),
    )
    _assert_summary_matches(
        summary,
        "rpet_m",
        math.sqrt(math.fsum(value * value for value in rel_trans) / len(rel_trans)),
    )
    _assert_summary_matches(
        summary,
        "rper_deg",
        math.sqrt(math.fsum(value * value for value in rel_rot) / len(rel_rot)),
    )
    pose_ids_name = payload.get("pose_provenance", {}).get("files", {}).get("frame_ids")
    pose_ids_path = Path(path).parent / str(pose_ids_name)
    if not pose_ids_name or not pose_ids_path.is_file():
        raise ValueError("Missing pose frame-ID sidecar declared by provenance.")
    if sha256(pose_ids_path) != summary.get("pose_frame_ids_sha256"):
        raise ValueError("Pose frame-ID sidecar hash mismatch.")
    with pose_ids_path.open("r", encoding="utf-8") as handle:
        if json.load(handle) != pose_frame_ids:
            raise ValueError("Pose frame-ID sidecar content mismatch.")
    determinism = payload.get("pose_provenance", {}).get("determinism", {})
    expected_determinism = {
        "random_seed": 0,
        "torch_deterministic_algorithms_enabled": True,
        "torch_deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    if determinism != expected_determinism:
        raise ValueError(
            f"Inference determinism provenance mismatch: expected={expected_determinism} "
            f"actual={determinism}"
        )
    if payload.get("pose_provenance", {}).get("run_kind") != run_kind:
        raise ValueError("Inference pose provenance run_kind does not match the matrix contract.")
    evaluator_determinism = (
        payload.get("derivation_provenance", {})
        .get("runtime", {})
        .get("determinism", {})
    )
    required_evaluator_determinism = {
        "random_seed": 0,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms_enabled": True,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in required_evaluator_determinism.items():
        if evaluator_determinism.get(key) != expected:
            raise ValueError(
                f"Evaluator determinism mismatch for {key}: "
                f"expected={expected} actual={evaluator_determinism.get(key)}"
            )
    if eval_pose is not None:
        expected_eval_path = str(Path(eval_pose).resolve())
        expected_eval_hash = sha256(eval_pose)
        if str(Path(summary.get("eval_pose_file", "")).resolve()) != expected_eval_path:
            raise ValueError("Evaluator-reported pose path does not match the launch evaluation pose.")
        if summary.get("eval_pose_file_sha256") != expected_eval_hash:
            raise ValueError("Evaluator-observed pose hash does not match the launch evaluation pose hash.")
    if not pose_only:
        recon_frames = summary.get("num_reconstruction_frames")
        per_frame = payload.get("per_frame_reconstruction")
        source_counts = payload.get("reconstruction_source_counts")
        if recon_frames != pose_frames:
            raise ValueError(
                f"Reconstruction/pose frame count mismatch: reconstruction={recon_frames} pose={pose_frames}"
            )
        if not isinstance(per_frame, list) or len(per_frame) != recon_frames:
            raise ValueError(
                f"Per-frame reconstruction count mismatch: rows={len(per_frame) if isinstance(per_frame, list) else None} "
                f"summary={recon_frames}"
            )
        expected_counts = {
            "gt": recon_frames,
            "runtime_gt": recon_frames,
            "render": recon_frames,
            "mask": recon_frames,
        }
        if source_counts != expected_counts:
            raise ValueError(
                f"Reconstruction source count mismatch: expected={expected_counts} actual={source_counts}"
            )
        for idx, row in enumerate(per_frame):
            if not isinstance(row, dict) or row.get("frame_idx") != idx:
                raise ValueError(f"Invalid per-frame reconstruction row at index {idx}")
            for metric_key in ("psnr", "ssim", "lpips"):
                _finite_metric(row.get(metric_key), f"reconstruction[{idx}].{metric_key}")
            binding = row.get("frame_binding") if isinstance(row, dict) else None
            if not isinstance(binding, dict):
                raise ValueError(f"Missing frame identity binding at reconstruction row {idx}")
            if binding.get("local_frame_idx") != idx:
                raise ValueError(f"Invalid local frame identity at reconstruction row {idx}")
            if binding.get("dataset_frame_id") != binding.get("pose_frame_id"):
                raise ValueError(f"Pose/image identity mismatch at reconstruction row {idx}")
            if binding.get("pose_frame_id") != pose_frame_ids[idx]:
                raise ValueError(f"Pose sidecar/frame binding mismatch at reconstruction row {idx}")
            for key in (
                "gt_sha256",
                "runtime_gt_sha256",
                "render_sha256",
                "mask_sha256",
                "right_rgb_sha256",
                "semantic_sha256",
            ):
                value = binding.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"Missing {key} at reconstruction row {idx}")
            if binding["gt_sha256"] != binding["runtime_gt_sha256"]:
                raise ValueError(f"Dataset/runtime GT hash mismatch at reconstruction row {idx}")
            for path_key, hash_key in (
                ("gt_path", "gt_sha256"),
                ("runtime_gt_path", "runtime_gt_sha256"),
                ("render_path", "render_sha256"),
                ("mask_path", "mask_sha256"),
                ("right_rgb_path", "right_rgb_sha256"),
                ("semantic_path", "semantic_sha256"),
            ):
                artifact_path = binding.get(path_key)
                if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
                    raise ValueError(f"Missing bound artifact {path_key} at reconstruction row {idx}")
                if sha256(artifact_path) != binding[hash_key]:
                    raise ValueError(f"Bound artifact hash mismatch for {path_key} at reconstruction row {idx}")
        for metric_key in ("psnr", "ssim", "lpips"):
            values = [float(row[metric_key]) for row in per_frame]
            _assert_summary_matches(summary, metric_key, math.fsum(values) / len(values))
    if expected_evaluation_kind == "independent_verification":
        comparison = payload.get("reference_comparison")
        if not isinstance(comparison, dict) or comparison.get("status") != "match":
            raise ValueError("Independent verification did not attest a matching primary summary.")
        if comparison.get("reference_per_frame_rows_read") is not False:
            raise ValueError("Independent verification must not consume primary per-frame rows.")
        if comparison.get("computed_before_reference_read") is not True:
            raise ValueError("Independent metrics were not derived before reading the reference.")
    return payload


def validate_completion_artifacts(run_dir, run_kind, expected_frames):
    run_dir = Path(run_dir)
    required_files = [
        "effective_config.yaml",
        "environment_lock.json",
        "launch_provenance.json",
        "pre_inference_provenance.json",
        "metrics.json",
        "pose_provenance.json",
        "input_c2w.npy",
        "initialized_c2w.npy",
        "optimized_c2w.npy",
        "pose_frame_ids.json",
        "source_snapshot.zip",
        "tracked.pckl",
    ]
    if not is_pose_only(run_kind):
        required_files.append("metrics_verification.json")
    missing = [relative for relative in required_files if not (run_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completion artifacts: {missing}")

    if not is_pose_only(run_kind):
        frame_artifacts = {
            "raw_rgb/gt": "*.png",
            "raw_rgb/render": "*.png",
            "raw_depth/gt": "*.npy",
            "raw_depth/render": "*.npy",
            "mapping": "*.jpg",
            "semantic": "*.jpg",
        }
        for relative_dir, pattern in frame_artifacts.items():
            paths = sorted((run_dir / relative_dir).glob(pattern))
            if len(paths) != int(expected_frames):
                raise ValueError(
                    f"Completion artifact count mismatch for {relative_dir}: "
                    f"expected={expected_frames} actual={len(paths)}"
                )


def output_artifact_hashes(run_dir, run_kind, expected_frames):
    run_dir = Path(run_dir)
    validate_completion_artifacts(run_dir, run_kind, expected_frames)
    excluded = {"run_status.json", ".phase0.lock"}
    result = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"Completion tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Completion tree contains a non-regular artifact: {relative}")
        if relative in excluded or path.name.endswith(".tmp"):
            continue
        result[relative] = sha256(path)
    if not result:
        raise ValueError("Completion artifact tree is empty.")
    return result


def _find_forbidden_keys(value, forbidden, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                found.append(child_prefix)
            found.extend(_find_forbidden_keys(child, forbidden, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, forbidden, f"{prefix}[{index}]"))
    return found


def validate_inference_evaluation_isolation(run_dir, mode, eval_pose):
    if mode == "clean_pose":
        return
    forbidden = {"evaluation_pose_file", "evaluation_pose_sha256"}
    config_path = Path(run_dir) / "effective_config.yaml"
    provenance_path = Path(run_dir) / "pose_provenance.json"
    prelaunch_path = Path(run_dir) / "pre_inference_provenance.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    with prelaunch_path.open("r", encoding="utf-8") as handle:
        prelaunch = json.load(handle)
    found = _find_forbidden_keys(config, forbidden, "effective_config")
    found.extend(_find_forbidden_keys(provenance, forbidden, "pose_provenance"))
    found.extend(_find_forbidden_keys(prelaunch, forbidden, "pre_inference_provenance"))
    eval_path = str(Path(eval_pose).resolve())
    eval_hash = sha256(eval_pose)
    serialized = (
        yaml.safe_dump(config, sort_keys=True)
        + json.dumps(provenance, sort_keys=True)
        + json.dumps(prelaunch, sort_keys=True)
    )
    if eval_path in serialized or eval_hash in serialized:
        found.append("evaluation pose path/hash byte value")
    if found:
        raise ValueError(
            "Non-clean inference artifacts expose evaluation-pose metadata: " + ", ".join(found)
        )
    if prelaunch.get("evaluation_reference_provenance") != "deferred_until_post_inference":
        raise ValueError("Pre-inference provenance does not attest deferred evaluation access.")


def validate_first_frame_gate_outcome(run_dir, mode):
    if mode not in ("light_noise", "heavy_noise"):
        return {"required": False, "status": "not_applicable"}
    path = Path(run_dir) / "tool_motion_score.json"
    if not path.is_file():
        raise FileNotFoundError("Missing tool_motion_score.json for pose-trust-gated condition.")
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError("Pose-trust gate diagnostics have no first-frame record.")
    gate = rows[0]
    if gate.get("ff_pose_gate_evaluated") is not True:
        raise ValueError("First-frame pose-trust gate was not evaluated.")
    if gate.get("ff_pose_gate_observations_finite") is not True:
        raise ValueError("First-frame pose-trust gate observations were not all finite.")
    for key in (
        "ff_pose_gate_init_psnr",
        "ff_pose_gate_init_ssim",
        "ff_pose_gate_final_psnr",
        "ff_pose_gate_final_ssim",
        "ff_pose_gate_psnr_drop",
        "ff_pose_gate_ssim_drop",
    ):
        _finite_metric(gate.get(key), f"gate.{key}")
    triggered = gate.get("ff_pose_gate_triggered")
    fallback = gate.get("ff_pose_gate_fallback_applied")
    reasons = gate.get("ff_pose_gate_reasons")
    if not isinstance(triggered, bool) or not isinstance(fallback, bool):
        raise ValueError("First-frame gate outcome flags are invalid.")
    if not isinstance(reasons, str) or not reasons:
        raise ValueError("First-frame gate reason is missing.")
    if triggered and reasons == "pass":
        raise ValueError("Triggered first-frame gate is mislabeled as pass.")
    if not triggered and reasons != "pass":
        raise ValueError("Passing first-frame gate carries failure reasons.")
    if triggered and gate.get("ff_pose_gate_fallback_mode") == "no_prior" and not fallback:
        raise ValueError("Triggered no-prior first-frame gate did not apply its fallback.")
    return {
        "required": True,
        "status": "verified",
        "artifact": path.name,
        "artifact_sha256": sha256(path),
        "triggered": triggered,
        "fallback_applied": fallback,
        "reasons": reasons,
    }


def _metric_values_match(left, right, key):
    absolute = {
        "psnr": 2e-6,
        "ssim": 2e-6,
        "lpips": 1e-6,
        "absolute_translation_error_m": 1e-10,
        "absolute_rotation_error_deg": 1e-7,
        "relative_translation_error_m": 1e-10,
        "relative_rotation_error_deg": 1e-7,
    }[key]
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=absolute)


def compare_metric_derivations(primary_path, verification_path):
    with Path(primary_path).open("r", encoding="utf-8") as handle:
        primary = json.load(handle)
    with Path(verification_path).open("r", encoding="utf-8") as handle:
        verification = json.load(handle)
    comparison = verification.get("reference_comparison", {})
    if comparison.get("reference_metrics_path") != str(Path(primary_path).resolve()):
        raise ValueError("Verification references a different primary metrics artifact.")
    if comparison.get("reference_metrics_sha256") != sha256(primary_path):
        raise ValueError("Verification reference hash does not match primary metrics bytes.")
    if primary.get("pose_frame_ids") != verification.get("pose_frame_ids"):
        raise ValueError("Primary/verification pose frame IDs differ.")
    if primary.get("reconstruction_source_counts") != verification.get(
        "reconstruction_source_counts"
    ):
        raise ValueError("Primary/verification reconstruction source counts differ.")

    pose_keys = (
        "absolute_translation_error_m",
        "absolute_rotation_error_deg",
        "relative_translation_error_m",
        "relative_rotation_error_deg",
    )
    primary_pose = primary.get("per_frame_pose", [])
    verification_pose = verification.get("per_frame_pose", [])
    if len(primary_pose) != len(verification_pose):
        raise ValueError("Primary/verification pose row counts differ.")
    for index, (left, right) in enumerate(zip(primary_pose, verification_pose)):
        if left.get("frame_idx") != right.get("frame_idx") or left.get(
            "pose_frame_id"
        ) != right.get("pose_frame_id"):
            raise ValueError(f"Primary/verification pose identity differs at row {index}.")
        for key in pose_keys:
            if left.get(key) is None or right.get(key) is None:
                if left.get(key) is not None or right.get(key) is not None:
                    raise ValueError(f"Primary/verification {key} nullability differs at row {index}.")
            elif not _metric_values_match(left[key], right[key], key):
                raise ValueError(f"Primary/verification {key} differs at row {index}.")

    primary_recon = primary.get("per_frame_reconstruction", [])
    verification_recon = verification.get("per_frame_reconstruction", [])
    if len(primary_recon) != len(verification_recon):
        raise ValueError("Primary/verification reconstruction row counts differ.")
    for index, (left, right) in enumerate(zip(primary_recon, verification_recon)):
        if left.get("frame_idx") != right.get("frame_idx") or left.get(
            "frame_binding"
        ) != right.get("frame_binding"):
            raise ValueError(f"Primary/verification reconstruction identity differs at row {index}.")
        for key in ("psnr", "ssim", "lpips"):
            if not _metric_values_match(left.get(key), right.get(key), key):
                raise ValueError(f"Primary/verification {key} differs at row {index}.")
    return {
        "status": "verified",
        "primary_sha256": sha256(primary_path),
        "verification_sha256": sha256(verification_path),
        "pose_rows_compared": len(primary_pose),
        "reconstruction_rows_compared": len(primary_recon),
        "metrics_per_reconstruction_row": ["psnr", "ssim", "lpips"],
        "metrics_per_pose_row": list(pose_keys),
    }


def validate_launch_provenance(run_dir, seq, mode, run_kind, expected_pose_lineage):
    path = Path(run_dir) / "launch_provenance.json"
    if not path.is_file():
        raise ValueError("Missing launch_provenance.json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "inference_completed" or payload.get("returncode") != 0:
        raise ValueError(
            f"Launcher provenance is not completed: status={payload.get('status')} "
            f"returncode={payload.get('returncode')}"
        )
    if payload.get("sequence") != seq or payload.get("mode") != mode:
        raise ValueError("Launcher provenance sequence/mode mismatch.")
    if payload.get("run_kind") != run_kind:
        raise ValueError("Launcher provenance run_kind mismatch.")
    if payload.get("pose_condition_contract") != pose_condition_contract(mode):
        raise ValueError("Launcher pose-condition contract mismatch.")
    if payload.get("input_pose_file") != expected_pose_lineage["input_pose_file"]:
        raise ValueError("Launcher provenance input-pose path mismatch.")
    if payload.get("input_pose_file_sha256") != expected_pose_lineage["input_pose_sha256"]:
        raise ValueError("Launcher provenance input-pose hash mismatch.")
    if payload.get("eval_pose_file") != expected_pose_lineage["evaluation_pose_file"]:
        raise ValueError("Launcher provenance evaluation-pose path mismatch.")
    if payload.get("eval_pose_file_sha256") != expected_pose_lineage["evaluation_pose_sha256"]:
        raise ValueError("Launcher provenance evaluation-pose hash mismatch.")
    if payload.get("dedicated_eval_reference_argument_passed_to_inference") is not False:
        raise ValueError("Dedicated evaluation-pose argument reached the inference command.")
    expected_oracle = mode == "clean_pose"
    if payload.get("evaluation_pose_bytes_used_as_oracle_input") is not expected_oracle:
        raise ValueError("Launcher provenance oracle-input role mismatch.")
    if not expected_oracle and payload.get("evaluation_reference_provenance") != (
        "attached_post_inference"
    ):
        raise ValueError("Evaluation-reference provenance was not attached after inference.")
    initial = payload.get("condition_input_verification")
    final = payload.get("post_inference_condition_input_verification")
    if not isinstance(initial, dict) or initial.get("status") != "verified" or final != initial:
        raise ValueError("Launcher condition verification is missing or changed during inference.")
    if initial.get("declared_mode") != mode or initial.get("sequence") != seq:
        raise ValueError("Launcher condition verification label mismatch.")
    config_path = Path(run_dir) / "effective_config.yaml"
    if payload.get("config_sha256") != sha256(config_path):
        raise ValueError("Launcher provenance effective-config hash mismatch.")
    return payload


def resume_validation_errors(
    previous,
    run_dir,
    metrics_path,
    run_kind,
    expected_frames,
    launch_cmd,
    current_code_hashes,
    current_runtime_lineage,
    current_input_data_lineage,
    input_pose,
    eval_pose,
):
    errors = []
    if previous.get("status") != completed_status(run_kind):
        errors.append(f"status={previous.get('status')}")
    if previous.get("run_contract", {}).get("run_kind") != run_kind:
        errors.append("run kind changed")
    if previous.get("run_contract", {}).get("expected_num_frames") != expected_frames:
        errors.append("run contract frame count changed")
    if previous.get("code_sha256") != current_code_hashes:
        errors.append("code hashes changed")
    if previous.get("runtime_lineage") != current_runtime_lineage:
        errors.append("runtime source, weights, environment, or seed changed")
    if previous.get("input_data_lineage") != current_input_data_lineage:
        errors.append("left/right RGB, mask, semantic, or frame identities changed")
    if previous.get("launch_command") != launch_cmd:
        errors.append("launch command changed")
    expected_input_hash = sha256(input_pose) if input_pose is not None else None
    if previous.get("input_pose_sha256") != expected_input_hash:
        errors.append("input pose hash changed")
    if previous.get("eval_pose_sha256") != sha256(eval_pose):
        errors.append("evaluation pose hash changed")
    current_pose_lineage = pose_file_lineage(input_pose, eval_pose)
    if previous.get("pose_file_lineage") != current_pose_lineage:
        errors.append("input/evaluation pose lineage changed")
    current_condition_lineage = None
    try:
        current_condition_lineage = condition_dataset_lineage(previous.get("sequence"))
        if previous.get("condition_dataset_lineage") != current_condition_lineage:
            errors.append("light/heavy condition dataset lineage changed")
    except Exception as exc:
        errors.append(f"condition dataset lineage validation failed: {exc}")
    snapshot_path = Path(run_dir) / "source_snapshot.zip"
    try:
        validate_source_snapshot(snapshot_path, current_runtime_lineage)
        if previous.get("source_snapshot_sha256") != sha256(snapshot_path):
            errors.append("source snapshot hash changed")
    except Exception as exc:
        errors.append(f"source snapshot validation failed: {exc}")
    environment_lock_path = Path(run_dir) / "environment_lock.json"
    try:
        with environment_lock_path.open("r", encoding="utf-8") as handle:
            environment_lock = json.load(handle)
        if environment_lock.get("schema_version") != 2:
            errors.append("environment lock schema changed")
        if environment_lock.get("run_contract") != previous.get("run_contract"):
            errors.append("environment lock run contract changed")
        if environment_lock.get("runtime_lineage") != current_runtime_lineage:
            errors.append("environment lock runtime lineage changed")
        if environment_lock.get("condition_dataset_lineage") != current_condition_lineage:
            errors.append("environment lock condition dataset lineage changed")
        if environment_lock.get("gpu_physical_id") != previous.get("gpu_physical_id"):
            errors.append("environment lock GPU identity changed")
        if environment_lock.get("source_snapshot_sha256") != previous.get(
            "source_snapshot_sha256"
        ):
            errors.append("environment lock source snapshot changed")
        if previous.get("environment_lock_sha256") != sha256(environment_lock_path):
            errors.append("environment lock hash changed")
    except Exception as exc:
        errors.append(f"environment lock validation failed: {exc}")
    try:
        validate_metrics_file(
            metrics_path,
            run_kind,
            eval_pose=eval_pose,
            expected_frames=expected_frames,
            expected_evaluation_kind=(
                "pose_only_diagnostic" if is_pose_only(run_kind) else "reconstruction_and_pose"
            ),
        )
        if not is_pose_only(run_kind):
            verification_path = Path(run_dir) / "metrics_verification.json"
            validate_metrics_file(
                verification_path,
                run_kind,
                eval_pose=eval_pose,
                expected_frames=expected_frames,
                expected_evaluation_kind="independent_verification",
            )
            compare_metric_derivations(metrics_path, verification_path)
    except Exception as exc:
        errors.append(f"metric validation failed: {exc}")
    try:
        validate_launch_provenance(
            run_dir,
            previous.get("sequence"),
            previous.get("mode"),
            run_kind,
            current_pose_lineage,
        )
        validate_inference_evaluation_isolation(
            run_dir,
            previous.get("mode"),
            eval_pose,
        )
        gate_attestation = validate_first_frame_gate_outcome(run_dir, previous.get("mode"))
        if previous.get("first_frame_gate_attestation") != gate_attestation:
            errors.append("first-frame gate attestation changed")
    except Exception as exc:
        errors.append(f"launch provenance validation failed: {exc}")
    try:
        current_artifacts = output_artifact_hashes(run_dir, run_kind, expected_frames)
        if previous.get("output_artifact_sha256") != current_artifacts:
            errors.append("output artifact hashes changed or are incomplete")
    except Exception as exc:
        errors.append(f"output artifact validation failed: {exc}")
    return errors


def live_lineage_errors(
    args,
    seq,
    mode,
    run_dir,
    expected_code,
    expected_runtime,
    expected_inputs,
    expected_pose_lineage,
    expected_condition_lineage,
    expected_source_snapshot_sha256,
    expected_environment_lock,
):
    errors = []
    if code_hashes() != expected_code:
        errors.append("code hashes changed during execution")
    if runtime_lineage() != expected_runtime:
        errors.append("runtime source or model weights changed during execution")
    if input_data_lineage(seq, args) != expected_inputs:
        errors.append("left/right RGB, masks, semantics, or frame identities changed during execution")
    input_pose, eval_pose = pose_files(seq, mode)
    if pose_file_lineage(input_pose, eval_pose) != expected_pose_lineage:
        errors.append("input or evaluation pose changed during execution")
    try:
        if condition_dataset_lineage(seq) != expected_condition_lineage:
            errors.append("light/heavy condition dataset lineage changed during execution")
    except Exception as exc:
        errors.append(f"condition dataset lineage invalid: {exc}")
    snapshot_path = Path(run_dir) / "source_snapshot.zip"
    try:
        validate_source_snapshot(snapshot_path, expected_runtime)
        if sha256(snapshot_path) != expected_source_snapshot_sha256:
            errors.append("source snapshot hash changed during execution")
    except Exception as exc:
        errors.append(f"source snapshot invalid: {exc}")
    environment_lock_path = Path(run_dir) / "environment_lock.json"
    try:
        with environment_lock_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != expected_environment_lock:
                errors.append("environment lock changed during execution")
    except Exception as exc:
        errors.append(f"environment lock invalid: {exc}")
    config_path = Path(run_dir) / "effective_config.yaml"
    if not config_path.is_file():
        errors.append("effective config is missing")
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        data = config.get("data", {})
        expected_slice = expected_inputs["frame_slice"]
        try:
            actual_slice = {key: int(data.get(key)) for key in ("start", "stop", "step")}
        except (TypeError, ValueError):
            errors.append("effective config has an invalid frame slice")
        else:
            if actual_slice != expected_slice:
                errors.append(
                    f"effective config frame slice changed: expected={expected_slice} actual={actual_slice}"
                )
        runtime_cfg = config.get("track2map_runtime", {})
        if runtime_cfg.get("run_kind") != args.run_kind:
            errors.append("effective config run_kind changed or is missing")
        if args.run_kind == "full_evaluation":
            training = config.get("training", {})
            if int(training.get("iters", -1)) != 100 or int(training.get("iters_first", -1)) != 1000:
                errors.append("full_evaluation effective config is not using 100/1000 iterations")
    try:
        validate_launch_provenance(run_dir, seq, mode, args.run_kind, expected_pose_lineage)
        validate_inference_evaluation_isolation(run_dir, mode, eval_pose)
        validate_first_frame_gate_outcome(run_dir, mode)
    except Exception as exc:
        errors.append(f"launch provenance invalid: {exc}")
    return errors


def pose_files(seq, mode):
    eval_pose = DATA_ROOT / seq / "groundtruth.txt"
    if mode == "clean_pose":
        input_pose = eval_pose
    elif mode == "light_noise":
        input_pose = LIGHT_ROOT / seq / "groundtruth_noisy.txt"
    elif mode == "heavy_noise":
        input_pose = HEAVY_ROOT / seq / "groundtruth_noisy.txt"
    elif mode == "no_pose":
        input_pose = None
    else:
        raise ValueError(mode)
    return input_pose, eval_pose


def _read_pose_table(path):
    ids = []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.replace(",", " ").split()
            if not values or values[0].startswith("#"):
                continue
            if len(values) < 8:
                raise ValueError(f"Malformed pose row in {path}: {line.rstrip()}")
            ids.append(values[0])
            rows.append([float(value) for value in values[1:8]])
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    return ids, rows


def condition_dataset_lineage(seq):
    gt_path = DATA_ROOT / seq / "groundtruth.txt"
    light_path = LIGHT_ROOT / seq / "groundtruth_noisy.txt"
    heavy_path = HEAVY_ROOT / seq / "groundtruth_noisy.txt"
    gt_ids, gt_rows = _read_pose_table(gt_path)
    light_ids, light_rows = _read_pose_table(light_path)
    heavy_ids, heavy_rows = _read_pose_table(heavy_path)
    if gt_ids != light_ids or gt_ids != heavy_ids:
        raise ValueError(f"GT/light/heavy pose IDs differ for {seq}.")

    max_translation_relation_error = 0.0
    max_rotation_difference = 0.0
    max_light_translation_delta = 0.0
    for gt_row, light_row, heavy_row in zip(gt_rows, light_rows, heavy_rows):
        for axis in range(3):
            light_delta = light_row[axis] - gt_row[axis]
            heavy_delta = heavy_row[axis] - gt_row[axis]
            max_light_translation_delta = max(max_light_translation_delta, abs(light_delta))
            max_translation_relation_error = max(
                max_translation_relation_error,
                abs(heavy_delta - 10.0 * light_delta),
            )
        for component in range(3, 7):
            max_rotation_difference = max(
                max_rotation_difference,
                abs(heavy_row[component] - light_row[component]),
            )
    if max_light_translation_delta <= 0.0:
        raise ValueError(f"Light pose file has no translation perturbation for {seq}.")
    if max_translation_relation_error > 1e-10:
        raise ValueError(
            f"Heavy translation perturbation is not 10x light for {seq}: "
            f"max_error={max_translation_relation_error}"
        )
    if max_rotation_difference > 1e-12:
        raise ValueError(
            f"Heavy/light rotations differ for {seq}: max_error={max_rotation_difference}"
        )
    return {
        "sequence": seq,
        "num_pose_rows": len(gt_ids),
        "frame_ids_sha256": hashlib.sha256("\n".join(gt_ids).encode("utf-8")).hexdigest(),
        "ground_truth_sha256": sha256(gt_path),
        "light_pose_sha256": sha256(light_path),
        "heavy_pose_sha256": sha256(heavy_path),
        "heavy_translation_perturbation_multiplier": 10.0,
        "max_translation_relation_abs_error": max_translation_relation_error,
        "heavy_light_rotation_relationship": "identical",
        "max_rotation_abs_difference": max_rotation_difference,
        "condition_scope": "translation magnitude and gate policy bundle",
    }


def build_launch_command(args, seq, mode, run_dir):
    input_pose, eval_pose = pose_files(seq, mode)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_track2map.py"),
        "--mode",
        mode,
        "--seq",
        seq,
        "--input-folder",
        str(DATA_ROOT / seq),
        "--output",
        str(run_dir),
        "--eval-pose-file",
        str(eval_pose),
        "--depth-input-source",
        "foundation_stereo",
        "--foundation-root",
        str(FOUNDATION_ROOT),
        "--foundation-ckpt",
        str(FOUNDATION_CKPT),
        "--foundation-cfg",
        str(FOUNDATION_CFG),
        "--foundation-intrinsic-file",
        str(FOUNDATION_INTRINSIC),
        "--flow-init-source",
        "raft",
        "--cotracker-runtime-mode",
        "online_prefix",
        "--run-kind",
        args.run_kind,
        "--save-config",
        str(run_dir / "effective_config.yaml"),
    ]
    if input_pose is not None:
        cmd.extend(["--pose-file", str(input_pose)])
    if args.start is not None:
        cmd.extend(["--start", str(args.start)])
    if args.stop is not None:
        cmd.extend(["--stop", str(args.stop)])
    if args.step is not None:
        cmd.extend(["--step", str(args.step)])
    if args.iters is not None:
        cmd.extend(["--iters", str(args.iters)])
    if args.iters_first is not None:
        cmd.extend(["--iters-first", str(args.iters_first)])
    if args.debug:
        cmd.append("--debug")
    if not is_pose_only(args.run_kind):
        cmd.append("--visualize")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def build_eval_command(args, seq, run_dir, eval_pose, evaluation_kind=None):
    with (run_dir / "effective_config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data = cfg["data"]
    if evaluation_kind is None:
        evaluation_kind = (
            "pose_only_diagnostic"
            if is_pose_only(args.run_kind)
            else "reconstruction_and_pose"
        )
    output_name = (
        "metrics_verification.json"
        if evaluation_kind == "independent_verification"
        else "metrics.json"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_track2map_metrics.py"),
        "--seq",
        seq,
        "--input-root",
        str(DATA_ROOT),
        "--output-dir",
        str(run_dir),
        "--eval-pose-file",
        str(eval_pose),
        "--pose-start",
        str(data["start"]),
        "--pose-stop",
        str(data["stop"]),
        "--pose-step",
        str(data["step"]),
        "--evaluation-kind",
        evaluation_kind,
        "--output-json",
        str(run_dir / output_name),
    ]
    if evaluation_kind == "independent_verification":
        cmd.extend(["--reference-metrics-json", str(run_dir / "metrics.json")])
    if evaluation_kind == "pose_only_diagnostic":
        cmd.extend(["--device", "cpu"])
    else:
        cmd.extend(["--device", "cuda"])
    return cmd


def validate_inputs(jobs):
    missing = [
        str(path)
        for path in (FOUNDATION_ROOT, FOUNDATION_CKPT, FOUNDATION_CFG, FOUNDATION_INTRINSIC)
        if not path.exists()
    ]
    for seq, mode in jobs:
        seq_dir = DATA_ROOT / seq
        input_pose, eval_pose = pose_files(seq, mode)
        for path in (seq_dir, eval_pose, input_pose):
            if path is not None and not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing Phase 0 inputs:\n" + "\n".join(sorted(set(missing))))


def run_job(args, seq, mode, output_root, env):
    run_dir = output_root / "runs" / seq / mode
    log_path = output_root / "logs" / f"{seq}_{mode}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(run_dir):
        return _run_job_locked(args, seq, mode, output_root, env, run_dir, log_path)


def _run_job_locked(args, seq, mode, output_root, env, run_dir, log_path):
    status_path = run_dir / "run_status.json"
    metrics_path = run_dir / "metrics.json"
    verification_path = run_dir / "metrics_verification.json"

    launch_cmd = build_launch_command(args, seq, mode, run_dir)
    input_pose, eval_pose = pose_files(seq, mode)
    contract = validate_run_contract(args)
    current_code_hashes = code_hashes()
    current_runtime_lineage = runtime_lineage()
    current_input_data_lineage = input_data_lineage(seq, args)
    current_pose_file_lineage = pose_file_lineage(input_pose, eval_pose)
    current_condition_lineage = condition_dataset_lineage(seq)
    current_pose_contract = pose_condition_contract(mode)
    inference_pose_lineage = inference_visible_pose_lineage(
        current_pose_file_lineage,
        mode,
    )
    inference_launch_command = redact_evaluation_reference_from_command(launch_cmd, mode)

    if status_path.is_file():
        with status_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous_status = previous.get("status")
        if previous_status in SEALED_STATUSES:
            if args.rerun:
                raise RuntimeError(
                    f"Refusing to overwrite sealed result {seq}/{mode}; use a fresh output root."
                )
            errors = resume_validation_errors(
                previous,
                run_dir,
                metrics_path,
                args.run_kind,
                current_input_data_lineage["num_frames"],
                launch_cmd,
                current_code_hashes,
                current_runtime_lineage,
                current_input_data_lineage,
                input_pose,
                eval_pose,
            )
            if errors:
                raise RuntimeError(
                    f"Refusing to resume stale or unbound result {seq}/{mode}: "
                    + "; ".join(errors)
                )
            print(f"[skip] {seq}/{mode} already {previous_status}", flush=True)
            return True
        raise RuntimeError(
            f"Run directory contains non-sealed status={previous_status!r} for {seq}/{mode}; "
            "use a fresh output root."
        )

    stale_entries = [
        path.relative_to(run_dir).as_posix()
        for path in run_dir.iterdir()
        if path.name != ".phase0.lock"
    ]
    if stale_entries:
        raise RuntimeError(
            f"Run directory is non-empty without a status for {seq}/{mode}: {sorted(stale_entries)}"
        )

    source_snapshot_path = run_dir / "source_snapshot.zip"
    source_snapshot_sha256 = write_source_snapshot(source_snapshot_path)
    environment_lock = {
        "schema_version": 2,
        "gpu_physical_id": str(args.gpu),
        "run_contract": contract,
        "runtime_lineage": current_runtime_lineage,
        "condition_dataset_lineage": current_condition_lineage,
        "source_snapshot_file": source_snapshot_path.name,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    environment_lock_path = run_dir / "environment_lock.json"
    atomic_json(environment_lock_path, environment_lock)

    status = {
        "sequence": seq,
        "mode": mode,
        "run_contract": contract,
        "gpu_physical_id": str(args.gpu),
        "status": "dry_run" if args.dry_run else "running",
        "started_at": utc_now(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "code_sha256": current_code_hashes,
        "runtime_lineage": current_runtime_lineage,
        "input_data_lineage": current_input_data_lineage,
        "pose_file_lineage": inference_pose_lineage,
        "pose_condition_contract": current_pose_contract,
        "condition_dataset_lineage": current_condition_lineage,
        "source_snapshot_sha256": source_snapshot_sha256,
        "environment_lock_sha256": sha256(environment_lock_path),
        "input_pose_file": str(input_pose) if input_pose is not None else None,
        "input_pose_sha256": sha256(input_pose) if input_pose is not None else None,
        "dedicated_eval_reference_argument_passed_to_inference": False,
        "evaluation_pose_bytes_used_as_oracle_input": mode == "clean_pose",
        "condition_interpretation": {
            "clean_pose": "oracle_pose_mapping_control",
            "light_noise": "light_noise_plus_1x_gate_policy_bundle",
            "heavy_noise": "heavy_translation_noise_plus_10x_gate_policy_bundle",
            "no_pose": "identity_initialization_with_causal_raft_vo",
        }[mode],
        "launch_command": inference_launch_command,
        "log_file": str(log_path),
    }
    if mode != "clean_pose":
        status["evaluation_reference_provenance"] = "deferred_until_post_inference"
    else:
        status["eval_pose_file"] = str(eval_pose)
        status["eval_pose_sha256"] = sha256(eval_pose)
    atomic_json(status_path, status)
    print(f"[start] gpu={args.gpu} {seq}/{mode}", flush=True)
    start_time = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] LAUNCH {' '.join(inference_launch_command)}\n")
        log.flush()
        launch = subprocess.run(
            launch_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        status["launch_returncode"] = int(launch.returncode)
        status["pose_file_lineage"] = current_pose_file_lineage
        status["eval_pose_file"] = str(eval_pose)
        status["eval_pose_sha256"] = sha256(eval_pose)
        status["launch_command"] = launch_cmd
        status["evaluation_reference_provenance"] = (
            "attached_after_launcher_dry_run" if args.dry_run else "attached_post_inference"
        )
        atomic_json(status_path, status)
        if args.dry_run:
            status["status"] = "dry_run"
            status["finished_at"] = utc_now()
            status["elapsed_seconds"] = time.time() - start_time
            atomic_json(status_path, status)
            return launch.returncode == 0
        if launch.returncode != 0:
            status["status"] = "failed_inference"
            status["finished_at"] = utc_now()
            status["elapsed_seconds"] = time.time() - start_time
            atomic_json(status_path, status)
            print(f"[fail] {seq}/{mode} inference rc={launch.returncode}", flush=True)
            return False

        lineage_errors = live_lineage_errors(
            args,
            seq,
            mode,
            run_dir,
            current_code_hashes,
            current_runtime_lineage,
            current_input_data_lineage,
            current_pose_file_lineage,
            current_condition_lineage,
            source_snapshot_sha256,
            environment_lock,
        )
        if lineage_errors:
            status["status"] = "failed_lineage"
            status["lineage_validation_stage"] = "post_inference"
            status["lineage_validation_errors"] = lineage_errors
            status["finished_at"] = utc_now()
            status["elapsed_seconds"] = time.time() - start_time
            atomic_json(status_path, status)
            log.write(f"\n[{utc_now()}] LINEAGE_VALIDATION_FAILED {'; '.join(lineage_errors)}\n")
            log.flush()
            print(f"[fail] {seq}/{mode} lineage changed after inference", flush=True)
            return False
        status["post_inference_lineage_revalidated"] = True

        eval_cmd = build_eval_command(args, seq, run_dir, eval_pose)
        status["eval_command"] = eval_cmd
        atomic_json(status_path, status)
        log.write(f"\n[{utc_now()}] EVAL {' '.join(eval_cmd)}\n")
        log.flush()
        evaluation = subprocess.run(
            eval_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        status["eval_returncode"] = int(evaluation.returncode)
        evaluation_ok = evaluation.returncode == 0

        if evaluation_ok:
            try:
                metrics = validate_metrics_file(
                    metrics_path,
                    args.run_kind,
                    eval_pose=eval_pose,
                    expected_frames=current_input_data_lineage["num_frames"],
                    expected_evaluation_kind=(
                        "pose_only_diagnostic"
                        if is_pose_only(args.run_kind)
                        else "reconstruction_and_pose"
                    ),
                )
                lineage_errors = live_lineage_errors(
                    args,
                    seq,
                    mode,
                    run_dir,
                    current_code_hashes,
                    current_runtime_lineage,
                    current_input_data_lineage,
                    current_pose_file_lineage,
                    current_condition_lineage,
                    source_snapshot_sha256,
                    environment_lock,
                )
                if lineage_errors:
                    raise ValueError("Lineage changed before completion: " + "; ".join(lineage_errors))
                status["validated_metrics"] = metrics["summary"]
                if not is_pose_only(args.run_kind):
                    verification_cmd = build_eval_command(
                        args,
                        seq,
                        run_dir,
                        eval_pose,
                        evaluation_kind="independent_verification",
                    )
                    status["metrics_verification_command"] = verification_cmd
                    atomic_json(status_path, status)
                    log.write(
                        f"\n[{utc_now()}] VERIFY_METRICS {' '.join(verification_cmd)}\n"
                    )
                    log.flush()
                    verification = subprocess.run(
                        verification_cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    status["metrics_verification_returncode"] = int(
                        verification.returncode
                    )
                    if verification.returncode != 0:
                        raise ValueError(
                            "Independent raw-artifact metric verification failed with "
                            f"return code {verification.returncode}."
                        )
                    validate_metrics_file(
                        verification_path,
                        args.run_kind,
                        eval_pose=eval_pose,
                        expected_frames=current_input_data_lineage["num_frames"],
                        expected_evaluation_kind="independent_verification",
                    )
                    status["metrics_derivation_comparison"] = compare_metric_derivations(
                        metrics_path,
                        verification_path,
                    )
                status["first_frame_gate_attestation"] = validate_first_frame_gate_outcome(
                    run_dir,
                    mode,
                )
                validate_inference_evaluation_isolation(run_dir, mode, eval_pose)
                lineage_errors = live_lineage_errors(
                    args,
                    seq,
                    mode,
                    run_dir,
                    current_code_hashes,
                    current_runtime_lineage,
                    current_input_data_lineage,
                    current_pose_file_lineage,
                    current_condition_lineage,
                    source_snapshot_sha256,
                    environment_lock,
                )
                if lineage_errors:
                    raise ValueError(
                        "Lineage changed during metric verification: "
                        + "; ".join(lineage_errors)
                    )
                status["completion_lineage_revalidated"] = True
            except Exception as exc:
                evaluation_ok = False
                status["eval_returncode"] = 1
                status["metric_validation_error"] = str(exc)
                log.write(f"\n[{utc_now()}] METRIC_VALIDATION_FAILED {exc}\n")
                log.flush()

    status["finished_at"] = utc_now()
    status["elapsed_seconds"] = time.time() - start_time
    status["status"] = completed_status(args.run_kind) if evaluation_ok else "failed_evaluation"
    if (run_dir / "effective_config.yaml").is_file():
        status["effective_config_sha256"] = sha256(run_dir / "effective_config.yaml")
    if evaluation_ok:
        try:
            status["output_artifact_sha256"] = output_artifact_hashes(
                run_dir,
                args.run_kind,
                current_input_data_lineage["num_frames"],
            )
        except Exception as exc:
            evaluation_ok = False
            status["status"] = "failed_sealing"
            status["artifact_sealing_error"] = str(exc)
    atomic_json(status_path, status)
    print(f"[{status['status']}] {seq}/{mode} elapsed={status['elapsed_seconds']:.1f}s", flush=True)
    return evaluation_ok


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable Track2Map Phase 0 matrix worker")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-kind", required=True, choices=RUN_KINDS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--light-pose-root", type=Path, default=LIGHT_ROOT)
    parser.add_argument("--heavy-pose-root", type=Path, default=HEAVY_ROOT)
    parser.add_argument("--foundation-root", type=Path, default=FOUNDATION_ROOT)
    parser.add_argument("--foundation-ckpt", type=Path)
    parser.add_argument("--foundation-cfg", type=Path)
    parser.add_argument("--foundation-intrinsic-file", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--iters-first", type=int, default=None)
    parser.add_argument(
        "--pose-only-eval",
        action="store_true",
        help="Deprecated compatibility flag; requires --run-kind pose_only_diagnostic.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_paths(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    os.environ["PYTHONUNBUFFERED"] = "1"
    contract = validate_run_contract(args)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards.")

    all_jobs = [(seq, mode) for seq in args.sequences for mode in args.modes]
    jobs = [job for index, job in enumerate(all_jobs) if index % args.num_shards == args.shard_index]
    validate_inputs(jobs)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(),
        "gpu_physical_id": str(args.gpu),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "jobs": [{"sequence": seq, "mode": mode} for seq, mode in jobs],
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "code_sha256": code_hashes(),
        "runtime_lineage": runtime_lineage(),
        "run_contract": contract,
        "requested_run_type": "dry_run" if args.dry_run else args.run_kind,
    }
    atomic_json(output_root / f"shard_{args.shard_index}_manifest.json", manifest)

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    failures = []
    for seq, mode in jobs:
        if not run_job(args, seq, mode, output_root, env):
            failures.append(f"{seq}/{mode}")
            if args.fail_fast:
                break

    manifest["finished_at"] = utc_now()
    job_statuses = {}
    for seq, mode in jobs:
        status_path = output_root / "runs" / seq / mode / "run_status.json"
        if status_path.is_file():
            with status_path.open("r", encoding="utf-8") as handle:
                job_statuses[f"{seq}/{mode}"] = json.load(handle).get("status")
        else:
            job_statuses[f"{seq}/{mode}"] = "missing"
    manifest["job_statuses"] = job_statuses
    if args.dry_run:
        manifest["status"] = "dry_run" if not failures else "dry_run_with_failures"
    else:
        expected_status = completed_status(args.run_kind)
        incomplete = [job for job, status in job_statuses.items() if status != expected_status]
        for job in incomplete:
            if job not in failures:
                failures.append(job)
        manifest["status"] = expected_status if not failures else f"{expected_status}_with_failures"
    manifest["failures"] = failures
    atomic_json(output_root / f"shard_{args.shard_index}_manifest.json", manifest)
    if failures:
        print("Failed jobs: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
