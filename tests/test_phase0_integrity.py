import argparse
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.eval_track2map_metrics import (
    bind_reconstruction_frames,
    compute_pose_metrics,
    compute_pose_metrics_with_rows,
    compute_recon_metrics,
    load_pose_trajectory,
    load_pose_trajectory_with_ids,
)
from scripts.run_phase0_matrix import (
    canonical_json_sha256,
    compare_metric_derivations,
    condition_dataset_lineage,
    exclusive_run_lock,
    inference_visible_pose_lineage,
    input_data_lineage,
    native_extension_path,
    output_artifact_hashes,
    pose_file_lineage,
    run_job,
    runtime_lineage,
    validate_first_frame_gate_outcome,
    validate_launch_provenance,
    validate_metrics_file,
    validate_run_contract,
    validate_source_snapshot,
    write_source_snapshot,
)
from scripts.run_track2map import (
    build_config,
    inference_condition_view,
    inference_visible_launch_provenance,
    verify_condition_inputs,
)
from src.utils.inference_policy import (
    STRICT_GT_FREE,
    enforce_inference_policy,
    pose_condition_contract,
    poses_to_metric,
    resolve_track_annotation_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_attested_metrics(path, payload):
    payload.pop("attestation", None)
    payload["attestation"] = {
        "canonical_payload_sha256": canonical_json_sha256(payload),
        "hash_scope": "canonical JSON payload excluding the attestation object",
        "atomic_write": True,
    }
    Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def launcher_args(mode, stop=None):
    return argparse.Namespace(
        seq="P1_1",
        mode=mode,
        output="/tmp/phase0-test",
        depth_input_source="raft_stereo",
        foundation_root="/unused/foundation",
        foundation_ckpt="/unused/model.pth",
        foundation_cfg="/unused/cfg.yaml",
        foundation_intrinsic_file="/unused/K.txt",
        foundation_valid_iters=32,
        cotracker_repo=str(REPO_ROOT / "cotracker"),
        cotracker_runtime_mode="online_prefix",
        flow_init_source="raft",
        start=None,
        stop=stop,
        step=None,
        iters=None,
        iters_first=None,
        gate_profile="auto",
        run_kind="smoke",
        pose_file="/tmp/input_pose.txt" if mode != "no_pose" else None,
    )


class StrictPolicyTests(unittest.TestCase):
    def test_strict_policy_never_probes_annotation(self):
        cfg = enforce_inference_policy(
            {
                "track2map_runtime": {"inference_policy": STRICT_GT_FREE},
                "training": {
                    "cotracker_flow_init_query_source": "anchor",
                    "optical_flow_init_source": "raft",
                    "pose_optimization": {},
                },
            }
        )

        def poisoned_probe(_):
            raise AssertionError("strict policy probed track_pts.pckl")

        path, probed = resolve_track_annotation_path("/poisoned", cfg, poisoned_probe)
        self.assertIsNone(path)
        self.assertFalse(probed)

    def test_strict_policy_rejects_gt_queries(self):
        cfg = {
            "track2map_runtime": {"inference_policy": STRICT_GT_FREE},
            "training": {
                "cotracker_flow_init_query_source": "gt",
                "optical_flow_init_source": "raft",
                "pose_optimization": {},
            },
        }
        with self.assertRaisesRegex(ValueError, "forbids"):
            enforce_inference_policy(cfg)

    def test_strict_policy_rejects_external_pose_initialization(self):
        cfg = {
            "track2map_runtime": {"inference_policy": STRICT_GT_FREE},
            "training": {
                "cotracker_flow_init_query_source": "anchor",
                "optical_flow_init_source": "raft",
                "pose_optimization": {"pose_no_prior_external_init_file": "/tmp/oracle.npy"},
            },
        }
        with self.assertRaisesRegex(ValueError, "external_init"):
            enforce_inference_policy(cfg)

    def test_strict_policy_disables_all_gt_track_controls(self):
        cfg = {
            "track2map_runtime": {"inference_policy": STRICT_GT_FREE},
            "training": {
                "cotracker_flow_init_query_source": "anchor",
                "optical_flow_init_source": "raft",
                "pt_cotracker_gs_refine_enabled": True,
                "pose_optimization": {
                    "w_track_2d": 0.45,
                    "pose_track_guard_enabled": True,
                    "pose_track_adaptive_weight_enabled": True,
                    "pose_track_constraint_enabled": True,
                    "pose_track_constraint_weight": 0.3,
                    "pose_freeze_if_track_loss_invalid": True,
                    "pose_final_revert_on_track_worse": True,
                    "pose_step_adaptive_enabled": True,
                    "pose_no_prior_vo_corr_source": "auto",
                },
            },
        }
        pose = enforce_inference_policy(cfg)["training"]["pose_optimization"]
        self.assertEqual(pose["w_track_2d"], 0.0)
        self.assertEqual(pose["pose_track_constraint_weight"], 0.0)
        self.assertEqual(pose["pose_no_prior_vo_corr_source"], "raft")
        for key in (
            "pose_track_guard_enabled",
            "pose_track_adaptive_weight_enabled",
            "pose_track_constraint_enabled",
            "pose_freeze_if_track_loss_invalid",
            "pose_final_revert_on_track_worse",
            "pose_step_adaptive_enabled",
        ):
            self.assertFalse(pose[key], key)
        self.assertFalse(cfg["training"]["pt_cotracker_gs_refine_enabled"])
        runtime = cfg["track2map_runtime"]
        self.assertEqual(runtime["track_annotation_access"], "forbidden")
        self.assertEqual(runtime["dataset_tool_mask_access"], "allowed_and_recorded")
        self.assertEqual(runtime["semantic_prediction_access"], "allowed_and_recorded")
        self.assertNotIn("annotation_access", runtime)


class EffectiveConfigTests(unittest.TestCase):
    def test_clean_pose_is_a_frozen_dataset_pose_condition(self):
        cfg = build_config(launcher_args("clean_pose"), REPO_ROOT)
        pose = cfg["training"]["pose_optimization"]
        self.assertFalse(pose["enabled"])
        self.assertEqual(pose["pose_init_mode"], "dataset")
        self.assertFalse(pose["tool_motion_gate_enabled"])
        self.assertFalse(pose["pose_step_limit_enabled"])
        self.assertEqual(cfg["data"]["pose_source"], "file")

    def test_noise_and_no_pose_sources_are_explicit(self):
        light = build_config(launcher_args("light_noise"), REPO_ROOT)
        heavy = build_config(launcher_args("heavy_noise"), REPO_ROOT)
        no_pose = build_config(launcher_args("no_pose"), REPO_ROOT)
        self.assertEqual(light["track2map_runtime"]["gate_profile_resolved"], "1x")
        self.assertEqual(heavy["track2map_runtime"]["gate_profile_resolved"], "10x")
        self.assertEqual(light["data"]["pose_source"], "file")
        self.assertEqual(heavy["data"]["pose_source"], "file")
        self.assertEqual(no_pose["data"]["pose_source"], "identity")
        self.assertEqual(no_pose["training"]["pose_optimization"]["pose_init_mode"], "no_prior")

    def test_prefix_time_denominator_is_invariant_to_smoke_stop(self):
        full = build_config(launcher_args("clean_pose"), REPO_ROOT)
        smoke = build_config(launcher_args("clean_pose", stop=5), REPO_ROOT)
        self.assertEqual(
            full["data"]["deformation_time_denominator"],
            smoke["data"]["deformation_time_denominator"],
        )

    def test_phase0_default_slice_is_200_frames_for_every_mode(self):
        for mode in ("clean_pose", "light_noise", "heavy_noise", "no_pose"):
            data = build_config(launcher_args(mode), REPO_ROOT)["data"]
            self.assertEqual((data["start"], data["stop"], data["step"]), (0, 200, 1))


class PoseMetricTests(unittest.TestCase):
    def test_internal_scale_is_removed_before_metric_output(self):
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
        poses[:, 0, 3] = 0.1
        metric = poses_to_metric(poses, internal_scale=10.0)
        self.assertTrue(np.allclose(metric[:, 0, 3], 0.01))

        gt = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
        result = compute_pose_metrics(metric, gt)
        self.assertAlmostEqual(result["ate_m"], 0.01, places=8)

    def test_external_eval_pose_loader_normalizes_and_slices(self):
        rows = [
            "0 1 0 0 0 0 0 1\n",
            "1 1.01 0 0 0 0 0 1\n",
            "2 1.02 0 0 0 0 0 1\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groundtruth.txt"
            path.write_text("".join(rows), encoding="utf-8")
            poses, frame_ids = load_pose_trajectory_with_ids(str(path), start=0, stop=2, step=1)
        self.assertEqual(poses.shape, (2, 4, 4))
        self.assertEqual(frame_ids, ["0", "1"])
        self.assertAlmostEqual(float(poses[0, 0, 3]), 0.0, places=8)
        self.assertAlmostEqual(float(poses[1, 0, 3]), 0.01, places=8)

    def test_nonfinite_and_nonrigid_pose_rows_are_rejected(self):
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 3, axis=0)
        nonfinite = poses.copy()
        nonfinite[-1, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_pose_metrics(nonfinite, poses)

        nonrigid = poses.copy()
        nonrigid[-1, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "non-rigid"):
            compute_pose_metrics(nonrigid, poses)

    def test_pose_metric_rows_recompute_the_reported_summary(self):
        gt = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 3, axis=0)
        est = gt.copy()
        est[1:, 0, 3] = [0.01, 0.03]
        summary, rows = compute_pose_metrics_with_rows(est, gt, ["1", "2", "3"])
        self.assertEqual([row["pose_frame_id"] for row in rows], ["1", "2", "3"])
        ate = np.sqrt(np.mean([row["absolute_translation_error_m"] ** 2 for row in rows]))
        rpet = np.sqrt(
            np.mean([row["relative_translation_error_m"] ** 2 for row in rows[1:]])
        )
        self.assertAlmostEqual(summary["ate_m"], ate)
        self.assertAlmostEqual(summary["rpet_m"], rpet)


class MetricCompletenessTests(unittest.TestCase):
    def test_reconstruction_count_mismatch_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            compute_recon_metrics(
                ["gt0", "gt1"],
                ["render0"],
                ["mask0", "mask1"],
                "nonzero_is_bg",
                "cpu",
                None,
            )

    def test_matrix_requires_finite_pose_and_reconstruction_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_paths = {}
            artifact_hashes = {}
            for name, content in (
                ("gt", b"gt"),
                ("runtime_gt", b"gt"),
                ("render", b"render"),
                ("mask", b"mask"),
                ("right_rgb", b"right"),
                ("semantic", b"semantic"),
            ):
                artifact = root / f"{name}.bin"
                artifact.write_bytes(content)
                artifact_paths[f"{name}_path"] = str(artifact)
                artifact_hashes[f"{name}_sha256"] = hashlib.sha256(content).hexdigest()
            binding = {
                "local_frame_idx": 0,
                "dataset_frame_id": "000001",
                "pose_frame_id": "000001",
                **artifact_paths,
                **artifact_hashes,
            }
            pose_rows = [
                {
                    "frame_idx": 0,
                    "pose_frame_id": "000001",
                    "absolute_translation_error_m": 0.01,
                    "absolute_rotation_error_deg": 1.0,
                    "relative_translation_error_m": None,
                    "relative_rotation_error_deg": None,
                },
                {
                    "frame_idx": 1,
                    "pose_frame_id": "000002",
                    "absolute_translation_error_m": 0.01,
                    "absolute_rotation_error_deg": 1.0,
                    "relative_translation_error_m": 0.002,
                    "relative_rotation_error_deg": 0.5,
                },
            ]
            payload = {
                "schema_version": "track2map.metrics.v2",
                "artifact_kind": "primary_metrics",
                "evaluation_kind": "reconstruction_and_pose",
                "complete_seven_metric_set": True,
                "completion_eligible": None,
                "summary": {
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "lpips": 0.2,
                    "ate_m": 0.01,
                    "are_deg": 1.0,
                    "rpet_m": 0.002,
                    "rper_deg": 0.5,
                    "num_pose_frames": 2,
                    "num_reconstruction_frames": 2,
                },
                "pose_provenance": {
                    "run_kind": "smoke",
                    "files": {"frame_ids": "pose_frame_ids.json"},
                    "determinism": {
                        "random_seed": 0,
                        "torch_deterministic_algorithms_enabled": True,
                        "torch_deterministic_warn_only": True,
                        "cudnn_deterministic": True,
                        "cudnn_benchmark": False,
                        "cublas_workspace_config": ":4096:8",
                    },
                },
                "pose_frame_ids": ["000001", "000002"],
                "per_frame_pose": pose_rows,
                "reconstruction_source_counts": {
                    "gt": 2,
                    "runtime_gt": 2,
                    "render": 2,
                    "mask": 2,
                },
                "per_frame_reconstruction": [
                    {
                        "frame_idx": 0,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "lpips": 0.2,
                        "frame_binding": binding,
                    },
                    {
                        "frame_idx": 1,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "lpips": 0.2,
                        "frame_binding": {
                            **binding,
                            "local_frame_idx": 1,
                            "dataset_frame_id": "000002",
                            "pose_frame_id": "000002",
                        }
                    },
                ],
                "derivation_provenance": {
                    "runtime": {
                        "determinism": {
                            "random_seed": 0,
                            "cublas_workspace_config": ":4096:8",
                            "torch_deterministic_algorithms_enabled": True,
                            "torch_deterministic_warn_only": False,
                            "cudnn_deterministic": True,
                            "cudnn_benchmark": False,
                        }
                    }
                },
            }
            pose_ids_path = root / "pose_frame_ids.json"
            pose_ids_path.write_text(json.dumps(payload["pose_frame_ids"], indent=2), encoding="utf-8")
            payload["summary"]["pose_frame_ids_sha256"] = hashlib.sha256(
                pose_ids_path.read_bytes()
            ).hexdigest()
            path = Path(tmp) / "metrics.json"
            write_attested_metrics(path, payload)
            validate_metrics_file(path, "smoke", expected_frames=2)
            eval_pose = root / "eval_pose.txt"
            eval_pose.write_text("eval-v1", encoding="utf-8")
            payload["summary"]["eval_pose_file"] = str(eval_pose.resolve())
            payload["summary"]["eval_pose_file_sha256"] = hashlib.sha256(
                eval_pose.read_bytes()
            ).hexdigest()
            write_attested_metrics(path, payload)
            validate_metrics_file(path, "smoke", eval_pose=eval_pose, expected_frames=2)
            eval_pose.write_text("eval-v2", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Evaluator-observed pose hash"):
                validate_metrics_file(path, "smoke", eval_pose=eval_pose, expected_frames=2)
            eval_pose.write_text("eval-v1", encoding="utf-8")
            Path(binding["render_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_metrics_file(path, "smoke", expected_frames=2)
            Path(binding["render_path"]).write_bytes(b"render")
            payload["summary"]["ssim"] = None
            write_attested_metrics(path, payload)
            with self.assertRaisesRegex(ValueError, "ssim"):
                validate_metrics_file(path, "smoke", expected_frames=2)

            payload["summary"]["ssim"] = 0.8
            payload["summary"]["psnr"] = 20.1
            write_attested_metrics(path, payload)
            with self.assertRaisesRegex(ValueError, "Summary/per-frame mismatch for psnr"):
                validate_metrics_file(path, "smoke", expected_frames=2)

            payload["summary"]["psnr"] = 20.0
            payload["per_frame_reconstruction"][1]["lpips"] = float("nan")
            payload.pop("attestation", None)
            path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-standard JSON"):
                validate_metrics_file(path, "smoke", expected_frames=2)

            payload["per_frame_reconstruction"][1]["lpips"] = 0.2
            payload["per_frame_pose"][1]["absolute_translation_error_m"] = 0.02
            write_attested_metrics(path, payload)
            with self.assertRaisesRegex(ValueError, "Summary/per-frame mismatch for ate_m"):
                validate_metrics_file(path, "smoke", expected_frames=2)

    def test_frame_binding_rejects_shifted_pose_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "014341l.png"
            mask = root / "014341l-mask-placeholder"
            runtime_gt = root / "00000.png"
            render_dir = root / "render"
            render_dir.mkdir()
            render = render_dir / "00000.png"
            content = b"same-image"
            gt.write_bytes(content)
            runtime_gt.write_bytes(content)
            render.write_bytes(b"render")
            mask.write_bytes(b"mask")

            with self.assertRaisesRegex(ValueError, "GT/mask"):
                bind_reconstruction_frames([gt], [runtime_gt], [render], [mask], ["014341"])

            mask = root / "014341l.png.mask-parent" / "014341l.png"
            mask.parent.mkdir()
            mask.write_bytes(b"mask")
            with self.assertRaisesRegex(ValueError, "Pose/image"):
                bind_reconstruction_frames([gt], [runtime_gt], [render], [mask], ["014347"])


class LineageTests(unittest.TestCase):
    def test_runtime_and_input_lineage_are_concrete(self):
        runtime = runtime_lineage()
        self.assertIsInstance(runtime, dict)
        self.assertEqual(len(runtime["source_tree_sha256"]), 64)
        self.assertIn("raft_large_C_T_SKHT_V2-ff5fadd5.pth", " ".join(runtime["model_weight_sha256"]))
        self.assertIsInstance(runtime["environment"]["packages"]["torch"], str)
        self.assertIsNotNone(runtime["environment"]["torch_build"]["torch_cuda_version"])
        self.assertEqual(len(runtime["native_extensions"]), 1)
        native = runtime["native_extensions"][0]
        self.assertEqual(native["backend"], "diff_gaussian_rasterization")
        self.assertEqual(len(native["binary_sha256"]), 64)
        self.assertEqual(native["binary_path"], str(native_extension_path()))
        self.assertGreater(native["native_source_file_count"], 0)

        inputs = input_data_lineage("P1_1", argparse.Namespace(start=0, stop=3, step=1))
        self.assertEqual(inputs["num_frames"], 3)
        self.assertEqual(inputs["artifact_count"], 12)
        self.assertEqual(inputs["frame_ids"], ["014341", "014347", "014353"])
        self.assertEqual(len(inputs["combined_sha256"]), 64)

        condition = condition_dataset_lineage("P1_1")
        self.assertEqual(condition["num_pose_rows"], 300)
        self.assertEqual(condition["heavy_translation_perturbation_multiplier"], 10.0)
        self.assertLess(condition["max_translation_relation_abs_error"], 1e-10)

    def test_source_snapshot_archives_the_exact_runtime_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "source_snapshot.zip"
            runtime = runtime_lineage()
            snapshot_hash = write_source_snapshot(snapshot)
            self.assertEqual(len(snapshot_hash), 64)
            validate_source_snapshot(snapshot, runtime)
            with zipfile.ZipFile(snapshot, "r") as archive:
                self.assertIn(
                    native_extension_path().relative_to(REPO_ROOT).as_posix(),
                    archive.namelist(),
                )
            snapshot.write_bytes(b"tampered")
            with self.assertRaises(Exception):
                validate_source_snapshot(snapshot, runtime)

    def test_pose_lineage_changes_when_either_pose_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_pose = root / "input.txt"
            eval_pose = root / "eval.txt"
            input_pose.write_text("input-v1", encoding="utf-8")
            eval_pose.write_text("eval-v1", encoding="utf-8")
            expected = pose_file_lineage(input_pose, eval_pose)
            input_pose.write_text("input-v2", encoding="utf-8")
            self.assertNotEqual(pose_file_lineage(input_pose, eval_pose), expected)
            input_pose.write_text("input-v1", encoding="utf-8")
            eval_pose.write_text("eval-v2", encoding="utf-8")
            self.assertNotEqual(pose_file_lineage(input_pose, eval_pose), expected)

    def test_launch_provenance_requires_identical_pre_and_post_condition_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = run_dir / "effective_config.yaml"
            config.write_text("data: {}\n", encoding="utf-8")
            input_pose = run_dir / "input.txt"
            eval_pose = run_dir / "eval.txt"
            input_pose.write_text("input", encoding="utf-8")
            eval_pose.write_text("eval", encoding="utf-8")
            lineage = pose_file_lineage(input_pose, eval_pose)
            condition = {
                "status": "verified",
                "declared_mode": "light_noise",
                "sequence": "P1_1",
            }
            provenance = {
                "status": "inference_completed",
                "returncode": 0,
                "sequence": "P1_1",
                "mode": "light_noise",
                "run_kind": "smoke",
                "pose_condition_contract": pose_condition_contract("light_noise"),
                "input_pose_file": lineage["input_pose_file"],
                "input_pose_file_sha256": lineage["input_pose_sha256"],
                "eval_pose_file": lineage["evaluation_pose_file"],
                "eval_pose_file_sha256": lineage["evaluation_pose_sha256"],
                "dedicated_eval_reference_argument_passed_to_inference": False,
                "evaluation_pose_bytes_used_as_oracle_input": False,
                "evaluation_reference_provenance": "attached_post_inference",
                "condition_input_verification": condition,
                "post_inference_condition_input_verification": condition,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            provenance_path = run_dir / "launch_provenance.json"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            validate_launch_provenance(run_dir, "P1_1", "light_noise", "smoke", lineage)
            provenance["post_inference_condition_input_verification"] = {
                **condition,
                "status": "changed",
            }
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                validate_launch_provenance(
                    run_dir,
                    "P1_1",
                    "light_noise",
                    "smoke",
                    lineage,
                )


class ConditionVerificationTests(unittest.TestCase):
    def _args(self, root, mode, pose_file):
        input_folder = root / "steremis_tracking" / "P1_1"
        input_folder.mkdir(parents=True, exist_ok=True)
        eval_pose = input_folder / "groundtruth.txt"
        eval_pose.write_text("evaluation", encoding="utf-8")
        return argparse.Namespace(
            seq="P1_1",
            mode=mode,
            input_folder=str(input_folder),
            eval_pose_file=str(eval_pose),
            pose_file=None if pose_file is None else str(pose_file),
        )

    def test_clean_and_no_pose_roles_are_content_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, "clean_pose", None)
            args.pose_file = args.eval_pose_file
            verified = verify_condition_inputs(args)
            self.assertEqual(verified["condition_role"], "oracle_pose_mapping_control")
            self.assertTrue(verified["input_pose_bytes_equal_evaluation_pose"])

            args.mode = "no_pose"
            with self.assertRaisesRegex(ValueError, "forbids"):
                verify_condition_inputs(args)
            args.pose_file = None
            verified = verify_condition_inputs(args)
            self.assertEqual(
                verified["condition_role"], "identity_initialization_without_pose_file"
            )

    def test_light_and_heavy_roots_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            light = root / "stereomis_noisy_light" / "P1_1" / "groundtruth_noisy.txt"
            heavy = (
                root
                / "stereomis_noisy_light_transx10"
                / "P1_1"
                / "groundtruth_noisy.txt"
            )
            light.parent.mkdir(parents=True)
            heavy.parent.mkdir(parents=True)
            light.write_text("light", encoding="utf-8")
            heavy.write_text("heavy", encoding="utf-8")

            args = self._args(root, "light_noise", light)
            self.assertEqual(
                verify_condition_inputs(args)["canonical_input_pose_root"],
                "stereomis_noisy_light",
            )
            args.pose_file = str(heavy)
            with self.assertRaisesRegex(ValueError, "light_noise"):
                verify_condition_inputs(args)

            args.mode = "heavy_noise"
            self.assertEqual(
                verify_condition_inputs(args)["canonical_input_pose_root"],
                "stereomis_noisy_light_transx10",
            )
            args.pose_file = str(light)
            with self.assertRaisesRegex(ValueError, "heavy_noise"):
                verify_condition_inputs(args)

    def test_nonclean_inference_view_excludes_evaluation_pose_capability(self):
        verification = {
            "status": "verified",
            "declared_mode": "light_noise",
            "sequence": "P1_1",
            "evaluation_pose_file": "/dataset/P1_1/groundtruth.txt",
            "evaluation_pose_sha256": "a" * 64,
            "evaluation_pose_is_canonical_sequence_groundtruth": True,
            "input_pose_bytes_equal_evaluation_pose": False,
            "input_pose_file": "/noise/P1_1/groundtruth_noisy.txt",
            "input_pose_sha256": "b" * 64,
        }
        inference_view = inference_condition_view(verification, "light_noise")
        self.assertNotIn("evaluation_pose_file", inference_view)
        self.assertNotIn("evaluation_pose_sha256", inference_view)
        self.assertNotIn("input_pose_bytes_equal_evaluation_pose", inference_view)
        self.assertEqual(inference_view["input_pose_sha256"], "b" * 64)
        self.assertEqual(
            inference_condition_view(verification, "clean_pose"),
            verification,
        )

        launch = {
            "eval_pose_file": verification["evaluation_pose_file"],
            "eval_pose_file_sha256": verification["evaluation_pose_sha256"],
            "condition_input_verification": verification,
        }
        visible = inference_visible_launch_provenance(launch, "no_pose")
        self.assertNotIn("eval_pose_file", visible)
        self.assertNotIn("eval_pose_file_sha256", visible)
        self.assertEqual(
            visible["evaluation_reference_provenance"],
            "deferred_until_post_inference",
        )

        lineage = {
            "input_pose_file": None,
            "input_pose_sha256": None,
            "evaluation_pose_file": verification["evaluation_pose_file"],
            "evaluation_pose_sha256": verification["evaluation_pose_sha256"],
        }
        inference_lineage = inference_visible_pose_lineage(lineage, "no_pose")
        self.assertNotIn("evaluation_pose_file", inference_lineage)
        self.assertNotIn("evaluation_pose_sha256", inference_lineage)

    def test_pose_condition_claim_eligibility_is_explicit(self):
        clean = pose_condition_contract("clean_pose")
        no_pose = pose_condition_contract("no_pose")
        light = pose_condition_contract("light_noise")
        heavy = pose_condition_contract("heavy_noise")
        self.assertFalse(clean["pose_optimization_enabled"])
        self.assertFalse(clean["pose_gt_free_claim_eligible"])
        self.assertTrue(no_pose["pose_gt_free_claim_eligible"])
        self.assertFalse(light["pose_gt_free_claim_eligible"])
        self.assertFalse(heavy["pose_gt_free_claim_eligible"])
        self.assertTrue(light["pose_prior_derived_from_evaluation_gt"])
        self.assertTrue(heavy["pose_prior_derived_from_evaluation_gt"])


class RunContractAndSealingTests(unittest.TestCase):
    def _args(self, run_kind, **overrides):
        values = {
            "run_kind": run_kind,
            "start": None,
            "stop": None,
            "step": None,
            "iters": None,
            "iters_first": None,
            "pose_only_eval": False,
            "debug": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_full_evaluation_contract_rejects_short_or_low_iteration_runs(self):
        contract = validate_run_contract(self._args("full_evaluation"))
        self.assertEqual(contract["expected_num_frames"], 200)
        self.assertEqual(contract["completed_status"], "completed")
        with self.assertRaisesRegex(ValueError, "exact frame slice"):
            validate_run_contract(self._args("full_evaluation", stop=3))
        with self.assertRaisesRegex(ValueError, "iteration overrides"):
            validate_run_contract(self._args("full_evaluation", iters=2))

    def test_smoke_and_pose_diagnostic_have_distinct_completion_contracts(self):
        smoke = validate_run_contract(
            self._args("smoke", start=0, stop=3, step=1, iters=2, iters_first=2)
        )
        diagnostic_args = self._args("pose_only_diagnostic", stop=3)
        diagnostic = validate_run_contract(diagnostic_args)
        self.assertEqual(smoke["completed_status"], "completed_smoke")
        self.assertFalse(smoke["pose_only"])
        self.assertEqual(
            diagnostic["completed_status"],
            "completed_pose_diagnostic",
        )
        self.assertTrue(diagnostic["pose_only"])
        self.assertTrue(diagnostic_args.pose_only_eval)

    def test_run_directory_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with exclusive_run_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with exclusive_run_lock(root):
                        pass

    def test_recursive_seal_binds_unlisted_artifacts_and_excludes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
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
                "raw_depth/gt/00000.npy",
                "run_status.json",
                ".phase0.lock",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("ascii"))
            hashes = output_artifact_hashes(root, "pose_only_diagnostic", 2)
            self.assertIn("raw_depth/gt/00000.npy", hashes)
            self.assertNotIn("run_status.json", hashes)
            self.assertNotIn(".phase0.lock", hashes)

    def test_nonfinite_first_frame_gate_cannot_be_sealed(self):
        valid_gate = {
            "ff_pose_gate_evaluated": True,
            "ff_pose_gate_observations_finite": True,
            "ff_pose_gate_init_psnr": 20.0,
            "ff_pose_gate_init_ssim": 0.7,
            "ff_pose_gate_final_psnr": 19.0,
            "ff_pose_gate_final_ssim": 0.6,
            "ff_pose_gate_psnr_drop": 1.0,
            "ff_pose_gate_ssim_drop": 0.1,
            "ff_pose_gate_triggered": True,
            "ff_pose_gate_fallback_applied": True,
            "ff_pose_gate_fallback_mode": "no_prior",
            "ff_pose_gate_reasons": "psnr_below_min",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tool_motion_score.json"
            path.write_text(json.dumps([valid_gate]), encoding="utf-8")
            result = validate_first_frame_gate_outcome(root, "light_noise")
            self.assertEqual(result["status"], "verified")
            invalid_gate = dict(valid_gate)
            invalid_gate["ff_pose_gate_observations_finite"] = False
            invalid_gate["ff_pose_gate_final_psnr"] = None
            path.write_text(json.dumps([invalid_gate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not all finite"):
                validate_first_frame_gate_outcome(root, "light_noise")

    def test_completed_candidate_missing_provenance_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            run_dir = output_root / "runs" / "P1_1" / "clean_pose"
            run_dir.mkdir(parents=True)
            (run_dir / "run_status.json").write_text(
                json.dumps({"status": "completed_smoke"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_kind="smoke",
                gpu="2",
                start=0,
                stop=3,
                step=1,
                iters=2,
                iters_first=2,
                pose_only_eval=False,
                debug=False,
                dry_run=False,
                rerun=False,
            )
            with self.assertRaisesRegex(RuntimeError, "stale or unbound"):
                run_job(args, "P1_1", "clean_pose", output_root, {})
            self.assertFalse((run_dir / "source_snapshot.zip").exists())
            self.assertFalse((run_dir / "environment_lock.json").exists())

    def test_independent_verifier_detects_coordinated_per_row_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_path = root / "metrics.json"
            verification_path = root / "metrics_verification.json"
            binding = {"local_frame_idx": 0, "dataset_frame_id": "1"}
            primary = {
                "pose_frame_ids": ["1", "2"],
                "reconstruction_source_counts": {"gt": 1},
                "per_frame_pose": [
                    {
                        "frame_idx": 0,
                        "pose_frame_id": "1",
                        "absolute_translation_error_m": 9.0,
                        "absolute_rotation_error_deg": 9.0,
                        "relative_translation_error_m": None,
                        "relative_rotation_error_deg": None,
                    }
                ],
                "per_frame_reconstruction": [
                    {
                        "frame_idx": 0,
                        "frame_binding": binding,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "lpips": 0.2,
                    }
                ],
            }
            primary_path.write_text(json.dumps(primary), encoding="utf-8")
            verification = {
                **primary,
                "per_frame_pose": [
                    {
                        **primary["per_frame_pose"][0],
                        "absolute_translation_error_m": 1.0,
                    }
                ],
                "reference_comparison": {
                    "reference_metrics_path": str(primary_path.resolve()),
                    "reference_metrics_sha256": hashlib.sha256(
                        primary_path.read_bytes()
                    ).hexdigest(),
                },
            }
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absolute_translation_error_m"):
                compare_metric_derivations(primary_path, verification_path)


if __name__ == "__main__":
    unittest.main()
