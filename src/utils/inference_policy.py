import os
from copy import deepcopy


STRICT_GT_FREE = "strict_gt_free"
LEGACY_ORACLE_TRACKS = "legacy_oracle_tracks"
INFERENCE_POLICIES = (STRICT_GT_FREE, LEGACY_ORACLE_TRACKS)


POSE_CONDITION_CONTRACTS = {
    "clean_pose": {
        "condition_class": "oracle_pose_mapping_control",
        "pose_prior_source": "exact_dataset_ground_truth",
        "pose_prior_derived_from_evaluation_gt": True,
        "exact_evaluation_pose_available_to_inference": True,
        "pose_optimization_enabled": False,
        "pose_gt_free_claim_eligible": False,
        "pose_refinement_claim_eligible": False,
        "claim_ceiling": "oracle-pose mapping and reconstruction control only",
    },
    "light_noise": {
        "condition_class": "corrupted_oracle_pose_prior_control",
        "pose_prior_source": "dataset_ground_truth_plus_light_synthetic_noise",
        "pose_prior_derived_from_evaluation_gt": True,
        "exact_evaluation_pose_available_to_inference": False,
        "pose_optimization_enabled": True,
        "pose_gt_free_claim_eligible": False,
        "pose_refinement_claim_eligible": True,
        "claim_ceiling": "controlled pose-refinement under a GT-derived noisy prior",
    },
    "heavy_noise": {
        "condition_class": "corrupted_oracle_pose_prior_control",
        "pose_prior_source": "dataset_ground_truth_plus_heavy_synthetic_noise",
        "pose_prior_derived_from_evaluation_gt": True,
        "exact_evaluation_pose_available_to_inference": False,
        "pose_optimization_enabled": True,
        "pose_gt_free_claim_eligible": False,
        "pose_refinement_claim_eligible": True,
        "claim_ceiling": "controlled pose-refinement under a GT-derived noisy prior and 10x gate policy bundle",
    },
    "noisy": {
        "condition_class": "user_declared_noisy_pose_prior",
        "pose_prior_source": "user_declared_pose_file",
        "pose_prior_derived_from_evaluation_gt": None,
        "exact_evaluation_pose_available_to_inference": False,
        "pose_optimization_enabled": True,
        "pose_gt_free_claim_eligible": False,
        "pose_refinement_claim_eligible": True,
        "claim_ceiling": "ad-hoc pose-refinement diagnostic; prior provenance must be declared separately",
    },
    "noisy_auto_gate": {
        "condition_class": "user_declared_noisy_pose_prior",
        "pose_prior_source": "user_declared_pose_file",
        "pose_prior_derived_from_evaluation_gt": None,
        "exact_evaluation_pose_available_to_inference": False,
        "pose_optimization_enabled": True,
        "pose_gt_free_claim_eligible": False,
        "pose_refinement_claim_eligible": True,
        "claim_ceiling": "ad-hoc pose-refinement diagnostic; prior provenance must be declared separately",
    },
    "no_pose": {
        "condition_class": "pose_gt_free_causal_estimation",
        "pose_prior_source": "identity_then_causal_raft_visual_odometry",
        "pose_prior_derived_from_evaluation_gt": False,
        "exact_evaluation_pose_available_to_inference": False,
        "pose_optimization_enabled": True,
        "pose_gt_free_claim_eligible": True,
        "pose_refinement_claim_eligible": True,
        "claim_ceiling": "pose-GT-free causal trajectory estimation with post-inference real-GT evaluation",
    },
}


def pose_condition_contract(mode):
    try:
        return deepcopy(POSE_CONDITION_CONTRACTS[mode])
    except KeyError as exc:
        raise ValueError(f"No declared pose-condition contract for mode '{mode}'.") from exc


_STRICT_POSE_ZERO = (
    "w_track_2d",
    "stage2_track_scale",
    "stage3_track_scale",
    "pose_track_constraint_weight",
)

_STRICT_POSE_FALSE = (
    "pose_track_guard_enabled",
    "pose_track_adaptive_weight_enabled",
    "pose_track_constraint_enabled",
    "pose_freeze_if_track_loss_invalid",
    "pose_final_revert_on_track_worse",
    "pose_step_adaptive_enabled",
)


def inference_policy(cfg):
    runtime = cfg.get("track2map_runtime", {})
    policy = str(runtime.get("inference_policy", STRICT_GT_FREE)).lower()
    if policy not in INFERENCE_POLICIES:
        raise ValueError(
            f"Unsupported inference policy '{policy}'. Expected one of {INFERENCE_POLICIES}."
        )
    return policy


def enforce_inference_policy(cfg, copy_config=False):
    """Apply the load-bearing GT-access policy to an effective config."""
    if copy_config:
        cfg = deepcopy(cfg)

    runtime = cfg.setdefault("track2map_runtime", {})
    policy = inference_policy(cfg)
    runtime["inference_policy"] = policy
    runtime["inference_policy_scope"] = (
        "forbid_track_annotations_and_dedicated_evaluation_reference_reads; "
        "declared pose-prior access is governed by pose_condition_contract"
    )
    runtime["ground_truth_tracks_allowed"] = policy == LEGACY_ORACLE_TRACKS

    if policy != STRICT_GT_FREE:
        return cfg

    training = cfg.setdefault("training", {})
    pose_cfg = training.setdefault("pose_optimization", {})

    external_init = pose_cfg.get("pose_no_prior_external_init_file")
    if external_init not in (None, ""):
        raise ValueError(
            "strict_gt_free forbids pose_no_prior_external_init_file; "
            "the no-pose condition must initialize from the declared causal policy."
        )
    pose_cfg["pose_no_prior_external_init_file"] = None

    query_source = str(training.get("cotracker_flow_init_query_source", "anchor")).lower()
    if query_source == "gt":
        raise ValueError(
            "strict_gt_free forbids cotracker_flow_init_query_source=gt; use anchor queries."
        )
    if query_source != "anchor":
        raise ValueError(
            "strict_gt_free supports only cotracker_flow_init_query_source=anchor; "
            f"got '{query_source}'."
        )

    training["cotracker_flow_init_query_source"] = "anchor"
    training["pt_cotracker_gs_refine_enabled"] = False
    training["track_annotation_enabled"] = False

    pose_cfg["pose_no_prior_vo_corr_source"] = "raft"
    flow_source = str(training.get("optical_flow_init_source", "raft")).lower()
    if flow_source in ("cotracker3", "hybrid", "hybrid_foundation"):
        runtime_mode = str(training.get("pt_cotracker_runtime_mode", "precompute")).lower()
        if runtime_mode != "online_prefix":
            raise ValueError(
                "strict_gt_free requires pt_cotracker_runtime_mode=online_prefix when "
                f"optical_flow_init_source={flow_source}."
            )

    for key in _STRICT_POSE_ZERO:
        pose_cfg[key] = 0.0
    for key in _STRICT_POSE_FALSE:
        pose_cfg[key] = False

    runtime.pop("annotation_access", None)
    runtime["track_annotation_access"] = "forbidden"
    runtime["dataset_tool_mask_access"] = "allowed_and_recorded"
    runtime["semantic_prediction_access"] = "allowed_and_recorded"
    runtime["effective_pose_track_controls"] = "disabled"
    return cfg


def resolve_track_annotation_path(input_folder, cfg, path_exists=None):
    """Return a legacy annotation path without probing it in strict mode."""
    if inference_policy(cfg) == STRICT_GT_FREE:
        return None, False

    if not bool(cfg.get("training", {}).get("track_annotation_enabled", True)):
        return None, False

    exists = os.path.isfile if path_exists is None else path_exists
    track_file = os.path.join(input_folder, "track_pts.pckl")
    return (track_file if exists(track_file) else None), True


def poses_to_metric(poses, internal_scale):
    metric = poses.copy()
    scale = float(internal_scale)
    if scale <= 0.0:
        raise ValueError(f"internal_scale must be positive, got {scale}")
    metric[..., :3, 3] /= scale
    return metric
