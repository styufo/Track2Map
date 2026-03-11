import numpy as np
import os
import torch
import wandb
import warnings
import csv
import json
import cv2
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
from collections import deque
from torch.utils.data import DataLoader
from src.utils.camera import Camera
from src.utils.FrameVisualizer import FrameVisualizer
from src.utils.flow_utils import (
    get_scene_flow,
    get_depth_from_raft,
    get_depth_from_foundation,
    get_scene_flow_from_foundation,
)
from src.utils.PointTracker import PointTracker, mte, surv_2d, delta_2d
from src.utils.datasets import StereoMIS
from src.utils.loss_utils import l1_loss, ssim, lpips_loss
from src.utils.renderer import render, set_rasterizer_backend
from src.utils.pose_utils import apply_se3_delta
from src.scene.gaussian_model import GaussianModel


class SceneOptimizer():
    def __init__(self, cfg, args):
        self.total_iters = 0
        self.cfg = cfg
        self.args = args
        self.visualize = args.visualize
        self.save_widefield_ply_every = max(0, int(args.save_widefield_ply_every))
        self.widefield_ply_alpha_thr = float(args.widefield_ply_alpha_thr)
        self.widefield_ply_min_depth = float(args.widefield_ply_min_depth)
        self.widefield_ply_max_depth = float(args.widefield_ply_max_depth)
        self.scale = cfg['scale']
        self.device = cfg['device']
        self.output = cfg['data']['output']
        os.makedirs(self.output, exist_ok=True)

        self.frame_reader = StereoMIS(cfg, args, scale=self.scale)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=0 if args.debug else 4)
        self.net = GaussianModel(cfg=cfg['model'])
        self.camera = Camera(cfg['cam'])
        self.visualizer = FrameVisualizer(self.output, cfg, self.net)

        self.log_freq = args.log_freq
        self.log = args.log is not None
        self.run_id = wandb.util.generate_id()
        log_cfg = cfg.copy()
        log_cfg.update(vars(args))
        if self.log:
            wandb.init(id=self.run_id, name=args.log, config=log_cfg, project='gtracker', group=args.log_group)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.dbg = args.debug
        self.pt_tracker = None
        track_file = os.path.join(self.frame_reader.input_folder, 'track_pts.pckl')
        if os.path.isfile(track_file):
            self.pt_tracker = PointTracker(cfg, self.net, track_file)
        self.pt_tracker_backend = str(cfg['training'].get('pt_tracker_backend', 'gaussian3d')).lower()
        self.pt_cotracker_repo = str(cfg['training'].get('pt_cotracker_repo', '/path/to/co-tracker'))
        self.pt_cotracker_model = str(cfg['training'].get('pt_cotracker_model', 'cotracker3_offline'))
        self.pt_cotracker_tracks = None
        self.pt_cotracker_vis = None
        self.pt_cotracker_tracks_deform = None
        self.pt_cotracker_vis_deform = None
        self.pt_cotracker_gs_refine_enabled = bool(
            cfg['training'].get('pt_cotracker_gs_refine_enabled', False)
        )
        self.pt_cotracker_gs_refine_interval = max(
            1, int(cfg['training'].get('pt_cotracker_gs_refine_interval', 8))
        )
        self.pt_cotracker_gs_refine_alpha = float(
            np.clip(cfg['training'].get('pt_cotracker_gs_refine_alpha', 0.35), 0.0, 1.0)
        )
        self.pt_cotracker_gs_refine_max_res_px = float(
            cfg['training'].get('pt_cotracker_gs_refine_max_res_px', 10.0)
        )
        self.pt_cotracker_gs_refine_start_frame = max(
            0, int(cfg['training'].get('pt_cotracker_gs_refine_start_frame', 8))
        )
        self.pt_cotracker_gs_refine_clamp_to_image = bool(
            cfg['training'].get('pt_cotracker_gs_refine_clamp_to_image', True)
        )
        self.pt_cotracker_gs_refine_log = bool(
            cfg['training'].get('pt_cotracker_gs_refine_log', True)
        )
        self.cotracker_deform_query_pose0 = None
        self.pt_curr_2d = None
        self.pt_prev_track_frame = None
        self.last_frame = None
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.raft = self.raft.eval()
        self.baseline = cfg['cam']['stereo_baseline'] / 1000.0 * self.scale
        self.depth_input_source = str(cfg['data'].get('depth_input_source', 'raft_stereo')).lower()
        if self.depth_input_source not in ('raft_stereo', 'mono_precomputed', 'foundation_stereo'):
            warnings.warn(
                f"Unknown depth_input_source={self.depth_input_source}, fallback to raft_stereo."
            )
            self.depth_input_source = 'raft_stereo'
        self.precomputed_depth_dir = cfg['data'].get('precomputed_depth_dir', None)
        if self.precomputed_depth_dir is not None:
            self.precomputed_depth_dir = str(self.precomputed_depth_dir)
        self.precomputed_depth_ext = str(cfg['data'].get('precomputed_depth_ext', '.npy')).lower()
        if not self.precomputed_depth_ext.startswith('.'):
            self.precomputed_depth_ext = f'.{self.precomputed_depth_ext}'
        self.precomputed_depth_scale = float(cfg['data'].get('precomputed_depth_scale', 1.0))
        self.precomputed_depth_bias = float(cfg['data'].get('precomputed_depth_bias', 0.0))
        self.precomputed_depth_invert = bool(cfg['data'].get('precomputed_depth_invert', False))
        self.precomputed_depth_eps = float(cfg['data'].get('precomputed_depth_eps', 1e-6))
        self.precomputed_depth_clip_min = float(cfg['data'].get('precomputed_depth_clip_min', 1e-4))
        self.precomputed_depth_clip_max = float(cfg['data'].get('precomputed_depth_clip_max', 0.0))
        self.foundation_root = cfg['data'].get('foundation_root', None)
        self.foundation_ckpt = cfg['data'].get('foundation_ckpt', None)
        self.foundation_cfg = cfg['data'].get('foundation_cfg', None)
        self.foundation_intrinsic_file = cfg['data'].get('foundation_intrinsic_file', None)
        self.foundation_valid_iters = int(cfg['data'].get('foundation_valid_iters', 32))
        self.foundation_depth_formula = str(
            cfg['data'].get('foundation_depth_formula', 'raft_compatible')
        ).lower()
        self.foundation_use_sequence_camera_params = bool(
            cfg['data'].get('foundation_use_sequence_camera_params', True)
        )
        self.foundation_seq_fx = float(cfg['cam']['fx'])
        self.foundation_seq_fy = float(cfg['cam']['fy'])
        self.foundation_seq_cx = float(cfg['cam']['cx'])
        self.foundation_seq_cy = float(cfg['cam']['cy'])
        self.foundation_seq_baseline_m = float(cfg['cam']['stereo_baseline']) / 1000.0
        self.foundation_model = None
        self.foundation_args = None
        if self.foundation_root is not None:
            self.foundation_root = str(self.foundation_root)
        if self.foundation_ckpt is not None:
            self.foundation_ckpt = str(self.foundation_ckpt)
        if self.foundation_cfg is not None:
            self.foundation_cfg = str(self.foundation_cfg)
        if self.foundation_intrinsic_file is not None:
            self.foundation_intrinsic_file = str(self.foundation_intrinsic_file)
        need_foundation_model = (
            self.depth_input_source == 'foundation_stereo'
            or str(cfg['training'].get('optical_flow_init_source', 'raft')).lower() in ('foundation', 'hybrid_foundation')
        )
        if self.depth_input_source == 'mono_precomputed':
            if self.precomputed_depth_dir is None or (not os.path.isdir(self.precomputed_depth_dir)):
                raise FileNotFoundError(
                    f"mono_precomputed depth source requires valid precomputed_depth_dir, got: {self.precomputed_depth_dir}"
                )
        if need_foundation_model:
            from src.utils.foundation_model import get_foundation_stereo_model

            self.foundation_model, self.foundation_args = get_foundation_stereo_model(
                device=self.device,
                foundation_root=self.foundation_root,
                ckpt_path=self.foundation_ckpt,
                cfg_path=self.foundation_cfg,
                intrinsic_file=self.foundation_intrinsic_file,
                valid_iters=self.foundation_valid_iters,
            )
            if self.foundation_use_sequence_camera_params and self.foundation_args is not None:
                self.foundation_args.K = [
                    [self.foundation_seq_fx, 0.0, self.foundation_seq_cx],
                    [0.0, self.foundation_seq_fy, self.foundation_seq_cy],
                    [0.0, 0.0, 1.0],
                ]
                self.foundation_args.baseline = float(self.foundation_seq_baseline_m)
        self.rasterizer_backend = str(cfg['training'].get('rasterizer_backend', 'default')).lower()
        self.render_visibility_mask_enabled = bool(cfg['training'].get('render_visibility_mask_enabled', False))
        self.render_visibility_opacity_threshold = float(
            cfg['training'].get('render_visibility_opacity_threshold', 0.0)
        )
        self.render_visibility_keep_ratio = float(cfg['training'].get('render_visibility_keep_ratio', 1.0))
        self.render_visibility_keep_ratio = float(np.clip(self.render_visibility_keep_ratio, 0.0, 1.0))
        self.optical_flow_init_guard_enabled = bool(cfg['training'].get('optical_flow_init_guard_enabled', True))
        self.optical_flow_init_min_valid_points = int(cfg['training'].get('optical_flow_init_min_valid_points', 2048))
        self.optical_flow_init_min_extent = float(cfg['training'].get('optical_flow_init_min_extent', 1e-4))
        self.optical_flow_init_unique_ratio_min = float(
            cfg['training'].get('optical_flow_init_unique_ratio_min', 0.01)
        )
        self.optical_flow_init_unique_sample = int(cfg['training'].get('optical_flow_init_unique_sample', 20000))
        self.optical_flow_init_unique_round = int(cfg['training'].get('optical_flow_init_unique_round', 6))
        self.optical_flow_init_source = str(cfg['training'].get('optical_flow_init_source', 'raft')).lower()
        if self.optical_flow_init_source not in ('raft', 'foundation', 'cotracker3', 'hybrid', 'hybrid_foundation'):
            warnings.warn(
                f"Unknown optical_flow_init_source={self.optical_flow_init_source}, fallback to raft."
            )
            self.optical_flow_init_source = 'raft'
        self.cotracker_flow_init_min_valid_points = int(
            cfg['training'].get('cotracker_flow_init_min_valid_points', 64)
        )
        self.cotracker_flow_init_max_points = int(
            cfg['training'].get('cotracker_flow_init_max_points', 2048)
        )
        self.cotracker_flow_init_use_tool_mask = bool(
            cfg['training'].get('cotracker_flow_init_use_tool_mask', True)
        )
        self.cotracker_flow_init_query_source = str(
            cfg['training'].get('cotracker_flow_init_query_source', 'gt')
        ).lower()
        if self.cotracker_flow_init_query_source not in ('gt', 'anchor'):
            warnings.warn(
                f"Unknown cotracker_flow_init_query_source={self.cotracker_flow_init_query_source}, fallback to gt."
            )
            self.cotracker_flow_init_query_source = 'gt'
        self.cotracker_flow_init_query_max_points = int(
            cfg['training'].get('cotracker_flow_init_query_max_points', 512)
        )
        self.cotracker_flow_init_query_min_depth = float(
            cfg['training'].get('cotracker_flow_init_query_min_depth', 1e-4)
        )
        self.cotracker_flow_init_query_grid_cell = int(
            cfg['training'].get('cotracker_flow_init_query_grid_cell', 0)
        )
        set_rasterizer_backend(self.rasterizer_backend)

        pose_cfg = cfg['training'].get('pose_optimization', {})
        self.pose_opt_enabled = bool(pose_cfg.get('enabled', False))
        self.pose_optimize_first_frame = bool(pose_cfg.get('optimize_first_frame', False))
        self.pose_warmup_iters = int(pose_cfg.get('warmup_iters', 0))
        self.pose_opt_iters = int(pose_cfg.get('iters', 0))
        self.pose_lr = float(pose_cfg.get('lr', 5e-4))
        self.pose_w_trans = float(pose_cfg.get('w_trans_reg', 1e-3))
        self.pose_w_rot = float(pose_cfg.get('w_rot_reg', 1e-3))
        self.pose_w_track_2d = float(pose_cfg.get('w_track_2d', 0.0))
        self.pose_w_prior = float(pose_cfg.get('w_pose_prior', 0.0))
        self.pose_w_smooth = float(pose_cfg.get('w_pose_smooth', 0.0))
        self.pose_w_ssim = float(pose_cfg.get('w_ssim', 0.0))
        self.pose_w_lpips = float(pose_cfg.get('w_lpips', 0.0))
        self.pose_grad_clip = float(pose_cfg.get('grad_clip', 1.0))
        self.pose_only_iters = int(pose_cfg.get('pose_only_iters', 0))
        self.pose_stage1_map_only_iters = int(pose_cfg.get('stage1_map_only_iters', 0))
        self.pose_stage2_iters = int(pose_cfg.get('stage2_iters', 0))
        self.pose_stage2_lr_scale = float(pose_cfg.get('stage2_lr_scale', 1.0))
        self.pose_stage3_lr_scale = float(pose_cfg.get('stage3_lr_scale', 1.0))
        self.pose_stage2_track_scale = float(pose_cfg.get('stage2_track_scale', 1.0))
        self.pose_stage3_track_scale = float(pose_cfg.get('stage3_track_scale', 1.0))
        self.pose_stage2_prior_scale = float(pose_cfg.get('stage2_prior_scale', 1.0))
        self.pose_stage3_prior_scale = float(pose_cfg.get('stage3_prior_scale', 1.0))
        self.pose_stage2_smooth_scale = float(pose_cfg.get('stage2_smooth_scale', 1.0))
        self.pose_stage3_smooth_scale = float(pose_cfg.get('stage3_smooth_scale', 1.0))
        self.pose_stage2_recon_scale = float(pose_cfg.get('stage2_recon_scale', 1.0))
        self.pose_stage3_recon_scale = float(pose_cfg.get('stage3_recon_scale', 1.0))
        self.pose_save = bool(pose_cfg.get('save_poses', True))
        self.tool_motion_gate_enabled = bool(pose_cfg.get('tool_motion_gate_enabled', False))
        self.tool_motion_mode = str(pose_cfg.get('tool_motion_mode', 'hysteresis'))
        self.tool_motion_on = float(pose_cfg.get('tool_motion_on', 1.0))
        self.tool_motion_off = float(pose_cfg.get('tool_motion_off', 0.5))
        self.tool_motion_hysteresis = int(pose_cfg.get('tool_motion_hysteresis', 3))
        self.tool_motion_static_on = float(pose_cfg.get('tool_motion_static_on', -0.15))
        self.tool_motion_moving_on = float(pose_cfg.get('tool_motion_moving_on', 1.0))
        self.tool_motion_static_required = int(pose_cfg.get('tool_motion_static_required', 8))
        self.tool_motion_moving_required = int(pose_cfg.get('tool_motion_moving_required', 2))
        self.tool_motion_iou_weight = float(pose_cfg.get('tool_motion_iou_weight', 0.5))
        self.tool_motion_centroid_weight = float(pose_cfg.get('tool_motion_centroid_weight', 0.05))
        self.tool_motion_min_pixels = int(pose_cfg.get('tool_motion_min_pixels', 256))
        self.tool_motion_soft_gate_enabled = bool(pose_cfg.get('tool_motion_soft_gate_enabled', False))
        self.tool_motion_soft_power = float(pose_cfg.get('tool_motion_soft_power', 1.5))
        self.tool_motion_soft_min_pose_weight = float(pose_cfg.get('tool_motion_soft_min_pose_weight', 0.0))
        self.tool_motion_soft_lock_power = float(pose_cfg.get('tool_motion_soft_lock_power', 1.2))
        self.tool_motion_soft_lock_max = float(pose_cfg.get('tool_motion_soft_lock_max', 1.0))
        self.tool_motion_lock_translation_only = bool(pose_cfg.get('tool_motion_lock_translation_only', False))
        self.tool_motion_use_semantic_tool_mask = bool(pose_cfg.get('tool_motion_use_semantic_tool_mask', False))
        self.tool_semantic_channel = int(pose_cfg.get('tool_semantic_channel', 1))
        self.tool_semantic_threshold = float(pose_cfg.get('tool_semantic_threshold', 0.5))
        self.tool_motion_use_bg_for_pose_loss = bool(pose_cfg.get('tool_motion_use_bg_for_pose_loss', False))
        self.tool_motion_bg_switch_ratio = float(pose_cfg.get('tool_motion_bg_switch_ratio', 0.5))
        self.tool_motion_bg_min_pixels = int(pose_cfg.get('tool_motion_bg_min_pixels', 1024))
        self.pose_recon_bg_blend = float(pose_cfg.get('pose_recon_bg_blend', 0.5))
        self.tool_motion_dynamic_reg_enabled = bool(pose_cfg.get('tool_motion_dynamic_reg_enabled', False))
        self.tool_motion_dynamic_prior_max = float(pose_cfg.get('tool_motion_dynamic_prior_max', 1.0))
        self.tool_motion_dynamic_smooth_max = float(pose_cfg.get('tool_motion_dynamic_smooth_max', 1.0))
        self.tool_motion_dynamic_track_min = float(pose_cfg.get('tool_motion_dynamic_track_min', 1.0))
        self.tool_motion_dynamic_recon_min = float(pose_cfg.get('tool_motion_dynamic_recon_min', 1.0))
        self.tool_motion_dynamic_lr_min = float(pose_cfg.get('tool_motion_dynamic_lr_min', 1.0))
        self.pose_soft_enable_threshold = float(pose_cfg.get('pose_soft_enable_threshold', 1e-4))
        self.pose_step_limit_enabled = bool(pose_cfg.get('pose_step_limit_enabled', False))
        self.pose_step_max_trans = float(pose_cfg.get('pose_step_max_trans', 0.005))
        self.pose_step_max_rot_deg = float(pose_cfg.get('pose_step_max_rot_deg', 0.5))
        self.pose_step_limit_wrt_input = bool(pose_cfg.get('pose_step_limit_wrt_input', False))
        self.pose_step_adaptive_enabled = bool(pose_cfg.get('pose_step_adaptive_enabled', False))
        self.pose_step_adaptive_ref_track = float(pose_cfg.get('pose_step_adaptive_ref_track', 0.02))
        self.pose_step_adaptive_max_scale = float(pose_cfg.get('pose_step_adaptive_max_scale', 3.0))
        self.pose_step_adaptive_motion_ratio_max = float(pose_cfg.get('pose_step_adaptive_motion_ratio_max', 0.35))
        self.pose_step_adaptive_min_vo_inliers = int(pose_cfg.get('pose_step_adaptive_min_vo_inliers', 1500))
        self.pose_track_guard_enabled = bool(pose_cfg.get('pose_track_guard_enabled', False))
        self.pose_track_guard_ratio = float(pose_cfg.get('pose_track_guard_ratio', 1.2))
        self.pose_track_guard_margin = float(pose_cfg.get('pose_track_guard_margin', 0.0))
        self.pose_track_guard_decay = float(pose_cfg.get('pose_track_guard_decay', 0.5))
        self.pose_track_guard_max_triggers = int(pose_cfg.get('pose_track_guard_max_triggers', 0))
        self.pose_track_guard_abs_max = float(pose_cfg.get('pose_track_guard_abs_max', 0.0))
        self.pose_track_guard_decay = min(max(self.pose_track_guard_decay, 0.0), 1.0)
        self.pose_track_adaptive_weight_enabled = bool(pose_cfg.get('pose_track_adaptive_weight_enabled', False))
        self.pose_track_adaptive_ref = float(pose_cfg.get('pose_track_adaptive_ref', 0.02))
        self.pose_track_adaptive_power = float(pose_cfg.get('pose_track_adaptive_power', 1.0))
        self.pose_track_adaptive_min_scale = float(pose_cfg.get('pose_track_adaptive_min_scale', 0.1))
        self.pose_track_adaptive_min_scale = min(max(self.pose_track_adaptive_min_scale, 0.0), 1.0)
        self.pose_track_constraint_enabled = bool(pose_cfg.get('pose_track_constraint_enabled', False))
        self.pose_track_constraint_ratio = float(pose_cfg.get('pose_track_constraint_ratio', 1.2))
        self.pose_track_constraint_margin = float(pose_cfg.get('pose_track_constraint_margin', 0.0))
        self.pose_track_constraint_abs_max = float(pose_cfg.get('pose_track_constraint_abs_max', 0.0))
        self.pose_track_constraint_weight = float(pose_cfg.get('pose_track_constraint_weight', 0.0))
        self.pose_deform_decouple_enabled = bool(pose_cfg.get('pose_deform_decouple_enabled', False))
        self.pose_deform_static_l2 = float(pose_cfg.get('pose_deform_static_l2', 0.0))
        self.pose_deform_static_power = float(pose_cfg.get('pose_deform_static_power', 1.0))
        self.pose_deform_rot_ratio = float(pose_cfg.get('pose_deform_rot_ratio', 0.3))
        self.pose_init_mode = str(pose_cfg.get('pose_init_mode', 'dataset')).lower()
        self.pose_no_prior_use_prev_optimized = bool(pose_cfg.get('pose_no_prior_use_prev_optimized', True))
        self.pose_no_prior_vo_enabled = bool(pose_cfg.get('pose_no_prior_vo_enabled', False))
        self.pose_no_prior_vo_min_points = int(pose_cfg.get('pose_no_prior_vo_min_points', 2000))
        self.pose_no_prior_vo_max_points = int(pose_cfg.get('pose_no_prior_vo_max_points', 20000))
        self.pose_no_prior_vo_min_depth = float(pose_cfg.get('pose_no_prior_vo_min_depth', 1e-4))
        self.pose_no_prior_vo_max_depth = float(pose_cfg.get('pose_no_prior_vo_max_depth', 3.0 * self.scale))
        self.pose_no_prior_vo_max_flow = float(pose_cfg.get('pose_no_prior_vo_max_flow', 80.0))
        self.pose_no_prior_vo_max_trans = float(pose_cfg.get('pose_no_prior_vo_max_trans', 0.02))
        self.pose_no_prior_vo_max_rot_deg = float(pose_cfg.get('pose_no_prior_vo_max_rot_deg', 5.0))
        self.pose_no_prior_vo_use_tool_mask = bool(pose_cfg.get('pose_no_prior_vo_use_tool_mask', True))
        self.pose_no_prior_vo_refine_iters = int(pose_cfg.get('pose_no_prior_vo_refine_iters', 2))
        self.pose_no_prior_vo_inlier_quantile = float(pose_cfg.get('pose_no_prior_vo_inlier_quantile', 0.6))
        self.pose_no_prior_vo_inlier_max_res = float(pose_cfg.get('pose_no_prior_vo_inlier_max_res', 0.02))
        default_min_inliers = max(64, int(self.pose_no_prior_vo_min_points * 0.25))
        self.pose_no_prior_vo_min_inliers = int(
            pose_cfg.get('pose_no_prior_vo_min_inliers', default_min_inliers)
        )
        self.pose_no_prior_vo_ransac_enabled = bool(pose_cfg.get('pose_no_prior_vo_ransac_enabled', False))
        self.pose_no_prior_vo_ransac_iters = int(pose_cfg.get('pose_no_prior_vo_ransac_iters', 128))
        self.pose_no_prior_vo_ransac_sample_size = int(pose_cfg.get('pose_no_prior_vo_ransac_sample_size', 8))
        self.pose_no_prior_vo_ransac_inlier_res = float(pose_cfg.get('pose_no_prior_vo_ransac_inlier_res', 0.01))
        self.pose_no_prior_vo_ransac_min_inliers = int(
            pose_cfg.get('pose_no_prior_vo_ransac_min_inliers', max(64, int(self.pose_no_prior_vo_min_inliers)))
        )
        self.pose_no_prior_vo_ransac_eval_points = int(
            pose_cfg.get('pose_no_prior_vo_ransac_eval_points', 12000)
        )
        self.pose_no_prior_vo_solver = str(pose_cfg.get('pose_no_prior_vo_solver', 'svd')).lower()
        self.pose_no_prior_vo_corr_source = str(pose_cfg.get('pose_no_prior_vo_corr_source', 'raft')).lower()
        if self.pose_no_prior_vo_corr_source not in ('raft', 'cotracker3', 'auto'):
            warnings.warn(
                f"Unknown pose_no_prior_vo_corr_source={self.pose_no_prior_vo_corr_source}, fallback to raft."
            )
            self.pose_no_prior_vo_corr_source = 'raft'
        self.pose_no_prior_vo_pnp_reproj_err = float(pose_cfg.get('pose_no_prior_vo_pnp_reproj_err', 2.0))
        self.pose_no_prior_vo_pnp_iterations = int(pose_cfg.get('pose_no_prior_vo_pnp_iterations', 200))
        self.pose_no_prior_vo_pnp_confidence = float(pose_cfg.get('pose_no_prior_vo_pnp_confidence', 0.999))
        self.pose_no_prior_vo_pnp_refine = bool(pose_cfg.get('pose_no_prior_vo_pnp_refine', True))
        self.pose_no_prior_vo_essential_ransac_px = float(pose_cfg.get('pose_no_prior_vo_essential_ransac_px', 1.5))
        self.pose_no_prior_vo_essential_confidence = float(
            pose_cfg.get('pose_no_prior_vo_essential_confidence', 0.999)
        )
        self.pose_no_prior_vo_essential_min_inliers = int(
            pose_cfg.get('pose_no_prior_vo_essential_min_inliers', max(32, int(self.pose_no_prior_vo_min_inliers)))
        )
        self.pose_no_prior_vo_essential_use_depth_scale = bool(
            pose_cfg.get('pose_no_prior_vo_essential_use_depth_scale', True)
        )
        self.pose_no_prior_vo_chain_source = str(
            pose_cfg.get('pose_no_prior_vo_chain_source', 'optimized')
        ).lower()
        if self.pose_no_prior_vo_chain_source not in ('optimized', 'input'):
            warnings.warn(
                f"Unknown pose_no_prior_vo_chain_source={self.pose_no_prior_vo_chain_source}, fallback to optimized."
            )
            self.pose_no_prior_vo_chain_source = 'optimized'
        self.pose_track_robust_delta = float(pose_cfg.get('pose_track_robust_delta', 0.0))
        self.pose_debug_grad_enabled = bool(pose_cfg.get('pose_debug_grad_enabled', False))
        self.pose_debug_grad_log_freq = max(1, int(pose_cfg.get('pose_debug_grad_log_freq', 1)))
        self.pose_debug_grad_records = []
        self.pose_no_prior_external_init_apply_scale = bool(
            pose_cfg.get('pose_no_prior_external_init_apply_scale', True)
        )
        (
            self.pose_no_prior_external_init_file,
            self.pose_no_prior_external_init_poses,
        ) = self._load_external_no_prior_init_poses(
            pose_cfg.get('pose_no_prior_external_init_file', None),
            apply_scale=self.pose_no_prior_external_init_apply_scale,
        )
        self.flow_dir_gate_pose_on_std = float(pose_cfg.get('flow_dir_gate_pose_on_std', 0.95))
        self.flow_dir_gate_pose_off_std = float(pose_cfg.get('flow_dir_gate_pose_off_std', 1.20))
        self.flow_dir_gate_min_mag = float(pose_cfg.get('flow_dir_gate_min_mag', 0.5))
        self.flow_dir_gate_min_pixels = int(pose_cfg.get('flow_dir_gate_min_pixels', 4096))
        self.flow_dir_gate_max_points = int(pose_cfg.get('flow_dir_gate_max_points', 50000))
        self.flow_dir_gate_use_magnitude_weight = bool(pose_cfg.get('flow_dir_gate_use_magnitude_weight', True))
        self.flow_dir_gate_use_tool_mask = bool(pose_cfg.get('flow_dir_gate_use_tool_mask', False))
        self.pose_no_prior_vo_static_skip_enabled = bool(
            pose_cfg.get('pose_no_prior_vo_static_skip_enabled', False)
        )
        self.pose_no_prior_vo_static_skip_std = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_std', 0.75)
        )
        self.pose_no_prior_vo_static_skip_moving_std = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_moving_std', 0.50)
        )
        self.pose_no_prior_vo_static_skip_hysteresis = max(
            1, int(pose_cfg.get('pose_no_prior_vo_static_skip_hysteresis', 2))
        )
        self.pose_no_prior_vo_static_skip_use_tool_mask = bool(
            pose_cfg.get('pose_no_prior_vo_static_skip_use_tool_mask', False)
        )
        self.pose_no_prior_vo_static_skip_adaptive_enabled = bool(
            pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_enabled', False)
        )
        self.pose_no_prior_vo_static_skip_adaptive_window = max(
            8, int(pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_window', 64))
        )
        self.pose_no_prior_vo_static_skip_adaptive_min_samples = max(
            4, int(pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_min_samples', 16))
        )
        self.pose_no_prior_vo_static_skip_adaptive_static_up = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_static_up', 0.20)
        )
        self.pose_no_prior_vo_static_skip_adaptive_static_down = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_static_down', 0.05)
        )
        self.pose_no_prior_vo_static_skip_adaptive_moving_up = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_moving_up', 0.20)
        )
        self.pose_no_prior_vo_static_skip_adaptive_moving_down = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_adaptive_moving_down', 0.15)
        )
        self.pose_no_prior_vo_static_skip_mag_guard_enabled = bool(
            pose_cfg.get('pose_no_prior_vo_static_skip_mag_guard_enabled', False)
        )
        self.pose_no_prior_vo_static_skip_mag_static_quantile = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_mag_static_quantile', 0.60)
        )
        self.pose_no_prior_vo_static_skip_mag_moving_quantile = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_mag_moving_quantile', 0.80)
        )
        self.pose_no_prior_vo_static_skip_mag_static_scale = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_mag_static_scale', 1.0)
        )
        self.pose_no_prior_vo_static_skip_mag_moving_scale = float(
            pose_cfg.get('pose_no_prior_vo_static_skip_mag_moving_scale', 1.0)
        )
        self.pose_no_prior_vo_static_skip_mag_min_samples = max(
            4, int(pose_cfg.get('pose_no_prior_vo_static_skip_mag_min_samples', 16))
        )
        self.pose_no_prior_vo_conf_reject_enabled = bool(
            pose_cfg.get('pose_no_prior_vo_conf_reject_enabled', False)
        )
        self.pose_no_prior_vo_conf_min_inlier_ratio = float(
            pose_cfg.get('pose_no_prior_vo_conf_min_inlier_ratio', 0.03)
        )
        self.pose_no_prior_vo_conf_min_inliers = max(
            0, int(pose_cfg.get('pose_no_prior_vo_conf_min_inliers', 0))
        )
        self.pose_no_prior_vo_conf_max_reproj_px = float(
            pose_cfg.get('pose_no_prior_vo_conf_max_reproj_px', 0.0)
        )
        self.pose_no_prior_vo_conf_abs_max_trans = float(
            pose_cfg.get('pose_no_prior_vo_conf_abs_max_trans', 0.06)
        )
        self.pose_no_prior_vo_conf_abs_max_rot_deg = float(
            pose_cfg.get('pose_no_prior_vo_conf_abs_max_rot_deg', 12.0)
        )
        self.pose_no_prior_vo_conf_jump_hist_window = max(
            8, int(pose_cfg.get('pose_no_prior_vo_conf_jump_hist_window', 64))
        )
        self.pose_no_prior_vo_conf_jump_min_hist = max(
            4, int(pose_cfg.get('pose_no_prior_vo_conf_jump_min_hist', 8))
        )
        self.pose_no_prior_vo_conf_jump_k_trans = float(
            pose_cfg.get('pose_no_prior_vo_conf_jump_k_trans', 4.0)
        )
        self.pose_no_prior_vo_conf_jump_k_rot = float(
            pose_cfg.get('pose_no_prior_vo_conf_jump_k_rot', 4.0)
        )
        self.pose_freeze_on_vo_static_skip = bool(
            pose_cfg.get('pose_freeze_on_vo_static_skip', False)
        )
        self.pose_freeze_if_track_loss_invalid = bool(
            pose_cfg.get('pose_freeze_if_track_loss_invalid', False)
        )
        self.pose_final_revert_on_track_worse = bool(
            pose_cfg.get('pose_final_revert_on_track_worse', False)
        )
        self.pose_final_revert_ratio = float(pose_cfg.get('pose_final_revert_ratio', 1.02))
        self.pose_final_revert_margin = float(pose_cfg.get('pose_final_revert_margin', 0.0))
        self.pose_final_revert_ratio = max(1.0, self.pose_final_revert_ratio)
        self.pose_final_revert_margin = max(0.0, self.pose_final_revert_margin)
        self.first_frame_pose_trust_gate_enabled = bool(
            pose_cfg.get('first_frame_pose_trust_gate_enabled', False)
        )
        self.first_frame_pose_trust_gate_min_psnr = float(
            pose_cfg.get('first_frame_pose_trust_gate_min_psnr', 0.0)
        )
        self.first_frame_pose_trust_gate_min_ssim = float(
            pose_cfg.get('first_frame_pose_trust_gate_min_ssim', 0.0)
        )
        self.first_frame_pose_trust_gate_max_psnr_drop = float(
            pose_cfg.get('first_frame_pose_trust_gate_max_psnr_drop', 0.0)
        )
        self.first_frame_pose_trust_gate_max_ssim_drop = float(
            pose_cfg.get('first_frame_pose_trust_gate_max_ssim_drop', 0.0)
        )
        self.first_frame_pose_trust_gate_fallback_mode = str(
            pose_cfg.get('first_frame_pose_trust_gate_fallback_mode', 'no_prior')
        ).lower()
        self.first_frame_pose_trust_gate_chain_source = str(
            pose_cfg.get('first_frame_pose_trust_gate_chain_source', 'input')
        ).lower()
        if self.first_frame_pose_trust_gate_chain_source not in ('optimized', 'input'):
            warnings.warn(
                f"Unknown first_frame_pose_trust_gate_chain_source={self.first_frame_pose_trust_gate_chain_source}, "
                "fallback to input."
            )
            self.first_frame_pose_trust_gate_chain_source = 'input'
        self.first_frame_pose_trust_gate_enable_no_prior_vo = bool(
            pose_cfg.get('first_frame_pose_trust_gate_enable_no_prior_vo', True)
        )
        self.first_frame_pose_trust_gate_fallback_w_pose_prior = float(
            pose_cfg.get('first_frame_pose_trust_gate_fallback_w_pose_prior', -1.0)
        )
        self.first_frame_pose_trust_gate_info = None
        self.prev_input_pose_for_smooth = None
        self.prev_optimized_pose_for_smooth = None
        self.prev_depth_for_pose_init = None
        self.prev_tool_mask_for_motion = None
        self.pose_no_prior_vo_static_state = False
        self.pose_no_prior_vo_static_streak = 0
        self.pose_no_prior_vo_moving_streak = 0
        self.pose_no_prior_vo_flow_std_hist = deque(maxlen=self.pose_no_prior_vo_static_skip_adaptive_window)
        self.pose_no_prior_vo_flow_mag_hist = deque(maxlen=self.pose_no_prior_vo_static_skip_adaptive_window)
        self.pose_no_prior_vo_accept_trans_hist = deque(maxlen=self.pose_no_prior_vo_conf_jump_hist_window)
        self.pose_no_prior_vo_accept_rot_hist = deque(maxlen=self.pose_no_prior_vo_conf_jump_hist_window)
        self.tool_static_counter = 0
        self.tool_motion_gate_state = bool(self.tool_motion_mode == 'flow_dir_std')
        self.tool_motion_static_streak = 0
        self.tool_motion_moving_streak = 0
        self.tool_motion_records = []
        self.last_motion_info = None
        self.last_pose_init_info = None
        self.last_pose_runtime_info = None
        self.last_deform_init_info = None
        self.lpips_model = None
        if self.pose_w_lpips > 0.0:
            try:
                import lpips  # type: ignore

                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad_(False)
            except Exception as exc:
                warnings.warn(f"LPIPS init failed, disable LPIPS loss. reason: {exc}")
                self.pose_w_lpips = 0.0

    def _pose_enabled_for_frame(self, frame_id: int, incremental: bool):
        if not self.pose_opt_enabled:
            return False
        if incremental:
            return True
        return self.pose_optimize_first_frame and frame_id == 0

    @staticmethod
    def _identity_pose_like(reference_pose: torch.Tensor):
        return torch.eye(4, device=reference_pose.device, dtype=reference_pose.dtype).unsqueeze(0)

    def _resolve_precomputed_depth_path(self, frame_name: str, frame_id: int):
        if self.precomputed_depth_dir is None:
            return None, []
        stem = os.path.splitext(os.path.basename(frame_name))[0]
        stem_wo_side = stem[:-1] if len(stem) > 0 and stem[-1].lower() in ('l', 'r') else stem
        candidates = [
            f'{stem}{self.precomputed_depth_ext}',
            f'{stem_wo_side}{self.precomputed_depth_ext}',
            f'{frame_id:05d}{self.precomputed_depth_ext}',
            f'{frame_id:06d}{self.precomputed_depth_ext}',
            f'{frame_id:08d}{self.precomputed_depth_ext}',
        ]
        for rel in candidates:
            path = os.path.join(self.precomputed_depth_dir, rel)
            if os.path.isfile(path):
                return path, candidates
        return None, candidates

    def _load_precomputed_depth_map(self, frame_name: str, frame_id: int, device, dtype):
        depth_path, tried = self._resolve_precomputed_depth_path(frame_name, frame_id)
        if depth_path is None:
            raise FileNotFoundError(
                f'No precomputed depth found for frame={frame_name} id={frame_id}, '
                f'tried={tried} in dir={self.precomputed_depth_dir}'
            )

        if depth_path.lower().endswith('.npy'):
            depth_np = np.load(depth_path).astype(np.float32)
        else:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise RuntimeError(f'Failed to read depth image: {depth_path}')
            if depth_img.ndim == 3:
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
            depth_np = depth_img.astype(np.float32)

        if depth_np.ndim != 2:
            raise RuntimeError(f'Expected 2D depth map, got shape={depth_np.shape} at {depth_path}')

        H, W, _, _, _, _ = self.camera.get_params()
        if depth_np.shape[0] != H or depth_np.shape[1] != W:
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        if self.precomputed_depth_invert:
            depth_np = 1.0 / np.maximum(depth_np, self.precomputed_depth_eps)
        depth_np = depth_np * self.precomputed_depth_scale + self.precomputed_depth_bias

        if self.precomputed_depth_clip_max > self.precomputed_depth_clip_min and self.precomputed_depth_clip_max > 0.0:
            depth_np = np.clip(depth_np, self.precomputed_depth_clip_min, self.precomputed_depth_clip_max)
        elif self.precomputed_depth_clip_min > 0.0:
            depth_np = np.maximum(depth_np, self.precomputed_depth_clip_min)

        depth_np = np.where(np.isfinite(depth_np), depth_np, 0.0).astype(np.float32)
        depth_t = torch.from_numpy(depth_np).to(device=device, dtype=dtype).unsqueeze(0)
        valid_t = torch.isfinite(depth_t) & (depth_t > 0.0)
        return depth_t, valid_t

    def _get_input_depth(self, frame_id: int, gt_color: torch.Tensor, gt_color_r: torch.Tensor):
        if self.depth_input_source == 'raft_stereo':
            with torch.no_grad():
                stereo_depth, flow_valid = get_depth_from_raft(self.raft, gt_color, gt_color_r, self.baseline)
            return stereo_depth, flow_valid
        if self.depth_input_source == 'foundation_stereo':
            with torch.no_grad():
                stereo_depth, flow_valid = get_depth_from_foundation(
                    self.foundation_model,
                    gt_color,
                    gt_color_r,
                    gt_color.device,
                    self.foundation_args,
                    depth_scale=self.scale,
                    depth_formula=self.foundation_depth_formula,
                    override_fx=self.foundation_seq_fx if self.foundation_use_sequence_camera_params else None,
                    override_baseline=self.baseline if self.foundation_use_sequence_camera_params else None,
                )
            return stereo_depth, flow_valid

        frame_name = self.frame_reader.get_name(frame_id)
        stereo_depth, flow_valid = self._load_precomputed_depth_map(
            frame_name=frame_name,
            frame_id=frame_id,
            device=gt_color.device,
            dtype=gt_color.dtype,
        )
        return stereo_depth, flow_valid

    def _resolve_pose_file_path(self, pose_file_value):
        if pose_file_value is None:
            return None
        pose_file = str(pose_file_value).strip()
        if pose_file == '':
            return None
        pose_file = os.path.expanduser(pose_file)
        candidates = [pose_file]
        if not os.path.isabs(pose_file):
            candidates.append(os.path.join(self.frame_reader.input_folder, pose_file))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        warnings.warn(
            f"External pose init file not found: {pose_file_value}. "
            "Fallback to default no-prior initialization."
        )
        return None

    def _load_external_no_prior_init_poses(self, pose_file_value, apply_scale=True):
        pose_path = self._resolve_pose_file_path(pose_file_value)
        if pose_path is None:
            return None, None

        pose_rows = []
        with open(pose_path, 'r') as file:
            for line in file:
                row = line.strip().replace(',', ' ').replace('\t', ' ')
                if len(row) == 0 or row.startswith('#'):
                    continue
                tokens = [token for token in row.split(' ') if token != '']
                if len(tokens) < 8:
                    continue
                try:
                    frame_id = int(float(tokens[0]))
                    trans = np.asarray(tokens[1:4], dtype=np.float32)
                    quat = np.asarray(tokens[4:8], dtype=np.float32)
                    rot = Rotation.from_quat(quat).as_matrix().astype(np.float32)
                except Exception:
                    continue
                pose_mat = np.eye(4, dtype=np.float32)
                pose_mat[:3, :3] = rot
                trans_scale = float(self.scale) if apply_scale else 1.0
                pose_mat[:3, 3] = trans * trans_scale
                pose_rows.append((frame_id, torch.from_numpy(pose_mat)))

        if len(pose_rows) == 0:
            warnings.warn(
                f"External pose init file is empty/invalid: {pose_path}. "
                "Fallback to default no-prior initialization."
            )
            return pose_path, None

        pose_rows.sort(key=lambda item: item[0])
        first_pose_inv = torch.linalg.inv(pose_rows[0][1].float())
        pose_map = {}
        for frame_id, pose_mat in pose_rows:
            normalized = (first_pose_inv @ pose_mat.float()).unsqueeze(0)
            if torch.isfinite(normalized).all():
                pose_map[int(frame_id)] = normalized
        if len(pose_map) == 0:
            warnings.warn(
                f"External pose init file has no finite poses after normalization: {pose_path}. "
                "Fallback to default no-prior initialization."
            )
            return pose_path, None

        print(f"[PoseInit] Loaded {len(pose_map)} external no-prior poses from {pose_path}")
        return pose_path, pose_map

    def _build_render_visibility_mask(self):
        if not self.render_visibility_mask_enabled:
            return None
        if self.net is None or self.net.get_xyz is None or self.net.get_xyz.shape[0] == 0:
            return None
        opacity = self.net.get_opacity
        if opacity is None or opacity.numel() == 0:
            return None
        op = opacity.reshape(-1).detach()
        mask = torch.ones_like(op, dtype=torch.bool)
        if self.render_visibility_opacity_threshold > 0.0:
            mask &= op > self.render_visibility_opacity_threshold
        if self.render_visibility_keep_ratio < 1.0:
            k = int(max(1, round(op.shape[0] * self.render_visibility_keep_ratio)))
            if k < op.shape[0]:
                topk_idx = torch.topk(op, k, sorted=False).indices
                keep = torch.zeros_like(mask)
                keep[topk_idx] = True
                mask &= keep
        if int(mask.sum().item()) == 0:
            return None
        return mask

    def _render_scene(self, deform=True, render_deformation=False):
        visibility_mask = self._build_render_visibility_mask()
        return render(
            self.camera,
            self.net,
            self.background,
            deform=deform,
            render_deformation=render_deformation,
            visibility_mask=visibility_mask,
        )

    def _estimate_motion_from_flow_direction(self, prev_color, curr_color, tool_mask=None):
        if prev_color is None or curr_color is None:
            return None
        with torch.no_grad():
            flow = self.raft(
                2 * prev_color.permute(0, 3, 1, 2) - 1.0,
                2 * curr_color.permute(0, 3, 1, 2) - 1.0,
            )[-1]
        flow_x = flow[:, 0].squeeze(0)
        flow_y = flow[:, 1].squeeze(0)
        flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y)
        valid = torch.isfinite(flow_x) & torch.isfinite(flow_y) & torch.isfinite(flow_mag)
        valid &= flow_mag >= self.flow_dir_gate_min_mag

        if self.flow_dir_gate_use_tool_mask and tool_mask is not None:
            if tool_mask.ndim == 4:
                tool_valid = tool_mask[:, 0].squeeze(0).bool()
            elif tool_mask.ndim == 3:
                tool_valid = tool_mask.squeeze(0).bool()
            else:
                tool_valid = tool_mask.bool()
            valid &= tool_valid

        valid_idx = torch.where(valid)
        n_valid = int(valid_idx[0].numel())
        if n_valid < self.flow_dir_gate_min_pixels:
            return None

        if self.flow_dir_gate_max_points > 0 and n_valid > self.flow_dir_gate_max_points:
            perm = torch.randperm(n_valid, device=flow.device)[: self.flow_dir_gate_max_points]
            ys = valid_idx[0][perm]
            xs = valid_idx[1][perm]
        else:
            ys = valid_idx[0]
            xs = valid_idx[1]

        theta = torch.atan2(flow_y[ys, xs], flow_x[ys, xs])
        mag = flow_mag[ys, xs]
        if self.flow_dir_gate_use_magnitude_weight:
            weights = mag.clamp_min(1e-6)
            weights = weights / weights.sum().clamp_min(1e-6)
            mean_cos = torch.sum(torch.cos(theta) * weights)
            mean_sin = torch.sum(torch.sin(theta) * weights)
            mean_mag = float(torch.sum(mag * weights).item())
        else:
            mean_cos = torch.mean(torch.cos(theta))
            mean_sin = torch.mean(torch.sin(theta))
            mean_mag = float(torch.mean(mag).item())

        R = torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin).clamp(1e-6, 1.0)
        flow_dir_std = float(torch.sqrt(torch.clamp(-2.0 * torch.log(R), min=0.0)).item())
        flow_dir_mean = float(torch.atan2(mean_sin, mean_cos).item())
        return {
            'flow_dir_std': flow_dir_std,
            'flow_dir_mean': flow_dir_mean,
            'flow_dir_resultant': float(R.item()),
            'flow_mag_mean': mean_mag,
            'flow_valid_points': int(theta.shape[0]),
        }

    def _estimate_motion_from_flow_components(self, flow_x, flow_y, tool_mask=None):
        if flow_x is None or flow_y is None:
            return None
        if flow_x.ndim == 3:
            flow_x = flow_x.squeeze(0)
        if flow_y.ndim == 3:
            flow_y = flow_y.squeeze(0)
        flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y)
        valid = torch.isfinite(flow_x) & torch.isfinite(flow_y) & torch.isfinite(flow_mag)
        valid &= flow_mag >= self.flow_dir_gate_min_mag

        if self.pose_no_prior_vo_static_skip_use_tool_mask and tool_mask is not None:
            if tool_mask.ndim == 4:
                tool_valid = tool_mask[:, 0].squeeze(0).bool()
            elif tool_mask.ndim == 3:
                tool_valid = tool_mask.squeeze(0).bool()
            else:
                tool_valid = tool_mask.bool()
            valid &= tool_valid

        valid_idx = torch.where(valid)
        n_valid = int(valid_idx[0].numel())
        if n_valid < self.flow_dir_gate_min_pixels:
            return None

        if self.flow_dir_gate_max_points > 0 and n_valid > self.flow_dir_gate_max_points:
            perm = torch.randperm(n_valid, device=flow_x.device)[: self.flow_dir_gate_max_points]
            ys = valid_idx[0][perm]
            xs = valid_idx[1][perm]
        else:
            ys = valid_idx[0]
            xs = valid_idx[1]

        theta = torch.atan2(flow_y[ys, xs], flow_x[ys, xs])
        mag = flow_mag[ys, xs]
        if self.flow_dir_gate_use_magnitude_weight:
            weights = mag.clamp_min(1e-6)
            weights = weights / weights.sum().clamp_min(1e-6)
            mean_cos = torch.sum(torch.cos(theta) * weights)
            mean_sin = torch.sum(torch.sin(theta) * weights)
            mean_mag = float(torch.sum(mag * weights).item())
        else:
            mean_cos = torch.mean(torch.cos(theta))
            mean_sin = torch.mean(torch.sin(theta))
            mean_mag = float(torch.mean(mag).item())

        resultant = torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin).clamp(1e-6, 1.0)
        flow_dir_std = float(torch.sqrt(torch.clamp(-2.0 * torch.log(resultant), min=0.0)).item())
        flow_dir_mean = float(torch.atan2(mean_sin, mean_cos).item())
        return {
            'flow_dir_std': flow_dir_std,
            'flow_dir_mean': flow_dir_mean,
            'flow_dir_resultant': float(resultant.item()),
            'flow_mag_mean': mean_mag,
            'flow_valid_points': int(theta.shape[0]),
        }

    def _vo_static_skip_from_flow_info(self, flow_info):
        if not self.pose_no_prior_vo_static_skip_enabled:
            return False, {'vo_static_skip_enabled': False}

        base_static_thr = float(self.pose_no_prior_vo_static_skip_std)
        base_moving_thr = float(self.pose_no_prior_vo_static_skip_moving_std)
        static_thr = float(base_static_thr)
        moving_thr = float(base_moving_thr)

        info = {
            'vo_static_skip_enabled': True,
            'vo_static_skip_valid': bool(flow_info is not None),
            'vo_static_skip_state': bool(self.pose_no_prior_vo_static_state),
            'vo_static_skip_streak_static': int(self.pose_no_prior_vo_static_streak),
            'vo_static_skip_streak_moving': int(self.pose_no_prior_vo_moving_streak),
            'vo_static_skip_std_thr': float(static_thr),
            'vo_static_skip_moving_std_thr': float(moving_thr),
            'vo_static_skip_std_thr_base': float(base_static_thr),
            'vo_static_skip_moving_std_thr_base': float(base_moving_thr),
            'vo_static_skip_hysteresis': int(self.pose_no_prior_vo_static_skip_hysteresis),
            'vo_static_skip_adaptive_enabled': bool(self.pose_no_prior_vo_static_skip_adaptive_enabled),
            'vo_static_skip_mag_guard_enabled': bool(self.pose_no_prior_vo_static_skip_mag_guard_enabled),
        }
        if flow_info is None:
            info['vo_static_skip_reason'] = 'invalid_flow'
            return False, info

        std_v = float(flow_info.get('flow_dir_std', np.nan))
        mean_mag = float(flow_info.get('flow_mag_mean', np.nan))
        if np.isfinite(std_v):
            self.pose_no_prior_vo_flow_std_hist.append(float(std_v))
        if np.isfinite(mean_mag):
            self.pose_no_prior_vo_flow_mag_hist.append(float(mean_mag))

        std_hist_len = int(len(self.pose_no_prior_vo_flow_std_hist))
        mag_hist_len = int(len(self.pose_no_prior_vo_flow_mag_hist))
        adaptive_applied = False
        if (
            self.pose_no_prior_vo_static_skip_adaptive_enabled
            and std_hist_len >= self.pose_no_prior_vo_static_skip_adaptive_min_samples
        ):
            std_hist = np.asarray(list(self.pose_no_prior_vo_flow_std_hist), dtype=np.float64)
            q_static = float(np.quantile(std_hist, 0.70))
            q_moving = float(np.quantile(std_hist, 0.35))
            static_low = base_static_thr - float(self.pose_no_prior_vo_static_skip_adaptive_static_down)
            static_high = base_static_thr + float(self.pose_no_prior_vo_static_skip_adaptive_static_up)
            moving_low = base_moving_thr - float(self.pose_no_prior_vo_static_skip_adaptive_moving_down)
            moving_high = base_moving_thr + float(self.pose_no_prior_vo_static_skip_adaptive_moving_up)
            static_thr = float(np.clip(q_static, static_low, static_high))
            moving_thr = float(np.clip(q_moving, moving_low, moving_high))
            if moving_thr > static_thr - 1e-3:
                moving_thr = float(max(static_thr - 1e-3, moving_low))
            adaptive_applied = True
            info.update(
                {
                    'vo_static_skip_std_q70': q_static,
                    'vo_static_skip_std_q35': q_moving,
                }
            )

        mag_static_thr = float('nan')
        mag_moving_thr = float('nan')
        if (
            self.pose_no_prior_vo_static_skip_mag_guard_enabled
            and mag_hist_len >= self.pose_no_prior_vo_static_skip_mag_min_samples
        ):
            mag_hist = np.asarray(list(self.pose_no_prior_vo_flow_mag_hist), dtype=np.float64)
            q_static = float(
                np.quantile(
                    mag_hist,
                    np.clip(float(self.pose_no_prior_vo_static_skip_mag_static_quantile), 0.05, 0.95),
                )
            )
            q_moving = float(
                np.quantile(
                    mag_hist,
                    np.clip(float(self.pose_no_prior_vo_static_skip_mag_moving_quantile), 0.05, 0.99),
                )
            )
            mag_static_thr = q_static * float(max(self.pose_no_prior_vo_static_skip_mag_static_scale, 1e-6))
            mag_moving_thr = q_moving * float(max(self.pose_no_prior_vo_static_skip_mag_moving_scale, 1e-6))
            info.update(
                {
                    'vo_static_skip_mag_q_static': q_static,
                    'vo_static_skip_mag_q_moving': q_moving,
                }
            )

        static_conf_std = bool(np.isfinite(std_v) and std_v >= static_thr)
        moving_conf_std = bool(np.isfinite(std_v) and std_v <= moving_thr)
        static_conf_mag = bool((not np.isfinite(mag_static_thr)) or (np.isfinite(mean_mag) and mean_mag <= mag_static_thr))
        moving_conf_mag = bool(np.isfinite(mean_mag) and np.isfinite(mag_moving_thr) and mean_mag >= mag_moving_thr)

        if self.pose_no_prior_vo_static_skip_mag_guard_enabled and np.isfinite(mag_static_thr):
            static_confident = bool(static_conf_std and static_conf_mag)
        else:
            static_confident = bool(static_conf_std)
        if self.pose_no_prior_vo_static_skip_mag_guard_enabled and np.isfinite(mag_moving_thr):
            moving_confident = bool(moving_conf_std or moving_conf_mag)
        else:
            moving_confident = bool(moving_conf_std)

        if static_confident and moving_confident and np.isfinite(std_v):
            static_margin = float(std_v - static_thr)
            moving_margin = float(moving_thr - std_v)
            if static_margin >= moving_margin:
                moving_confident = False
            else:
                static_confident = False

        if static_confident:
            self.pose_no_prior_vo_static_streak += 1
        else:
            self.pose_no_prior_vo_static_streak = 0
        if moving_confident:
            self.pose_no_prior_vo_moving_streak += 1
        else:
            self.pose_no_prior_vo_moving_streak = 0

        if self.pose_no_prior_vo_static_streak >= self.pose_no_prior_vo_static_skip_hysteresis:
            self.pose_no_prior_vo_static_state = True
        if self.pose_no_prior_vo_moving_streak >= self.pose_no_prior_vo_static_skip_hysteresis:
            self.pose_no_prior_vo_static_state = False

        should_skip = bool(self.pose_no_prior_vo_static_state)
        info.update(
            {
                'vo_static_skip_reason': 'static' if should_skip else 'moving_or_uncertain',
                'vo_static_skip_state': bool(self.pose_no_prior_vo_static_state),
                'vo_static_skip_streak_static': int(self.pose_no_prior_vo_static_streak),
                'vo_static_skip_streak_moving': int(self.pose_no_prior_vo_moving_streak),
                'vo_static_skip_std': float(std_v) if np.isfinite(std_v) else float('nan'),
                'vo_static_skip_flow_mag_mean': float(mean_mag) if np.isfinite(mean_mag) else float('nan'),
                'vo_static_skip_std_thr': float(static_thr),
                'vo_static_skip_moving_std_thr': float(moving_thr),
                'vo_static_skip_mag_static_thr': float(mag_static_thr) if np.isfinite(mag_static_thr) else float('nan'),
                'vo_static_skip_mag_moving_thr': float(mag_moving_thr) if np.isfinite(mag_moving_thr) else float('nan'),
                'vo_static_skip_hist_std_len': int(std_hist_len),
                'vo_static_skip_hist_mag_len': int(mag_hist_len),
                'vo_static_skip_adaptive_applied': bool(adaptive_applied),
                'vo_static_skip_conf_static_std': bool(static_conf_std),
                'vo_static_skip_conf_moving_std': bool(moving_conf_std),
                'vo_static_skip_conf_static_mag': bool(static_conf_mag),
                'vo_static_skip_conf_moving_mag': bool(moving_conf_mag),
                'vo_static_skip_conf_static': bool(static_confident),
                'vo_static_skip_conf_moving': bool(moving_confident),
            }
        )
        return should_skip, info

    @staticmethod
    def _vo_robust_upper_limit(values, k: float):
        if values is None:
            return float('nan'), float('nan'), float('nan')
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size <= 0:
            return float('nan'), float('nan'), float('nan')
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        robust_sigma = float(max(1.4826 * mad, 1e-6))
        upper = float(med + max(float(k), 0.0) * robust_sigma)
        return med, robust_sigma, upper

    def _vo_confidence_reject(self, vo_valid_points: int, solver_info: dict, trans_norm: float, rot_deg: float):
        enabled = bool(self.pose_no_prior_vo_conf_reject_enabled)
        inlier_points = int((solver_info or {}).get('vo_inlier_points', vo_valid_points))
        inlier_ratio = float(inlier_points / max(int(vo_valid_points), 1))
        reproj_median = float((solver_info or {}).get('vo_reproj_median_px', float('nan')))
        info = {
            'vo_conf_reject_enabled': bool(enabled),
            'vo_conf_inlier_ratio': float(inlier_ratio),
            'vo_conf_inlier_points': int(inlier_points),
            'vo_conf_reproj_median_px': float(reproj_median) if np.isfinite(reproj_median) else float('nan'),
            'vo_conf_hist_len_trans': int(len(self.pose_no_prior_vo_accept_trans_hist)),
            'vo_conf_hist_len_rot': int(len(self.pose_no_prior_vo_accept_rot_hist)),
        }
        if not enabled:
            return False, 'disabled', info

        min_inlier_ratio = float(max(self.pose_no_prior_vo_conf_min_inlier_ratio, 0.0))
        min_inliers = int(max(self.pose_no_prior_vo_conf_min_inliers, 0))
        max_reproj_px = float(self.pose_no_prior_vo_conf_max_reproj_px)
        abs_max_trans = float(max(self.pose_no_prior_vo_conf_abs_max_trans, 0.0))
        abs_max_rot_deg = float(max(self.pose_no_prior_vo_conf_abs_max_rot_deg, 0.0))
        info.update(
            {
                'vo_conf_min_inlier_ratio': float(min_inlier_ratio),
                'vo_conf_min_inliers': int(min_inliers),
                'vo_conf_max_reproj_px': float(max_reproj_px),
                'vo_conf_abs_max_trans': float(abs_max_trans),
                'vo_conf_abs_max_rot_deg': float(abs_max_rot_deg),
            }
        )

        if min_inliers > 0 and inlier_points < min_inliers:
            return True, 'vo_conf_reject_few_inliers', info
        if min_inlier_ratio > 0.0 and inlier_ratio < min_inlier_ratio:
            return True, 'vo_conf_reject_low_inlier_ratio', info
        if max_reproj_px > 0.0 and np.isfinite(reproj_median) and reproj_median > max_reproj_px:
            return True, 'vo_conf_reject_high_reproj', info
        if abs_max_trans > 0.0 and np.isfinite(trans_norm) and trans_norm > abs_max_trans:
            return True, 'vo_conf_reject_abs_trans', info
        if abs_max_rot_deg > 0.0 and np.isfinite(rot_deg) and rot_deg > abs_max_rot_deg:
            return True, 'vo_conf_reject_abs_rot', info

        trans_med, trans_sigma, trans_upper = self._vo_robust_upper_limit(
            self.pose_no_prior_vo_accept_trans_hist,
            self.pose_no_prior_vo_conf_jump_k_trans,
        )
        rot_med, rot_sigma, rot_upper = self._vo_robust_upper_limit(
            self.pose_no_prior_vo_accept_rot_hist,
            self.pose_no_prior_vo_conf_jump_k_rot,
        )
        info.update(
            {
                'vo_conf_jump_trans_median': float(trans_med) if np.isfinite(trans_med) else float('nan'),
                'vo_conf_jump_trans_sigma': float(trans_sigma) if np.isfinite(trans_sigma) else float('nan'),
                'vo_conf_jump_trans_upper': float(trans_upper) if np.isfinite(trans_upper) else float('nan'),
                'vo_conf_jump_rot_median': float(rot_med) if np.isfinite(rot_med) else float('nan'),
                'vo_conf_jump_rot_sigma': float(rot_sigma) if np.isfinite(rot_sigma) else float('nan'),
                'vo_conf_jump_rot_upper': float(rot_upper) if np.isfinite(rot_upper) else float('nan'),
            }
        )
        enough_hist = (
            len(self.pose_no_prior_vo_accept_trans_hist) >= self.pose_no_prior_vo_conf_jump_min_hist
            and len(self.pose_no_prior_vo_accept_rot_hist) >= self.pose_no_prior_vo_conf_jump_min_hist
        )
        info['vo_conf_jump_hist_ready'] = bool(enough_hist)
        if enough_hist:
            if np.isfinite(trans_upper) and np.isfinite(trans_norm) and trans_norm > trans_upper:
                return True, 'vo_conf_reject_jump_trans', info
            if np.isfinite(rot_upper) and np.isfinite(rot_deg) and rot_deg > rot_upper:
                return True, 'vo_conf_reject_jump_rot', info
        return False, 'ok', info

    @staticmethod
    def _solve_rigid(src: torch.Tensor, dst: torch.Tensor):
        src_mean = src.mean(dim=0, keepdim=True)
        dst_mean = dst.mean(dim=0, keepdim=True)
        cov_m = (src - src_mean).transpose(0, 1) @ (dst - dst_mean)
        U, _, Vh = torch.linalg.svd(cov_m)
        rot_m = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
        if torch.det(rot_m) < 0:
            Vh[-1, :] *= -1
            rot_m = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
        trans_v = dst_mean.squeeze(0) - rot_m @ src_mean.squeeze(0)
        return rot_m, trans_v

    def _estimate_vo_transform_svd(self, X1: torch.Tensor, X2: torch.Tensor):
        src_fit = X1
        dst_fit = X2
        ransac_info = {}
        try:
            if self.pose_no_prior_vo_ransac_enabled and int(X1.shape[0]) >= max(3, int(self.pose_no_prior_vo_ransac_sample_size)):
                n_points = int(X1.shape[0])
                sample_size = int(np.clip(self.pose_no_prior_vo_ransac_sample_size, 3, n_points))
                max_iters = max(1, int(self.pose_no_prior_vo_ransac_iters))
                inlier_thr = max(float(self.pose_no_prior_vo_ransac_inlier_res), 1e-6)
                min_inliers = max(16, int(self.pose_no_prior_vo_ransac_min_inliers))
                eval_points = max(0, int(self.pose_no_prior_vo_ransac_eval_points))

                if eval_points > 0 and n_points > eval_points:
                    eval_idx = torch.randperm(n_points, device=X1.device)[:eval_points]
                    X1_eval = X1[eval_idx]
                    X2_eval = X2[eval_idx]
                else:
                    eval_idx = None
                    X1_eval = X1
                    X2_eval = X2

                best_count = -1
                best_med_err = float('inf')
                best_model = None
                for _ in range(max_iters):
                    idx = torch.randperm(n_points, device=X1.device)[:sample_size]
                    R_try, t_try = self._solve_rigid(X1[idx], X2[idx])
                    if (not torch.isfinite(R_try).all()) or (not torch.isfinite(t_try).all()):
                        continue
                    pred_eval = (X1_eval @ R_try.transpose(0, 1)) + t_try.unsqueeze(0)
                    err_eval = torch.linalg.norm(pred_eval - X2_eval, dim=1)
                    inlier_eval = err_eval <= inlier_thr
                    inlier_count = int(inlier_eval.sum().item())
                    if inlier_count <= 0:
                        continue
                    med_err = float(torch.median(err_eval[inlier_eval]).item())
                    if inlier_count > best_count or (inlier_count == best_count and med_err < best_med_err):
                        best_count = inlier_count
                        best_med_err = med_err
                        best_model = (R_try, t_try)

                if best_model is not None:
                    R_try, t_try = best_model
                    pred_all = (X1 @ R_try.transpose(0, 1)) + t_try.unsqueeze(0)
                    err_all = torch.linalg.norm(pred_all - X2, dim=1)
                    inlier_mask_all = err_all <= inlier_thr
                    inlier_count_all = int(inlier_mask_all.sum().item())
                    if inlier_count_all >= min_inliers:
                        src_fit = X1[inlier_mask_all]
                        dst_fit = X2[inlier_mask_all]
                        ransac_info = {
                            'vo_ransac_used': True,
                            'vo_ransac_inlier_points': int(inlier_count_all),
                            'vo_ransac_inlier_ratio': float(inlier_count_all / max(n_points, 1)),
                            'vo_ransac_inlier_res': float(inlier_thr),
                            'vo_ransac_eval_points': int(X1_eval.shape[0]),
                        }
                    else:
                        ransac_info = {
                            'vo_ransac_used': False,
                            'vo_ransac_reason': 'few_inliers',
                            'vo_ransac_inlier_points': int(inlier_count_all),
                            'vo_ransac_inlier_res': float(inlier_thr),
                            'vo_ransac_eval_points': int(X1_eval.shape[0]),
                        }
                else:
                    ransac_info = {
                        'vo_ransac_used': False,
                        'vo_ransac_reason': 'no_model',
                        'vo_ransac_inlier_points': 0,
                        'vo_ransac_inlier_res': float(inlier_thr),
                        'vo_ransac_eval_points': int(X1_eval.shape[0]),
                    }

            for it in range(max(self.pose_no_prior_vo_refine_iters, 1)):
                R, t = self._solve_rigid(src_fit, dst_fit)
                if it + 1 >= max(self.pose_no_prior_vo_refine_iters, 1):
                    break
                pred = (X1 @ R.transpose(0, 1)) + t.unsqueeze(0)
                err = torch.linalg.norm(pred - X2, dim=1)
                quant = float(np.clip(self.pose_no_prior_vo_inlier_quantile, 0.05, 0.95))
                thr = torch.quantile(err, quant)
                if self.pose_no_prior_vo_inlier_max_res > 0.0:
                    thr = torch.minimum(
                        thr,
                        torch.tensor(
                            self.pose_no_prior_vo_inlier_max_res,
                            device=thr.device,
                            dtype=thr.dtype,
                        ),
                    )
                inlier_mask = err <= thr
                min_inliers = max(64, int(self.pose_no_prior_vo_min_inliers))
                if int(inlier_mask.sum().item()) < min_inliers:
                    break
                src_fit = X1[inlier_mask]
                dst_fit = X2[inlier_mask]
        except RuntimeError:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'svd_failed',
                'vo_solver': 'svd',
                'vo_inlier_points': 0,
                **ransac_info,
            }
        if (not torch.isfinite(R).all()) or (not torch.isfinite(t).all()):
            return None, None, {
                'vo_used': False,
                'vo_reason': 'non_finite',
                'vo_solver': 'svd',
                'vo_inlier_points': 0,
                **ransac_info,
            }
        return R, t, {
            'vo_used': True,
            'vo_reason': 'ok',
            'vo_solver': 'svd',
            'vo_inlier_points': int(src_fit.shape[0]),
            **ransac_info,
        }

    def _estimate_vo_transform_pnp(
        self,
        X1: torch.Tensor,
        x2_sel: torch.Tensor,
        y2_sel: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        n_points = int(X1.shape[0])
        min_inliers = max(32, int(self.pose_no_prior_vo_min_inliers))
        if n_points < max(6, min_inliers):
            return None, None, {
                'vo_used': False,
                'vo_reason': 'few_points_for_pnp',
                'vo_solver': 'pnp',
                'vo_inlier_points': 0,
            }

        obj_pts = X1.detach().cpu().numpy().astype(np.float32)
        img_pts = torch.stack([x2_sel, y2_sel], dim=-1).detach().cpu().numpy().astype(np.float32)
        K = np.array(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=K,
                distCoeffs=None,
                reprojectionError=float(self.pose_no_prior_vo_pnp_reproj_err),
                confidence=float(np.clip(self.pose_no_prior_vo_pnp_confidence, 0.5, 0.9999)),
                iterationsCount=max(1, int(self.pose_no_prior_vo_pnp_iterations)),
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'pnp_exception',
                'vo_solver': 'pnp',
                'vo_inlier_points': 0,
            }

        if (not ok) or rvec is None or tvec is None:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'pnp_failed',
                'vo_solver': 'pnp',
                'vo_inlier_points': 0,
            }
        inlier_idx = np.arange(n_points, dtype=np.int32) if inliers is None else inliers.reshape(-1).astype(np.int32)
        if inlier_idx.size < min_inliers:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'few_pnp_inliers',
                'vo_solver': 'pnp',
                'vo_inlier_points': int(inlier_idx.size),
            }

        if self.pose_no_prior_vo_pnp_refine and inlier_idx.size >= 6:
            obj_in = obj_pts[inlier_idx]
            img_in = img_pts[inlier_idx]
            try:
                if hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K, None, rvec, tvec)
                else:
                    cv2.solvePnP(
                        obj_in,
                        img_in,
                        K,
                        None,
                        rvec,
                        tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
            except cv2.error:
                pass

        try:
            R_np, _ = cv2.Rodrigues(rvec)
            proj, _ = cv2.projectPoints(obj_pts[inlier_idx], rvec, tvec, K, None)
            reproj = np.linalg.norm(proj.reshape(-1, 2) - img_pts[inlier_idx], axis=1)
            reproj_med = float(np.median(reproj)) if reproj.size > 0 else float('nan')
        except cv2.error:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'pnp_projection_failed',
                'vo_solver': 'pnp',
                'vo_inlier_points': int(inlier_idx.size),
            }

        R = torch.from_numpy(R_np).to(device=X1.device, dtype=X1.dtype)
        t = torch.from_numpy(tvec.reshape(3)).to(device=X1.device, dtype=X1.dtype)
        if (not torch.isfinite(R).all()) or (not torch.isfinite(t).all()):
            return None, None, {
                'vo_used': False,
                'vo_reason': 'pnp_non_finite',
                'vo_solver': 'pnp',
                'vo_inlier_points': int(inlier_idx.size),
            }
        return R, t, {
            'vo_used': True,
            'vo_reason': 'ok',
            'vo_solver': 'pnp',
            'vo_inlier_points': int(inlier_idx.size),
            'vo_reproj_median_px': reproj_med,
        }

    @staticmethod
    def _to_1hw(tensor: torch.Tensor):
        if tensor is None:
            return None
        if tensor.ndim == 4:
            return tensor[:, 0]
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return None

    @staticmethod
    def _sample_points_1hw(map_1hw: torch.Tensor, x: torch.Tensor, y: torch.Tensor, H: int, W: int):
        grid_x = (2.0 * x / max(W - 1, 1)) - 1.0
        grid_y = (2.0 * y / max(H - 1, 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
        vals = F.grid_sample(
            map_1hw.unsqueeze(1),
            grid,
            mode='bilinear',
            align_corners=True,
        )
        return vals.view(-1)

    def _estimate_vo_transform_essential(
        self,
        x1_sel: torch.Tensor,
        y1_sel: torch.Tensor,
        x2_sel: torch.Tensor,
        y2_sel: torch.Tensor,
        X1: torch.Tensor,
        X2: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        n_points = int(X1.shape[0])
        min_inliers = max(16, int(self.pose_no_prior_vo_essential_min_inliers))
        if n_points < max(16, min_inliers):
            return None, None, {
                'vo_used': False,
                'vo_reason': 'few_points_for_essential',
                'vo_solver': 'essential',
                'vo_inlier_points': 0,
            }

        pts1 = torch.stack([x1_sel, y1_sel], dim=-1).detach().cpu().numpy().astype(np.float32)
        pts2 = torch.stack([x2_sel, y2_sel], dim=-1).detach().cpu().numpy().astype(np.float32)
        K = np.array(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        try:
            E, ransac_mask = cv2.findEssentialMat(
                pts1,
                pts2,
                cameraMatrix=K,
                method=cv2.RANSAC,
                prob=float(np.clip(self.pose_no_prior_vo_essential_confidence, 0.5, 0.9999)),
                threshold=float(max(self.pose_no_prior_vo_essential_ransac_px, 0.25)),
            )
        except cv2.error:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'essential_exception',
                'vo_solver': 'essential',
                'vo_inlier_points': 0,
            }

        if E is None:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'essential_failed',
                'vo_solver': 'essential',
                'vo_inlier_points': 0,
            }
        if ransac_mask is None:
            ransac_mask = np.ones((pts1.shape[0], 1), dtype=np.uint8)

        if E.ndim == 2:
            E_candidates = [E]
        elif E.ndim == 1 and E.size == 9:
            E_candidates = [E.reshape(3, 3)]
        elif E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
            E_candidates = [E[i : i + 3, :] for i in range(0, E.shape[0], 3)]
        else:
            E_candidates = [E.reshape(3, 3)]

        best_R = None
        best_t = None
        best_mask = None
        best_inliers = -1
        for e_mat in E_candidates:
            try:
                _, R_np, t_np, pose_mask = cv2.recoverPose(e_mat, pts1, pts2, K, mask=ransac_mask)
            except cv2.error:
                continue
            if pose_mask is None:
                continue
            inlier_count = int((pose_mask.reshape(-1) > 0).sum())
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_R = R_np
                best_t = t_np.reshape(3)
                best_mask = (pose_mask.reshape(-1) > 0)

        if best_R is None or best_t is None or best_mask is None or best_inliers < min_inliers:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'few_essential_inliers',
                'vo_solver': 'essential',
                'vo_inlier_points': int(max(best_inliers, 0)),
            }

        R = torch.from_numpy(best_R).to(device=X1.device, dtype=X1.dtype)
        t_dir = torch.from_numpy(best_t).to(device=X1.device, dtype=X1.dtype)
        t_norm = torch.linalg.norm(t_dir)
        if (not torch.isfinite(R).all()) or (not torch.isfinite(t_dir).all()) or float(t_norm.item()) < 1e-8:
            return None, None, {
                'vo_used': False,
                'vo_reason': 'essential_non_finite',
                'vo_solver': 'essential',
                'vo_inlier_points': int(best_inliers),
            }
        t_dir = t_dir / t_norm

        depth_scale = 1.0
        if self.pose_no_prior_vo_essential_use_depth_scale:
            inlier_mask = torch.from_numpy(best_mask).to(device=X1.device, dtype=torch.bool)
            X1_in = X1[inlier_mask]
            X2_in = X2[inlier_mask]
            if X1_in.shape[0] >= min_inliers:
                residual_t = X2_in - (X1_in @ R.transpose(0, 1))
                proj_scale = torch.sum(residual_t * t_dir.unsqueeze(0), dim=1)
                finite = torch.isfinite(proj_scale)
                if int(finite.sum().item()) >= min_inliers:
                    depth_scale = float(torch.median(proj_scale[finite]).item())
                else:
                    depth_scale = 0.0
            else:
                depth_scale = 0.0

        t = t_dir * float(depth_scale)
        if (not torch.isfinite(t).all()) or float(torch.linalg.norm(t).item()) < 1e-8:
            inlier_mask = torch.from_numpy(best_mask).to(device=X1.device, dtype=torch.bool)
            X1_in = X1[inlier_mask]
            X2_in = X2[inlier_mask]
            if X1_in.shape[0] >= min_inliers:
                t = torch.median(X2_in - (X1_in @ R.transpose(0, 1)), dim=0).values

        if (not torch.isfinite(R).all()) or (not torch.isfinite(t).all()):
            return None, None, {
                'vo_used': False,
                'vo_reason': 'essential_bad_translation',
                'vo_solver': 'essential',
                'vo_inlier_points': int(best_inliers),
            }

        return R, t, {
            'vo_used': True,
            'vo_reason': 'ok',
            'vo_solver': 'essential',
            'vo_inlier_points': int(best_inliers),
            'vo_essential_depth_scale': float(depth_scale),
        }

    def _estimate_relative_cam_transform_from_flow(
        self,
        prev_color,
        curr_color,
        prev_depth,
        curr_depth,
        tool_mask=None,
        frame_id=None,
    ):
        if prev_color is None or curr_color is None or prev_depth is None or curr_depth is None:
            return None, None

        H, W, fx, fy, cx, cy = self.camera.get_params()
        d1 = self._to_1hw(prev_depth)
        d2 = self._to_1hw(curr_depth)
        if d1 is None or d2 is None:
            return None, {'vo_used': False, 'vo_reason': 'bad_depth_shape'}
        d1 = d1.to(device=curr_color.device, dtype=curr_color.dtype)
        d2 = d2.to(device=curr_color.device, dtype=curr_color.dtype)

        tool_valid_1hw = None
        if tool_mask is not None and self.pose_no_prior_vo_use_tool_mask:
            tool_valid_1hw = self._to_1hw(tool_mask)
            if tool_valid_1hw is not None:
                tool_valid_1hw = tool_valid_1hw.to(device=d1.device, dtype=d1.dtype)

        x1_sel = None
        y1_sel = None
        x2_sel = None
        y2_sel = None
        d1_sel = None
        d2_sel = None
        corr_source_used = None
        vo_static_gate_info = {}

        use_cotracker = (
            self.pose_no_prior_vo_corr_source in ('cotracker3', 'auto')
            and frame_id is not None
            and frame_id > 0
            and self.pt_cotracker_tracks is not None
            and frame_id < int(self.pt_cotracker_tracks.shape[0])
        )
        if use_cotracker:
            pts_prev = self.pt_cotracker_tracks[frame_id - 1].to(device=d1.device, dtype=d1.dtype)
            pts_curr = self.pt_cotracker_tracks[frame_id].to(device=d1.device, dtype=d1.dtype)
            valid = (
                torch.isfinite(pts_prev[:, 0])
                & torch.isfinite(pts_prev[:, 1])
                & torch.isfinite(pts_curr[:, 0])
                & torch.isfinite(pts_curr[:, 1])
            )
            if self.pt_cotracker_vis is not None and frame_id < int(self.pt_cotracker_vis.shape[0]):
                vis_prev = self.pt_cotracker_vis[frame_id - 1].to(device=d1.device, dtype=torch.bool)
                vis_curr = self.pt_cotracker_vis[frame_id].to(device=d1.device, dtype=torch.bool)
                valid &= vis_prev & vis_curr

            x1_all = pts_prev[:, 0]
            y1_all = pts_prev[:, 1]
            x2_all = pts_curr[:, 0]
            y2_all = pts_curr[:, 1]
            valid &= (x1_all >= 0.0) & (x1_all <= (W - 1)) & (y1_all >= 0.0) & (y1_all <= (H - 1))
            valid &= (x2_all >= 0.0) & (x2_all <= (W - 1)) & (y2_all >= 0.0) & (y2_all <= (H - 1))

            if self.pose_no_prior_vo_max_flow > 0.0:
                flow_mag = torch.sqrt((x2_all - x1_all) ** 2 + (y2_all - y1_all) ** 2)
                valid &= flow_mag <= self.pose_no_prior_vo_max_flow

            d1_all = self._sample_points_1hw(d1, x1_all, y1_all, H, W)
            d2_all = self._sample_points_1hw(d2, x2_all, y2_all, H, W)
            valid &= torch.isfinite(d1_all) & torch.isfinite(d2_all)
            valid &= (d1_all > self.pose_no_prior_vo_min_depth) & (d2_all > self.pose_no_prior_vo_min_depth)
            if self.pose_no_prior_vo_max_depth > self.pose_no_prior_vo_min_depth:
                valid &= (d1_all < self.pose_no_prior_vo_max_depth) & (d2_all < self.pose_no_prior_vo_max_depth)
            if tool_valid_1hw is not None:
                tool_vals = self._sample_points_1hw(tool_valid_1hw, x1_all, y1_all, H, W)
                valid &= tool_vals > 0.5

            valid_idx = torch.where(valid)[0]
            if int(valid_idx.numel()) >= self.pose_no_prior_vo_min_points:
                if self.pose_no_prior_vo_max_points > 0 and int(valid_idx.numel()) > self.pose_no_prior_vo_max_points:
                    perm = torch.randperm(int(valid_idx.numel()), device=d1.device)[: self.pose_no_prior_vo_max_points]
                    valid_idx = valid_idx[perm]
                x1_sel = x1_all[valid_idx]
                y1_sel = y1_all[valid_idx]
                x2_sel = x2_all[valid_idx]
                y2_sel = y2_all[valid_idx]
                d1_sel = d1_all[valid_idx]
                d2_sel = d2_all[valid_idx]
                corr_source_used = 'cotracker3'
            elif self.pose_no_prior_vo_corr_source == 'cotracker3':
                return None, {
                    'vo_valid_points': int(valid_idx.numel()),
                    'vo_used': False,
                    'vo_reason': 'few_points_cotracker',
                    'vo_corr_source': 'cotracker3',
                }

        if x1_sel is not None and self.pose_no_prior_vo_static_skip_enabled:
            static_mask = tool_mask if self.pose_no_prior_vo_static_skip_use_tool_mask else None
            flow_info = self._estimate_motion_from_flow_direction(
                prev_color=prev_color,
                curr_color=curr_color,
                tool_mask=static_mask,
            )
            should_skip_vo, vo_static_gate_info = self._vo_static_skip_from_flow_info(flow_info)
            if flow_info is not None:
                vo_static_gate_info.update(
                    {
                        'flow_dir_std': float(flow_info.get('flow_dir_std', float('nan'))),
                        'flow_dir_mean': float(flow_info.get('flow_dir_mean', float('nan'))),
                        'flow_dir_resultant': float(flow_info.get('flow_dir_resultant', float('nan'))),
                        'flow_mag_mean': float(flow_info.get('flow_mag_mean', float('nan'))),
                        'flow_valid_points': int(flow_info.get('flow_valid_points', 0)),
                    }
                )
            if should_skip_vo:
                T21 = torch.eye(4, device=d1.device, dtype=d1.dtype)
                return T21, {
                    'vo_valid_points': int(flow_info.get('flow_valid_points', 0)) if flow_info is not None else 0,
                    'vo_inlier_points': 0,
                    'vo_used': False,
                    'vo_reason': 'camera_static_skip',
                    'vo_trans_norm': 0.0,
                    'vo_rot_deg': 0.0,
                    'vo_solver': 'skip_static',
                    'vo_corr_source': 'cotracker3',
                    **vo_static_gate_info,
                }

        if x1_sel is None:
            with torch.no_grad():
                flow = self.raft(
                    2 * prev_color.permute(0, 3, 1, 2) - 1.0,
                    2 * curr_color.permute(0, 3, 1, 2) - 1.0,
                )[-1]
            flow_x = flow[:, 0]
            flow_y = flow[:, 1]
            if self.pose_no_prior_vo_static_skip_enabled:
                static_mask = tool_mask if self.pose_no_prior_vo_static_skip_use_tool_mask else None
                flow_info = self._estimate_motion_from_flow_components(flow_x, flow_y, tool_mask=static_mask)
                should_skip_vo, vo_static_gate_info = self._vo_static_skip_from_flow_info(flow_info)
                if flow_info is not None:
                    vo_static_gate_info.update(
                        {
                            'flow_dir_std': float(flow_info.get('flow_dir_std', float('nan'))),
                            'flow_dir_mean': float(flow_info.get('flow_dir_mean', float('nan'))),
                            'flow_dir_resultant': float(flow_info.get('flow_dir_resultant', float('nan'))),
                            'flow_mag_mean': float(flow_info.get('flow_mag_mean', float('nan'))),
                            'flow_valid_points': int(flow_info.get('flow_valid_points', 0)),
                        }
                    )
                if should_skip_vo:
                    T21 = torch.eye(4, device=flow.device, dtype=flow.dtype)
                    return T21, {
                        'vo_valid_points': int(flow_info.get('flow_valid_points', 0)) if flow_info is not None else 0,
                        'vo_inlier_points': 0,
                        'vo_used': False,
                        'vo_reason': 'camera_static_skip',
                        'vo_trans_norm': 0.0,
                        'vo_rot_deg': 0.0,
                        'vo_solver': 'skip_static',
                        'vo_corr_source': 'raft',
                        **vo_static_gate_info,
                    }
            yy, xx = torch.meshgrid(
                torch.arange(H, device=flow.device, dtype=flow.dtype),
                torch.arange(W, device=flow.device, dtype=flow.dtype),
                indexing='ij',
            )
            xx = xx.unsqueeze(0)
            yy = yy.unsqueeze(0)
            x2 = xx + flow_x
            y2 = yy + flow_y

            valid = (x2 >= 0.0) & (x2 <= (W - 1)) & (y2 >= 0.0) & (y2 <= (H - 1))
            flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y)
            if self.pose_no_prior_vo_max_flow > 0.0:
                valid &= flow_mag <= self.pose_no_prior_vo_max_flow

            grid_x = (2.0 * x2 / max(W - 1, 1)) - 1.0
            grid_y = (2.0 * y2 / max(H - 1, 1)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
            d2_warp = F.grid_sample(
                d2.unsqueeze(1),
                grid,
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)

            valid &= torch.isfinite(d1) & torch.isfinite(d2_warp)
            valid &= (d1 > self.pose_no_prior_vo_min_depth) & (d2_warp > self.pose_no_prior_vo_min_depth)
            if self.pose_no_prior_vo_max_depth > self.pose_no_prior_vo_min_depth:
                valid &= (d1 < self.pose_no_prior_vo_max_depth) & (d2_warp < self.pose_no_prior_vo_max_depth)
            if tool_valid_1hw is not None:
                valid &= tool_valid_1hw.bool()

            valid_idx = torch.where(valid.squeeze(0))
            n_valid = int(valid_idx[0].numel())
            if n_valid < self.pose_no_prior_vo_min_points:
                info = {
                    'vo_valid_points': n_valid,
                    'vo_used': False,
                    'vo_reason': 'few_points',
                    'vo_corr_source': 'raft',
                }
                if vo_static_gate_info:
                    info.update(vo_static_gate_info)
                return None, info

            if self.pose_no_prior_vo_max_points > 0 and n_valid > self.pose_no_prior_vo_max_points:
                perm = torch.randperm(n_valid, device=flow.device)[: self.pose_no_prior_vo_max_points]
                ys = valid_idx[0][perm]
                xs = valid_idx[1][perm]
            else:
                ys = valid_idx[0]
                xs = valid_idx[1]

            x1_sel = xs.float()
            y1_sel = ys.float()
            x2_sel = x2.squeeze(0)[ys, xs]
            y2_sel = y2.squeeze(0)[ys, xs]
            d1_sel = d1.squeeze(0)[ys, xs]
            d2_sel = d2_warp.squeeze(0)[ys, xs]
            corr_source_used = 'raft'

        X1 = torch.stack(
            [
                (x1_sel - cx) / fx * d1_sel,
                (y1_sel - cy) / fy * d1_sel,
                d1_sel,
            ],
            dim=-1,
        )
        X2 = torch.stack(
            [
                (x2_sel - cx) / fx * d2_sel,
                (y2_sel - cy) / fy * d2_sel,
                d2_sel,
            ],
            dim=-1,
        )

        solver = self.pose_no_prior_vo_solver
        solver_info = None
        R = None
        t = None
        if solver == 'pnp':
            R, t, solver_info = self._estimate_vo_transform_pnp(X1, x2_sel, y2_sel, fx, fy, cx, cy)
        elif solver == 'essential':
            R, t, solver_info = self._estimate_vo_transform_essential(
                x1_sel, y1_sel, x2_sel, y2_sel, X1, X2, fx, fy, cx, cy
            )
            if R is None:
                R, t, pnp_info = self._estimate_vo_transform_pnp(X1, x2_sel, y2_sel, fx, fy, cx, cy)
                if R is not None:
                    solver_info = {**pnp_info, 'vo_solver_fallback_from': 'essential'}
                else:
                    R, t, svd_info = self._estimate_vo_transform_svd(X1, X2)
                    if R is not None:
                        solver_info = {**svd_info, 'vo_solver_fallback_from': 'essential'}
                    elif solver_info is None:
                        solver_info = svd_info
                    else:
                        solver_info = {
                            **solver_info,
                            'vo_reason_pnp': pnp_info.get('vo_reason', 'unknown'),
                            'vo_reason_svd': svd_info.get('vo_reason', 'unknown'),
                            'vo_reason': 'essential_failed_all',
                        }
        elif solver == 'hybrid':
            R, t, solver_info = self._estimate_vo_transform_pnp(X1, x2_sel, y2_sel, fx, fy, cx, cy)
            if R is None:
                R, t, svd_info = self._estimate_vo_transform_svd(X1, X2)
                if R is not None:
                    solver_info = {**svd_info, 'vo_solver_fallback_from': 'pnp'}
                elif solver_info is None:
                    solver_info = svd_info
                else:
                    solver_info = {
                        **solver_info,
                        'vo_reason_pnp': solver_info.get('vo_reason', 'unknown'),
                        'vo_reason_svd': svd_info.get('vo_reason', 'unknown'),
                        'vo_reason': 'hybrid_failed',
                    }
        else:
            R, t, solver_info = self._estimate_vo_transform_svd(X1, X2)

        if R is None or t is None:
            info = {
                'vo_valid_points': int(X1.shape[0]),
                'vo_used': False,
                'vo_reason': 'solver_failed',
                'vo_solver': solver,
                'vo_inlier_points': 0,
            }
            if solver_info is not None:
                info.update(solver_info)
            if vo_static_gate_info:
                info.update(vo_static_gate_info)
            return None, info

        trace = torch.clamp((torch.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        rot_deg_raw = float(torch.rad2deg(torch.acos(trace)).item())
        trans_norm_raw = float(torch.linalg.norm(t).item())
        if not torch.isfinite(R).all() or not torch.isfinite(t).all():
            info = {
                'vo_valid_points': int(X1.shape[0]),
                'vo_used': False,
                'vo_reason': 'non_finite',
                'vo_solver': solver,
            }
            if vo_static_gate_info:
                info.update(vo_static_gate_info)
            return None, info

        conf_reject, conf_reason, conf_info = self._vo_confidence_reject(
            vo_valid_points=int(X1.shape[0]),
            solver_info=solver_info if isinstance(solver_info, dict) else {},
            trans_norm=trans_norm_raw,
            rot_deg=rot_deg_raw,
        )
        if conf_reject:
            T21 = torch.eye(4, device=R.device, dtype=R.dtype)
            info = {
                'vo_valid_points': int(X1.shape[0]),
                'vo_inlier_points': int((solver_info or {}).get('vo_inlier_points', X1.shape[0])),
                'vo_used': False,
                'vo_reason': conf_reason,
                'vo_trans_norm': 0.0,
                'vo_rot_deg': 0.0,
                'vo_trans_norm_raw': float(trans_norm_raw),
                'vo_rot_deg_raw': float(rot_deg_raw),
                'vo_solver': 'conf_reject_hold',
                'vo_corr_source': corr_source_used if corr_source_used is not None else 'unknown',
                'vo_conf_reject': True,
            }
            if solver_info is not None:
                for k in ('vo_reproj_median_px', 'vo_solver_fallback_from', 'vo_essential_depth_scale'):
                    if k in solver_info:
                        info[k] = solver_info[k]
            if conf_info is not None:
                info.update(conf_info)
            if vo_static_gate_info:
                info.update(vo_static_gate_info)
            return T21, info

        trans_norm = float(trans_norm_raw)
        rot_deg = float(rot_deg_raw)
        if self.pose_no_prior_vo_max_trans > 0.0 and trans_norm > self.pose_no_prior_vo_max_trans:
            t = t * (self.pose_no_prior_vo_max_trans / max(trans_norm, 1e-12))
            trans_norm = float(self.pose_no_prior_vo_max_trans)
        if self.pose_no_prior_vo_max_rot_deg > 0.0 and rot_deg > self.pose_no_prior_vo_max_rot_deg:
            rot = Rotation.from_matrix(R.detach().cpu().numpy())
            rotvec = rot.as_rotvec()
            rotvec = rotvec * (self.pose_no_prior_vo_max_rot_deg / max(rot_deg, 1e-12))
            R = torch.from_numpy(Rotation.from_rotvec(rotvec).as_matrix()).to(device=R.device, dtype=R.dtype)
            rot_deg = float(self.pose_no_prior_vo_max_rot_deg)

        T21 = torch.eye(4, device=R.device, dtype=R.dtype)
        T21[:3, :3] = R
        T21[:3, 3] = t
        info = {
            'vo_valid_points': int(X1.shape[0]),
            'vo_inlier_points': int((solver_info or {}).get('vo_inlier_points', X1.shape[0])),
            'vo_used': True,
            'vo_reason': 'ok',
            'vo_trans_norm': trans_norm,
            'vo_rot_deg': rot_deg,
            'vo_trans_norm_raw': float(trans_norm_raw),
            'vo_rot_deg_raw': float(rot_deg_raw),
            'vo_solver': (solver_info or {}).get('vo_solver', solver),
            'vo_corr_source': corr_source_used if corr_source_used is not None else 'unknown',
            'vo_conf_reject': False,
        }
        if solver_info is not None:
            for k in ('vo_reproj_median_px', 'vo_solver_fallback_from', 'vo_essential_depth_scale'):
                if k in solver_info:
                    info[k] = solver_info[k]
        if conf_info is not None:
            info.update(conf_info)
        if self.pose_no_prior_vo_conf_reject_enabled:
            self.pose_no_prior_vo_accept_trans_hist.append(float(trans_norm))
            self.pose_no_prior_vo_accept_rot_hist.append(float(rot_deg))
        if vo_static_gate_info:
            info.update(vo_static_gate_info)
        return T21, info

    def _init_pose_for_frame(self, gt_c2w: torch.Tensor, frame_id: int, gt_color=None, stereo_depth=None, tool_mask=None):
        if self.pose_init_mode in ('dataset', 'input', 'gt'):
            self.last_pose_init_info = {'pose_init_mode': self.pose_init_mode, 'pose_init_used': 'dataset'}
            return gt_c2w
        if self.pose_init_mode in ('identity', 'no_prior'):
            vo_chain_pose = self.prev_optimized_pose_for_smooth
            vo_chain_source = 'optimized'
            if self.pose_no_prior_vo_chain_source == 'input' and self.prev_input_pose_for_smooth is not None:
                vo_chain_pose = self.prev_input_pose_for_smooth
                vo_chain_source = 'input'
            if self.pose_no_prior_external_init_poses is not None:
                external_pose = self.pose_no_prior_external_init_poses.get(int(frame_id), None)
                if external_pose is not None:
                    external_pose = external_pose.to(device=gt_c2w.device, dtype=gt_c2w.dtype)
                    if self._is_pose_matrix_valid(external_pose):
                        self.last_pose_init_info = {'pose_init_mode': self.pose_init_mode, 'pose_init_used': 'external'}
                        return external_pose
            if (
                frame_id > 0
                and self.pose_no_prior_vo_enabled
                and vo_chain_pose is not None
                and self.last_frame is not None
                and self.prev_depth_for_pose_init is not None
                and gt_color is not None
                and stereo_depth is not None
            ):
                vo_T21, vo_info = self._estimate_relative_cam_transform_from_flow(
                    prev_color=self.last_frame,
                    curr_color=gt_color,
                    prev_depth=self.prev_depth_for_pose_init,
                    curr_depth=stereo_depth,
                    tool_mask=tool_mask,
                    frame_id=frame_id,
                )
                if vo_T21 is not None:
                    prev_pose = vo_chain_pose.detach().clone().to(
                        device=gt_c2w.device,
                        dtype=gt_c2w.dtype,
                    )
                    vo_T21 = vo_T21.to(device=gt_c2w.device, dtype=gt_c2w.dtype).unsqueeze(0)
                    c2w_init = prev_pose @ torch.linalg.inv(vo_T21)
                    if self._is_pose_matrix_valid(c2w_init):
                        pose_init_used = 'vo'
                        if isinstance(vo_info, dict):
                            vo_reason = str(vo_info.get('vo_reason', ''))
                            if vo_reason == 'camera_static_skip' or vo_reason.startswith('vo_conf_reject'):
                                pose_init_used = 'vo_static_skip'
                        self.last_pose_init_info = {
                            'pose_init_mode': self.pose_init_mode,
                            'pose_init_used': pose_init_used,
                            'vo_chain_source': vo_chain_source,
                            **(vo_info or {}),
                        }
                        return c2w_init
            if (
                frame_id > 0
                and self.pose_no_prior_use_prev_optimized
                and self.prev_optimized_pose_for_smooth is not None
            ):
                init_pose = self.prev_optimized_pose_for_smooth.detach().clone().to(
                    device=gt_c2w.device,
                    dtype=gt_c2w.dtype,
                )
                self.last_pose_init_info = {'pose_init_mode': self.pose_init_mode, 'pose_init_used': 'prev_opt'}
                return init_pose
            init_pose = self._identity_pose_like(gt_c2w)
            self.last_pose_init_info = {'pose_init_mode': self.pose_init_mode, 'pose_init_used': 'identity'}
            return init_pose
        warnings.warn(f"Unknown pose_init_mode={self.pose_init_mode}, fallback to dataset pose.")
        self.last_pose_init_info = {'pose_init_mode': self.pose_init_mode, 'pose_init_used': 'dataset_fallback'}
        return gt_c2w

    def _compose_pose(self, base_c2w, pose_delta):
        base_pose = base_c2w.squeeze(0)
        refined = apply_se3_delta(base_pose, pose_delta)
        return refined.unsqueeze(0)

    def _pose_reg_loss(self, pose_delta):
        reg_rot = self.pose_w_rot * torch.sum(pose_delta[:3] ** 2)
        reg_trans = self.pose_w_trans * torch.sum(pose_delta[3:] ** 2)
        return reg_rot + reg_trans, reg_rot, reg_trans

    def _pose_prior_loss(self, pose_delta):
        return torch.sum(pose_delta ** 2)

    def _pose_smooth_loss(self, current_c2w, init_c2w):
        if self.prev_input_pose_for_smooth is None or self.prev_optimized_pose_for_smooth is None:
            return None
        prev_input = self.prev_input_pose_for_smooth.to(device=current_c2w.device, dtype=current_c2w.dtype)
        prev_opt = self.prev_optimized_pose_for_smooth.to(device=current_c2w.device, dtype=current_c2w.dtype)
        if not (
            self._is_pose_matrix_valid(prev_input)
            and self._is_pose_matrix_valid(prev_opt)
            and self._is_pose_matrix_valid(init_c2w)
            and self._is_pose_matrix_valid(current_c2w)
        ):
            return None

        rel_input = torch.linalg.solve(prev_input.squeeze(0), init_c2w.squeeze(0))
        rel_opt = torch.linalg.solve(prev_opt.squeeze(0), current_c2w.squeeze(0))

        trans_loss = torch.mean((rel_opt[:3, 3] - rel_input[:3, 3]) ** 2)
        rel_rot = rel_input[:3, :3].transpose(0, 1) @ rel_opt[:3, :3]
        ident = torch.eye(3, dtype=rel_rot.dtype, device=rel_rot.device)
        rot_loss = torch.mean((rel_rot - ident) ** 2)
        return trans_loss + rot_loss, trans_loss, rot_loss

    def _pose_stage_state(self, iter_idx: int, total_iters: int):
        stage1 = min(max(self.pose_stage1_map_only_iters, 0), max(total_iters - 1, 0))
        stage2 = min(max(self.pose_stage2_iters, 0), max(total_iters - stage1, 0))
        if iter_idx <= stage1:
            return {
                'pose_active': False,
                'lr_scale': 0.0,
                'track_scale': 0.0,
                'prior_scale': 0.0,
                'smooth_scale': 0.0,
                'recon_scale': 1.0,
            }
        if iter_idx <= stage1 + stage2:
            return {
                'pose_active': True,
                'lr_scale': self.pose_stage2_lr_scale,
                'track_scale': self.pose_stage2_track_scale,
                'prior_scale': self.pose_stage2_prior_scale,
                'smooth_scale': self.pose_stage2_smooth_scale,
                'recon_scale': self.pose_stage2_recon_scale,
            }
        return {
            'pose_active': True,
            'lr_scale': self.pose_stage3_lr_scale,
            'track_scale': self.pose_stage3_track_scale,
            'prior_scale': self.pose_stage3_prior_scale,
            'smooth_scale': self.pose_stage3_smooth_scale,
            'recon_scale': self.pose_stage3_recon_scale,
        }

    @staticmethod
    def _is_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor).all().item())

    def _is_pose_matrix_valid(self, c2w: torch.Tensor) -> bool:
        if not self._is_finite(c2w):
            return False
        mat = c2w.squeeze(0)[:3, :3]
        det = torch.det(mat)
        if not torch.isfinite(det):
            return False
        return bool(torch.abs(det).item() > 1e-8)

    @staticmethod
    def _mask_centroid(mask: torch.Tensor):
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            return None
        return torch.stack([xs.float().mean(), ys.float().mean()])

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _frame_time(self, frame_id: int) -> float:
        if self.n_img <= 1:
            return 0.0
        return float(frame_id) / float(self.n_img - 1)

    def _tool_motion_soft_weights(self, score: float):
        denom = max(self.tool_motion_moving_on - self.tool_motion_static_on, 1e-6)
        raw_ratio = self._clip01((score - self.tool_motion_static_on) / denom)
        pose_weight = 1.0 - (raw_ratio ** max(self.tool_motion_soft_power, 1e-6))
        pose_weight = max(self.tool_motion_soft_min_pose_weight, pose_weight)
        pose_weight = self._clip01(pose_weight)
        lock_weight = (raw_ratio ** max(self.tool_motion_soft_lock_power, 1e-6)) * self.tool_motion_soft_lock_max
        lock_weight = self._clip01(lock_weight)
        return pose_weight, lock_weight, raw_ratio

    def _motion_dynamic_scales(self, motion_ratio: float):
        ratio = self._clip01(motion_ratio)
        if not self.tool_motion_dynamic_reg_enabled:
            return {
                'prior_scale': 1.0,
                'smooth_scale': 1.0,
                'track_scale': 1.0,
                'recon_scale': 1.0,
                'lr_scale': 1.0,
            }

        prior_max = max(1.0, self.tool_motion_dynamic_prior_max)
        smooth_max = max(1.0, self.tool_motion_dynamic_smooth_max)
        track_min = self._clip01(self.tool_motion_dynamic_track_min)
        recon_min = self._clip01(self.tool_motion_dynamic_recon_min)
        lr_min = self._clip01(self.tool_motion_dynamic_lr_min)
        return {
            'prior_scale': 1.0 + ratio * (prior_max - 1.0),
            'smooth_scale': 1.0 + ratio * (smooth_max - 1.0),
            'track_scale': track_min + (1.0 - ratio) * (1.0 - track_min),
            'recon_scale': recon_min + (1.0 - ratio) * (1.0 - recon_min),
            'lr_scale': lr_min + (1.0 - ratio) * (1.0 - lr_min),
        }

    def _adaptive_pose_step_limits(self, pose_stats, motion_ratio: float):
        max_trans = float(self.pose_step_max_trans)
        max_rot_deg = float(self.pose_step_max_rot_deg)
        scale = 1.0
        reason = 'disabled'
        if (not self.pose_step_adaptive_enabled) or pose_stats is None:
            return max_trans, max_rot_deg, scale, reason

        init_track_loss = pose_stats.get('init_track_loss', float('nan'))
        if not np.isfinite(init_track_loss) or init_track_loss <= 0.0 or self.pose_step_adaptive_ref_track <= 0.0:
            return max_trans, max_rot_deg, scale, 'invalid_track_loss'

        if float(motion_ratio) > self.pose_step_adaptive_motion_ratio_max:
            return max_trans, max_rot_deg, scale, 'motion_ratio_high'

        if self.last_pose_init_info is not None and self.last_pose_init_info.get('pose_init_used') == 'vo':
            vo_inliers = int(self.last_pose_init_info.get('vo_inlier_points', 0))
            if vo_inliers < self.pose_step_adaptive_min_vo_inliers:
                return max_trans, max_rot_deg, scale, 'few_vo_inliers'

        raw_scale = float(init_track_loss) / max(self.pose_step_adaptive_ref_track, 1e-9)
        scale = float(np.clip(raw_scale, 1.0, max(self.pose_step_adaptive_max_scale, 1.0)))
        max_trans *= scale
        max_rot_deg *= scale
        return max_trans, max_rot_deg, scale, 'track_loss'

    def _blend_pose_towards_reference(self, base_c2w, ref_c2w, weight: float, translation_only: bool = False):
        w = self._clip01(weight)
        if w <= 0.0:
            return base_c2w
        if w >= 1.0:
            return ref_c2w.detach().clone()
        if (not self._is_pose_matrix_valid(base_c2w)) or (not self._is_pose_matrix_valid(ref_c2w)):
            return base_c2w

        base_np = base_c2w.squeeze(0).detach().cpu().numpy()
        ref_np = ref_c2w.squeeze(0).detach().cpu().numpy()

        t_new = (1.0 - w) * base_np[:3, 3] + w * ref_np[:3, 3]
        if translation_only:
            rot_new = base_np[:3, :3]
        else:
            rel_rot = Rotation.from_matrix(base_np[:3, :3]).inv() * Rotation.from_matrix(ref_np[:3, :3])
            rotvec = rel_rot.as_rotvec() * w
            rot_new = (Rotation.from_matrix(base_np[:3, :3]) * Rotation.from_rotvec(rotvec)).as_matrix()

        blended = base_np.copy()
        blended[:3, :3] = rot_new
        blended[:3, 3] = t_new
        return torch.from_numpy(blended).to(device=base_c2w.device, dtype=base_c2w.dtype).unsqueeze(0)

    def _extract_tool_semantic_mask(self, semantics):
        if semantics is None or semantics.ndim < 4:
            return None
        if self.tool_semantic_channel < 0 or self.tool_semantic_channel >= semantics.shape[-1]:
            return None
        tool_prob = semantics[..., self.tool_semantic_channel]
        return tool_prob > self.tool_semantic_threshold

    def _resolve_motion_mask(self, tool_mask, semantics):
        if (not self.tool_motion_use_semantic_tool_mask) or semantics is None:
            return tool_mask
        sem_tool = self._extract_tool_semantic_mask(semantics)
        if sem_tool is None:
            return tool_mask
        if tool_mask is not None:
            sem_tool = sem_tool & tool_mask
        if int(sem_tool.sum().item()) < self.tool_motion_min_pixels:
            return tool_mask
        return sem_tool

    def _build_pose_loss_mask(self, tool_mask, semantics, motion_info):
        if tool_mask is None or (not self.tool_motion_use_bg_for_pose_loss):
            return tool_mask
        motion_ratio = float(motion_info.get('tool_motion_ratio', 0.0))
        if motion_ratio < self.tool_motion_bg_switch_ratio:
            return tool_mask
        sem_tool = self._extract_tool_semantic_mask(semantics)
        if sem_tool is None:
            return tool_mask
        bg_mask = tool_mask & (~sem_tool)
        if int(bg_mask.sum().item()) < self.tool_motion_bg_min_pixels:
            return tool_mask
        return bg_mask

    @staticmethod
    def _safe_loss_mask(preferred_mask, fallback_ref):
        if preferred_mask is not None and int(preferred_mask.sum().item()) > 0:
            return preferred_mask
        return torch.ones_like(fallback_ref, dtype=torch.bool)

    @staticmethod
    def _to_bchw_image(image: torch.Tensor):
        if image is None:
            return None
        if image.ndim != 4:
            return None
        if image.shape[-1] in (1, 3, 4):
            return image.permute(0, 3, 1, 2).contiguous()
        if image.shape[1] in (1, 3, 4):
            return image.contiguous()
        return None

    def _compute_rgb_quality(self, gt_color: torch.Tensor, pred_color: torch.Tensor, eval_mask: torch.Tensor = None):
        psnr_val = float('nan')
        ssim_val = float('nan')
        if gt_color is None or pred_color is None:
            return psnr_val, ssim_val

        if gt_color.ndim == 3:
            gt_color = gt_color.unsqueeze(0)
        if pred_color.ndim == 3:
            pred_color = pred_color.unsqueeze(0)
        if gt_color.shape != pred_color.shape:
            return psnr_val, ssim_val

        use_mask = None
        if eval_mask is not None:
            use_mask = eval_mask
            if use_mask.ndim == 4:
                use_mask = use_mask[:, 0]
            if use_mask.ndim == 2:
                use_mask = use_mask.unsqueeze(0)
            if use_mask.ndim != 3 or use_mask.shape != gt_color.shape[:3]:
                use_mask = None
            elif int(use_mask.sum().item()) <= 0:
                use_mask = None

        if use_mask is not None:
            diff = pred_color[use_mask] - gt_color[use_mask]
        else:
            diff = pred_color.reshape(-1, pred_color.shape[-1]) - gt_color.reshape(-1, gt_color.shape[-1])
        if diff.numel() > 0:
            mse = torch.mean(diff * diff)
            if torch.isfinite(mse):
                psnr_val = float((-10.0 * torch.log10(torch.clamp(mse, min=1e-8))).item())

        gt_bchw = self._to_bchw_image(gt_color)
        pred_bchw = self._to_bchw_image(pred_color)
        if gt_bchw is not None and pred_bchw is not None and gt_bchw.shape == pred_bchw.shape:
            try:
                ssim_tensor = ssim(pred_bchw, gt_bchw)
                if torch.isfinite(ssim_tensor):
                    ssim_val = float(ssim_tensor.item())
            except Exception:
                pass
        return psnr_val, ssim_val

    def _maybe_apply_first_frame_pose_trust_gate(
        self,
        frame_id: int,
        gt_color: torch.Tensor,
        stereo_depth: torch.Tensor,
        input_c2w: torch.Tensor,
        optimized_c2w: torch.Tensor,
        pose_mask: torch.Tensor = None,
    ):
        if frame_id != 0:
            return
        if not self.first_frame_pose_trust_gate_enabled:
            return
        if isinstance(self.first_frame_pose_trust_gate_info, dict) and bool(
            self.first_frame_pose_trust_gate_info.get('ff_pose_gate_evaluated', False)
        ):
            return
        if gt_color is None or stereo_depth is None:
            return
        if not self._is_pose_matrix_valid(input_c2w) or not self._is_pose_matrix_valid(optimized_c2w):
            return

        eval_mask = self._safe_loss_mask(pose_mask, stereo_depth)
        init_psnr = float('nan')
        init_ssim = float('nan')
        final_psnr = float('nan')
        final_ssim = float('nan')

        with torch.no_grad():
            self.camera.set_c2w(input_c2w)
            init_render = self._render_scene(deform=True)['render'][None, ...]
            init_psnr, init_ssim = self._compute_rgb_quality(gt_color, init_render, eval_mask)
            self.camera.set_c2w(optimized_c2w)
            final_render = self._render_scene(deform=True)['render'][None, ...]
            final_psnr, final_ssim = self._compute_rgb_quality(gt_color, final_render, eval_mask)

        fail_reasons = []
        if np.isfinite(final_psnr) and self.first_frame_pose_trust_gate_min_psnr > 0.0:
            if final_psnr < self.first_frame_pose_trust_gate_min_psnr:
                fail_reasons.append('psnr_below_min')
        if np.isfinite(final_ssim) and self.first_frame_pose_trust_gate_min_ssim > 0.0:
            if final_ssim < self.first_frame_pose_trust_gate_min_ssim:
                fail_reasons.append('ssim_below_min')
        if (
            np.isfinite(init_psnr)
            and np.isfinite(final_psnr)
            and self.first_frame_pose_trust_gate_max_psnr_drop > 0.0
            and (init_psnr - final_psnr) > self.first_frame_pose_trust_gate_max_psnr_drop
        ):
            fail_reasons.append('psnr_drop_too_large')
        if (
            np.isfinite(init_ssim)
            and np.isfinite(final_ssim)
            and self.first_frame_pose_trust_gate_max_ssim_drop > 0.0
            and (init_ssim - final_ssim) > self.first_frame_pose_trust_gate_max_ssim_drop
        ):
            fail_reasons.append('ssim_drop_too_large')

        triggered = len(fail_reasons) > 0
        fallback_applied = False
        if triggered and self.first_frame_pose_trust_gate_fallback_mode == 'no_prior':
            self.pose_init_mode = 'no_prior'
            self.pose_no_prior_vo_chain_source = self.first_frame_pose_trust_gate_chain_source
            if self.first_frame_pose_trust_gate_enable_no_prior_vo:
                self.pose_no_prior_vo_enabled = True
            if self.first_frame_pose_trust_gate_fallback_w_pose_prior >= 0.0:
                self.pose_w_prior = float(self.first_frame_pose_trust_gate_fallback_w_pose_prior)
            fallback_applied = True

        self.first_frame_pose_trust_gate_info = {
            'ff_pose_gate_evaluated': True,
            'ff_pose_gate_triggered': bool(triggered),
            'ff_pose_gate_fallback_applied': bool(fallback_applied),
            'ff_pose_gate_fallback_mode': self.first_frame_pose_trust_gate_fallback_mode,
            'ff_pose_gate_init_psnr': float(init_psnr),
            'ff_pose_gate_init_ssim': float(init_ssim),
            'ff_pose_gate_final_psnr': float(final_psnr),
            'ff_pose_gate_final_ssim': float(final_ssim),
            'ff_pose_gate_psnr_drop': float(init_psnr - final_psnr)
            if np.isfinite(init_psnr) and np.isfinite(final_psnr)
            else float('nan'),
            'ff_pose_gate_ssim_drop': float(init_ssim - final_ssim)
            if np.isfinite(init_ssim) and np.isfinite(final_ssim)
            else float('nan'),
            'ff_pose_gate_reasons': '|'.join(fail_reasons) if triggered else 'pass',
            'ff_pose_gate_pose_init_mode_after': str(self.pose_init_mode),
            'ff_pose_gate_vo_enabled_after': bool(self.pose_no_prior_vo_enabled),
            'ff_pose_gate_vo_chain_after': str(self.pose_no_prior_vo_chain_source),
        }
        if self.dbg:
            print(
                "[PoseTrustGate] "
                f"triggered={bool(triggered)} reasons={self.first_frame_pose_trust_gate_info['ff_pose_gate_reasons']} "
                f"init_psnr={init_psnr:.4f} final_psnr={final_psnr:.4f} "
                f"init_ssim={init_ssim:.4f} final_ssim={final_ssim:.4f} "
                f"pose_init_mode={self.pose_init_mode}"
            )

    def _track_points_with_raft2d(self, prev_color, curr_color, pts_2d):
        if prev_color is None or curr_color is None or pts_2d is None:
            return pts_2d
        if pts_2d.numel() == 0:
            return pts_2d
        with torch.no_grad():
            flow = self.raft(
                2 * prev_color.permute(0, 3, 1, 2) - 1.0,
                2 * curr_color.permute(0, 3, 1, 2) - 1.0,
            )[-1]
        _, _, H, W = flow.shape
        pts_base = pts_2d.to(device=flow.device, dtype=flow.dtype)
        x = pts_base[:, 0].clamp(0, W - 1)
        y = pts_base[:, 1].clamp(0, H - 1)
        gx = 2.0 * x / max(W - 1, 1) - 1.0
        gy = 2.0 * y / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        delta = F.grid_sample(
            flow,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        delta = delta[0, :, :, 0].permute(1, 0)
        pts_next = pts_base + delta
        pts_next[:, 0].clamp_(0, W - 1)
        pts_next[:, 1].clamp_(0, H - 1)
        return pts_next

    def _prepare_cotracker_tracks(self):
        if self.pt_tracker is None:
            return False
        if self.pt_cotracker_tracks is not None:
            return True
        if getattr(self.frame_reader, 'color_paths', None) is None:
            return False
        if len(self.frame_reader.color_paths) == 0:
            return False
        try:
            frames = []
            for p in self.frame_reader.color_paths:
                im = cv2.imread(p)
                if im is None:
                    continue
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if len(frames) == 0:
                return False
            video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)[None, ...].float().to(self.device)
            q_xy = self.pt_tracker.gt_2d_pts[:, 0].detach().to(device=self.device, dtype=torch.float32)
            q_t = torch.zeros((q_xy.shape[0], 1), device=self.device, dtype=torch.float32)
            queries = torch.cat([q_t, q_xy], dim=1)[None, ...]
            cotracker = torch.hub.load(
                self.pt_cotracker_repo,
                self.pt_cotracker_model,
                source='local',
            ).to(self.device).eval()
            with torch.no_grad():
                model_name = self.pt_cotracker_model.lower()
                if "online" in model_name:
                    step = int(getattr(cotracker, "step", 8))
                    init_len = max(1, min(video.shape[1], step * 2))
                    cotracker(
                        video_chunk=video[:, :init_len],
                        is_first_step=True,
                        queries=queries,
                    )
                    tracks, vis = None, None
                    ran = False
                    for ind in range(0, max(video.shape[1] - step, 1), step):
                        ran = True
                        chunk = video[:, ind:min(video.shape[1], ind + step * 2)]
                        tracks, vis = cotracker(video_chunk=chunk, is_first_step=False)
                    if (not ran) or tracks is None:
                        tracks, vis = cotracker(video_chunk=video, is_first_step=False)
                else:
                    tracks, vis = cotracker(video, queries=queries)
            if tracks is None:
                return False
            self.pt_cotracker_tracks = tracks[0].detach()
            self.pt_cotracker_vis = vis[0].detach() if vis is not None else None
            return True
        except Exception as exc:
            warnings.warn(f"CoTracker init failed, fallback to gaussian3d tracker eval. reason: {exc}")
            self.pt_tracker_backend = 'gaussian3d'
            self.pt_cotracker_tracks = None
            self.pt_cotracker_vis = None
            return False

    def _build_cotracker_anchor_queries(self, ref_pose: torch.Tensor, query_time: int = 0):
        if ref_pose is None:
            return None, {'deform_init_query_reason': 'no_ref_pose'}

        means = self.net.get_xyz.detach()
        deform = getattr(self.net, "_deformation", None)
        anchor_ids = getattr(deform, "anchor_ids", None) if deform is not None else None
        if anchor_ids is not None and isinstance(anchor_ids, torch.Tensor) and int(anchor_ids.numel()) > 0:
            ids = anchor_ids.detach().long().to(device=means.device)
            ids = ids[(ids >= 0) & (ids < means.shape[0])]
            if int(ids.numel()) > 0:
                pts_world = means[ids]
            else:
                pts_world = means
        else:
            pts_world = means

        if pts_world.ndim != 2 or pts_world.shape[0] == 0:
            return None, {'deform_init_query_reason': 'no_anchor_points'}

        ref_pose = ref_pose.squeeze(0) if ref_pose.ndim == 3 else ref_pose
        if ref_pose.ndim != 2 or ref_pose.shape[0] != 4:
            return None, {'deform_init_query_reason': 'bad_ref_pose_shape'}

        ref_pose = ref_pose.to(device=pts_world.device, dtype=pts_world.dtype)
        w2c = torch.linalg.inv(ref_pose)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pts_cam = (pts_world @ R.transpose(0, 1)) + t.unsqueeze(0)

        H, W, fx, fy, cx, cy = self.camera.get_params()
        z = pts_cam[:, 2]
        valid = torch.isfinite(pts_cam).all(dim=-1)
        valid &= z > float(self.cotracker_flow_init_query_min_depth)

        x = fx * pts_cam[:, 0] / z + cx
        y = fy * pts_cam[:, 1] / z + cy
        valid &= torch.isfinite(x) & torch.isfinite(y)
        valid &= (x >= 0.0) & (x <= (W - 1)) & (y >= 0.0) & (y <= (H - 1))

        valid_idx = torch.where(valid)[0]
        n_valid = int(valid_idx.numel())
        if n_valid < 1:
            return None, {'deform_init_query_reason': 'no_valid_projected_anchor'}

        if self.cotracker_flow_init_query_max_points > 0 and n_valid > self.cotracker_flow_init_query_max_points:
            if self.cotracker_flow_init_query_grid_cell > 0:
                xy = torch.stack([x[valid_idx], y[valid_idx]], dim=-1).detach().cpu().numpy()
                cell = max(1, int(self.cotracker_flow_init_query_grid_cell))
                grid_ids = np.floor(xy / float(cell)).astype(np.int64)
                _, inv = np.unique(grid_ids, axis=0, return_inverse=True)
                selected_local = []
                for gid in range(int(inv.max()) + 1):
                    members = np.where(inv == gid)[0]
                    if members.size == 0:
                        continue
                    selected_local.append(int(np.random.choice(members)))
                selected_local = np.asarray(selected_local, dtype=np.int64)
                if selected_local.size == 0:
                    perm = torch.randperm(n_valid, device=valid_idx.device)[: self.cotracker_flow_init_query_max_points]
                    valid_idx = valid_idx[perm]
                else:
                    if selected_local.size > self.cotracker_flow_init_query_max_points:
                        selected_local = np.random.permutation(selected_local)[: self.cotracker_flow_init_query_max_points]
                    selected = valid_idx[torch.from_numpy(selected_local).to(device=valid_idx.device, dtype=torch.long)]
                    if int(selected.shape[0]) < self.cotracker_flow_init_query_max_points:
                        selected_set = set(selected.detach().cpu().tolist())
                        remain = [int(idx.item()) for idx in valid_idx if int(idx.item()) not in selected_set]
                        if len(remain) > 0:
                            add_num = min(self.cotracker_flow_init_query_max_points - int(selected.shape[0]), len(remain))
                            extra = np.random.permutation(np.asarray(remain, dtype=np.int64))[:add_num]
                            extra_t = torch.from_numpy(extra).to(device=valid_idx.device, dtype=torch.long)
                            selected = torch.cat([selected, extra_t], dim=0)
                    valid_idx = selected
            else:
                perm = torch.randperm(n_valid, device=valid_idx.device)[: self.cotracker_flow_init_query_max_points]
                valid_idx = valid_idx[perm]

        q_xy = torch.stack([x[valid_idx], y[valid_idx]], dim=-1).to(device=self.device, dtype=torch.float32)
        if int(q_xy.shape[0]) == 0:
            return None, {'deform_init_query_reason': 'empty_after_sampling'}
        q_t = torch.full((q_xy.shape[0], 1), float(max(query_time, 0)), device=self.device, dtype=torch.float32)
        queries = torch.cat([q_t, q_xy], dim=1)[None, ...]
        return queries, {
            'deform_init_query_reason': 'ok',
            'deform_init_query_points': int(q_xy.shape[0]),
            'deform_init_query_time': int(max(query_time, 0)),
        }

    def _prepare_cotracker_tracks_deform_anchor(self, ref_pose: torch.Tensor, query_time: int = 0):
        if self.pt_cotracker_tracks_deform is not None:
            return True
        if getattr(self.frame_reader, 'color_paths', None) is None:
            return False
        if len(self.frame_reader.color_paths) == 0:
            return False
        try:
            frames = []
            for p in self.frame_reader.color_paths:
                im = cv2.imread(p)
                if im is None:
                    continue
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if len(frames) == 0:
                return False
            video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)[None, ...].float().to(self.device)
            queries, _ = self._build_cotracker_anchor_queries(ref_pose=ref_pose, query_time=query_time)
            if queries is None:
                return False
            cotracker = torch.hub.load(
                self.pt_cotracker_repo,
                self.pt_cotracker_model,
                source='local',
            ).to(self.device).eval()
            with torch.no_grad():
                model_name = self.pt_cotracker_model.lower()
                if "online" in model_name:
                    step = int(getattr(cotracker, "step", 8))
                    init_len = max(1, min(video.shape[1], step * 2))
                    cotracker(
                        video_chunk=video[:, :init_len],
                        is_first_step=True,
                        queries=queries,
                    )
                    tracks, vis = None, None
                    ran = False
                    for ind in range(0, max(video.shape[1] - step, 1), step):
                        ran = True
                        chunk = video[:, ind:min(video.shape[1], ind + step * 2)]
                        tracks, vis = cotracker(video_chunk=chunk, is_first_step=False)
                    if (not ran) or tracks is None:
                        tracks, vis = cotracker(video_chunk=video, is_first_step=False)
                else:
                    tracks, vis = cotracker(video, queries=queries)
            if tracks is None:
                return False
            self.pt_cotracker_tracks_deform = tracks[0].detach()
            self.pt_cotracker_vis_deform = vis[0].detach() if vis is not None else None
            return True
        except Exception as exc:
            warnings.warn(f"CoTracker anchor-query init failed. reason: {exc}")
            self.pt_cotracker_tracks_deform = None
            self.pt_cotracker_vis_deform = None
            return False

    def _apply_sparse_deformation_init_from_world_motion(
        self,
        anchor_pts_world: torch.Tensor,
        motion_world: torch.Tensor,
        source: str,
        frame_id: int = None,
    ):
        if anchor_pts_world is None or motion_world is None:
            return False, {'deform_init_used': False, 'deform_init_source': source, 'deform_init_reason': 'none'}
        if anchor_pts_world.ndim != 2 or motion_world.ndim != 2:
            return False, {'deform_init_used': False, 'deform_init_source': source, 'deform_init_reason': 'bad_shape'}
        if anchor_pts_world.shape[0] < 3 or motion_world.shape[0] < 3:
            return False, {
                'deform_init_used': False,
                'deform_init_source': source,
                'deform_init_reason': 'few_points',
                'deform_init_points': int(anchor_pts_world.shape[0]),
            }

        finite = (
            torch.isfinite(anchor_pts_world).all(dim=-1)
            & torch.isfinite(motion_world).all(dim=-1)
        )
        if int(finite.sum().item()) < 3:
            return False, {
                'deform_init_used': False,
                'deform_init_source': source,
                'deform_init_reason': 'non_finite',
                'deform_init_points': int(finite.sum().item()),
            }

        anchor_pts_world = anchor_pts_world[finite]
        motion_world = motion_world[finite]
        if anchor_pts_world.shape[0] < 3:
            return False, {
                'deform_init_used': False,
                'deform_init_source': source,
                'deform_init_reason': 'few_finite_points',
                'deform_init_points': int(anchor_pts_world.shape[0]),
            }

        anchor_np = anchor_pts_world.detach().cpu().numpy()
        gauss_np = self.net._deformation.get_deformed_means(self.net.get_xyz).detach().cpu().numpy()
        k = int(max(1, min(3, anchor_np.shape[0])))
        tree = KDTree(anchor_np)
        neighbour_dists, neighbours = tree.query(gauss_np, k=k)
        if k == 1:
            neighbour_dists = neighbour_dists[:, None]
            neighbours = neighbours[:, None]

        dev = motion_world.device
        dtype = motion_world.dtype
        weights = torch.exp(-50.0 * torch.from_numpy(neighbour_dists).to(device=dev, dtype=dtype))
        neighbours_t = torch.from_numpy(neighbours).to(device=dev, dtype=torch.long)
        deformation = motion_world[neighbours_t]
        self.net._deformation.init_from_flow(deformation.clamp(-0.01, 0.01), weights)

        info = {
            'deform_init_used': True,
            'deform_init_source': source,
            'deform_init_reason': 'ok',
            'deform_init_points': int(anchor_pts_world.shape[0]),
            'deform_init_k': int(k),
        }
        if frame_id is not None:
            info['deform_init_frame'] = int(frame_id)
        return True, info

    def _init_deformation_from_cotracker(
        self,
        frame_id: int,
        frame_pose: torch.Tensor,
        stereo_depth: torch.Tensor,
        tool_mask: torch.Tensor = None,
    ):
        if frame_id <= 0:
            return False, {'deform_init_used': False, 'deform_init_source': 'cotracker3', 'deform_init_reason': 'first_frame'}
        if self.prev_depth_for_pose_init is None:
            return False, {'deform_init_used': False, 'deform_init_source': 'cotracker3', 'deform_init_reason': 'no_prev_depth'}
        if self.prev_optimized_pose_for_smooth is None:
            return False, {'deform_init_used': False, 'deform_init_source': 'cotracker3', 'deform_init_reason': 'no_prev_pose'}
        tracks = None
        vis_tracks = None
        query_source = self.cotracker_flow_init_query_source
        if query_source == 'anchor':
            ref_pose = self.cotracker_deform_query_pose0
            if ref_pose is None:
                ref_pose = self.prev_optimized_pose_for_smooth
            if self.pt_cotracker_tracks_deform is None and (not self._prepare_cotracker_tracks_deform_anchor(ref_pose=ref_pose, query_time=0)):
                return False, {
                    'deform_init_used': False,
                    'deform_init_source': 'cotracker3',
                    'deform_init_query_source': 'anchor',
                    'deform_init_reason': 'no_tracks_anchor',
                }
            tracks = self.pt_cotracker_tracks_deform
            vis_tracks = self.pt_cotracker_vis_deform
        else:
            if self.pt_cotracker_tracks is None and (not self._prepare_cotracker_tracks()):
                return False, {
                    'deform_init_used': False,
                    'deform_init_source': 'cotracker3',
                    'deform_init_query_source': 'gt',
                    'deform_init_reason': 'no_tracks',
                }
            tracks = self.pt_cotracker_tracks
            vis_tracks = self.pt_cotracker_vis

        if tracks is None:
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'cotracker3',
                'deform_init_query_source': query_source,
                'deform_init_reason': 'track_init_failed',
            }
        if frame_id >= int(tracks.shape[0]):
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'cotracker3',
                'deform_init_query_source': query_source,
                'deform_init_reason': 'frame_oob',
            }

        H, W, fx, fy, cx, cy = self.camera.get_params()
        depth_prev = self._to_1hw(self.prev_depth_for_pose_init)
        depth_curr = self._to_1hw(stereo_depth)
        if depth_prev is None or depth_curr is None:
            return False, {'deform_init_used': False, 'deform_init_source': 'cotracker3', 'deform_init_reason': 'bad_depth_shape'}

        depth_prev = depth_prev.to(device=frame_pose.device, dtype=frame_pose.dtype)
        depth_curr = depth_curr.to(device=frame_pose.device, dtype=frame_pose.dtype)

        pts_prev = tracks[frame_id - 1].to(device=frame_pose.device, dtype=frame_pose.dtype)
        pts_curr = tracks[frame_id].to(device=frame_pose.device, dtype=frame_pose.dtype)
        valid = (
            torch.isfinite(pts_prev[:, 0])
            & torch.isfinite(pts_prev[:, 1])
            & torch.isfinite(pts_curr[:, 0])
            & torch.isfinite(pts_curr[:, 1])
        )
        if vis_tracks is not None and frame_id < int(vis_tracks.shape[0]):
            vis_prev = vis_tracks[frame_id - 1].to(device=frame_pose.device, dtype=torch.bool)
            vis_curr = vis_tracks[frame_id].to(device=frame_pose.device, dtype=torch.bool)
            valid &= vis_prev & vis_curr

        x1 = pts_prev[:, 0]
        y1 = pts_prev[:, 1]
        x2 = pts_curr[:, 0]
        y2 = pts_curr[:, 1]
        valid &= (x1 >= 0.0) & (x1 <= (W - 1)) & (y1 >= 0.0) & (y1 <= (H - 1))
        valid &= (x2 >= 0.0) & (x2 <= (W - 1)) & (y2 >= 0.0) & (y2 <= (H - 1))

        if self.pose_no_prior_vo_max_flow > 0.0:
            flow_mag = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            valid &= flow_mag <= self.pose_no_prior_vo_max_flow

        d1 = self._sample_points_1hw(depth_prev, x1, y1, H, W)
        d2 = self._sample_points_1hw(depth_curr, x2, y2, H, W)
        valid &= torch.isfinite(d1) & torch.isfinite(d2)
        valid &= (d1 > self.pose_no_prior_vo_min_depth) & (d2 > self.pose_no_prior_vo_min_depth)
        if self.pose_no_prior_vo_max_depth > self.pose_no_prior_vo_min_depth:
            valid &= (d1 < self.pose_no_prior_vo_max_depth) & (d2 < self.pose_no_prior_vo_max_depth)

        if tool_mask is not None and self.cotracker_flow_init_use_tool_mask:
            tool_valid = self._to_1hw(tool_mask)
            if tool_valid is not None:
                tool_valid = tool_valid.to(device=frame_pose.device, dtype=frame_pose.dtype)
                tool_vals = self._sample_points_1hw(tool_valid, x2, y2, H, W)
                valid &= tool_vals > 0.5

        valid_idx = torch.where(valid)[0]
        n_valid = int(valid_idx.numel())
        if n_valid < self.cotracker_flow_init_min_valid_points:
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'cotracker3',
                'deform_init_query_source': query_source,
                'deform_init_reason': 'few_points',
                'deform_init_points': int(n_valid),
            }

        if self.cotracker_flow_init_max_points > 0 and n_valid > self.cotracker_flow_init_max_points:
            perm = torch.randperm(n_valid, device=frame_pose.device)[: self.cotracker_flow_init_max_points]
            valid_idx = valid_idx[perm]

        x1 = x1[valid_idx]
        y1 = y1[valid_idx]
        x2 = x2[valid_idx]
        y2 = y2[valid_idx]
        d1 = d1[valid_idx]
        d2 = d2[valid_idx]

        X1_cam = torch.stack(
            [
                (x1 - cx) / fx * d1,
                (y1 - cy) / fy * d1,
                d1,
            ],
            dim=-1,
        )
        X2_cam = torch.stack(
            [
                (x2 - cx) / fx * d2,
                (y2 - cy) / fy * d2,
                d2,
            ],
            dim=-1,
        )

        prev_pose = self.prev_optimized_pose_for_smooth.squeeze(0).to(device=frame_pose.device, dtype=frame_pose.dtype)
        curr_pose = frame_pose.squeeze(0).to(device=frame_pose.device, dtype=frame_pose.dtype)
        R1, t1 = prev_pose[:3, :3], prev_pose[:3, 3]
        R2, t2 = curr_pose[:3, :3], curr_pose[:3, 3]
        X1_world = (X1_cam @ R1.transpose(0, 1)) + t1.unsqueeze(0)
        X2_world = (X2_cam @ R2.transpose(0, 1)) + t2.unsqueeze(0)
        motion_world = X2_world - X1_world

        used, info = self._apply_sparse_deformation_init_from_world_motion(
            anchor_pts_world=X1_world,
            motion_world=motion_world,
            source='cotracker3',
            frame_id=frame_id,
        )
        if not used:
            info['deform_init_query_source'] = query_source
            return False, info
        info['deform_init_query_source'] = query_source
        info['deform_init_points_raw'] = int(n_valid)
        return True, info

    def _init_deformation_from_raft(
        self,
        frame_id: int,
        gt_color: torch.Tensor,
        stereo_depth: torch.Tensor,
        tool_mask: torch.Tensor,
        flow_valid: torch.Tensor,
        render_pkg: dict,
    ):
        scene_flow, anchor_pts, valid = get_scene_flow(
            self.raft,
            render_pkg['render'][None, ...],
            gt_color,
            render_pkg['depth'][None, ...],
            stereo_depth,
            tool_mask,
            self.camera,
        )
        valid &= tool_mask.squeeze(0) & flow_valid.squeeze(0)
        anchor_valid = anchor_pts[valid].detach().cpu().numpy()
        use_flow_init = anchor_valid.shape[0] >= 3
        skip_reason = None
        if self.optical_flow_init_guard_enabled:
            if anchor_valid.shape[0] < self.optical_flow_init_min_valid_points:
                use_flow_init = False
                skip_reason = 'few_points'
            elif not np.isfinite(anchor_valid).all():
                use_flow_init = False
                skip_reason = 'non_finite'
            else:
                extent = np.ptp(anchor_valid, axis=0)
                max_extent = float(np.max(extent))
                if max_extent < self.optical_flow_init_min_extent:
                    use_flow_init = False
                    skip_reason = 'collapsed_extent'
                else:
                    sample = anchor_valid
                    if (
                        self.optical_flow_init_unique_sample > 0
                        and sample.shape[0] > self.optical_flow_init_unique_sample
                    ):
                        idx = np.random.choice(
                            sample.shape[0],
                            self.optical_flow_init_unique_sample,
                            replace=False,
                        )
                        sample = sample[idx]
                    sample_q = np.round(sample, decimals=self.optical_flow_init_unique_round)
                    unique_ratio = float(np.unique(sample_q, axis=0).shape[0]) / float(sample_q.shape[0])
                    if unique_ratio < self.optical_flow_init_unique_ratio_min:
                        use_flow_init = False
                        skip_reason = 'low_unique_ratio'

        if not use_flow_init:
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'raft',
                'deform_init_reason': skip_reason if skip_reason is not None else 'guard_rejected',
                'deform_init_points': int(anchor_valid.shape[0]),
                'deform_init_frame': int(frame_id),
            }

        anchor_pts_world = anchor_pts[valid]
        motion_world = scene_flow[valid]
        used, info = self._apply_sparse_deformation_init_from_world_motion(
            anchor_pts_world=anchor_pts_world,
            motion_world=motion_world,
            source='raft',
            frame_id=frame_id,
        )
        if not used:
            return False, info
        info['deform_init_points_raw'] = int(anchor_valid.shape[0])
        return True, info

    def _init_deformation_from_foundation(
        self,
        frame_id: int,
        gt_color: torch.Tensor,
        stereo_depth: torch.Tensor,
        tool_mask: torch.Tensor,
        flow_valid: torch.Tensor,
        render_pkg: dict,
    ):
        if self.foundation_model is None or self.foundation_args is None:
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'foundation',
                'deform_init_reason': 'foundation_model_unavailable',
                'deform_init_points': 0,
                'deform_init_frame': int(frame_id),
            }

        scene_flow, anchor_pts, valid = get_scene_flow_from_foundation(
            self.foundation_model,
            render_pkg['render'][None, ...],
            gt_color,
            render_pkg['depth'][None, ...],
            stereo_depth,
            tool_mask,
            self.camera,
            self.foundation_args,
            gt_color.device,
        )
        valid &= tool_mask.squeeze(0) & flow_valid.squeeze(0)
        anchor_valid = anchor_pts[valid].detach().cpu().numpy()
        use_flow_init = anchor_valid.shape[0] >= 3
        skip_reason = None
        if self.optical_flow_init_guard_enabled:
            if anchor_valid.shape[0] < self.optical_flow_init_min_valid_points:
                use_flow_init = False
                skip_reason = 'few_points'
            elif not np.isfinite(anchor_valid).all():
                use_flow_init = False
                skip_reason = 'non_finite'
            else:
                extent = np.ptp(anchor_valid, axis=0)
                max_extent = float(np.max(extent))
                if max_extent < self.optical_flow_init_min_extent:
                    use_flow_init = False
                    skip_reason = 'collapsed_extent'
                else:
                    sample = anchor_valid
                    if (
                        self.optical_flow_init_unique_sample > 0
                        and sample.shape[0] > self.optical_flow_init_unique_sample
                    ):
                        idx = np.random.choice(
                            sample.shape[0],
                            self.optical_flow_init_unique_sample,
                            replace=False,
                        )
                        sample = sample[idx]
                    sample_q = np.round(sample, decimals=self.optical_flow_init_unique_round)
                    unique_ratio = float(np.unique(sample_q, axis=0).shape[0]) / float(sample_q.shape[0])
                    if unique_ratio < self.optical_flow_init_unique_ratio_min:
                        use_flow_init = False
                        skip_reason = 'low_unique_ratio'

        if not use_flow_init:
            return False, {
                'deform_init_used': False,
                'deform_init_source': 'foundation',
                'deform_init_reason': skip_reason if skip_reason is not None else 'guard_rejected',
                'deform_init_points': int(anchor_valid.shape[0]),
                'deform_init_frame': int(frame_id),
            }

        anchor_pts_world = anchor_pts[valid]
        motion_world = scene_flow[valid]
        used, info = self._apply_sparse_deformation_init_from_world_motion(
            anchor_pts_world=anchor_pts_world,
            motion_world=motion_world,
            source='foundation',
            frame_id=frame_id,
        )
        if not used:
            return False, info
        info['deform_init_points_raw'] = int(anchor_valid.shape[0])
        return True, info

    def _estimate_tool_motion(self, prev_color, curr_color, prev_tool_mask, curr_tool_mask):
        default_allowed = True
        if self.tool_motion_gate_enabled and self.tool_motion_mode == 'high_conf_streak':
            default_allowed = self.tool_motion_gate_state
        base = {
            'tool_motion_valid': False,
            'tool_motion_score': 0.0,
            'tool_flow_tool': 0.0,
            'tool_flow_bg': 0.0,
            'tool_iou_delta': 0.0,
            'tool_centroid_shift': 0.0,
            'tool_static_confident': False,
            'tool_moving_confident': False,
            'tool_static_streak': int(self.tool_motion_static_streak),
            'tool_moving_streak': int(self.tool_motion_moving_streak),
            'tool_gate_state': float(self.tool_motion_gate_state),
            'tool_moving': bool(not default_allowed),
            'pose_opt_allowed': bool(default_allowed),
            'tool_motion_ratio': 0.0,
            'pose_opt_weight': 1.0 if default_allowed else 0.0,
            'pose_lock_weight': 0.0 if default_allowed else 1.0,
        }
        if self.tool_motion_gate_enabled and self.tool_motion_mode == 'flow_dir_std':
            flow_info = self._estimate_motion_from_flow_direction(
                prev_color=prev_color,
                curr_color=curr_color,
                tool_mask=curr_tool_mask if self.flow_dir_gate_use_tool_mask else None,
            )
            if flow_info is None:
                return base

            std_on = float(self.flow_dir_gate_pose_on_std)
            std_off = float(self.flow_dir_gate_pose_off_std)
            if std_off <= std_on:
                std_off = std_on + 1e-6
            dir_std = float(flow_info['flow_dir_std'])
            static_conf = dir_std <= std_on
            moving_conf = dir_std >= std_off

            if self.tool_motion_soft_gate_enabled:
                ratio = self._clip01((dir_std - std_on) / max(std_off - std_on, 1e-6))
                pose_opt_weight = 1.0 - (ratio ** max(self.tool_motion_soft_power, 1e-6))
                pose_opt_weight = self._clip01(max(self.tool_motion_soft_min_pose_weight, pose_opt_weight))
                pose_lock_weight = self._clip01(
                    (ratio ** max(self.tool_motion_soft_lock_power, 1e-6)) * self.tool_motion_soft_lock_max
                )
                pose_opt_allowed = bool(pose_opt_weight > self.pose_soft_enable_threshold)
                moving = not pose_opt_allowed
                motion_ratio = ratio
            else:
                if static_conf:
                    self.tool_motion_gate_state = True
                elif moving_conf:
                    self.tool_motion_gate_state = False
                pose_opt_allowed = bool(self.tool_motion_gate_state)
                moving = not pose_opt_allowed
                motion_ratio = 1.0 if moving else 0.0
                pose_opt_weight = 1.0 if pose_opt_allowed else 0.0
                pose_lock_weight = 1.0 if moving else 0.0

            base.update(
                {
                    'tool_motion_valid': True,
                    'tool_motion_score': dir_std,
                    'tool_flow_tool': float(flow_info['flow_mag_mean']),
                    'tool_flow_bg': 0.0,
                    'tool_iou_delta': 0.0,
                    'tool_centroid_shift': 0.0,
                    'tool_static_confident': bool(static_conf),
                    'tool_moving_confident': bool(moving_conf),
                    'tool_static_streak': int(self.tool_motion_static_streak),
                    'tool_moving_streak': int(self.tool_motion_moving_streak),
                    'tool_gate_state': float(self.tool_motion_gate_state),
                    'tool_moving': bool(moving),
                    'pose_opt_allowed': bool(pose_opt_allowed),
                    'tool_motion_ratio': float(motion_ratio),
                    'pose_opt_weight': float(pose_opt_weight),
                    'pose_lock_weight': float(pose_lock_weight),
                    'flow_dir_std': float(flow_info['flow_dir_std']),
                    'flow_dir_mean': float(flow_info['flow_dir_mean']),
                    'flow_dir_resultant': float(flow_info['flow_dir_resultant']),
                    'flow_mag_mean': float(flow_info['flow_mag_mean']),
                    'flow_valid_points': float(flow_info['flow_valid_points']),
                }
            )
            return base

        if (
            (not self.tool_motion_gate_enabled)
            or prev_color is None
            or prev_tool_mask is None
            or curr_tool_mask is None
        ):
            return base

        prev_mask = prev_tool_mask.squeeze(0).bool()
        curr_mask = curr_tool_mask.squeeze(0).bool()
        if (
            int(prev_mask.sum().item()) < self.tool_motion_min_pixels
            or int(curr_mask.sum().item()) < self.tool_motion_min_pixels
        ):
            return base

        with torch.no_grad():
            flow = self.raft(
                2 * prev_color.permute(0, 3, 1, 2) - 1.0,
                2 * curr_color.permute(0, 3, 1, 2) - 1.0,
            )[-1]
            flow_mag = torch.linalg.norm(flow, dim=1).squeeze(0)

            tool_vals = flow_mag[curr_mask]
            bg_mask = ~curr_mask
            if int(bg_mask.sum().item()) > self.tool_motion_min_pixels:
                bg_vals = flow_mag[bg_mask]
                flow_bg = bg_vals.median().item()
            else:
                flow_bg = 0.0
            flow_tool = tool_vals.median().item()

            inter = (prev_mask & curr_mask).sum().float()
            union = (prev_mask | curr_mask).sum().float()
            iou = (inter / union.clamp_min(1.0)).item()
            iou_delta = 1.0 - iou

            prev_cent = self._mask_centroid(prev_mask)
            curr_cent = self._mask_centroid(curr_mask)
            if prev_cent is None or curr_cent is None:
                centroid_shift = 0.0
            else:
                centroid_shift = torch.linalg.norm(curr_cent - prev_cent).item()

            score = (flow_tool - flow_bg) + self.tool_motion_iou_weight * iou_delta + self.tool_motion_centroid_weight * centroid_shift

        if self.tool_motion_mode == 'high_conf_streak':
            static_conf = score <= self.tool_motion_static_on
            moving_conf = score >= self.tool_motion_moving_on
            if static_conf:
                self.tool_motion_static_streak += 1
            else:
                self.tool_motion_static_streak = 0
            if moving_conf:
                self.tool_motion_moving_streak += 1
            else:
                self.tool_motion_moving_streak = 0

            if (not self.tool_motion_gate_state) and (self.tool_motion_static_streak >= self.tool_motion_static_required):
                self.tool_motion_gate_state = True
            if self.tool_motion_gate_state and (self.tool_motion_moving_streak >= self.tool_motion_moving_required):
                self.tool_motion_gate_state = False
            pose_opt_allowed = bool(self.tool_motion_gate_state)
            moving = not pose_opt_allowed
        else:
            if score <= self.tool_motion_off:
                self.tool_static_counter += 1
            else:
                self.tool_static_counter = 0
            moving = (score >= self.tool_motion_on) or (self.tool_static_counter < self.tool_motion_hysteresis)
            pose_opt_allowed = not moving
            static_conf = score <= self.tool_motion_off
            moving_conf = score >= self.tool_motion_on

        if self.tool_motion_soft_gate_enabled:
            pose_opt_weight, pose_lock_weight, motion_ratio = self._tool_motion_soft_weights(float(score))
            pose_opt_allowed = bool(pose_opt_weight > self.pose_soft_enable_threshold)
            moving = bool(motion_ratio >= 0.5)
        else:
            motion_ratio = 1.0 if moving else 0.0
            pose_opt_weight = 1.0 if pose_opt_allowed else 0.0
            pose_lock_weight = 1.0 if moving else 0.0

        base.update(
            {
                'tool_motion_valid': True,
                'tool_motion_score': float(score),
                'tool_flow_tool': float(flow_tool),
                'tool_flow_bg': float(flow_bg),
                'tool_iou_delta': float(iou_delta),
                'tool_centroid_shift': float(centroid_shift),
                'tool_static_confident': bool(static_conf),
                'tool_moving_confident': bool(moving_conf),
                'tool_static_streak': int(self.tool_motion_static_streak),
                'tool_moving_streak': int(self.tool_motion_moving_streak),
                'tool_gate_state': float(self.tool_motion_gate_state),
                'tool_moving': bool(moving),
                'pose_opt_allowed': bool(pose_opt_allowed),
                'tool_motion_ratio': float(motion_ratio),
                'pose_opt_weight': float(pose_opt_weight),
                'pose_lock_weight': float(pose_lock_weight),
            }
        )
        return base

    def _limit_pose_step(self, prev_c2w, curr_c2w, max_trans: float = None, max_rot_deg: float = None):
        stats = {
            'pose_step_trans_before': 0.0,
            'pose_step_rot_deg_before': 0.0,
            'pose_step_trans_after': 0.0,
            'pose_step_rot_deg_after': 0.0,
            'pose_step_clamped': False,
        }
        step_max_trans = self.pose_step_max_trans if max_trans is None else float(max_trans)
        step_max_rot_deg = self.pose_step_max_rot_deg if max_rot_deg is None else float(max_rot_deg)
        if (not self.pose_step_limit_enabled) or prev_c2w is None:
            return curr_c2w, stats
        if step_max_trans <= 0.0 and step_max_rot_deg <= 0.0:
            return curr_c2w, stats

        prev = prev_c2w.squeeze(0).detach().cpu().numpy()
        curr = curr_c2w.squeeze(0).detach().cpu().numpy()

        t_prev = prev[:3, 3]
        t_curr = curr[:3, 3]
        delta_t = t_curr - t_prev
        trans_norm = float(np.linalg.norm(delta_t))

        rel_rot = Rotation.from_matrix(prev[:3, :3]).inv() * Rotation.from_matrix(curr[:3, :3])
        rot_deg = float(np.rad2deg(rel_rot.magnitude()))

        trans_scale = 1.0
        if step_max_trans > 0.0 and trans_norm > step_max_trans:
            trans_scale = step_max_trans / max(trans_norm, 1e-12)
        t_new = t_prev + delta_t * trans_scale

        rot_scale = 1.0
        if step_max_rot_deg > 0.0 and rot_deg > step_max_rot_deg:
            rot_scale = step_max_rot_deg / max(rot_deg, 1e-12)
        rel_rotvec = rel_rot.as_rotvec() * rot_scale
        rot_new = (Rotation.from_matrix(prev[:3, :3]) * Rotation.from_rotvec(rel_rotvec)).as_matrix()

        limited = curr.copy()
        limited[:3, :3] = rot_new
        limited[:3, 3] = t_new

        rel_after = Rotation.from_matrix(prev[:3, :3]).inv() * Rotation.from_matrix(limited[:3, :3])
        trans_after = float(np.linalg.norm(limited[:3, 3] - t_prev))
        rot_after = float(np.rad2deg(rel_after.magnitude()))

        stats.update(
            {
                'pose_step_trans_before': trans_norm,
                'pose_step_rot_deg_before': rot_deg,
                'pose_step_trans_after': trans_after,
                'pose_step_rot_deg_after': rot_after,
                'pose_step_clamped': bool((trans_scale < 1.0) or (rot_scale < 1.0)),
                'pose_step_max_trans_used': float(step_max_trans),
                'pose_step_max_rot_deg_used': float(step_max_rot_deg),
            }
        )
        limited_t = torch.from_numpy(limited).to(device=curr_c2w.device, dtype=curr_c2w.dtype).unsqueeze(0)
        return limited_t, stats

    def _pose_track_loss(self, c2w, frame_id: int):
        if self.pt_tracker is None or not self.pt_tracker.is_initialized():
            return None
        if frame_id >= int(self.pt_tracker.gt_2d_pts.shape[1]):
            return None
        valid_np = self.pt_tracker.valid[:, frame_id]
        if valid_np is None or int(np.sum(valid_np)) == 0:
            return None

        pts_gt = self.pt_tracker.gt_2d_pts[:, frame_id]
        _, pts_2d = self.pt_tracker.get_2d_pts(c2w, detach_scene=True, clamp_to_image=False)
        pts_gt = pts_gt.to(device=pts_2d.device, dtype=pts_2d.dtype)
        valid = torch.from_numpy(valid_np).to(device=pts_2d.device, dtype=torch.bool)
        H, W = self.camera.get_params()[:2]
        norm = torch.tensor([W, H], dtype=pts_2d.dtype, device=pts_2d.device)
        error = torch.abs((pts_2d - pts_gt) / norm).sum(dim=-1)
        valid_error = error[valid]
        valid_error = valid_error[torch.isfinite(valid_error)]
        if valid_error.numel() == 0:
            return None
        if self.pose_track_robust_delta > 0.0:
            delta = torch.tensor(
                self.pose_track_robust_delta,
                device=valid_error.device,
                dtype=valid_error.dtype,
            )
            scaled = valid_error / torch.clamp(delta, min=1e-9)
            valid_error = (delta * delta) * (torch.sqrt(1.0 + scaled * scaled) - 1.0)
        return valid_error.mean()

    def _maybe_init_pt_tracker_for_gs_refine(self, frame_id: int, c2w, stereo_depth):
        if self.pt_tracker is None:
            return False
        if self.pt_tracker.is_initialized():
            return True
        if int(frame_id) != 0:
            return False
        try:
            self.pt_tracker.init_tracking_points(c2w, stereo_depth.squeeze(0))
            return True
        except Exception as exc:
            warnings.warn(f"PointTracker init failed for cotracker gs refine. reason: {exc}")
            return False

    def _maybe_refine_cotracker_with_gs(self, frame_id: int, cotracker_pts_2d, c2w):
        info = {
            'applied': False,
            'reason': 'disabled',
            'num_refined': 0,
            'num_points': int(cotracker_pts_2d.shape[0]),
            'mean_raw_residual_px': float('nan'),
            'mean_shift_px': 0.0,
        }
        if not self.pt_cotracker_gs_refine_enabled:
            return cotracker_pts_2d, info
        if self.pt_tracker is None or (not self.pt_tracker.is_initialized()):
            info['reason'] = 'pt_not_initialized'
            return cotracker_pts_2d, info
        if int(frame_id) < int(self.pt_cotracker_gs_refine_start_frame):
            info['reason'] = 'before_start_frame'
            return cotracker_pts_2d, info
        if (int(frame_id) % int(self.pt_cotracker_gs_refine_interval)) != 0:
            info['reason'] = 'interval_skip'
            return cotracker_pts_2d, info
        if self.pt_cotracker_gs_refine_alpha <= 0.0:
            info['reason'] = 'alpha_zero'
            return cotracker_pts_2d, info
        if self.pt_cotracker_gs_refine_max_res_px <= 0.0:
            info['reason'] = 'max_res_non_positive'
            return cotracker_pts_2d, info

        gs_pts_2d = None
        try:
            _, gs_pts_2d = self.pt_tracker.get_2d_pts(
                c2w,
                detach_scene=True,
                clamp_to_image=self.pt_cotracker_gs_refine_clamp_to_image,
            )
        except Exception as exc:
            info['reason'] = f'gs_proj_failed:{exc}'
            return cotracker_pts_2d, info

        if gs_pts_2d is None:
            info['reason'] = 'gs_proj_none'
            return cotracker_pts_2d, info

        gs_pts_2d = gs_pts_2d.to(device=cotracker_pts_2d.device, dtype=cotracker_pts_2d.dtype)
        finite_mask = torch.isfinite(gs_pts_2d).all(dim=-1) & torch.isfinite(cotracker_pts_2d).all(dim=-1)
        if int(finite_mask.sum().item()) <= 0:
            info['reason'] = 'no_finite_points'
            return cotracker_pts_2d, info

        raw_residual = torch.linalg.norm(gs_pts_2d - cotracker_pts_2d, dim=-1)
        residual_mask = raw_residual <= float(self.pt_cotracker_gs_refine_max_res_px)
        refine_mask = finite_mask & residual_mask
        num_refined = int(refine_mask.sum().item())
        if num_refined <= 0:
            info['reason'] = 'no_points_under_residual_gate'
            valid_raw = raw_residual[finite_mask]
            if valid_raw.numel() > 0:
                info['mean_raw_residual_px'] = float(valid_raw.mean().item())
            return cotracker_pts_2d, info

        alpha = float(self.pt_cotracker_gs_refine_alpha)
        refined = cotracker_pts_2d.clone()
        refined[refine_mask] = (1.0 - alpha) * cotracker_pts_2d[refine_mask] + alpha * gs_pts_2d[refine_mask]
        shift = torch.linalg.norm(refined - cotracker_pts_2d, dim=-1)

        valid_raw = raw_residual[finite_mask]
        info.update(
            {
                'applied': True,
                'reason': 'ok',
                'num_refined': num_refined,
                'mean_raw_residual_px': float(valid_raw.mean().item()) if valid_raw.numel() > 0 else float('nan'),
                'mean_shift_px': float(shift[refine_mask].mean().item()),
            }
        )
        return refined, info

    def warmup_pose(self, gt_color, stereo_depth, init_c2w, tool_mask, pose_mask=None, pose_opt_weight=1.0, frame_id: int = None):
        pose_weight = self._clip01(float(pose_opt_weight))
        if (
            (not self.pose_opt_enabled)
            or self.pose_warmup_iters <= 0
            or pose_weight <= self.pose_soft_enable_threshold
        ):
            return init_c2w, None

        if frame_id is not None:
            frame_time = self._frame_time(int(frame_id))
            self.net.set_deformation_time(frame_time)
            self.camera.set_time(frame_time)

        loss_mask = self._safe_loss_mask(pose_mask if pose_mask is not None else tool_mask, stereo_depth)
        pose_delta = torch.zeros(6, device=init_c2w.device, dtype=init_c2w.dtype, requires_grad=True)
        pose_optimizer = torch.optim.Adam([pose_delta], lr=self.pose_lr)
        max_iters = self.pose_warmup_iters

        for _ in range(max_iters):
            pose_optimizer.zero_grad(set_to_none=True)
            if self.net.optimizer is not None:
                self.net.optimizer.zero_grad(set_to_none=True)

            effective_delta = pose_delta * pose_weight
            cur_c2w = self._compose_pose(init_c2w, effective_delta)
            if not self._is_pose_matrix_valid(cur_c2w):
                pose_delta.data.zero_()
                cur_c2w = init_c2w
            self.camera.set_c2w(cur_c2w)
            render_pkg = self._render_scene(deform=True)
            color = render_pkg['render'][None, ...]
            depth = render_pkg['depth'][None, ...]

            loss_color = self.cfg['training']['w_color'] * l1_loss(color[loss_mask], gt_color[loss_mask])
            loss_depth = self.cfg['training']['w_depth'] * l1_loss(
                depth[loss_mask] / self.scale, stereo_depth[loss_mask] / self.scale
            )
            pose_reg, _, _ = self._pose_reg_loss(effective_delta)
            loss = loss_color + loss_depth + pose_reg
            loss.backward()
            if self.pose_grad_clip > 0.0 and pose_delta.grad is not None:
                torch.nn.utils.clip_grad_norm_([pose_delta], self.pose_grad_clip)
            pose_optimizer.step()

        with torch.no_grad():
            effective_delta = pose_delta * pose_weight
            final_pose = self._compose_pose(init_c2w, effective_delta).detach()
            if not self._is_pose_matrix_valid(final_pose):
                final_pose = init_c2w.detach()
            warmup_stats = {
                'delta_rot_norm': effective_delta[:3].norm().item(),
                'delta_trans_norm': effective_delta[3:].norm().item(),
                'pose_opt_weight': pose_weight,
            }
        return final_pose, warmup_stats

    def fit(self, frame, iters, incremental, pose_mask=None, pose_opt_weight=1.0, motion_ratio=0.0):
        def _scalar_or_nan(value):
            if value is None:
                return float('nan')
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return float('nan')
                return float(value.detach().item())
            try:
                return float(value)
            except Exception:
                return float('nan')

        self.net.reset_optimizer()
        av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
        idx, gt_color, stereo_depth, init_c2w, tool_mask = frame
        frame_id = int(idx.item()) if torch.is_tensor(idx) else int(idx)
        frame_time = self._frame_time(frame_id)
        self.net.set_deformation_time(frame_time)
        self.camera.set_time(frame_time)
        pose_opt_weight = self._clip01(float(pose_opt_weight))
        dynamic_scales = self._motion_dynamic_scales(float(motion_ratio))
        pose_recon_bg_blend = self._clip01(self.pose_recon_bg_blend)
        map_mask = self._safe_loss_mask(tool_mask, stereo_depth)
        pose_loss_mask = self._safe_loss_mask(pose_mask if pose_mask is not None else map_mask, stereo_depth)
        use_pose_opt = (
            self._pose_enabled_for_frame(frame_id=frame_id, incremental=incremental)
            and (pose_opt_weight > self.pose_soft_enable_threshold)
        )
        pose_init_used = ''
        if isinstance(self.last_pose_init_info, dict):
            pose_init_used = str(self.last_pose_init_info.get('pose_init_used', ''))
        pose_freeze_reason = ''
        if use_pose_opt and self.pose_freeze_on_vo_static_skip and pose_init_used == 'vo_static_skip':
            use_pose_opt = False
            pose_freeze_reason = 'vo_static_skip'

        pose_delta = None
        pose_optimizer = None
        pose_steps = iters if self.pose_opt_iters <= 0 else min(self.pose_opt_iters, iters)
        pose_only_iters = 0
        if use_pose_opt:
            pose_delta = torch.zeros(6, device=init_c2w.device, dtype=init_c2w.dtype, requires_grad=True)
            pose_optimizer = torch.optim.Adam([pose_delta], lr=self.pose_lr)
            pose_only_iters = min(max(self.pose_only_iters, 0), max(iters - 1, 0))

        init_track_loss_value = None
        need_init_track_loss = (
            use_pose_opt
            and self.pose_w_track_2d > 0.0
            and self.pt_tracker is not None
            and (
                self.pose_track_guard_enabled
                or self.pose_track_adaptive_weight_enabled
                or self.pose_track_constraint_enabled
                or self.pose_freeze_if_track_loss_invalid
                or self.pose_final_revert_on_track_worse
            )
        )
        if need_init_track_loss:
            with torch.no_grad():
                init_track_loss = self._pose_track_loss(init_c2w, frame_id)
            if init_track_loss is not None and torch.isfinite(init_track_loss):
                init_track_loss_value = float(init_track_loss.item())

        pose_track_adaptive_scale = 1.0
        if (
            use_pose_opt
            and self.pose_track_adaptive_weight_enabled
            and init_track_loss_value is not None
            and init_track_loss_value > 1e-9
        ):
            ratio = self.pose_track_adaptive_ref / max(init_track_loss_value, 1e-9)
            pose_track_adaptive_scale = min(1.0, float(ratio ** self.pose_track_adaptive_power))
            pose_track_adaptive_scale = max(self.pose_track_adaptive_min_scale, pose_track_adaptive_scale)
        pose_opt_weight_runtime = pose_opt_weight * pose_track_adaptive_scale
        if (
            use_pose_opt
            and self.pose_freeze_if_track_loss_invalid
            and self.pose_w_track_2d > 0.0
            and self.pt_tracker is not None
            and init_track_loss_value is None
        ):
            use_pose_opt = False
            pose_freeze_reason = 'invalid_init_track_loss'
            pose_delta = None
            pose_optimizer = None
            pose_steps = 0
            pose_only_iters = 0
        if not use_pose_opt:
            pose_opt_weight_runtime = 0.0

        guard_trigger_count = 0
        current_c2w = init_c2w
        self.camera.set_c2w(current_c2w)
        debug_pose = bool(self.pose_debug_grad_enabled)
        for iter in range(1, iters + 1):
            stage_state = self._pose_stage_state(iter, iters)
            pose_lr_current = float('nan')
            if pose_optimizer is not None:
                pose_optimizer.zero_grad(set_to_none=True)
                if iter <= pose_steps:
                    for pg in pose_optimizer.param_groups:
                        pg['lr'] = self.pose_lr * stage_state['lr_scale'] * dynamic_scales['lr_scale'] * pose_opt_weight_runtime
                if len(pose_optimizer.param_groups) > 0:
                    pose_lr_current = float(pose_optimizer.param_groups[0].get('lr', float('nan')))

            effective_pose_delta = pose_delta * pose_opt_weight_runtime if pose_delta is not None else None
            if pose_delta is not None and stage_state['pose_active']:
                current_c2w = self._compose_pose(init_c2w, effective_pose_delta)
            else:
                current_c2w = init_c2w
            if not self._is_pose_matrix_valid(current_c2w):
                current_c2w = init_c2w
                if pose_delta is not None:
                    pose_delta.data.zero_()

            guard_triggered = False
            guard_loss_value = None
            guard_threshold = None
            if (
                pose_delta is not None
                and stage_state['pose_active']
                and init_track_loss_value is not None
            ):
                with torch.no_grad():
                    guard_track_loss = self._pose_track_loss(current_c2w.detach(), frame_id)
                if guard_track_loss is not None and torch.isfinite(guard_track_loss):
                    guard_loss_value = float(guard_track_loss.item())
                    guard_threshold = (
                        self.pose_track_guard_ratio * init_track_loss_value + self.pose_track_guard_margin
                    )
                    if self.pose_track_guard_abs_max > 0.0:
                        guard_threshold = min(guard_threshold, self.pose_track_guard_abs_max)
                    allow_guard = (
                        self.pose_track_guard_max_triggers <= 0
                        or guard_trigger_count < self.pose_track_guard_max_triggers
                    )
                    if allow_guard and guard_loss_value > guard_threshold:
                        guard_triggered = True
                        guard_trigger_count += 1
                        current_c2w = init_c2w
                        if self.pose_track_guard_decay < 1.0 and pose_delta is not None:
                            pose_delta.data.mul_(self.pose_track_guard_decay)

            self.camera.set_c2w(current_c2w)
            if self.cfg['training']['spherical_harmonics'] and iter > iters / 2:
                self.net.enable_spherical_harmonics()
            self.total_iters += 1
            self.net.train(iter == 1)
            render_pkg = self._render_scene(deform=incremental)
            self.net.eval()
            color = render_pkg['render'][None, ...]
            depth = render_pkg['depth'][None, ...]
            rgb_pred = color.permute(0, 3, 1, 2).clamp(0.0, 1.0)
            rgb_gt = gt_color.permute(0, 3, 1, 2).clamp(0.0, 1.0)

            # Loss
            Ll1_map = self.cfg['training']['w_color'] * l1_loss(color[map_mask], gt_color[map_mask])
            Ll1_depth_map = self.cfg['training']['w_depth'] * l1_loss(
                depth[map_mask] / self.scale, stereo_depth[map_mask] / self.scale
            )
            Ll1 = Ll1_map
            Ll1_depth = Ll1_depth_map
            if pose_delta is not None and stage_state['pose_active']:
                Ll1_pose = self.cfg['training']['w_color'] * l1_loss(color[pose_loss_mask], gt_color[pose_loss_mask])
                Ll1_depth_pose = self.cfg['training']['w_depth'] * l1_loss(
                    depth[pose_loss_mask] / self.scale, stereo_depth[pose_loss_mask] / self.scale
                )
                Ll1 = (1.0 - pose_recon_bg_blend) * Ll1_map + pose_recon_bg_blend * Ll1_pose
                Ll1_depth = (1.0 - pose_recon_bg_blend) * Ll1_depth_map + pose_recon_bg_blend * Ll1_depth_pose
            loss = Ll1 + Ll1_depth

            percept_mask = pose_loss_mask if (pose_delta is not None and stage_state['pose_active']) else map_mask
            if percept_mask is not None:
                percept_mask_4d = percept_mask.unsqueeze(1).to(dtype=rgb_pred.dtype)
            else:
                percept_mask_4d = torch.ones_like(rgb_pred[:, :1, :, :], dtype=rgb_pred.dtype)

            ssim_term = torch.zeros_like(Ll1)
            if self.pose_w_ssim > 0.0:
                ssim_term = 1.0 - ssim(rgb_pred * percept_mask_4d, rgb_gt * percept_mask_4d)
                loss += self.pose_w_ssim * stage_state['recon_scale'] * dynamic_scales['recon_scale'] * ssim_term

            lpips_term = torch.zeros_like(Ll1)
            if self.pose_w_lpips > 0.0 and self.lpips_model is not None:
                lpips_term = lpips_loss(
                    (rgb_pred * percept_mask_4d) * 2.0 - 1.0,
                    (rgb_gt * percept_mask_4d) * 2.0 - 1.0,
                    self.lpips_model,
                )
                loss += self.pose_w_lpips * stage_state['recon_scale'] * dynamic_scales['recon_scale'] * lpips_term

            if incremental:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = self.net.compute_regulation(render_pkg["visibility_filter"])
                def_loss = (
                    self.cfg['training']['w_def']['rigid'] * l_rigidtrans
                    + self.cfg['training']['w_def']['iso'] * l_iso
                    + self.cfg['training']['w_def']['rot'] * l_rigidrot
                    + self.cfg['training']['w_def']['nvisible'] * l_visible
                )
                loss += def_loss
                deform_static_decouple = torch.zeros_like(Ll1)
                if self.pose_deform_decouple_enabled and self.pose_deform_static_l2 > 0.0:
                    static_factor = (1.0 - float(motion_ratio)) ** self.pose_deform_static_power
                    if static_factor > 0.0:
                        mean_def = self.net._deformation.get_mean_def(self.net.get_xyz)
                        rot_def = getattr(self.net._deformation, "rot_def", None)
                        deform_energy = torch.mean(mean_def * mean_def)
                        if rot_def is not None:
                            deform_energy = deform_energy + self.pose_deform_rot_ratio * torch.mean(rot_def * rot_def)
                        deform_static_decouple = (
                            self.pose_deform_static_l2
                            * static_factor
                            * stage_state['recon_scale']
                            * dynamic_scales['recon_scale']
                            * deform_energy
                        )
                        loss += deform_static_decouple
            else:
                l_rigidtrans = torch.zeros_like(Ll1)
                l_rigidrot = torch.zeros_like(Ll1)
                l_iso = torch.zeros_like(Ll1)
                l_visible = torch.zeros_like(Ll1)
                deform_static_decouple = torch.zeros_like(Ll1)

            pose_prior = torch.zeros_like(Ll1)
            pose_smooth = None
            pose_smooth_trans = torch.zeros_like(Ll1)
            pose_smooth_rot = torch.zeros_like(Ll1)
            if pose_delta is not None and stage_state['pose_active']:
                pose_reg, pose_reg_rot, pose_reg_trans = self._pose_reg_loss(effective_pose_delta)
                loss += pose_reg
                pose_track = None
                if self.pose_w_track_2d > 0.0:
                    pose_track = self._pose_track_loss(current_c2w, frame_id)
                    if pose_track is not None:
                        loss += self.pose_w_track_2d * stage_state['track_scale'] * dynamic_scales['track_scale'] * pose_track
                pose_track_constraint = torch.zeros_like(Ll1)
                pose_track_constraint_threshold = None
                if (
                    self.pose_track_constraint_enabled
                    and self.pose_track_constraint_weight > 0.0
                    and pose_track is not None
                    and init_track_loss_value is not None
                ):
                    pose_track_constraint_threshold = (
                        self.pose_track_constraint_ratio * init_track_loss_value + self.pose_track_constraint_margin
                    )
                    if self.pose_track_constraint_abs_max > 0.0:
                        pose_track_constraint_threshold = min(
                            pose_track_constraint_threshold, self.pose_track_constraint_abs_max
                        )
                    threshold_t = torch.tensor(
                        pose_track_constraint_threshold,
                        device=pose_track.device,
                        dtype=pose_track.dtype,
                    )
                    pose_track_constraint = torch.relu(pose_track - threshold_t) ** 2
                    loss += self.pose_track_constraint_weight * pose_track_constraint
                if self.pose_w_prior > 0.0:
                    pose_prior = self._pose_prior_loss(effective_pose_delta)
                    loss += self.pose_w_prior * stage_state['prior_scale'] * dynamic_scales['prior_scale'] * pose_prior
                if self.pose_w_smooth > 0.0:
                    pose_smooth = self._pose_smooth_loss(current_c2w=current_c2w, init_c2w=init_c2w)
                    if pose_smooth is not None:
                        pose_smooth_total, pose_smooth_trans, pose_smooth_rot = pose_smooth
                        loss += self.pose_w_smooth * stage_state['smooth_scale'] * dynamic_scales['smooth_scale'] * pose_smooth_total
            else:
                pose_reg_rot = torch.zeros_like(Ll1)
                pose_reg_trans = torch.zeros_like(Ll1)
                pose_track = None
                pose_reg = torch.zeros_like(Ll1)
                pose_track_constraint = torch.zeros_like(Ll1)
                pose_track_constraint_threshold = None

            loss.backward()
            pose_grad_norm_pre = float('nan')
            pose_grad_norm_rot_pre = float('nan')
            pose_grad_norm_trans_pre = float('nan')
            pose_grad_norm_post = float('nan')
            pose_grad_norm_rot_post = float('nan')
            pose_grad_norm_trans_post = float('nan')
            pose_grad_clip_total_norm = float('nan')
            if pose_delta is not None and pose_delta.grad is not None:
                pose_grad_norm_pre = float(pose_delta.grad.norm().item())
                pose_grad_norm_rot_pre = float(pose_delta.grad[:3].norm().item())
                pose_grad_norm_trans_pre = float(pose_delta.grad[3:].norm().item())
            if pose_delta is not None and self.pose_grad_clip > 0.0 and pose_delta.grad is not None:
                pose_grad_clip_total_norm = float(torch.nn.utils.clip_grad_norm_([pose_delta], self.pose_grad_clip).item())
            if pose_delta is not None and pose_delta.grad is not None:
                pose_grad_norm_post = float(pose_delta.grad.norm().item())
                pose_grad_norm_rot_post = float(pose_delta.grad[:3].norm().item())
                pose_grad_norm_trans_post = float(pose_delta.grad[3:].norm().item())
            viewspace_point_tensor_grad = torch.zeros_like(render_pkg["viewspace_points"])
            viewspace_point_tensor_grad += render_pkg["viewspace_points"].grad

            ########### Logging & Evaluation ###################
            with torch.no_grad():
                av_loss[0] += Ll1.item()
                av_loss[1] += Ll1_depth.item()
                av_loss[2] += l_rigidtrans.item()
                av_loss[3] += l_rigidrot.item()
                av_loss[4] += l_iso.item()
                av_loss[5] += l_visible.item()
                av_loss[-1] += 1
                if ((self.total_iters % self.log_freq) == 0) and self.log:
                    log_vals = {
                        'color_loss': av_loss[0] / av_loss[-1],
                        'depth_loss': av_loss[1] / av_loss[-1],
                        'rigidtrans_loss': av_loss[2] / av_loss[-1],
                        'rigidrot_loss': av_loss[3] / av_loss[-1],
                        'iso_loss': av_loss[4] / av_loss[-1],
                        'visible_loss': av_loss[5] / av_loss[-1],
                        'loss': sum(av_loss[:-1]) / av_loss[-1],
                    }
                    if pose_delta is not None:
                        log_vals.update(
                            {
                                'pose_rot_reg': pose_reg_rot.item(),
                                'pose_trans_reg': pose_reg_trans.item(),
                                'pose_rot_norm': effective_pose_delta[:3].norm().item(),
                                'pose_trans_norm': effective_pose_delta[3:].norm().item(),
                                'pose_stage_lr': self.pose_lr * stage_state['lr_scale'] * dynamic_scales['lr_scale'],
                                'pose_opt_weight': pose_opt_weight_runtime,
                                'pose_opt_weight_base': pose_opt_weight,
                                'pose_track_adaptive_scale': pose_track_adaptive_scale,
                                'pose_motion_ratio': float(motion_ratio),
                                'pose_dyn_track_scale': float(dynamic_scales['track_scale']),
                                'pose_dyn_prior_scale': float(dynamic_scales['prior_scale']),
                                'pose_dyn_smooth_scale': float(dynamic_scales['smooth_scale']),
                                'pose_dyn_recon_scale': float(dynamic_scales['recon_scale']),
                                'pose_prior_loss': pose_prior.item() if torch.is_tensor(pose_prior) else 0.0,
                                'pose_track_guard_triggers': float(guard_trigger_count),
                            }
                        )
                        if guard_threshold is not None:
                            log_vals['pose_track_guard_threshold'] = float(guard_threshold)
                        if guard_loss_value is not None:
                            log_vals['pose_track_guard_loss'] = float(guard_loss_value)
                        if guard_triggered:
                            log_vals['pose_track_guard_hit'] = 1.0
                        if pose_track is not None:
                            log_vals['pose_track_2d_loss'] = pose_track.item()
                        if self.pose_track_constraint_enabled and self.pose_track_constraint_weight > 0.0:
                            log_vals['pose_track_constraint_loss'] = pose_track_constraint.item()
                            if pose_track_constraint_threshold is not None:
                                log_vals['pose_track_constraint_threshold'] = float(pose_track_constraint_threshold)
                        if pose_smooth is not None:
                            log_vals['pose_smooth_trans_loss'] = pose_smooth_trans.item()
                            log_vals['pose_smooth_rot_loss'] = pose_smooth_rot.item()
                    if self.pose_w_ssim > 0.0:
                        log_vals['ssim_loss'] = ssim_term.item()
                    if self.pose_w_lpips > 0.0 and self.lpips_model is not None:
                        log_vals['lpips_loss'] = lpips_term.item()
                    if self.pose_deform_decouple_enabled and self.pose_deform_static_l2 > 0.0:
                        log_vals['pose_deform_static_decouple_loss'] = deform_static_decouple.item()
                    wandb.log(log_vals, step=self.total_iters)
                    av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]

                self.net.add_densification_stats(viewspace_point_tensor_grad, render_pkg["visibility_filter"])
                if not incremental:
                    if (
                        iter > self.cfg["training"]["densify_from_iter"]
                        and iter % self.cfg["training"]["densification_interval"] == 0
                    ):
                        self.net.densify(self.cfg["training"]["densify_grad_threshold"])

                pose_step_applied = False
                pose_step_will_apply = False
                pose_update_norm = float('nan')
                pose_update_rot_norm = float('nan')
                pose_update_trans_norm = float('nan')
                pose_delta_norm_before_step = float('nan')
                pose_delta_norm_after_step = float('nan')
                map_update_applied = False
                if iter < iters:
                    pose_step_will_apply = bool(
                        pose_optimizer is not None
                        and iter <= pose_steps
                        and stage_state['pose_active']
                        and stage_state['lr_scale'] > 0.0
                        and pose_opt_weight_runtime > self.pose_soft_enable_threshold
                    )
                    if pose_step_will_apply:
                        pose_before_step = pose_delta.detach().clone() if pose_delta is not None else None
                        if pose_before_step is not None:
                            pose_delta_norm_before_step = float(pose_before_step.norm().item())
                        pose_optimizer.step()
                        pose_step_applied = True
                        if pose_delta is not None and pose_before_step is not None:
                            pose_after_step = pose_delta.detach()
                            pose_delta_norm_after_step = float(pose_after_step.norm().item())
                            pose_step_vec = pose_after_step - pose_before_step
                            pose_update_norm = float(pose_step_vec.norm().item())
                            pose_update_rot_norm = float(pose_step_vec[:3].norm().item())
                            pose_update_trans_norm = float(pose_step_vec[3:].norm().item())
                    update_map = not (pose_optimizer is not None and iter <= pose_only_iters)
                    if update_map:
                        self.net.optimizer.step()
                    map_update_applied = bool(update_map)
                    self.net.optimizer.zero_grad(set_to_none=True)

                if (
                    debug_pose
                    and pose_delta is not None
                    and ((iter == 1) or (iter == iters) or ((iter % self.pose_debug_grad_log_freq) == 0))
                ):
                    self.pose_debug_grad_records.append(
                        {
                            'frame': int(frame_id),
                            'iter': int(iter),
                            'iters_total': int(iters),
                            'incremental': int(bool(incremental)),
                            'pose_active': int(bool(stage_state['pose_active'])),
                            'pose_steps_limit': int(pose_steps),
                            'pose_only_iters': int(pose_only_iters),
                            'pose_opt_weight_base': float(pose_opt_weight),
                            'pose_opt_weight_runtime': float(pose_opt_weight_runtime),
                            'pose_track_adaptive_scale': float(pose_track_adaptive_scale),
                            'motion_ratio': float(motion_ratio),
                            'dynamic_lr_scale': float(dynamic_scales['lr_scale']),
                            'dynamic_track_scale': float(dynamic_scales['track_scale']),
                            'dynamic_prior_scale': float(dynamic_scales['prior_scale']),
                            'dynamic_smooth_scale': float(dynamic_scales['smooth_scale']),
                            'dynamic_recon_scale': float(dynamic_scales['recon_scale']),
                            'stage_lr_scale': float(stage_state['lr_scale']),
                            'stage_track_scale': float(stage_state['track_scale']),
                            'stage_prior_scale': float(stage_state['prior_scale']),
                            'stage_smooth_scale': float(stage_state['smooth_scale']),
                            'stage_recon_scale': float(stage_state['recon_scale']),
                            'pose_lr_current': float(pose_lr_current),
                            'loss_total': _scalar_or_nan(loss),
                            'loss_l1': _scalar_or_nan(Ll1),
                            'loss_depth': _scalar_or_nan(Ll1_depth),
                            'loss_ssim': _scalar_or_nan(ssim_term),
                            'loss_lpips': _scalar_or_nan(lpips_term),
                            'loss_pose_track': _scalar_or_nan(pose_track),
                            'loss_pose_prior': _scalar_or_nan(pose_prior),
                            'loss_pose_reg_rot': _scalar_or_nan(pose_reg_rot),
                            'loss_pose_reg_trans': _scalar_or_nan(pose_reg_trans),
                            'loss_pose_smooth_trans': _scalar_or_nan(pose_smooth_trans),
                            'loss_pose_smooth_rot': _scalar_or_nan(pose_smooth_rot),
                            'loss_pose_track_constraint': _scalar_or_nan(pose_track_constraint),
                            'pose_track_guard_triggered': int(bool(guard_triggered)),
                            'pose_track_guard_triggers_total': int(guard_trigger_count),
                            'pose_track_guard_loss': _scalar_or_nan(guard_loss_value),
                            'pose_track_guard_threshold': _scalar_or_nan(guard_threshold),
                            'init_track_loss': _scalar_or_nan(init_track_loss_value),
                            'pose_grad_norm_pre': float(pose_grad_norm_pre),
                            'pose_grad_rot_norm_pre': float(pose_grad_norm_rot_pre),
                            'pose_grad_trans_norm_pre': float(pose_grad_norm_trans_pre),
                            'pose_grad_norm_post': float(pose_grad_norm_post),
                            'pose_grad_rot_norm_post': float(pose_grad_norm_rot_post),
                            'pose_grad_trans_norm_post': float(pose_grad_norm_trans_post),
                            'pose_grad_clip_total_norm': float(pose_grad_clip_total_norm),
                            'pose_update_will_apply': int(bool(pose_step_will_apply)),
                            'pose_update_applied': int(bool(pose_step_applied)),
                            'pose_update_norm': float(pose_update_norm),
                            'pose_update_rot_norm': float(pose_update_rot_norm),
                            'pose_update_trans_norm': float(pose_update_trans_norm),
                            'pose_delta_norm_before_step': float(pose_delta_norm_before_step),
                            'pose_delta_norm_after_step': float(pose_delta_norm_after_step),
                            'effective_delta_rot_norm': _scalar_or_nan(
                                effective_pose_delta[:3].norm() if effective_pose_delta is not None else None
                            ),
                            'effective_delta_trans_norm': _scalar_or_nan(
                                effective_pose_delta[3:].norm() if effective_pose_delta is not None else None
                            ),
                            'map_update_applied': int(bool(map_update_applied)),
                        }
                    )

        with torch.no_grad():
            final_reverted_to_init = False
            final_track_loss_value = float('nan')
            final_track_threshold = float('nan')
            if pose_delta is not None:
                effective_pose_delta = pose_delta * pose_opt_weight_runtime
                final_c2w = self._compose_pose(init_c2w, effective_pose_delta).detach()
                if not self._is_pose_matrix_valid(final_c2w):
                    final_c2w = init_c2w.detach()
                if (
                    self.pose_final_revert_on_track_worse
                    and init_track_loss_value is not None
                    and self.pose_w_track_2d > 0.0
                    and self.pt_tracker is not None
                ):
                    final_track_loss = self._pose_track_loss(final_c2w, frame_id)
                    if final_track_loss is not None and torch.isfinite(final_track_loss):
                        final_track_loss_value = float(final_track_loss.item())
                        final_track_threshold = (
                            self.pose_final_revert_ratio * init_track_loss_value + self.pose_final_revert_margin
                        )
                        if final_track_loss_value > final_track_threshold:
                            final_reverted_to_init = True
                            final_c2w = init_c2w.detach()
                            effective_pose_delta = torch.zeros_like(effective_pose_delta)
                pose_stats = {
                    'delta_rot_norm': effective_pose_delta[:3].norm().item(),
                    'delta_trans_norm': effective_pose_delta[3:].norm().item(),
                    'pose_opt_weight': pose_opt_weight_runtime,
                    'pose_track_adaptive_scale': pose_track_adaptive_scale,
                    'init_track_loss': float(init_track_loss_value) if init_track_loss_value is not None else float('nan'),
                    'final_track_loss': float(final_track_loss_value),
                    'final_track_threshold': float(final_track_threshold),
                    'pose_reverted_to_init': float(final_reverted_to_init),
                    'pose_frozen_reason': pose_freeze_reason,
                    'motion_ratio': float(motion_ratio),
                }
            else:
                final_c2w = init_c2w.detach()
                pose_stats = {
                    'delta_rot_norm': 0.0,
                    'delta_trans_norm': 0.0,
                    'pose_opt_weight': 0.0,
                    'pose_track_adaptive_scale': pose_track_adaptive_scale,
                    'init_track_loss': float(init_track_loss_value) if init_track_loss_value is not None else float('nan'),
                    'final_track_loss': float('nan'),
                    'final_track_threshold': float('nan'),
                    'pose_reverted_to_init': 0.0,
                    'pose_frozen_reason': pose_freeze_reason,
                    'motion_ratio': float(motion_ratio),
                }
            self.last_pose_runtime_info = {
                'pose_runtime_use_pose_opt': bool(pose_delta is not None),
                'pose_runtime_freeze_reason': pose_freeze_reason,
                'pose_runtime_init_track_loss_valid': bool(init_track_loss_value is not None),
                'pose_runtime_reverted_to_init': bool(final_reverted_to_init),
            }
        return final_c2w, pose_stats

    def run(self):
        torch.cuda.empty_cache()
        pt_track_stats = {"pred_2d": [], "pred_3d": []}
        input_pose_history = []
        gt_pose_history = []
        optimized_pose_history = []
        self.prev_input_pose_for_smooth = None
        self.prev_optimized_pose_for_smooth = None
        self.prev_depth_for_pose_init = None
        self.prev_tool_mask_for_motion = None
        self.pose_no_prior_vo_static_state = False
        self.pose_no_prior_vo_static_streak = 0
        self.pose_no_prior_vo_moving_streak = 0
        self.pose_no_prior_vo_flow_std_hist.clear()
        self.pose_no_prior_vo_flow_mag_hist.clear()
        self.pose_no_prior_vo_accept_trans_hist.clear()
        self.pose_no_prior_vo_accept_rot_hist.clear()
        self.tool_static_counter = 0
        self.tool_motion_gate_state = bool(self.tool_motion_mode == 'flow_dir_std')
        self.tool_motion_static_streak = 0
        self.tool_motion_moving_streak = 0
        self.tool_motion_records = []
        self.last_motion_info = None
        self.last_pose_init_info = None
        self.last_pose_runtime_info = None
        self.last_deform_init_info = None
        self.first_frame_pose_trust_gate_info = None
        self.pose_debug_grad_records = []
        self.pt_curr_2d = None
        self.pt_prev_track_frame = None
        self.pt_cotracker_tracks = None
        self.pt_cotracker_vis = None
        self.pt_cotracker_tracks_deform = None
        self.pt_cotracker_vis_deform = None
        self.cotracker_deform_query_pose0 = None

        if self.pt_tracker is not None and self.pt_tracker_backend.startswith('cotracker3'):
            self._prepare_cotracker_tracks()

        for ids, gt_color, gt_color_r, gt_c2w, tool_mask, semantics in tqdm(self.frame_loader, total=self.n_img):
            frame_id = int(ids.item())
            gt_color = gt_color.cuda()
            gt_color_r = gt_color_r.cuda()
            gt_c2w = gt_c2w.cuda()
            tool_mask = tool_mask.cuda() if tool_mask is not None else None
            semantics = semantics.float().cuda() if semantics is not None else None
            stereo_depth, flow_valid = self._get_input_depth(
                frame_id=frame_id,
                gt_color=gt_color,
                gt_color_r=gt_color_r,
            )

            warmup_stats = None
            pose_step_stats = None
            pose_locked_to_prev_opt = False
            frame_pose = self._init_pose_for_frame(
                gt_c2w,
                frame_id,
                gt_color=gt_color,
                stereo_depth=stereo_depth,
                tool_mask=tool_mask,
            )
            frame_time = self._frame_time(frame_id)
            self.net.set_deformation_time(frame_time)
            self.camera.set_time(frame_time)
            self.visualizer.camera.set_time(frame_time)
            self.visualizer.widefield_camera.set_time(frame_time)
            motion_mask = self._resolve_motion_mask(tool_mask=tool_mask, semantics=semantics)
            motion_info = self._estimate_tool_motion(
                prev_color=self.last_frame,
                curr_color=gt_color,
                prev_tool_mask=self.prev_tool_mask_for_motion,
                curr_tool_mask=motion_mask,
            )
            pose_loss_mask = self._build_pose_loss_mask(tool_mask=tool_mask, semantics=semantics, motion_info=motion_info)
            pose_opt_allowed = bool(motion_info['pose_opt_allowed'])
            pose_opt_weight = float(motion_info.get('pose_opt_weight', 1.0 if pose_opt_allowed else 0.0))
            pose_lock_weight = float(motion_info.get('pose_lock_weight', 0.0 if pose_opt_allowed else 1.0))
            pose_motion_ratio = float(motion_info.get('tool_motion_ratio', 0.0))
            if frame_id == 0:
                pose_opt_allowed = True
                pose_opt_weight = 1.0
                pose_lock_weight = 0.0
                pose_motion_ratio = 0.0
            elif self.prev_optimized_pose_for_smooth is not None and pose_lock_weight > 0.0:
                frame_pose = self._blend_pose_towards_reference(
                    base_c2w=frame_pose,
                    ref_c2w=self.prev_optimized_pose_for_smooth,
                    weight=pose_lock_weight,
                    translation_only=self.tool_motion_lock_translation_only,
                )
                pose_locked_to_prev_opt = bool(pose_lock_weight > 1e-3)

            if frame_id == 0:
                self.net.create_from_pcd(gt_color, stereo_depth, frame_pose, self.camera, tool_mask, semantics=semantics)
                self.net.training_setup(self.cfg['training'])
                frame = ids, gt_color, stereo_depth, frame_pose, tool_mask
                optimized_c2w, pose_stats = self.fit(
                    frame,
                    iters=self.cfg['training']['iters_first'],
                    incremental=False,
                    pose_mask=pose_loss_mask,
                    pose_opt_weight=pose_opt_weight,
                    motion_ratio=pose_motion_ratio,
                )
            else:
                if frame_id == 1 and self.cfg['training']['grad_weighing']:
                    self.net.enable_grad_weighing(True)

                if pose_opt_weight > self.pose_soft_enable_threshold:
                    frame_pose, warmup_stats = self.warmup_pose(
                        gt_color,
                        stereo_depth,
                        frame_pose,
                        tool_mask,
                        pose_mask=pose_loss_mask,
                        pose_opt_weight=pose_opt_weight,
                        frame_id=frame_id,
                    )

                with torch.no_grad():
                    self.camera.set_c2w(frame_pose)
                    render_pkg = self._render_scene(deform=True)
                    mask = render_pkg['alpha'][None, ..., None].squeeze(-1) < 0.95
                    if tool_mask is not None:
                        mask &= tool_mask
                    if self.cfg['training']['add_points']:
                        self.net.add_from_pcd(gt_color, stereo_depth, frame_pose, self.camera, mask, semantics=semantics)

                    self.last_deform_init_info = {
                        'deform_init_used': False,
                        'deform_init_source': self.optical_flow_init_source,
                        'deform_init_reason': 'disabled',
                        'deform_init_frame': int(frame_id),
                    }
                    if self.cfg['training']['optical_flow_init']:
                        used = False
                        last_info = None
                        if self.optical_flow_init_source in ('cotracker3', 'hybrid', 'hybrid_foundation'):
                            used, info_ct = self._init_deformation_from_cotracker(
                                frame_id=frame_id,
                                frame_pose=frame_pose,
                                stereo_depth=stereo_depth,
                                tool_mask=tool_mask,
                            )
                            last_info = info_ct
                        if (not used) and self.optical_flow_init_source in ('raft', 'hybrid'):
                            used, info_rf = self._init_deformation_from_raft(
                                frame_id=frame_id,
                                gt_color=gt_color,
                                stereo_depth=stereo_depth,
                                tool_mask=tool_mask,
                                flow_valid=flow_valid,
                                render_pkg=render_pkg,
                            )
                            if (last_info is not None) and (not last_info.get('deform_init_used', False)):
                                for k in ('deform_init_reason', 'deform_init_points'):
                                    if k in last_info and k not in info_rf:
                                        info_rf[f'prev_{k}'] = last_info[k]
                            last_info = info_rf
                        if (not used) and self.optical_flow_init_source in ('foundation', 'hybrid_foundation'):
                            used, info_fd = self._init_deformation_from_foundation(
                                frame_id=frame_id,
                                gt_color=gt_color,
                                stereo_depth=stereo_depth,
                                tool_mask=tool_mask,
                                flow_valid=flow_valid,
                                render_pkg=render_pkg,
                            )
                            if (last_info is not None) and (not last_info.get('deform_init_used', False)):
                                for k in ('deform_init_reason', 'deform_init_points'):
                                    if k in last_info and k not in info_fd:
                                        info_fd[f'prev_{k}'] = last_info[k]
                            last_info = info_fd
                        if last_info is not None:
                            self.last_deform_init_info = {
                                **self.last_deform_init_info,
                                **last_info,
                            }
                        if (not used) and self.dbg:
                            print(
                                f"[FlowInit] skip frame={frame_id} source={self.optical_flow_init_source} "
                                f"reason={self.last_deform_init_info.get('deform_init_reason', 'unknown')} "
                                f"points={self.last_deform_init_info.get('deform_init_points', -1)}"
                            )

                frame = ids, gt_color, stereo_depth, frame_pose, tool_mask
                optimized_c2w, pose_stats = self.fit(
                    frame,
                    iters=self.cfg['training']['iters'],
                    incremental=True,
                    pose_mask=pose_loss_mask,
                    pose_opt_weight=pose_opt_weight,
                    motion_ratio=pose_motion_ratio,
                )

            if pose_opt_weight > self.pose_soft_enable_threshold and self.pose_step_limit_enabled:
                pose_step_ref = frame_pose if self.pose_step_limit_wrt_input else self.prev_optimized_pose_for_smooth
                step_max_trans, step_max_rot_deg, step_scale, step_reason = self._adaptive_pose_step_limits(
                    pose_stats=pose_stats,
                    motion_ratio=pose_motion_ratio,
                )
                optimized_c2w, pose_step_stats = self._limit_pose_step(
                    pose_step_ref,
                    optimized_c2w,
                    max_trans=step_max_trans,
                    max_rot_deg=step_max_rot_deg,
                )
                if pose_step_stats is not None:
                    pose_step_stats['pose_step_adaptive_scale'] = float(step_scale)
                    pose_step_stats['pose_step_adaptive_reason'] = step_reason

            self._maybe_apply_first_frame_pose_trust_gate(
                frame_id=frame_id,
                gt_color=gt_color,
                stereo_depth=stereo_depth,
                input_c2w=frame_pose,
                optimized_c2w=optimized_c2w,
                pose_mask=pose_loss_mask,
            )

            if (
                self.pt_tracker is not None
                and (not self.pt_tracker.is_initialized())
                and (
                    (self.pose_w_track_2d > 0.0)
                    or (self.pt_cotracker_gs_refine_enabled and frame_id == 0)
                )
            ):
                try:
                    self.pt_tracker.init_tracking_points(optimized_c2w, stereo_depth.squeeze(0))
                except Exception as exc:
                    warnings.warn(f"PointTracker init failed for pose optimization. reason: {exc}")

            self.last_frame = gt_color.detach()
            self.prev_depth_for_pose_init = stereo_depth.detach()
            self.prev_tool_mask_for_motion = motion_mask.detach() if motion_mask is not None else None
            input_pose_history.append(frame_pose.squeeze(0).detach().cpu())
            gt_pose_history.append(gt_c2w.squeeze(0).detach().cpu())
            optimized_pose_history.append(optimized_c2w.squeeze(0).detach().cpu())
            if frame_id == 0 and self.cotracker_deform_query_pose0 is None:
                self.cotracker_deform_query_pose0 = optimized_c2w.detach()
            self.prev_input_pose_for_smooth = frame_pose.detach()
            self.prev_optimized_pose_for_smooth = optimized_c2w.detach()
            motion_record = {
                'frame': frame_id,
                **motion_info,
                'pose_locked_to_prev_opt': bool(pose_locked_to_prev_opt),
                'pose_opt_weight_applied': float(pose_opt_weight),
                'pose_lock_weight_applied': float(pose_lock_weight),
            }
            if isinstance(self.last_pose_init_info, dict):
                for k, v in self.last_pose_init_info.items():
                    motion_record[k] = v
            if isinstance(self.last_pose_runtime_info, dict):
                for k, v in self.last_pose_runtime_info.items():
                    motion_record[k] = v
            if isinstance(self.last_deform_init_info, dict):
                for k, v in self.last_deform_init_info.items():
                    motion_record[k] = v
            if isinstance(self.first_frame_pose_trust_gate_info, dict):
                for k, v in self.first_frame_pose_trust_gate_info.items():
                    motion_record[k] = v
            if pose_step_stats is not None:
                motion_record.update(pose_step_stats)
            self.last_motion_info = motion_record
            self.tool_motion_records.append(motion_record)

            # eval
            with torch.no_grad():
                log_dict = {}
                if self.pt_tracker is not None:
                    if self.pt_tracker_backend == 'raft2d':
                        frame_index = int(ids.item())
                        if self.pt_curr_2d is None:
                            self.pt_curr_2d = self.pt_tracker.gt_2d_pts[:, 0].detach().clone()
                        else:
                            self.pt_curr_2d = self._track_points_with_raft2d(
                                self.pt_prev_track_frame,
                                gt_color,
                                self.pt_curr_2d,
                            )
                        self.pt_prev_track_frame = gt_color.detach()
                        pts_2d = self.pt_curr_2d
                        pts_2d_gt = self.pt_tracker.gt_2d_pts[:, frame_index]
                        valid_np = self.pt_tracker.valid[:, frame_index]
                        valid_t = torch.from_numpy(valid_np).to(device=pts_2d.device, dtype=torch.bool)
                        if int(valid_t.sum().item()) > 0:
                            l2_2d = torch.linalg.norm(pts_2d_gt - pts_2d, ord=2, dim=-1)[valid_t].mean()
                        else:
                            l2_2d = torch.tensor(float('nan'), device=pts_2d.device, dtype=pts_2d.dtype)
                        pts_3d = torch.zeros((pts_2d.shape[0], 3), device=pts_2d.device, dtype=pts_2d.dtype)
                        pt_track_stats["pred_2d"].append(pts_2d.detach().cpu().numpy())
                        pt_track_stats["pred_3d"].append(pts_3d.detach().cpu().numpy())
                        log_dict.update({'pt_track_l2_2d': l2_2d, 'frame': ids[0].item()})
                    elif self.pt_tracker_backend.startswith('cotracker3'):
                        frame_index = int(ids.item())
                        if self.pt_cotracker_tracks is None:
                            self._prepare_cotracker_tracks()
                        if self.pt_cotracker_tracks is not None and frame_index < int(self.pt_cotracker_tracks.shape[0]):
                            pts_2d = self.pt_cotracker_tracks[frame_index]
                            if self.pt_cotracker_gs_refine_enabled:
                                self._maybe_init_pt_tracker_for_gs_refine(
                                    frame_id=frame_index,
                                    c2w=optimized_c2w,
                                    stereo_depth=stereo_depth,
                                )
                                pts_2d, gs_refine_info = self._maybe_refine_cotracker_with_gs(
                                    frame_id=frame_index,
                                    cotracker_pts_2d=pts_2d,
                                    c2w=optimized_c2w,
                                )
                                if self.pt_cotracker_gs_refine_log:
                                    log_dict.update(
                                        {
                                            'pt_ct3_gs_refine_applied': float(gs_refine_info.get('applied', False)),
                                            'pt_ct3_gs_refine_num': float(gs_refine_info.get('num_refined', 0)),
                                            'pt_ct3_gs_refine_mean_raw_res_px': float(
                                                gs_refine_info.get('mean_raw_residual_px', float('nan'))
                                            ),
                                            'pt_ct3_gs_refine_mean_shift_px': float(
                                                gs_refine_info.get('mean_shift_px', 0.0)
                                            ),
                                        }
                                    )
                            pts_2d_gt = self.pt_tracker.gt_2d_pts[:, frame_index]
                            valid_np = self.pt_tracker.valid[:, frame_index]
                            valid_t = torch.from_numpy(valid_np).to(device=pts_2d.device, dtype=torch.bool)
                            if self.pt_cotracker_vis is not None and frame_index < int(self.pt_cotracker_vis.shape[0]):
                                valid_t &= self.pt_cotracker_vis[frame_index].to(device=pts_2d.device, dtype=torch.bool)
                            if int(valid_t.sum().item()) > 0:
                                l2_2d = torch.linalg.norm(pts_2d_gt - pts_2d, ord=2, dim=-1)[valid_t].mean()
                            else:
                                l2_2d = torch.tensor(float('nan'), device=pts_2d.device, dtype=pts_2d.dtype)
                            pts_3d = torch.zeros((pts_2d.shape[0], 3), device=pts_2d.device, dtype=pts_2d.dtype)
                            pt_track_stats["pred_2d"].append(pts_2d.detach().cpu().numpy())
                            pt_track_stats["pred_3d"].append(pts_3d.detach().cpu().numpy())
                            log_dict.update({'pt_track_l2_2d': l2_2d, 'frame': ids[0].item()})
                        else:
                            pts_2d, pts_2d_gt = None, None
                    else:
                        if not self.pt_tracker.is_initialized():
                            self.pt_tracker.init_tracking_points(optimized_c2w, stereo_depth.squeeze(0))
                        pts_3d_gt, pts_3d, pts_2d, l2_3d, l2_2d, pts_2d_gt = self.pt_tracker.eval(optimized_c2w, ids.item())
                        pt_track_stats["pred_2d"].append(pts_2d.cpu().numpy())
                        pt_track_stats["pred_3d"].append(pts_3d.cpu().numpy())
                        log_dict.update({'pt_track_l2_2d': l2_2d, 'frame': ids[0].item()})
                else:
                    pts_2d, pts_2d_gt = None, None

                if pose_stats is not None:
                    log_dict.update(
                        {
                            'pose_delta_rot_norm': pose_stats['delta_rot_norm'],
                            'pose_delta_trans_norm': pose_stats['delta_trans_norm'],
                            'pose_runtime_reverted_to_init': float(pose_stats.get('pose_reverted_to_init', 0.0)),
                        }
                    )
                    if 'final_track_loss' in pose_stats:
                        log_dict['pose_final_track_loss'] = float(pose_stats['final_track_loss'])
                    if 'final_track_threshold' in pose_stats:
                        log_dict['pose_final_track_threshold'] = float(pose_stats['final_track_threshold'])
                if warmup_stats is not None:
                    log_dict.update(
                        {
                            'pose_warmup_rot_norm': warmup_stats['delta_rot_norm'],
                            'pose_warmup_trans_norm': warmup_stats['delta_trans_norm'],
                        }
                    )
                if pose_step_stats is not None:
                    log_dict.update(
                        {
                            'pose_step_trans_before': pose_step_stats['pose_step_trans_before'],
                            'pose_step_rot_deg_before': pose_step_stats['pose_step_rot_deg_before'],
                            'pose_step_trans_after': pose_step_stats['pose_step_trans_after'],
                            'pose_step_rot_deg_after': pose_step_stats['pose_step_rot_deg_after'],
                            'pose_step_clamped': float(pose_step_stats['pose_step_clamped']),
                        }
                    )
                if self.tool_motion_gate_enabled:
                    log_dict.update(
                        {
                            'tool_motion_score': motion_info['tool_motion_score'],
                            'tool_flow_tool': motion_info['tool_flow_tool'],
                            'tool_flow_bg': motion_info['tool_flow_bg'],
                            'tool_iou_delta': motion_info['tool_iou_delta'],
                            'tool_centroid_shift': motion_info['tool_centroid_shift'],
                            'tool_static_confident': float(motion_info['tool_static_confident']),
                            'tool_moving_confident': float(motion_info['tool_moving_confident']),
                            'tool_static_streak': motion_info['tool_static_streak'],
                            'tool_moving_streak': motion_info['tool_moving_streak'],
                            'tool_gate_state': motion_info['tool_gate_state'],
                            'tool_moving': float(motion_info['tool_moving']),
                            'pose_opt_allowed': float(pose_opt_allowed),
                            'pose_opt_weight': float(pose_opt_weight),
                            'pose_lock_weight': float(pose_lock_weight),
                            'pose_locked_to_prev_opt': float(pose_locked_to_prev_opt),
                        }
                    )
                if self.last_pose_init_info is not None:
                    log_dict.update(
                        {
                            'pose_init_used_external': float(
                                self.last_pose_init_info.get('pose_init_used') == 'external'
                            ),
                            'pose_init_used_vo': float(self.last_pose_init_info.get('pose_init_used') == 'vo'),
                            'pose_init_used_vo_static_skip': float(
                                self.last_pose_init_info.get('pose_init_used') == 'vo_static_skip'
                            ),
                            'pose_init_used_prev_opt': float(
                                self.last_pose_init_info.get('pose_init_used') == 'prev_opt'
                            ),
                            'pose_init_used_identity': float(
                                self.last_pose_init_info.get('pose_init_used') == 'identity'
                            ),
                        }
                    )
                    if 'vo_trans_norm' in self.last_pose_init_info:
                        log_dict['pose_init_vo_trans_norm'] = float(self.last_pose_init_info['vo_trans_norm'])
                    if 'vo_rot_deg' in self.last_pose_init_info:
                        log_dict['pose_init_vo_rot_deg'] = float(self.last_pose_init_info['vo_rot_deg'])
                    if 'vo_valid_points' in self.last_pose_init_info:
                        log_dict['pose_init_vo_valid_points'] = float(self.last_pose_init_info['vo_valid_points'])
                if self.first_frame_pose_trust_gate_info is not None:
                    for k, v in self.first_frame_pose_trust_gate_info.items():
                        if isinstance(v, (bool, int, float, np.floating, np.integer)):
                            log_dict[k] = float(v)

                if self.visualize:
                    outmap, outsem, outrack = self.visualizer.save_imgs(
                        ids.item(), stereo_depth, gt_color, optimized_c2w, pts_2d, pts_2d_gt
                    )
                    if self.log:
                        log_dict.update(
                            {
                                'mapping': wandb.Image(outmap),
                                'tracking': wandb.Image(outrack) if outrack is not None else None,
                                'semantic': wandb.Image(outsem),
                            }
                        )
                if self.save_widefield_ply_every > 0:
                    self.visualizer.save_widefield_ply(
                        idx=ids.item(),
                        c2w=optimized_c2w,
                        every_n=self.save_widefield_ply_every,
                        alpha_threshold=self.widefield_ply_alpha_thr,
                        min_depth=self.widefield_ply_min_depth,
                        max_depth=self.widefield_ply_max_depth,
                    )
                if self.log:
                    wandb.log(log_dict)

        if self.log and self.pt_tracker is not None:
            gt_2d, valid = self.pt_tracker.get_gt_2d_pts()
            pred_2d = np.stack(pt_track_stats["pred_2d"], axis=1)
            H, W = self.camera.get_params()[:2]
            wandb.summary['MTE_2D'] = mte(pred_2d, gt_2d, valid)
            wandb.summary['delta_2D'] = delta_2d(pred_2d, gt_2d, valid, H, W)
            wandb.summary['survival_2D'] = surv_2d(pred_2d, gt_2d, valid, H, W)

        with open(os.path.join(self.output, 'tracked.pckl'), 'wb') as f:
            pickle.dump(pt_track_stats, f)

        if self.pose_opt_enabled and self.pose_save and len(optimized_pose_history) > 0:
            input_poses = torch.stack(input_pose_history, dim=0).numpy()
            optimized_poses = torch.stack(optimized_pose_history, dim=0).numpy()
            gt_poses = torch.stack(gt_pose_history, dim=0).numpy() if len(gt_pose_history) == len(optimized_pose_history) else None
            np.save(os.path.join(self.output, 'input_c2w.npy'), input_poses)
            np.save(os.path.join(self.output, 'optimized_c2w.npy'), optimized_poses)
            if gt_poses is not None:
                np.save(os.path.join(self.output, 'gt_c2w.npy'), gt_poses)
            input_trans = torch.from_numpy(input_poses[:, :3, 3])
            optimized_trans = torch.from_numpy(optimized_poses[:, :3, 3])
            pose_delta_trans = torch.linalg.norm(optimized_trans - input_trans, dim=1)
            input_rot = torch.from_numpy(input_poses[:, :3, :3])
            optimized_rot = torch.from_numpy(optimized_poses[:, :3, :3])
            rel_rot = torch.matmul(input_rot.transpose(1, 2), optimized_rot)
            cos_angle = ((rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]) - 1.0) * 0.5
            cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
            pose_delta_rot_deg = torch.rad2deg(torch.acos(cos_angle))
            print(
                "[PoseOpt] input->optimized "
                f"mean_trans={pose_delta_trans.mean().item():.6g}, "
                f"max_trans={pose_delta_trans.max().item():.6g}, "
                f"mean_rot_deg={pose_delta_rot_deg.mean().item():.6g}, "
                f"max_rot_deg={pose_delta_rot_deg.max().item():.6g}"
            )
            if gt_poses is not None:
                gt_trans = torch.from_numpy(gt_poses[:, :3, 3])
                pose_opt_to_gt_trans = torch.linalg.norm(optimized_trans - gt_trans, dim=1)
                gt_rot = torch.from_numpy(gt_poses[:, :3, :3])
                rel_opt_gt = torch.matmul(gt_rot.transpose(1, 2), optimized_rot)
                cos_opt_gt = ((rel_opt_gt[:, 0, 0] + rel_opt_gt[:, 1, 1] + rel_opt_gt[:, 2, 2]) - 1.0) * 0.5
                cos_opt_gt = torch.clamp(cos_opt_gt, min=-1.0, max=1.0)
                pose_opt_to_gt_rot_deg = torch.rad2deg(torch.acos(cos_opt_gt))
                print(
                    "[PoseOpt] optimized->gt "
                    f"mean_trans={pose_opt_to_gt_trans.mean().item():.6g}, "
                    f"max_trans={pose_opt_to_gt_trans.max().item():.6g}, "
                    f"mean_rot_deg={pose_opt_to_gt_rot_deg.mean().item():.6g}, "
                    f"max_rot_deg={pose_opt_to_gt_rot_deg.max().item():.6g}"
                )
            if self.log:
                wandb.summary['pose_input_to_opt_mean_trans'] = pose_delta_trans.mean().item()
                wandb.summary['pose_input_to_opt_max_trans'] = pose_delta_trans.max().item()
                wandb.summary['pose_input_to_opt_mean_rot_deg'] = pose_delta_rot_deg.mean().item()
                wandb.summary['pose_input_to_opt_max_rot_deg'] = pose_delta_rot_deg.max().item()
                if gt_poses is not None:
                    wandb.summary['pose_opt_to_gt_mean_trans'] = pose_opt_to_gt_trans.mean().item()
                    wandb.summary['pose_opt_to_gt_max_trans'] = pose_opt_to_gt_trans.max().item()
                    wandb.summary['pose_opt_to_gt_mean_rot_deg'] = pose_opt_to_gt_rot_deg.mean().item()
                    wandb.summary['pose_opt_to_gt_max_rot_deg'] = pose_opt_to_gt_rot_deg.max().item()

        if self.tool_motion_gate_enabled and len(self.tool_motion_records) > 0:
            motion_csv = os.path.join(self.output, 'tool_motion_score.csv')
            motion_json = os.path.join(self.output, 'tool_motion_score.json')
            motion_fields = []
            for record in self.tool_motion_records:
                for key in record.keys():
                    if key not in motion_fields:
                        motion_fields.append(key)
            with open(motion_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=motion_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.tool_motion_records)
            with open(motion_json, 'w') as f:
                json.dump(self.tool_motion_records, f, indent=2)

        if len(self.pose_debug_grad_records) > 0:
            grad_csv = os.path.join(self.output, 'pose_grad_debug.csv')
            grad_json = os.path.join(self.output, 'pose_grad_debug.json')
            grad_fields = []
            for record in self.pose_debug_grad_records:
                for key in record.keys():
                    if key not in grad_fields:
                        grad_fields.append(key)
            with open(grad_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=grad_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.pose_debug_grad_records)
            with open(grad_json, 'w') as f:
                json.dump(self.pose_debug_grad_records, f, indent=2)

        print('...finished')


if __name__ == "__main__":
    from src.config import load_config
    import random

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('config', type=str)
    parser.add_argument(
        '--input_folder',
        type=str,
        help='input folder, this have higher priority, can overwrite the one in config file',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='output folder, this have higher priority, can overwrite the one in config file',
    )
    parser.add_argument(
        '--pose_file',
        type=str,
        help='optional pose file path (e.g. groundtruth_noisy.txt), defaults to groundtruth.txt in input folder',
    )
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--log', type=str)
    parser.add_argument('--log_group', type=str, default='default')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save_widefield_ply_every', type=int, default=0)
    parser.add_argument('--widefield_ply_alpha_thr', type=float, default=0.8)
    parser.add_argument('--widefield_ply_min_depth', type=float, default=1e-6)
    parser.add_argument('--widefield_ply_max_depth', type=float, default=0.0)

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/base.yaml')
    cfg['data']['output'] = args.output if args.output else cfg['data']['output']

    trainer = SceneOptimizer(cfg, args)
    trainer.run()
