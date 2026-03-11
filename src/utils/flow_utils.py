import numpy as np
import torch
import cv2
import os
import sys
from pathlib import Path

try:
    from FoundationStereo.core.utils.utils import InputPadder
except Exception:
    InputPadder = None


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    b = c2w.shape[0]
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1).to(device)

    dirs = dirs.reshape(1, -1, 1, 3).expand(b, -1, -1, -1)
    rays_d = torch.sum(dirs * c2w[:, None, :3, :3], -1)
    rays_o = c2w[:, :3, -1][:, None, :].expand(rays_d.shape)
    return rays_o, rays_d, dirs.squeeze(-2)


def get_surface_pts(depth, fx, fy, cx, cy, c2w, device):
    b, H, W = depth.shape
    rays_o, rays_d, _ = get_rays(H, W, fx, fy, cx, cy, c2w, device)
    pts = rays_o + depth.view(b, -1, 1)*rays_d
    return pts.view(b,H,W,3)


def reproject(pts2d, depth, fx, fy, cx, cy, c2ws):
    dirs = torch.stack([(pts2d[..., 0] - cx) / fx, (pts2d[...,1] - cy) / fy, torch.ones_like(pts2d[...,0], device=pts2d[0].device)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2ws[:, :3, :3], -1)
    rays_o = c2ws[:, :3, -1].expand(rays_d.shape)
    pts = rays_o + depth[pts2d[..., 1], pts2d[..., 0], None] * rays_d
    return pts


def remap_from_flow(x, flow):
    # get optical flow correspondences
    n, _, h, w = flow.shape
    row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    flow_off = torch.empty_like(flow)
    flow_off[:, 1] = 2 * (flow[:, 1] + row_coords.to(flow.device)) / (h - 1) - 1
    flow_off[:, 0] = 2 * (flow[:, 0] + col_coords.to(flow.device)) / (w - 1) - 1
    x = torch.nn.functional.grid_sample(x, flow_off.permute(0, 2, 3, 1), align_corners=True)
    valid = (x > 0).any(dim=1).unsqueeze(1)
    return x, valid


def get_scene_flow(raft, img1, img2, depth1, depth2, mask, camera):
    optical_flow = raft(2 * img1.permute(0, 3, 1, 2) - 1.0, 2 * img2.permute(0, 3, 1, 2) - 1.0)[-1]
    depth_interp = depth2.clone()
    depth_interp[~mask] = depth1[~mask]
    H, W, fx, fy, cx, cy = camera.get_params()
    src_pts = get_surface_pts(depth1, fx, fy, cx, cy, camera.c2w, depth1.device)
    target_pts = get_surface_pts(depth2, fx, fy, cx, cy, camera.c2w, depth2.device)
    target_remapped, valid = remap_from_flow(target_pts.permute(0,3,1,2), optical_flow)
    scene_flow = target_remapped.permute(0,2,3,1) - src_pts
    return scene_flow.squeeze(), src_pts.squeeze(), valid.squeeze()


def get_depth_from_raft(raft, img1, img2, baseline):
    flow = raft(2 * img1.permute(0, 3, 1, 2) - 1.0, 2 * img2.permute(0, 3, 1, 2) - 1.0)[-1]
    baseline_t = baseline * torch.ones_like(flow[:, 0])
    depth = baseline_t / -flow[:, 0]
    depth = torch.from_numpy(
        cv2.bilateralFilter(depth.cpu().numpy().squeeze(), d=-1, sigmaColor=2.5, sigmaSpace=2.5)).cuda().unsqueeze(0)
    valid = flow[:, 1].abs() < 1.5
    return depth, valid


def _require_foundation():
    global InputPadder
    if InputPadder is None:
        candidates = []
        env_root = os.environ.get("FOUNDATION_STEREO_ROOT", "").strip()
        if env_root:
            candidates.append(os.path.abspath(os.path.expanduser(env_root)))
        repo_default = Path(__file__).resolve().parents[2] / "foundationstereo"
        candidates.append(str(repo_default.resolve()))

        for root in candidates:
            if not root:
                continue
            if root not in sys.path:
                sys.path.append(root)
            root_parent = os.path.dirname(root)
            if root_parent and root_parent not in sys.path:
                sys.path.append(root_parent)
            try:
                from FoundationStereo.core.utils.utils import InputPadder as _InputPadder
                InputPadder = _InputPadder
                break
            except Exception:
                continue

    if InputPadder is None:
        raise ImportError(
            "FoundationStereo is not available. Set data.depth_input_source=raft_stereo "
            "or install FoundationStereo and provide foundation config paths."
        )


def get_depth_from_foundation(
    depth_model,
    color_data,
    right_color_data,
    device,
    fs_args,
    depth_scale=10.0,
    depth_formula='legacy_fx_baseline',
    override_fx=None,
    override_baseline=None,
):
    _require_foundation()
    _, H, W, _ = color_data.shape
    img0 = color_data * 255.0
    img1 = right_color_data * 255.0
    img0 = img0.permute(0, 3, 1, 2).to(device)
    img1 = img1.permute(0, 3, 1, 2).to(device)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.amp.autocast('cuda', enabled=True):
        disp = depth_model.forward(img0, img1, iters=fs_args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())

    disp_np = disp.data.cpu().numpy().reshape(H, W)
    disp_np = np.where(np.isfinite(disp_np), disp_np, 0.0).astype(np.float32)

    disp_safe = np.maximum(disp_np, 1e-6)
    formula = str(depth_formula).lower().strip()
    if formula == 'raft_compatible':
        # Match the legacy RAFT-stereo convention in this repository:
        # depth = baseline_scaled / disparity
        # where baseline_scaled already contains global scene scale.
        baseline_scaled = float(override_baseline) if override_baseline is not None else float(fs_args.baseline) * float(depth_scale)
        depth_np = baseline_scaled / disp_safe
    elif formula == 'seq_fx_baseline':
        # Sequence-specific calibrated conversion:
        # depth = fx * baseline / disparity * depth_scale.
        fx = float(override_fx) if override_fx is not None else float(fs_args.K[0][0])
        baseline = float(override_baseline) if override_baseline is not None else float(fs_args.baseline)
        depth_np = (fx * baseline / disp_safe) * float(depth_scale)
    else:
        # Backward compatible behavior.
        fx = float(fs_args.K[0][0])
        baseline = float(fs_args.baseline)
        depth_np = (fx * baseline / disp_safe) * float(depth_scale)
    depth_np = cv2.bilateralFilter(depth_np, d=-1, sigmaColor=2.5, sigmaSpace=2.5)

    depth = torch.from_numpy(depth_np).to(device).unsqueeze(0)

    disp_tensor = torch.from_numpy(disp_np).to(device)
    valid = disp_tensor > 0
    valid_disp = disp_tensor[valid]
    if valid_disp.numel() > 0:
        disp_max = torch.quantile(valid_disp, 0.95) * 2.0
        valid = valid & (disp_tensor < disp_max)
    xx = torch.arange(W, device=device).expand(H, W)
    us_right = xx - disp_tensor
    valid = valid & (us_right >= 0)
    valid = valid.unsqueeze(0)
    return depth, valid


def get_scene_flow_from_foundation(
    foundation_model,
    rendered_left,
    gt_left,
    rendered_depth,
    gt_depth,
    tool_mask,
    camera,
    fs_args,
    device,
):
    """
    Compute scene flow using FoundationStereo-based pseudo optical flow.

    This mirrors the online-dynamic-gs integration path:
    - run FoundationStereo on (rendered_left, gt_left)
    - interpret disparity as horizontal flow component
    - backproject + remap to obtain 3D scene flow.
    """
    _require_foundation()
    _, H, W, _ = rendered_left.shape
    img1_fs = rendered_left.permute(0, 3, 1, 2) * 255.0
    img2_fs = gt_left.permute(0, 3, 1, 2) * 255.0

    padder = InputPadder(img1_fs.shape, divis_by=32, force_square=False)
    img1_pad, img2_pad = padder.pad(img1_fs, img2_fs)

    with torch.amp.autocast('cuda', enabled=True):
        disp = foundation_model.forward(img1_pad, img2_pad, iters=fs_args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())

    if disp.dim() == 2:
        disp = disp.unsqueeze(0)

    optical_flow = torch.zeros(img1_fs.shape[0], 2, H, W, device=device)
    optical_flow[:, 0] = -disp

    depth_interp = gt_depth.clone()
    depth_interp[~tool_mask] = rendered_depth[~tool_mask]
    H, W, fx, fy, cx, cy = camera.get_params()
    src_pts = get_surface_pts(rendered_depth, fx, fy, cx, cy, camera.c2w, rendered_depth.device)
    target_pts = get_surface_pts(depth_interp, fx, fy, cx, cy, camera.c2w, depth_interp.device)
    target_remapped, valid = remap_from_flow(target_pts.permute(0, 3, 1, 2), optical_flow)
    scene_flow = target_remapped.permute(0, 2, 3, 1) - src_pts

    disp_valid = disp > 0
    if disp_valid.dim() != valid.dim():
        if disp_valid.dim() < valid.dim():
            while disp_valid.dim() < valid.dim():
                disp_valid = disp_valid.unsqueeze(1)
        else:
            disp_valid = disp_valid.squeeze()
            if disp_valid.dim() < valid.dim():
                disp_valid = disp_valid.unsqueeze(1)
    valid = valid & disp_valid

    return scene_flow.squeeze(), src_pts.squeeze(), valid.squeeze()
