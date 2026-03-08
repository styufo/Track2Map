import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import KDTree
from src.utils.general_utils import build_rotation
from math import ceil


class ExplicitDeformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.means_def = torch.nn.Parameter(torch.zeros(0, 3))
        self.rot_def = torch.nn.Parameter(torch.zeros(0, 4))
        self.cuda()
        self.means_cache, self.rot_cache = [None,None], [None,None]
        self.neighbour_dists = None
        self.neighbours = None
        self.neighbour_weights = None

    def forward(self, means, scales, rot, init=False, time=None):
        means = means + self.means_def
        rot = rot + self.rot_def
        if init:
            self.means_cache[1] = means.detach()
            self.rot_cache[1] = rot.detach()
        self.means_cache[0] = means
        self.rot_cache[0] = rot
        return means, scales, rot

    def get_mean_def(self, means):
        return self.means_def

    def get_deformed_means(self, means):
        return means + self.means_def

    def add_gaussians(self, n_new_gaussians: int, means):
        self.means_def = torch.nn.Parameter(torch.cat((self.means_def.data, torch.zeros(n_new_gaussians, 3, device='cuda')), dim=0))
        self.rot_def = torch.nn.Parameter(torch.cat((self.rot_def.data, torch.zeros(n_new_gaussians, 4, device='cuda')), dim=0))
        self.update_topology(means) if self.neighbours is not None else self.init_topology(means)

    def replace(self, param_list, means, reinit):
        self.means_def = param_list[0]
        self.rot_def = param_list[1]
        if reinit or self.neighbours is None:
            self.init_topology(means)
        else:
            self.update_topology(means)

    @torch.no_grad()
    def init_topology(self, means, k=20):
        """
            init topology finding k-nearest neighbours for each point to regularize deformation
        """
        tree = KDTree(means.detach().cpu().numpy())
        neighbour_dists, neighbours = tree.query(means.detach().cpu().numpy(), k=k)#, eps=0.1)
        self.neighbour_dists = torch.tensor(neighbour_dists[:, 1:], device="cuda")
        self.neighbours = torch.tensor(neighbours[:, 1:], device="cuda")
        self.neighbour_weights = torch.exp(-50*(self.neighbour_dists))
        self.means_cache, self.rot_cache = [None, None], [None, None]

    @torch.no_grad()
    def update_topology(self, means, k=20):
        # assume that new points are added to the end of the tensor and none are removed
        new_means = means[self.neighbours.shape[0]:]
        if new_means.shape[0] > 0:
            tree = KDTree(means.detach().cpu().numpy())
            neighbour_dists, neighbours = tree.query(new_means.detach().cpu().numpy(), k=k)#, eps=0.1)
            self.neighbour_dists = torch.cat((self.neighbour_dists, torch.tensor(neighbour_dists[:, 1:], device="cuda")), dim=0)
            self.neighbours = torch.cat((self.neighbours, torch.tensor(neighbours[:, 1:], device="cuda")), dim=0)
            self.neighbour_weights = torch.exp(-10.0*(self.neighbour_dists))
            self.means_cache, self.rot_cache = [None, None], [None, None]

    def reg_loss(self, visibility_filter):
        if self.means_cache[1] is None:
            l = torch.zeros(1, device="cuda").squeeze()
            return l, l, l, l
        prev_rot = build_rotation(self.rot_cache[1])
        cur_rot = build_rotation(self.rot_cache[0])
        rel_rot = prev_rot @ cur_rot.transpose(1,2)
        cur_offset = self.means_cache[0][self.neighbours] - self.means_cache[0][:,None]
        last_offset = self.means_cache[1][self.neighbours] - self.means_cache[1][:,None]
        rot_offset = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(rot_offset, dim=-1) * self.neighbour_weights).mean()

        l_rot = torch.sqrt(((rel_rot[:,None]-rel_rot[self.neighbours]) ** 2).sum(-1).sum(-1) * self.neighbour_weights + 1e-20).mean()
        curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists)*self.neighbour_weights).mean()
        ids = ~visibility_filter
        l_visible = self.means_def[ids].abs().sum() / (ids.sum()+1)
        return l_rigid, l_rot, l_iso, l_visible

    def get_new_params(self, shape):
        new_shape = shape[0]
        new_means = torch.zeros(new_shape, 3, device='cuda')
        new_rot = torch.zeros(new_shape, 4, device='cuda')
        return new_means, new_rot

    @torch.no_grad()
    def init_from_flow(self, deformation, weights):
        weight_sum = weights.sum(-1)
        deformation = (weights[..., None] * deformation).sum(1) / weight_sum[..., None]
        deformation[weight_sum < 0.1] = 0.0
        self.means_def += deformation.clamp(-0.01, 0.01)


class ExplicitSparseDeformation(ExplicitDeformation):
    def __init__(
        self,
        subsample: int = 64,
        anchor_sampling: str = "random",
        anchor_grid_cell_scale: float = 1.0,
    ):
        super().__init__()
        self.anchor_ids = None
        self.subsample = subsample
        self.control_pts = None
        self.anchor_sampling = str(anchor_sampling).lower()
        self.anchor_grid_cell_scale = float(max(anchor_grid_cell_scale, 1e-3))

    def interpolate(self, time=None):
        """
            fast approximation to Gaussian Kernel Interpolation using pre-computed weights of k-most important
            control points
        """
        assert self.means_def.shape[0] > self.neighbours.max()
        means_def = (self.neighbour_weights[...,None] * self.means_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[...,None]
        rot_def = (self.neighbour_weights[...,None] * self.rot_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[...,None]
        return means_def, rot_def

    def forward(self, means, scales, rot, init=False, time=None):
        means_def, rot_def = self.interpolate(time=time)
        means_deformed = means + means_def
        rot_deformed = rot + rot_def
        if init:
            self.means_cache[1] = (means[self.anchor_ids] + self.means_def).detach()
            self.rot_cache[1] = (rot[self.anchor_ids] + self.rot_def).detach()
        self.means_cache[0] = means[self.anchor_ids] + self.means_def
        self.rot_cache[0] = rot[self.anchor_ids] + self.rot_def
        return means_deformed, scales, rot_deformed

    def get_mean_def(self, means):
        return self.interpolate()[0]

    def get_deformed_means(self, means):
        return means + self.interpolate()[0]

    def get_new_params(self, shape):
        new_shape = ceil(shape[0]/self.subsample)
        new_means = torch.zeros(new_shape, 3, device='cuda')
        new_rot = torch.zeros(new_shape, 4, device='cuda')
        return new_means, new_rot

    def _select_anchor_ids(self, candidate_ids: np.ndarray, means_np: np.ndarray) -> np.ndarray:
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids.size == 0:
            return candidate_ids
        target = int(max(1, ceil(candidate_ids.size / max(self.subsample, 1))))

        if self.anchor_sampling != "grid_random":
            sampled = np.random.permutation(candidate_ids)[:: self.subsample]
            if sampled.size == 0:
                sampled = candidate_ids[:1]
            return sampled.astype(np.int64, copy=False)

        pts = means_np[candidate_ids]
        pts_min = pts.min(axis=0, keepdims=True)
        pts_max = pts.max(axis=0, keepdims=True)
        extent = np.maximum(pts_max - pts_min, 1e-6)
        volume = float(extent[0, 0] * extent[0, 1] * extent[0, 2])
        base_cell = max((volume / max(target, 1)) ** (1.0 / 3.0), 1e-6)
        cell_size = base_cell * self.anchor_grid_cell_scale
        cell_size = max(cell_size, 1e-6)

        grid_ids = np.floor((pts - pts_min) / cell_size).astype(np.int64)
        _, inv = np.unique(grid_ids, axis=0, return_inverse=True)
        selected_local = []
        for group_idx in range(int(inv.max()) + 1):
            members = np.where(inv == group_idx)[0]
            if members.size == 0:
                continue
            selected_local.append(int(np.random.choice(members)))
        selected_local = np.asarray(selected_local, dtype=np.int64)
        if selected_local.size == 0:
            selected = candidate_ids[:1]
        else:
            selected = candidate_ids[selected_local]

        if selected.size > target:
            keep = np.random.permutation(selected.shape[0])[:target]
            selected = selected[keep]
        elif selected.size < target:
            selected_set = set(selected.tolist())
            remain = np.asarray([idx for idx in candidate_ids.tolist() if idx not in selected_set], dtype=np.int64)
            if remain.size > 0:
                add_num = min(target - selected.size, remain.size)
                extra = remain[np.random.permutation(remain.shape[0])[:add_num]]
                selected = np.concatenate([selected, extra], axis=0)

        if selected.size == 0:
            selected = candidate_ids[:1]
        return selected.astype(np.int64, copy=False)

    def add_gaussians(self, n_new_gaussians: int, means: torch.Tensor):
        means_np = means.detach().cpu().numpy()
        if self.neighbours is not None:
            anchor_ids = self._select_anchor_ids(
                np.arange(self.neighbours.shape[0], means_np.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(anchor_ids, device='cuda')))
        else:
            anchor_ids = self._select_anchor_ids(
                np.arange(means_np.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda')

        self.means_def = torch.nn.Parameter(torch.cat((self.means_def.data, torch.zeros(self.anchor_ids.shape[0], 3, device='cuda')), dim=0))
        self.rot_def = torch.nn.Parameter(torch.cat((self.rot_def.data, torch.zeros(self.anchor_ids.shape[0], 4, device='cuda')), dim=0))
        self.update_topology(means, self.anchor_ids) if self.neighbours is not None else self.init_topology(means, self.anchor_ids)

    def replace(self, param_list, means, reinit):
        self.means_def = param_list[0]
        self.rot_def = param_list[1]
        means_np = means.detach().cpu().numpy()
        if reinit or self.neighbours is None:
            anchor_ids = self._select_anchor_ids(
                np.arange(means.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda')
            self.init_topology(means, self.anchor_ids)
        else:
            anchor_ids = self._select_anchor_ids(
                np.arange(self.neighbours.shape[0], means.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(anchor_ids, device='cuda')))
            self.update_topology(means, self.anchor_ids)

    @torch.no_grad()
    def init_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        """
            select a subset of points from the collection of Gaussians as control points
            init topology between control points
            init Gaussian kernel weights for all Gaussians for fast interpolation
        """
        means_np = means.detach().cpu().numpy()
        anchor_ids_np = anchor_ids.cpu().numpy()
        tree = KDTree(means_np[anchor_ids_np])
        neighbour_dists, neighbours = tree.query(means_np, k=k, eps=eps)

        self.control_pts = means[anchor_ids].detach()
        self.neighbour_dists = torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)
        self.neighbours = torch.tensor(neighbours, device="cuda")
        self.neighbour_weights = torch.exp(-4.5*(self.neighbour_dists))
        self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
        self.means_cache, self.rot_cache = [None, None], [None, None]

    @torch.no_grad()
    def update_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        # assume that new points are added to the end of the tensor and none are removed
        means_np = means.detach().cpu().numpy()
        new_means = means_np[self.neighbours.shape[0]:]
        anchor_ids_np = anchor_ids.cpu().numpy()
        if new_means.shape[0] > 0:
            tree = KDTree(means_np[anchor_ids_np])
            neighbour_dists, neighbours = tree.query(new_means, k=k, eps=eps)
            self.control_pts = means[anchor_ids].detach()
            self.neighbour_dists = torch.cat((self.neighbour_dists, torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)), dim=0)
            self.neighbours = torch.cat((self.neighbours, torch.tensor(neighbours, device="cuda")), dim=0)
            self.neighbour_weights = torch.exp(-4.5*(self.neighbour_dists))
            self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
            self.means_cache, self.rot_cache = [None, None], [None, None]

    def reg_loss(self, visibility_filter):
        if self.means_cache[1] is None:
            l = torch.zeros(1, device="cuda").squeeze()
            return l, l, l, l
        prev_rot = build_rotation(self.rot_cache[1])
        cur_rot = build_rotation(self.rot_cache[0])
        rel_rot = prev_rot @ cur_rot.transpose(1,2)
        cur_offset = self.means_cache[0][self.neighbours[self.anchor_ids]] - self.means_cache[0][:,None]
        last_offset = self.means_cache[1][self.neighbours[self.anchor_ids]] - self.means_cache[1][:,None]
        rot_offset = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(rot_offset, dim=-1) * self.neighbour_weights[self.anchor_ids]).mean()

        l_rot = torch.sqrt(((rel_rot[:,None]-rel_rot[self.neighbours[self.anchor_ids]]) ** 2).sum(-1).sum(-1) * self.neighbour_weights[self.anchor_ids] + 1e-20).mean()

        curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists[self.anchor_ids])*self.neighbour_weights[self.anchor_ids]).mean()
        ids = ~visibility_filter[self.anchor_ids]
        l_visible = self.means_def[ids].abs().sum() / (ids.sum()+1) # avoid nan for mean() if empty slice
        return l_rigid, l_rot, l_iso, l_visible

    @torch.no_grad()
    def init_from_flow(self, deformation, weights):
        weight_sum = weights.sum(-1)
        deformation = (weights[..., None] * deformation).sum(1) / weight_sum[..., None]
        deformation[weight_sum < 0.1] = 0.0
        self.means_def += deformation[self.anchor_ids].clamp(-0.01, 0.01).float()


class ExplicitSparseFDMDeformation(ExplicitSparseDeformation):
    def __init__(
        self,
        subsample: int = 64,
        basis_num: int = 9,
        basis_sigma: float = 0.18,
        normalize_basis: bool = True,
        anchor_sampling: str = "random",
        anchor_grid_cell_scale: float = 1.0,
    ):
        super().__init__(
            subsample=subsample,
            anchor_sampling=anchor_sampling,
            anchor_grid_cell_scale=anchor_grid_cell_scale,
        )
        self.basis_num = int(max(2, basis_num))
        self.basis_sigma = float(max(1e-3, basis_sigma))
        self.normalize_basis = bool(normalize_basis)
        self.current_time = 0.0

        self.register_buffer("basis_centers", torch.linspace(0.0, 1.0, self.basis_num))
        self.register_buffer("basis_sigmas", torch.full((self.basis_num,), self.basis_sigma))
        self.means_def = torch.nn.Parameter(torch.zeros(0, 3, self.basis_num, device='cuda'))
        self.rot_def = torch.nn.Parameter(torch.zeros(0, 4, self.basis_num, device='cuda'))

    def set_time(self, time: float):
        if time is None:
            return
        self.current_time = float(np.clip(time, 0.0, 1.0))

    def _basis_weights(self, time: float = None):
        t = self.current_time if time is None else float(np.clip(time, 0.0, 1.0))
        centers = self.basis_centers
        sigmas = self.basis_sigmas
        t_tensor = torch.tensor(t, device=centers.device, dtype=centers.dtype)
        basis = torch.exp(-0.5 * ((t_tensor - centers) / (sigmas + 1e-8)) ** 2)
        if self.normalize_basis:
            basis = basis / basis.sum().clamp_min(1e-8)
        return basis

    def _anchor_def_at_time(self, time: float = None):
        basis = self._basis_weights(time=time).to(device=self.means_def.device, dtype=self.means_def.dtype)
        means_def = torch.tensordot(self.means_def, basis, dims=([2], [0]))
        rot_def = torch.tensordot(self.rot_def, basis, dims=([2], [0]))
        return means_def, rot_def

    def interpolate(self, n_points: int = None, time: float = None):
        if self.neighbours is None or self.neighbour_weights is None or self.means_def.shape[0] == 0:
            count = 0 if n_points is None else int(n_points)
            return (
                torch.zeros(count, 3, device='cuda', dtype=torch.float32),
                torch.zeros(count, 4, device='cuda', dtype=torch.float32),
            )
        anchor_means_def, anchor_rot_def = self._anchor_def_at_time(time=time)
        means_def = (self.neighbour_weights[..., None] * anchor_means_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[..., None]
        rot_def = (self.neighbour_weights[..., None] * anchor_rot_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[..., None]
        return means_def, rot_def

    def forward(self, means, scales, rot, init=False, time=None):
        if time is not None:
            self.set_time(time)
        means_def, rot_def = self.interpolate(n_points=means.shape[0], time=self.current_time)
        means_deformed = means + means_def
        rot_deformed = rot + rot_def
        anchor_means_def, anchor_rot_def = self._anchor_def_at_time(time=self.current_time)
        if init:
            self.means_cache[1] = (means[self.anchor_ids] + anchor_means_def).detach()
            self.rot_cache[1] = (rot[self.anchor_ids] + anchor_rot_def).detach()
        self.means_cache[0] = means[self.anchor_ids] + anchor_means_def
        self.rot_cache[0] = rot[self.anchor_ids] + anchor_rot_def
        return means_deformed, scales, rot_deformed

    def get_mean_def(self, means):
        means_def, _ = self.interpolate(n_points=means.shape[0], time=self.current_time)
        return means_def

    def get_deformed_means(self, means):
        means_def, _ = self.interpolate(n_points=means.shape[0], time=self.current_time)
        return means + means_def

    def get_new_params(self, shape):
        new_shape = ceil(shape[0] / self.subsample)
        new_means = torch.zeros(new_shape, 3, self.basis_num, device='cuda')
        new_rot = torch.zeros(new_shape, 4, self.basis_num, device='cuda')
        return new_means, new_rot

    def add_gaussians(self, n_new_gaussians: int, means: torch.Tensor):
        means_np = means.detach().cpu().numpy()
        if self.neighbours is not None:
            new_candidates = np.arange(self.neighbours.shape[0], means_np.shape[0], dtype=np.int64)
            new_anchor_ids = self._select_anchor_ids(new_candidates, means_np=means_np)
            if new_anchor_ids.size > 0:
                self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(new_anchor_ids, device='cuda', dtype=torch.long)))
                self.means_def = torch.nn.Parameter(
                    torch.cat((self.means_def.data, torch.zeros(new_anchor_ids.shape[0], 3, self.basis_num, device='cuda')), dim=0)
                )
                self.rot_def = torch.nn.Parameter(
                    torch.cat((self.rot_def.data, torch.zeros(new_anchor_ids.shape[0], 4, self.basis_num, device='cuda')), dim=0)
                )
            self.update_topology(means, self.anchor_ids)
        else:
            anchor_ids = self._select_anchor_ids(
                np.arange(means_np.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda', dtype=torch.long)
            self.means_def = torch.nn.Parameter(torch.zeros(self.anchor_ids.shape[0], 3, self.basis_num, device='cuda'))
            self.rot_def = torch.nn.Parameter(torch.zeros(self.anchor_ids.shape[0], 4, self.basis_num, device='cuda'))
            self.init_topology(means, self.anchor_ids)

    def replace(self, param_list, means, reinit):
        self.means_def = param_list[0]
        self.rot_def = param_list[1]
        means_np = means.detach().cpu().numpy()
        if reinit or self.neighbours is None:
            anchor_ids = self._select_anchor_ids(
                np.arange(means.shape[0], dtype=np.int64),
                means_np=means_np,
            )
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda', dtype=torch.long)
            self.init_topology(means, self.anchor_ids)
        else:
            new_candidates = np.arange(self.neighbours.shape[0], means.shape[0], dtype=np.int64)
            new_anchor_ids = self._select_anchor_ids(new_candidates, means_np=means_np)
            if new_anchor_ids.size > 0:
                self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(new_anchor_ids, device='cuda', dtype=torch.long)))
            self.update_topology(means, self.anchor_ids)

    @torch.no_grad()
    def init_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        means_np = means.detach().cpu().numpy()
        anchor_ids_np = anchor_ids.cpu().numpy()
        if anchor_ids_np.shape[0] == 0:
            anchor_ids_np = np.array([0], dtype=np.int64)
            self.anchor_ids = torch.tensor(anchor_ids_np, device='cuda', dtype=torch.long)
        k_eff = int(max(1, min(k, anchor_ids_np.shape[0])))
        tree = KDTree(means_np[anchor_ids_np])
        neighbour_dists, neighbours = tree.query(means_np, k=k_eff, eps=eps)
        if k_eff == 1:
            neighbour_dists = neighbour_dists[:, None]
            neighbours = neighbours[:, None]
        self.control_pts = means[self.anchor_ids].detach()
        self.neighbour_dists = torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)
        self.neighbours = torch.tensor(neighbours, device="cuda", dtype=torch.long)
        self.neighbour_weights = torch.exp(-4.5 * (self.neighbour_dists))
        self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
        self.means_cache, self.rot_cache = [None, None], [None, None]

    @torch.no_grad()
    def update_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        means_np = means.detach().cpu().numpy()
        new_means = means_np[self.neighbours.shape[0]:]
        anchor_ids_np = anchor_ids.cpu().numpy()
        if new_means.shape[0] > 0:
            k_eff = int(max(1, min(k, anchor_ids_np.shape[0])))
            tree = KDTree(means_np[anchor_ids_np])
            neighbour_dists, neighbours = tree.query(new_means, k=k_eff, eps=eps)
            if k_eff == 1:
                neighbour_dists = neighbour_dists[:, None]
                neighbours = neighbours[:, None]
            self.control_pts = means[anchor_ids].detach()
            self.neighbour_dists = torch.cat((self.neighbour_dists, torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)), dim=0)
            self.neighbours = torch.cat((self.neighbours, torch.tensor(neighbours, device="cuda", dtype=torch.long)), dim=0)
            self.neighbour_weights = torch.exp(-4.5 * (self.neighbour_dists))
            self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
            self.means_cache, self.rot_cache = [None, None], [None, None]
        else:
            self.control_pts = means[anchor_ids].detach()

    def reg_loss(self, visibility_filter):
        if self.means_cache[1] is None:
            l = torch.zeros(1, device="cuda").squeeze()
            return l, l, l, l
        prev_rot = build_rotation(self.rot_cache[1])
        cur_rot = build_rotation(self.rot_cache[0])
        rel_rot = prev_rot @ cur_rot.transpose(1, 2)
        cur_offset = self.means_cache[0][self.neighbours[self.anchor_ids]] - self.means_cache[0][:, None]
        last_offset = self.means_cache[1][self.neighbours[self.anchor_ids]] - self.means_cache[1][:, None]
        rot_offset = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(rot_offset, dim=-1) * self.neighbour_weights[self.anchor_ids]).mean()
        l_rot = torch.sqrt(
            ((rel_rot[:, None] - rel_rot[self.neighbours[self.anchor_ids]]) ** 2).sum(-1).sum(-1)
            * self.neighbour_weights[self.anchor_ids]
            + 1e-20
        ).mean()
        curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists[self.anchor_ids]) * self.neighbour_weights[self.anchor_ids]).mean()
        ids = ~visibility_filter[self.anchor_ids]
        means_energy = self.means_cache[0][ids].abs().sum()
        l_visible = means_energy / (ids.sum() + 1)
        return l_rigid, l_rot, l_iso, l_visible

    @torch.no_grad()
    def init_from_flow(self, deformation, weights):
        weight_sum = weights.sum(-1)
        deformation = (weights[..., None] * deformation).sum(1) / weight_sum[..., None]
        deformation[weight_sum < 0.1] = 0.0
        deformation = deformation[self.anchor_ids].clamp(-0.01, 0.01).float()
        basis = self._basis_weights(time=self.current_time).to(device=deformation.device, dtype=deformation.dtype)
        norm = (basis * basis).sum().clamp_min(1e-6)
        scale = basis / norm
        self.means_def += deformation[..., None] * scale[None, None, :]
