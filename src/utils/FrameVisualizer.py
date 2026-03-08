import os
import torch
import numpy as np
from imageio import imsave
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from src.utils.renderer import render
from src.utils.camera import Camera
from src.utils.semantic_utils import SemanticDecoder


class FrameVisualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:

    """

    def __init__(self, outpath, cfg, net):
        self.outmap = os.path.join(outpath, 'mapping')
        self.outrack = os.path.join(outpath, 'tracking')
        self.outsem = os.path.join(outpath, 'semantic')
        self.out_raw_rgb_gt = os.path.join(outpath, 'raw_rgb', 'gt')
        self.out_raw_rgb_render = os.path.join(outpath, 'raw_rgb', 'render')
        self.out_raw_depth_gt = os.path.join(outpath, 'raw_depth', 'gt')
        self.out_raw_depth_render = os.path.join(outpath, 'raw_depth', 'render')
        self.out_widefield_ply = os.path.join(outpath, 'pointclouds_widefield')
        os.makedirs(self.outmap, exist_ok=True)
        os.makedirs(self.outrack, exist_ok=True)
        os.makedirs(self.outsem , exist_ok=True)
        os.makedirs(self.out_raw_rgb_gt, exist_ok=True)
        os.makedirs(self.out_raw_rgb_render, exist_ok=True)
        os.makedirs(self.out_raw_depth_gt, exist_ok=True)
        os.makedirs(self.out_raw_depth_render, exist_ok=True)
        os.makedirs(self.out_widefield_ply, exist_ok=True)
        self.camera = Camera(cfg['cam'])
        self.widefield_camera = Camera(cfg['cam_widefield'])
        self.decoder = SemanticDecoder()
        self.net = net
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    def save_imgs(self, idx, stereo_depth, gt_color, c2w, pts_pred=None, pts_gt=None):
        """
        Visualization of depth and color images and save to file.
        Args:

        """
        self.camera.set_c2w(c2w)
        render_pkg = render(self.camera, self.net, self.background, deform=True)
        self.save_raw_frames(idx, render_pkg['depth'], render_pkg['render'], stereo_depth, gt_color)
        self.plot_mapping(render_pkg['depth'], render_pkg['render'], stereo_depth, gt_color)
        outmap = os.path.join(self.outmap,f'{idx:05d}.jpg')
        plt.savefig(outmap, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close()

        img_sem = self.plot_semantics(c2w)
        outsem = os.path.join(self.outsem,f'{idx:05d}.jpg')
        imsave(outsem, img_sem)

        if pts_pred is not None:
            img_track = self.plot_tracking(pts_pred, render_pkg['render'], gt_color, pts_gt)
            outrack = os.path.join(self.outrack,f'{idx:05d}.jpg')
            imsave(outrack, img_track)
        else:
            outrack = None
        return outmap, outsem, outrack

    @staticmethod
    def _write_ply(path, points_xyz, points_rgb):
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points_xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points_xyz, points_rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    @torch.no_grad()
    def save_widefield_ply(
        self,
        idx,
        c2w,
        every_n=10,
        alpha_threshold=0.8,
        min_depth=1e-6,
        max_depth=0.0,
    ):
        if every_n <= 0 or (idx % every_n) != 0:
            return None
        self.widefield_camera.set_c2w(c2w)
        render_pkg = render(self.widefield_camera, self.net, self.background, deform=True)

        depth = render_pkg["depth"]
        color = render_pkg["render"]
        alpha = render_pkg.get("alpha", None)

        H, W, fx, fy, cx, cy = self.widefield_camera.get_params()
        ys, xs = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )

        z = depth
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        pts_cam = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

        c2w_mat = c2w.squeeze(0)
        rot = c2w_mat[:3, :3]
        trans = c2w_mat[:3, 3]
        pts_world = pts_cam @ rot.T + trans[None, :]

        valid = torch.isfinite(z) & (z > float(min_depth))
        if max_depth is not None and float(max_depth) > 0:
            valid &= z < float(max_depth)
        if alpha is not None and float(alpha_threshold) > 0:
            valid &= alpha > float(alpha_threshold)
        valid = valid.reshape(-1)

        if valid.sum().item() == 0:
            return None

        pts_world_np = pts_world[valid].detach().cpu().numpy().astype(np.float32)
        pts_rgb_np = (
            (color.reshape(-1, 3)[valid].detach().cpu().numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )

        ply_path = os.path.join(self.out_widefield_ply, f"frame_{idx:05d}_widefield_world.ply")
        self._write_ply(ply_path, pts_world_np, pts_rgb_np)
        return ply_path

    def save_raw_frames(self, idx, depth, color, stereo_depth, gt_color):
        gt_color_np = np.clip(gt_color.squeeze(0).detach().cpu().numpy(), 0.0, 1.0)
        color_np = np.clip(color.detach().cpu().numpy(), 0.0, 1.0)

        gt_rgb_u8 = cv2.cvtColor((gt_color_np * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        pred_rgb_u8 = cv2.cvtColor((color_np * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(self.out_raw_rgb_gt, f'{idx:05d}.png'), gt_rgb_u8)
        cv2.imwrite(os.path.join(self.out_raw_rgb_render, f'{idx:05d}.png'), pred_rgb_u8)

        stereo_depth_np = stereo_depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
        depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(self.out_raw_depth_gt, f'{idx:05d}.npy'), stereo_depth_np)
        np.save(os.path.join(self.out_raw_depth_render, f'{idx:05d}.npy'), depth_np)

    def plot_mapping(self, depth, color, stereo_depth, gt_color):
        stereo_depth_np = stereo_depth.squeeze(0).cpu().numpy()
        gt_color_np = gt_color.squeeze(0).cpu().numpy()
        depth_np = depth.squeeze(0).cpu().numpy()
        color_np = color.squeeze(0).cpu().numpy()
        depth_residual = np.abs(stereo_depth_np - depth_np)
        depth_residual[stereo_depth_np == 0.0] = 0.0
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[stereo_depth_np == 0.0] = 0.0

        fig, axs = plt.subplots(2, 3)
        max_depth = np.max(stereo_depth_np)

        axs[0, 0].imshow(stereo_depth_np, vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, vmin=0, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np)
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np)
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual)
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        return fig, axs


    @torch.no_grad()
    def plot_semantics(self, c2w, thr=0.8):
        self.widefield_camera.set_c2w(c2w)
        self.camera.set_c2w(c2w)
        render_pkg = render(self.widefield_camera, self.net, self.background)
        # mask gaussians that are in areas with very low opacity
        render_pkg['render'][render_pkg['alpha'] < thr] = 0.0
        semantics = torch.argmax(render_pkg['semantics'], dim=0)
        semantics[render_pkg['alpha'] < thr] = 0
        semantics = self.decoder.colorize_label(semantics.cpu().numpy()) / 255.0
        vis_img = 0.7*render_pkg['render'].cpu().numpy() + 0.3*semantics
        vis_img = (255*vis_img).clip(0, 255).astype(np.uint8)
        return vis_img

    def plot_tracking(self, pts, img_rend, img_gt=None, pts_gt=None, baseline_pts=None, size=15, thickness=1):
        class MplColorHelper:
            def __init__(self, cmap_name, start_val, stop_val):
                self.cmap_name = cmap_name
                self.cmap = plt.get_cmap(cmap_name)
                self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
                self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

            def get_rgb(self, val):
                return [255 * n for n in self.scalarMap.to_rgba(val)[:3]]
        m = MplColorHelper('jet', 0, pts.shape[0])
        h,w = img_rend.shape[:2]

        # plot canonical scene
        vis_img_rend = (255.0 * img_rend.cpu().numpy().squeeze()).clip(0,255).astype(np.uint8).copy()
        vis_img_gt = (255.0 * img_gt.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8).copy() if img_gt is not None else None
        for i, pt in enumerate(pts):
            if baseline_pts is not None:
                pt_gt = baseline_pts[i]
                if (pt_gt > 0).all() and (pt_gt[0] < w) and (pt_gt[1] < h):
                    cv2.drawMarker(vis_img_rend, (int(pt_gt[0].item()), int(pt_gt[1].item())), m.get_rgb(i),
                                   cv2.MARKER_SQUARE, size, thickness)
                    if img_gt is not None:
                        cv2.drawMarker(vis_img_gt, (int(pt_gt[0].item()), int(pt_gt[1].item())), m.get_rgb(i),
                                       cv2.MARKER_SQUARE, size, thickness)
            if pts_gt is not None:
                pt_gt = pts_gt[i]
                if (pt_gt > 0).all() and (pt_gt[0] < w) and (pt_gt[1] < h):
                    cv2.drawMarker(vis_img_rend, (int(pt_gt[0].item()), int(pt_gt[1].item())), m.get_rgb(i), cv2.MARKER_TRIANGLE_UP, size, thickness)
                    if img_gt is not None:
                        cv2.drawMarker(vis_img_gt, (int(pt_gt[0].item()), int(pt_gt[1].item())), m.get_rgb(i),
                                       cv2.MARKER_TRIANGLE_UP, size, thickness)
            if (pt > 0).all() and (pt[0] < w) and (pt[1] < h):
                cv2.drawMarker(vis_img_rend, (int(pt[0].item()), int(pt[1].item())), m.get_rgb(i), cv2.MARKER_CROSS, size, thickness)
                if img_gt is not None:
                    cv2.drawMarker(vis_img_gt, (int(pt[0].item()), int(pt[1].item())), m.get_rgb(i), cv2.MARKER_CROSS, size,
                                   thickness)
        vis_img = np.concatenate((vis_img_gt, vis_img_rend), axis=1) if img_gt is not None else vis_img_rend
        return vis_img
