import glob
import os
import cv2
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
from src.utils.semantic_utils import SemanticDecoder


class StereoMIS(Dataset):
    def __init__(self, cfg, args, scale):
        super(StereoMIS, self).__init__()
        self.name = cfg['dataset']
        self.scale = scale
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'video_frames*', '*l.png')))
        self.total_frames = len(self.color_paths)
        frame_slice = slice(cfg['data']['start'], cfg['data']['stop'], cfg['data']['step'])
        all_frame_ids = [os.path.splitext(os.path.basename(path))[0].removesuffix('l') for path in self.color_paths]
        self.frame_ids = all_frame_ids[frame_slice]
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("StereoMIS image frame IDs must be unique after slicing.")
        self.pose_source = str(cfg['data'].get('pose_source', 'file')).lower()
        self.pose_path = None
        if self.pose_source == 'identity':
            self.load_identity_poses(len(self.color_paths), frame_slice)
        elif self.pose_source == 'file':
            if hasattr(args, 'pose_file') and args.pose_file is not None:
                self.pose_path = os.path.abspath(os.path.expanduser(args.pose_file))
            else:
                pose_filename = cfg['data'].get('pose_file', 'groundtruth.txt')
                self.pose_path = os.path.join(self.input_folder, pose_filename)
            if os.path.isfile(self.pose_path):
                self.load_poses(self.pose_path, frame_slice)
            elif str(cfg.get('track2map_runtime', {}).get('inference_policy', '')).lower() == 'strict_gt_free':
                raise FileNotFoundError(f"Required input pose file not found: {self.pose_path}")
            else:
                warnings.warn(f"Pose file not found: {self.pose_path}. Falling back to identity poses.")
                self.pose_source = 'identity_fallback'
                self.load_identity_poses(len(self.color_paths), frame_slice)
        else:
            raise ValueError(f"Unsupported data.pose_source={self.pose_source}; expected file or identity.")
        self.color_paths = self.color_paths[frame_slice]
        self.n_img = len(self.color_paths)
        if len(self.poses) != self.n_img:
            raise ValueError(
                f"Pose/image count mismatch after slicing: poses={len(self.poses)} images={self.n_img}."
            )
        self.deformation_time_denominator = float(
            cfg['data'].get('deformation_time_denominator', max(self.n_img - 1, 1))
        )
        if self.deformation_time_denominator <= 0.0:
            raise ValueError("data.deformation_time_denominator must be positive.")
        self.semantic_decoder = SemanticDecoder()

    def load_poses(self, path: str, sclice):
        with open(path, 'r') as f:
            data = f.read()
            lines = data.replace(",", " ").replace("\t", " ").split("\n")
            list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                    len(line) > 0 and line[0] != "#"]

        trans = np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float)
        quat = np.asarray([l[4:] for l in list if len(l) > 0], dtype=float)
        pose_frame_ids = [l[0] for l in list if len(l) > 0]
        self.pose_frame_ids = pose_frame_ids[sclice]
        if self.pose_frame_ids != self.frame_ids:
            mismatch = next(
                (
                    idx,
                    self.frame_ids[idx] if idx < len(self.frame_ids) else None,
                    self.pose_frame_ids[idx] if idx < len(self.pose_frame_ids) else None,
                )
                for idx in range(max(len(self.frame_ids), len(self.pose_frame_ids)))
                if idx >= len(self.frame_ids)
                or idx >= len(self.pose_frame_ids)
                or self.frame_ids[idx] != self.pose_frame_ids[idx]
            )
            raise ValueError(
                "Input pose/image frame ID mismatch at local index "
                f"{mismatch[0]}: image={mismatch[1]} pose={mismatch[2]}."
            )
        self.poses = np.eye(4)[None, ...].repeat(quat.shape[0], axis=0)
        self.poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        self.poses[:, :3, 3] = trans
        self.poses = torch.from_numpy(self.poses[sclice]).float()
        self.poses = torch.linalg.inv(self.poses[0])[None, ...] @ self.poses
        self.poses[:, :3, 3] *= self.scale

    def load_identity_poses(self, num_frames: int, sclice):
        poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
        self.poses = poses[sclice]
        self.pose_frame_ids = list(self.frame_ids)

    def __getitem__(self, index):
        color_data = cv2.cvtColor(cv2.imread(self.color_paths[index]), cv2.COLOR_BGR2RGB)
        color_data = torch.from_numpy(color_data)
        color_data = color_data / 255.

        color_data_r = cv2.cvtColor(cv2.imread(self.color_paths[index].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        color_data_r = torch.from_numpy(color_data_r)
        color_data_r = color_data_r / 255.

        # add tool mask if existing
        mask_path = self.color_paths[index].replace("video_frames", "masks")
        if os.path.isfile(mask_path):
            mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_data = cv2.resize(mask_data, (color_data.shape[1], color_data.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_data = torch.from_numpy(binary_erosion(mask_data, iterations=7, border_value=1) > 0).bool()
        else:
            mask_data = None

        # add semantic segmentation if existing
        sm_path = self.color_paths[index].replace("video_frames", "semantic_predictions")
        if os.path.isfile(sm_path):
            semantics = cv2.cvtColor(cv2.imread(sm_path), cv2.COLOR_BGR2RGB)
            semantics = cv2.resize(semantics, (color_data.shape[1], color_data.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            semantics = self.semantic_decoder(semantics)
        else:
            semantics = None
        pose = self.poses[index]
        return index, color_data, color_data_r, pose, mask_data, semantics

    def get_name(self, index):
        return os.path.basename(self.color_paths[index])

    def __len__(self):
        return self.n_img


# extend default collate for None elements
def collate_none_fn(batch, *, collate_fn_map = None):
    return None
from torch.utils.data._utils.collate import default_collate_fn_map
default_collate_fn_map.update({type(None):collate_none_fn})
