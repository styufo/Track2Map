import os
import sys
import numpy as np
import torch
from typing import Optional


def _append_once(path: str):
    if path and path not in sys.path:
        sys.path.append(path)


def _resolve_foundation_root(foundation_root: Optional[str]):
    if foundation_root is not None and str(foundation_root).strip() != "":
        return os.path.abspath(os.path.expanduser(str(foundation_root)))
    env_root = os.environ.get("FOUNDATION_STEREO_ROOT", "")
    if env_root.strip() != "":
        return os.path.abspath(os.path.expanduser(env_root))
    return None


def get_foundation_stereo_model(
    device: str = "cuda",
    foundation_root: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    intrinsic_file: Optional[str] = None,
    valid_iters: int = 32,
):
    root = _resolve_foundation_root(foundation_root)
    if root is None:
        raise FileNotFoundError(
            "foundation_root is not provided. Set data.foundation_root or env FOUNDATION_STEREO_ROOT."
        )

    _append_once(root)
    _append_once(os.path.dirname(root))

    try:
        from omegaconf import OmegaConf
        from FoundationStereo.core.foundation_stereo import FoundationStereo
    except Exception as exception:
        raise ImportError(
            "Failed to import FoundationStereo/omegaconf. Ensure FoundationStereo repo "
            "and dependencies are available in the runtime environment."
        ) from exception

    ckpt = ckpt_path or os.path.join(root, "weights", "model_best_bp2.pth")
    cfg = cfg_path or os.path.join(root, "weights", "cfg.yaml")
    intrinsic = intrinsic_file or os.path.join(root, "assets", "K.txt")

    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"Foundation cfg file not found: {cfg}")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Foundation checkpoint not found: {ckpt}")
    if not os.path.isfile(intrinsic):
        raise FileNotFoundError(f"Foundation intrinsic file not found: {intrinsic}")

    fs_args = OmegaConf.load(cfg)
    if "vit_size" not in fs_args:
        fs_args["vit_size"] = "vitl"
    fs_args["ckpt_dir"] = ckpt
    fs_args["intrinsic_file"] = intrinsic
    fs_args["valid_iters"] = int(valid_iters)

    model = FoundationStereo(fs_args)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    with open(intrinsic, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip() != ""]
    if len(lines) < 2:
        raise RuntimeError(f"Invalid Foundation intrinsic file format: {intrinsic}")
    K = np.array(list(map(float, lines[0].split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])
    fs_args.K = K.tolist()
    fs_args.baseline = baseline
    return model, fs_args
