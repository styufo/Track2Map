# Track2Map

Track2Map is a cleaned release of our online tracking-to-mapping pipeline with **FoundationStereo (FixScale)** depth.

This repo keeps only the final runnable paths for 3 practical modes:

1. **`clean_pose`**: use dataset camera poses directly, no pose initialization from scratch, no pose optimization.
2. **`noisy_auto_gate`**: start from noisy pose, and use first-frame trust gate to switch between:
   - noisy-pose optimization route (for light noise, e.g. 1x),
   - no-pose-prior route (for heavy noise, e.g. 10x).
3. **`no_pose`**: no camera pose provided; start from no prior and estimate+optimize online.

---

## 1) Environment (from scratch)

This repo is tested with a clean conda environment (no cloning from existing envs).

### Option A (recommended): conda yaml

```bash
conda env create -f environment.yml
conda activate track2map
```

### Option B: manual install

```bash
conda create -y -n track2map python=3.10 pip
conda activate track2map
pip install -r requirements.txt
conda install -y -c nvidia cuda-nvcc=12.1 cuda-cccl=12.1
```

### Build CUDA submodules (required)

`run.py` depends on `src/submodules/simple-knn` and `src/submodules/gaussian-rasterization`.

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

pip install --no-build-isolation -e src/submodules/simple-knn
pip install --no-build-isolation -e src/submodules/gaussian-rasterization
```

### Quick sanity check

```bash
python -c "import torch,cv2,open3d,omegaconf,timm; print(torch.__version__, torch.version.cuda)"
```

---

## 2) Repo layout

- `run.py`: main training / mapping entry (supports `--visualize`)
- `src/`: core modules
- `configs/base.yaml`: base runtime config
- `configs/final/*_auto_gate_base.yaml`: final per-seq base configs
- `scripts/run_track2map.py`: unified launcher for the 3 modes

Final sequence-policy mapping kept in this repo:
- `P1_1 -> GateC`
- `P2_0/P2_1 -> P2S_staticheavy`
- `P3_1/P3_2 -> P4`

---

## 3) Unified launcher

Use `scripts/run_track2map.py` for all modes.

### Common args

- `--seq`: `P1_1 | P2_0 | P2_1 | P3_1 | P3_2`
- `--input-folder`: sequence folder
- `--output`: output folder
- `--flow-init-source`: `raft | cotracker3 | hybrid | foundation | hybrid_foundation` (default: `hybrid`)
- `--visualize`: enable online visualization outputs
- `--pose-file`: required in `clean_pose` and `noisy_auto_gate`

### Mode A: clean pose

```bash
python scripts/run_track2map.py \
  --mode clean_pose \
  --seq P3_1 \
  --input-folder /path/to/steremis_tracking/P3_1 \
  --pose-file /path/to/steremis_tracking/P3_1/groundtruth.txt \
  --output /path/to/output/p31_clean_found \
  --visualize
```

Behavior:
- uses FoundationStereo(FixScale) depth,
- disables pose optimization,
- follows provided clean pose.

### Mode B: noisy + auto gate

```bash
python scripts/run_track2map.py \
  --mode noisy_auto_gate \
  --seq P3_1 \
  --input-folder /path/to/steremis_tracking/P3_1 \
  --pose-file /path/to/stereomis_noisy_light_transx10/P3_1/groundtruth_noisy.txt \
  --output /path/to/output/p31_noisy_autogate_found \
  --gate-profile auto \
  --visualize
```

Gate profiles:
- `--gate-profile 1x`: relaxed thresholds (target: mostly no trigger),
- `--gate-profile 10x`: strict thresholds (target: high trigger rate),
- `--gate-profile auto` (default): infer by pose-file name (`x10/transx10/noisyx10` -> `10x`, else `1x`).

When triggered, fallback route is:
- `pose_init_mode = no_prior`
- `w_pose_prior = 0.0`
- no-prior VO chain enabled.

### Mode C: no pose prior

```bash
python scripts/run_track2map.py \
  --mode no_pose \
  --seq P3_1 \
  --input-folder /path/to/steremis_tracking/P3_1 \
  --output /path/to/output/p31_nopose_found \
  --visualize
```

Behavior:
- ignores external pose file,
- starts from no prior and optimizes online.

---

## 4) FoundationStereo paths

Launcher defaults:

- `--foundation-root /home/tianyi/external/FoundationStereo`
- `--foundation-ckpt /home/tianyi/external/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth`
- `--foundation-cfg /home/tianyi/external/FoundationStereo/pretrained_models/23-51-11/cfg.yaml`
- `--foundation-intrinsic-file /home/tianyi/external/FoundationStereo/assets/K.txt`

Override them in command line if needed.

---

## 5) Visual outputs

`--visualize` keeps the same online visualization path as prior workflow (render/mapping videos and related visualization outputs in each run folder).

For reconstruction metrics (`PSNR/SSIM/LPIPS`), run with `--visualize`; otherwise `raw_rgb/raw_depth` may stay empty.

---

## 6) Troubleshooting

- `No module named 'simple_knn'` / `diff_gaussian_rasterization`: reinstall the two submodules with the commands above.
- CUDA mismatch during build: ensure `nvcc` is from the active env (`which nvcc`) and matches torch CUDA (`python -c "import torch; print(torch.version.cuda)"`).
- `wandb` + `numpy` compatibility: use `wandb>=0.25.0`.
- FoundationStereo import errors: verify `--foundation-root`, checkpoint, cfg, and intrinsic paths.

---

## 7) Publish to GitHub

```bash
cd Track2Map
git init
git add .
git commit -m "Initial clean Track2Map release"
git branch -M main
git remote add origin git@github.com:<your_user>/Track2Map.git
git push -u origin main
```
