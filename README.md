# Track2Map

Official repository for Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery.

This repo keeps runnable paths for 3 practical modes:

1. **`clean_pose`**: use dataset camera poses directly, no pose initialization from scratch, no pose optimization.
2. **`noisy_auto_gate`**: start from noisy pose.
3. **`no_pose`**: no camera pose provided; start from no prior and estimate+optimize online.

---

## 1) Environment

```bash
conda env create -f environment.yml
conda activate track2map

pip install -e src/submodules/gaussian-rasterization
pip install -e src/submodules/simple-knn
```



---



## 2) Unified launcher

Use `scripts/run_track2map.py` for all modes.

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



### Mode B: noisy

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

### Mode C: no pose prior

```bash
python scripts/run_track2map.py \
  --mode no_pose \
  --seq P3_1 \
  --input-folder /path/to/steremis_tracking/P3_1 \
  --output /path/to/output/p31_nopose_found \
  --visualize
```



---


## 5) Visual outputs

`--visualize` render/mapping videos and related visualization outputs in each run folder.

For reconstruction metrics (`PSNR/SSIM/LPIPS`), run with `--visualize`; otherwise `raw_rgb/raw_depth` may stay empty.

---


