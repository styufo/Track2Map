# Track2Map

Official repository for Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery.

## Demo Video

You can download/watch the demo video here:
[Track2Map_demo.mp4](assets/vis.mov)

https://github.com/user-attachments/assets/123937e6-a2f9-4f25-b45f-451067698e5f

This repo keeps runnable paths for 3 practical modes:

1. **`clean_pose`**: use dataset camera poses directly, no pose initialization from scratch, no pose optimization.
2. **`noisy`**: start from noisy pose (auto gate enabled).
3. **`no_pose`**: no camera pose provided; start from no prior and estimate+optimize online.
---

## 1) Environment

```bash
conda env create -f environment.yml
conda activate track2map

pip install -e src/submodules/gaussian-rasterization
pip install -e src/submodules/simple-knn
```
Clone external dependencies:

```bash
git clone https://github.com/facebookresearch/co-tracker.git cotracker
git clone https://github.com/NVlabs/FoundationStereo.git foundationstereo
```

---
## 2) Dataset prepare
Download the data from [StereoMIS Tracking](https://zenodo.org/records/10867949) and unpack it in the repository base folder.

If you want to test the model's ability under noisy pose, you can generate the noisy pose `groundtruth_noisy.txt` from gt camera pose `groundtruth.txt` by the [script](scripts/perturb_stereomis_groundtruth.py) we provide.

### 1x noisy pose (light noise)
```bash
python scripts/perturb_stereomis_groundtruth.py \
  --config configs/StereoMIS/noise/noisy_pose_1x.yaml \
  --input-root /path/to/steremis_tracking \
  --out-root /path/to/stereomis_noisy_light
```

### 10x noisy pose (heavy noise)
```bash
python scripts/perturb_stereomis_groundtruth.py \
  --config configs/StereoMIS/noise/noisy_pose_10x.yaml \
  --input-root /path/to/steremis_tracking \
  --out-root /path/to/stereomis_noisy_light_transx10
```

## 3) Run

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
  --mode noisy \
  --seq P3_1 \
  --input-folder /path/to/steremis_tracking/P3_1 \
  --pose-file /path/to/stereomis_noisy_light_transx10/P3_1/groundtruth_noisy.txt \
  --output /path/to/output/p31_noisy_found \
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


## 4) Visual outputs

`--visualize` render/mapping videos and related visualization outputs in each run folder.

For reconstruction metrics (`PSNR/SSIM/LPIPS`), run with `--visualize`; otherwise `raw_rgb/raw_depth` may stay empty.

---

## 5) Acknowledgements
Our code is based on [Online-endo-track](https://github.com/mhayoz/online_endo_track), our depth estimation is based on [FoundationStereo](https://github.com/NVlabs/FoundationStereo), and our tracking method is based on [CoTracker3](https://github.com/facebookresearch/co-tracker). We thank the authors for their excellent work!
