# BioModule

A biomechanical criteria predictor that plugs on top of any frozen 3D human pose estimator.
Given a sliding window of 3D skeletal poses, BioModule regresses **17 biomechanical criteria per frame** — spanning kinematics, kinetics, and neuromuscular signals — without requiring force plates or motion-capture instrumentation at test time.

---

## Overview

BioModule decouples pose estimation from biomechanical inference.

```
Video
  │
  ▼
3D Pose Estimator  (MHFormer, TCPFormer, PoseMamba, VideoPose3D, …)
  │
  ▼  (T, 17, 3) — root-centred, metres
BioModule  [sliding window W=27]
  │
  ▼
17 × biomechanical criteria  (B, 27, dim)  →  centre-frame predictions
```

The model is trained on ground-truth 3D keypoints (Human3.6M) and evaluated on poses predicted by any estimator, introducing a deliberate domain gap that mirrors real-world deployment.

---

## Architecture

**3,824,073 parameters** — transformer encoder with independent per-criterion output heads.

| Block | Config | Params |
|---|---|---|
| Input embedding | Linear(51 → 256) + LayerNorm | 13,312 |
| Positional encoding | Sinusoidal, d=256, max_len=512 | 0 |
| Transformer encoder ×4 | Pre-LN, 8 heads, FFN 256→1024→256 | 3,159,040 |
| Final LayerNorm | d=256 | 512 |
| Output heads ×17 | Linear(256→128) → GELU → Drop → Linear(128→dim) | 610,209 |

See [`docs/architecture.md`](docs/architecture.md) for the full specification.

---

## Output Criteria (17)

| Tier | Criteria | Loss |
|---|---|---|
| Kinematic | `coordinate`, `speed`, `acceleration` | MSE |
| Kinetic | `active_torque`, `passive_torque`, `ideal_torque`, `instantaneous_power`, `instantaneous_power_raw`, `ground_reaction`, `seat_reaction`, `touch` | MSE / BCE |
| Neuromuscular | `activation_signal`, `excitation_signal`, `normalized_active_torque`, `angle_scaling`, `velocity_scaling`, `maximum_joint_torque` | MSE |

**Total output: 617 values per frame.**

---

## Installation

```bash
conda create -n biomodule python=3.10
conda activate biomodule
pip install torch torchvision   # follow https://pytorch.org for your CUDA version
pip install -r requirements.txt
```

---

## Training

### Mode A — pre-normalized directory (recommended)

```bash
python -m bio_module.train \
    --normalized-dir /path/to/normalized \
    --checkpoint-dir checkpoint/bio_gt
```

Or use the convenience script:

```bash
./scripts/train.sh --mode A --normalized-dir /path/to/normalized
```

### Mode B — raw data + pose estimator at test time

```bash
python -m bio_module.train \
    --pose-3d        /path/to/positions_3d.npy \
    --pose-2d        /path/to/data_2d_h36m_cpn_ft_h36m_dbb.npz \
    --processed-dir  /path/to/processed_all/processed_all \
    --model-weights  /path/to/mhformer_weights/ \
    --checkpoint-dir checkpoint/bio_gt
```

### Key hyperparameters

| Hyperparameter | Default |
|---|---|
| Window size (`--bio-win`) | 27 |
| Transformer dim (`--d-model`) | 256 |
| Heads (`--nhead`) | 8 |
| Layers (`--nlayers`) | 4 |
| Dropout (`--dropout`) | 0.1 |
| Epochs (`--epochs`) | 50 |
| Learning rate (`--lr`) | 3e-4 |
| Batch size (`--batch-size`) | 64 |

---

## Evaluation

Plug in poses from any estimator:

```bash
python eval.py \
    --poses         my_model_poses.npy \
    --processed-dir /path/to/processed_all/processed_all \
    --norm-stats    checkpoint/bio_gt/norm_stats.json \
    --checkpoint    checkpoint/bio_gt/bio_gt_best.pth \
    --subjects      S9 S11
```

**Pose file format** — a `.npy` file saved with `allow_pickle=True`:

```python
import numpy as np
poses = {
    'S9':  {'Walking 1': array_shape_T_17_3, ...},
    'S11': {'Directions': array_shape_T_17_3, ...},
}
np.save('my_model_poses.npy', poses)
```

J can be 17 (H36M 17-joint) or 32 (full H36M, joints are auto-selected).

---

## Repository Structure

```
BioModule/
├── bio_module/                    # Core Python package
│   ├── model.py                   # BioModuleV3 architecture
│   ├── dataset.py                 # BioDataset (Mode A & B)
│   ├── loss.py                    # Mixed MSE / BCE loss + evaluation metrics
│   ├── train.py                   # Training pipeline (3-phase)
│   └── pose_estimator.py          # Wrappers for MHFormer, TCPFormer, etc.
├── eval.py                        # Standalone evaluation script
├── scripts/
│   ├── build_aligned_dataset.py   # Build aligned.npz from H36M + H36M+ CSVs
│   ├── save_norm_stats.py         # Compute z-score statistics for training
│   ├── precompute_poses.py        # Run any pose estimator → poses .npy
│   ├── train.sh                   # Training convenience wrapper
│   ├── eval.sh                    # Evaluation convenience wrapper
│   └── assemble_poses.py          # Reassemble cached per-clip pose files
├── docs/
│   ├── architecture.md            # Full architecture specification
│   └── preprocessing.md           # Data acquisition + full processing pipeline
├── checkpoint/
│   └── norm_stats.json            # Pre-computed z-score stats (all 16 criteria)
├── data/
│   ├── sample/                    # Reference clip: S1/Sitting1, all criteria
│   └── README.md                  # Data layout + download links
├── results/
│   ├── figures/                   # Result figures
│   ├── eval_gt.json
│   ├── eval_mhformer.json
│   ├── eval_posemamba.json
│   └── eval_tcpformer.json
├── requirements.txt
└── .gitignore
```

---

## Data Setup

BioModule depends on three data sources. Two are restricted; one is freely
downloadable. The table below summarises access and the processing step each
feeds into.

| Source | License | Size | Used for |
|--------|---------|------|---------|
| [Human3.6M](http://vision.imar.ro/human3.6m/) poses | Restricted — request access | ~700 MB | training input, alignment |
| H36M+ OpenSim biomechanical CSVs | Contact authors | ~3 GB raw | ground-truth labels |
| [Human3.6M camera parameters](https://github.com/karfly/human36m-camera-parameters) | Public (MIT) | <1 MB | alignment projection |

### Quick start — pre-processed data (recommended)

If you want to train or evaluate without rebuilding everything from source,
download `processed_all.zip` from the GitHub Release:

```bash
wget https://github.com/eghbalian-ayda/BioModule/releases/download/v1.0-data/processed_all.zip
unzip processed_all.zip   # → processed_all/processed_all/{S1,S5,S6,S7,S8,S9,S11}/
```

Then train directly in Mode A:

```bash
./scripts/train.sh --mode A --normalized-dir processed_all/processed_all
```

### Rebuilding from raw data

If you have access to the H36M+ CSV database and want to reproduce the full
preprocessing pipeline:

**Step 1 — Obtain Human3.6M**

Request access at http://vision.imar.ro/human3.6m/ and download:
- `data_3d_h36m.npz`
- `data_2d_h36m_cpn_ft_h36m_dbb.npz`
- `data_2d_h36m_gt.npz`

**Step 2 — Obtain camera parameters**

```bash
git clone https://github.com/karfly/human36m-camera-parameters
```

**Step 3 — Build `aligned.npz` files**

```bash
python scripts/build_aligned_dataset.py \
    --db-root  /path/to/h36m_database \
    --pose-2d  dataset/data_2d_h36m_gt.npz \
    --pose-3d  dataset/data_3d_h36m.npz \
    --cam-json human36m-camera-parameters/camera-parameters.json
```

This writes one `aligned/aligned.npz` per (subject, action) folder inside
`h36m_database/`. Each file contains frame-aligned 3D poses, 2D projections,
marker pixel coordinates, and joint DoF values.

**Step 4 — Compute normalisation statistics**

```bash
python scripts/save_norm_stats.py \
    --db-root /path/to/h36m_database \
    --out     checkpoint/norm_stats.json
```

A pre-computed `checkpoint/norm_stats.json` is already committed to this
repository — skip this step unless you change the subject split.

**Step 5 — Precompute pose estimator outputs** *(evaluation only)*

```bash
python scripts/precompute_poses.py \
    --pose-2d      dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz \
    --model-type   mhformer \
    --model-weights model/model_4294.pth \
    --frames       351 \
    --subjects     S9 S11 \
    --out          poses_mhformer.npy
```

For other estimators edit `load_custom_model()` in `scripts/precompute_poses.py`.

See [`docs/preprocessing.md`](docs/preprocessing.md) for the complete
specification of all criteria, DoF orderings, normalisation conventions,
and frame-alignment details.

---

## Citation

```bibtex
@inproceedings{yourname2025biomodule,
  title     = {Kinetic Pose Estimation: Predicting Biomechanical Criteria
               from 3D Human Pose Sequences},
  author    = {Author One and Author Two and Author Three},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on
               Applications of Computer Vision (WACV)},
  year      = {2025},
}
```
