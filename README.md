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
├── bio_module/            # Core Python package
│   ├── model.py           # BioModuleV3 architecture
│   ├── dataset.py         # BioDataset (Mode A & B)
│   ├── loss.py            # Mixed MSE / BCE loss + evaluation metrics
│   ├── train.py           # Training pipeline (3-phase)
│   └── pose_estimator.py  # Wrappers for MHFormer, TCPFormer, etc.
├── eval.py                # Standalone evaluation script
├── scripts/
│   ├── train.sh           # Training convenience wrapper
│   ├── eval.sh            # Evaluation convenience wrapper
│   └── assemble_poses.py  # Reassemble cached per-clip pose files
├── docs/
│   ├── architecture.md    # Full V3 architecture specification
│   └── preprocessing.md   # H36M + OpenSim data pipeline
├── results/
│   ├── figures/           # Result figures
│   ├── eval_gt.json       # GT-input evaluation (Phase 1)
│   ├── eval_mhformer.json # MHFormer-input evaluation
│   ├── eval_posemamba.json
│   └── eval_tcpformer.json
├── requirements.txt
└── .gitignore
```

---

## Data

- **Poses:** [Human3.6M](http://vision.imar.ro/human3.6m/) — 17-joint subset, root-centred, metres, 50 Hz.
- **Biomechanical criteria:** Human3.6M+ OpenSim simulation CSVs (torque, power, GRF, muscle activation, etc.).
- **Train subjects:** S1, S5, S6, S7, S8 — **Test subjects:** S9, S11.

See [`docs/preprocessing.md`](docs/preprocessing.md) for the full data preparation pipeline.

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
