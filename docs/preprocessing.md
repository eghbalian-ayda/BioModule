# Data Acquisition and Preprocessing

This document describes every data source BioModule depends on, where to obtain
it, which parts are restricted, and the full processing pipeline to reproduce
the `aligned data/` database from scratch.

---

## 1. Data Sources

### 1a. Human3.6M — pose data (restricted)

**License required.** Request access at:
> http://vision.imar.ro/human3.6m/

After approval you will receive download links. The files BioModule needs are:

| File | Content | Used by |
|------|---------|---------|
| `data_3d_h36m.npz` | 3D world-space joint positions, 32-joint skeleton, metres | training, alignment |
| `data_2d_h36m_cpn_ft_h36m_dbb.npz` | CPN 2D detections | pose estimator inference |
| `data_2d_h36m_gt.npz` | GT 2D projections | alignment |

Place them under a `dataset/` folder next to the repo:

```
dataset/
  data_3d_h36m.npz
  data_2d_h36m_cpn_ft_h36m_dbb.npz
  data_2d_h36m_gt.npz
```

The H36M skeleton convention: joint 0 = pelvis (root), world-space coordinates
in metres, Z-up. BioModule selects 17 joints at runtime:
`[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]`.

---

### 1b. Human3.6M camera parameters (public)

```bash
git clone https://github.com/karfly/human36m-camera-parameters
```

This gives `camera-parameters.json`, which `build_aligned_dataset.py` uses to
project 3D joints into each of the four H36M camera views.

---

### 1c. Human3.6M+ OpenSim biomechanical data

The biomechanical ground-truth labels are OpenSim musculoskeletal simulation
outputs, computed per subject and action at 50 Hz to match H36M capture rate.

**Repository:** https://github.com/ainlamyae/Human3.6Mplus

**Database layout:**
```
h36m_database/
  S1/ S5/ S6/ S7/ S8/ S9/ S11/
    {Action}/
      {Action}_{gender}_{criterion}.csv
```

Each CSV has one row per frame and one column per degree of freedom.
The full list of criteria and their dimensionalities is in `docs/architecture.md`.

**To obtain:** contact the authors. The raw CSVs are not redistributed
because they are derived from subject-specific OpenSim models whose licensing
is intertwined with the H36M participant data.

**Pre-processed alternative:** if you only need to train or evaluate BioModule
without reconstructing the aligned database from scratch, download
`processed_all.zip` from the GitHub Release instead:

```bash
wget https://github.com/eghbalian-ayda/BioModule/releases/download/v1.0-data/processed_all.zip
unzip processed_all.zip          # → processed_all/processed_all/{S1,...}/
```

This archive (~948 MB compressed, ~2.4 GB unzipped) contains the pre-processed
`.npy` criterion files for all subjects and actions, ready for BioModule v1
training (`bio_module/train.py`).

---

## 2. Processing Pipeline

The pipeline has three independent stages. Run them in order the first time;
afterwards only stage 3 is needed for new pose estimators.

```
Stage 1:  H36M npz + H36M+ CSVs + camera params
               ↓ build_aligned_dataset.py
          h36m_database/.../aligned/aligned.npz  (one per action)

Stage 2:  aligned.npz (training subjects only)
               ↓ save_norm_stats.py
          checkpoint/norm_stats.json

Stage 3:  data_2d_h36m_cpn_ft_h36m_dbb.npz + pose estimator weights
               ↓ precompute_poses.py
          poses_{model}.npy
```

---

### Stage 1 — Build `aligned.npz` files

Reads H36M 3D/2D poses and the per-action coordinate CSV from the H36M+
database, reprojects joints through each camera, and saves a single compressed
`.npz` per action containing all aligned arrays.

```bash
python scripts/build_aligned_dataset.py \
    --db-root  /path/to/h36m_database \
    --pose-2d  dataset/data_2d_h36m_gt.npz \
    --pose-3d  dataset/data_3d_h36m.npz \
    --cam-json human36m-camera-parameters/camera-parameters.json
```

Optional: process only specific subjects:
```bash
python scripts/build_aligned_dataset.py \
    --db-root  /path/to/h36m_database \
    --pose-2d  dataset/data_2d_h36m_gt.npz \
    --pose-3d  dataset/data_3d_h36m.npz \
    --cam-json human36m-camera-parameters/camera-parameters.json \
    --subjects S9 S11
```

**Output per action:**
```
h36m_database/{Subject}/{Action}/aligned/aligned.npz
```

| Array | Shape | Description |
|-------|-------|-------------|
| `poses_2d` | `(N, 4, 17, 2)` | GT 2D keypoints, 4 cameras |
| `poses_3d` | `(N, 32, 3)` | 3D world joints, metres |
| `poses_3d_px` | `(N, 4, 32, 2)` | 3D joints projected into each camera |
| `markers_px` | `(N, 4, 19, 2)` | Surface marker pixel coords |
| `dofs` | `(N, 40)` | Joint DoF values from coordinate CSV |
| `cam_ids` | `(4,)` | Camera ID strings |
| `n_frames` | scalar | Usable frames (`min(N_h36m, N_csv)`) |

**Frame alignment note:** OpenSim simulations are typically 1–48 frames shorter
than the H36M capture for the same action. `build_aligned_dataset.py` takes the
shorter length (`min`). The criteria CSVs are **not** interpolated at this stage;
`bio_module_v3/dataset.py` interpolates them at load time.

---

### Stage 2 — Compute normalisation statistics

Reads the aligned database for training subjects, computes per-criterion
z-score statistics, and writes `checkpoint/norm_stats.json`.

The pre-computed file is already committed to this repository at
`checkpoint/norm_stats.json`. Only rerun this if you retrain on a different
subject split.

```bash
python scripts/save_norm_stats.py \
    --db-root /path/to/h36m_database \
    --out     checkpoint/norm_stats.json
```

---

### Stage 3 — Precompute pose-estimator outputs

Runs a frozen 3D pose estimator on H36M CPN 2D detections and saves the
per-subject/action 3D pose arrays used by the BioModule pipeline.

**MHFormer (351-frame window):**
```bash
python scripts/precompute_poses.py \
    --pose-2d      dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz \
    --model-type   mhformer \
    --model-weights model/model_4294.pth \
    --frames       351 \
    --subjects     S9 S11 \
    --out          poses_mhformer.npy
```

**Any other model:** edit `load_custom_model()` in `scripts/precompute_poses.py`
to load your architecture, then:
```bash
python scripts/precompute_poses.py \
    --pose-2d      dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz \
    --model-type   custom \
    --model-weights /path/to/weights.pth \
    --frames       243 \
    --subjects     S9 S11 \
    --out          poses_mymodel.npy
```

The output `.npy` contains:
```python
{
  'S9':  {'Walking 1': array(T, 17, 3), 'Directions': array(T, 17, 3), ...},
  'S11': {...},
}
```

---

## 3. Criterion Details

### 3a. Overview

| Criterion | Dims | Type | Source gender |
|-----------|------|------|---------------|
| `active_torque` | 34 | torque | male |
| `passive_torque` | 34 | torque | male |
| `ideal_torque` | 34 | torque | male |
| `normalized_active_torque` | 68 | torque | male |
| `instantaneous_power` | 34 | power | male |
| `instantaneous_power_raw` | 34 | power | male |
| `acceleration` | 34 | kinematic | female |
| `speed` | 40 | kinematic | male |
| `coordinate` | 40 | kinematic | female/male |
| `ground_reaction` | 12 | GRF | female |
| `seat_reaction` | 1 | force | male |
| `angle_scaling` | 68 | scaling | male |
| `velocity_scaling` | 68 | scaling | male |
| `maximum_joint_torque` | 68 | torque | male |
| `touch` | 2 | binary contact | female |
| `activation_signal` | 68 | muscle | — (zero-filled) |
| `excitation_signal` | 68 | muscle | — (zero-filled) |

`activation_signal` and `excitation_signal` were not exported from OpenSim and
are zero-filled. They are excluded from training (not present in `norm_stats.json`).

### 3b. DoF ordering — 34-dim criteria

Applies to: `active_torque`, `passive_torque`, `ideal_torque`,
`instantaneous_power`, `acceleration`.

```
Index   Joint              DoFs  Cumulative
 0-2    lumbar              3     3
 3-5    neck                3     6
 6-8    right_shoulder      3     9
 9-10   right_elbow         2    11
11-12   right_wrist         2    13
13-15   left_shoulder       3    16
16-17   left_elbow          2    18
18-19   left_wrist          2    20
20-22   right_hip           3    23
23      right_knee          1    24
24-26   right_ankle         3    27
27-29   left_hip            3    30
30      left_knee           1    31
31-33   left_ankle          3    34
```

### 3c. DoF ordering — 40-dim (speed, coordinate)

Prepends 6 pelvis DoFs before the 34 joint DoFs:
```
0-2   pelvis translation (x, y, z)
3-5   pelvis rotation (rx, ry, rz)
6-39  same 34-joint ordering as §3b (all indices shift by +6)
```

### 3d. DoF ordering — 68-dim (flexor/extensor pairs)

Applies to: `normalized_active_torque`, `angle_scaling`, `velocity_scaling`,
`maximum_joint_torque`.
```
Indices  0-33:  flexors   (same 34-joint ordering as §3b)
Indices 34-67:  extensors (same ordering, offset by 34)
```

### 3e. Ground-reaction forces — 12-dim

```
0-5   left foot:  [Fx, Fy, Fz, Mx, My, Mz]
6-11  right foot: [Fx, Fy, Fz, Mx, My, Mz]
```

Forces in N, moments in N·m, both made dimensionless by body-weight × body-height.

---

## 4. Dimensionless Normalisation

All OpenSim quantities are stored dimensionless:

```
value_dimensionless = value_physical / (body_weight × body_height^n)
```

| Quantity | n | Typical range |
|----------|---|---------------|
| Torque | 1 | [−12, 12] |
| Power | 2 | [−55, 34] |
| Force (GRF) | 0 | [0, 1.1] |
| Angular speed | 0 (rad/s) | [−11, 13] |
| Acceleration | 0 (rad/s²) | [−89, 91] |
| Marker position | 1 | [−2.6, 2.6] |
| Scaling functions | — | [0, 1.4] |

Z-score normalisation (mean/std across training subjects) is applied at
training time using the statistics in `checkpoint/norm_stats.json`.

---

## 5. Train / Test Split

| Split | Subjects | Pose input |
|-------|----------|-----------|
| Train | S1, S5, S6, S7, S8 | GT 3D keypoints |
| Test | S9, S11 | Pose estimator output |

The intentional domain gap (GT poses at train time, estimated poses at test time)
mirrors real deployment conditions.
