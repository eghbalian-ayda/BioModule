# Data Preprocessing — H36M+ BioModule Pipeline

This document describes how the raw source data (Human3.6M video + OpenSim simulation CSVs)
was converted into the `processed_all/` directory consumed by BioModule training and evaluation.

---

## 1. Source Data

### 1a. Human3.6M poses

| File | Content |
|------|---------|
| `dataset/data_3d_h36m.npz` | 3D world-space joint positions, 32-joint skeleton, metres |
| `dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz` | CPN 2D detections (used at test time for pose estimators) |
| `dataset/data_2d_h36m_gt.npz` | GT 2D projections (used for visualization / alignment checks) |

The 32-joint H36M skeleton uses the convention: joint 0 = pelvis (root). World-space coordinates
are in metres, Y-up.

### 1b. Human3.6M+ OpenSim simulation CSVs

Stored on Google Drive:
```
Colab Notebooks/MHFormer+h3.6plus/h36m_database/{Subject}/{Action}/
    {Action}_{gender}_{criterion_name}.csv
```

Each CSV has one row per frame (50 Hz) and one column per degree of freedom (or marker).
The `{gender}` tag (`female` / `male`) reflects which OpenSim musculoskeletal model variant
was used for that simulation run. It is inconsistent across criteria — see §3.

Processing was performed in Google Colab. The original output path visible in `metadata.json`
is `/content/drive/MyDrive/Colab Notebooks/MHFormer+h3.6plus/...`.

---

## 2. Output: `processed_all/` Directory

```
processed_all/processed_all/
  {Subject}/
    {Action}/
      {Subject}_{Action}.npy         # 3D poses, (T, 32, 3), float32, metres
      criterion_*.npy                # one file per criterion
      metadata.json                  # shapes, source CSVs, has_real_data flags
```

Subjects: S1, S5, S6, S7, S8, S9, S11
Actions:  all H36M actions (Directions, Discussion, Eating, Greeting, Phoning,
          Photo, Posing, Purchases, Sitting, SittingDown, Smoking, Waiting,
          WalkDog, WalkTogether, Walking — each with a '1' or '2' variant)

---

## 3. Criterion Files

### 3a. Criterion overview

| File | Shape | DoF dim | Type | Source CSV gender |
|------|-------|---------|------|-------------------|
| `criterion_dimensionless_active_torque.npy` | (T, 34) | 34 | torque | **male** |
| `criterion_dimensionless_passive_torque.npy` | (T, 34) | 34 | torque | male |
| `criterion_dimensionless_ideal_torque.npy` | (T, 34) | 34 | torque | male |
| `criterion_normalized_active_torque.npy` | (T, 68) | 68 | torque | male |
| `criterion_dimensionless_instantaneous_power.npy` | (T, 34) | 34 | power | male |
| `criterion_dimensionless_acceleration.npy` | (T, 34) | 34 | kinematic | female |
| `criterion_dimensionless_speed.npy` | (T, 40) | 40 | kinematic | **male** |
| `criterion_active_torque_angle_scaling_function.npy` | (T, 68) | 68 | scaling | **male** |
| `criterion_active_torque_angular_velocity_scaling_function.npy` | (T, 68) | 68 | scaling | male |
| `criterion_dimensionless_ground_reaction.npy` | (T, 12) | 12 | GRF | **female** |
| `criterion_dimensionless_seat_reaction.npy` | (T, 1) | 1 | seat force | male |
| `criterion_dimensionless_marker.npy` | (T, 60) | 60 | marker pos | male |
| `criterion_touch.npy` | (T, 2) | 2 | contact | **female** |
| `criterion_seat.npy` | (T, 1) | 1 | contact | male (seat reaction) |

Gender selection is inconsistent across criteria — it reflects which OpenSim model variant
produced each simulation run. The gender used for each criterion was verified by checking
which CSV produces zero interpolation error against the stored `.npy`.

Three criteria have **no real data** and are filled with zeros:

| File | Reason |
|------|--------|
| `criterion_activation_signals.npy` | OpenSim muscle activations were not exported |
| `criterion_excitation_signals.npy` | OpenSim muscle excitations were not exported |
| `criterion_dimensionless_instantaneous_power_normalized.npy` | Not computed |

These three are excluded from BioModule training (they never appear in `norm_stats.json`).

### 3b. DoF ordering — 34-dim criteria

Applies to: `dimensionless_active_torque`, `dimensionless_passive_torque`,
`dimensionless_ideal_torque`, `dimensionless_instantaneous_power`,
`dimensionless_acceleration`.

```
Index  Joint           #DoFs  Cumulative
 0-2   lumbar           3      3
 3-5   neck             3      6
 6-8   right_shoulder   3      9
 9-10  right_elbow      2     11
11-12  right_wrist      2     13
13-15  left_shoulder    3     16
16-17  left_elbow       2     18
18-19  left_wrist       2     20
20-22  right_hip        3     23
23     right_knee       1     24
24-26  right_ankle      3     27
27-29  left_hip         3     30
30     left_knee        1     31
31-33  left_ankle       3     34
```

**Key index:** right_knee = 23, left_knee = 30.

### 3c. DoF ordering — 40-dim (speed)

`dimensionless_speed` prepends 6 pelvis DoFs (3 translation + 3 rotation) before the
34 joint DoFs. All joint indices shift by +6.

```
0-2    pelvis translation (x, y, z)
3-5    pelvis rotation (rx, ry, rz)
6-8    lumbar
9-11   neck
12-14  right_shoulder
15-16  right_elbow
17-18  right_wrist
19-21  left_shoulder
22-23  left_elbow
24-25  left_wrist
26-28  right_hip
29     right_knee          ← index 29
30-32  right_ankle
33-35  left_hip
36     left_knee           ← index 36
37-39  left_ankle
```

### 3d. DoF ordering — 68-dim (flexor/extensor pairs)

Applies to: `normalized_active_torque`, `active_torque_angle_scaling_function`,
`active_torque_angular_velocity_scaling_function`.

```
Indices  0-33:  flexors  (same 34-joint ordering as §3b)
Indices 34-67:  extensors (same ordering, offset by 34)
```

Right-knee flexor = index 23, right-knee extensor = index 57.

### 3e. Ground-reaction forces — 12-dim

`dimensionless_ground_reaction` is a 12-element vector per frame:

```
0-5   left foot:  [Fx, Fy, Fz, Mx, My, Mz]
6-11  right foot: [Fx, Fy, Fz, Mx, My, Mz]
```

Forces (F) in the original model are in Newtons; moments (M) in N·m.
Both are made dimensionless by body-weight × body-height (see §4).

### 3f. Markers — 60-dim

`dimensionless_marker` encodes 20 surface marker positions × 3 coordinates = 60 values.

```
Markers 0-18: real OpenSim surface markers (19 markers)
Marker 19:    placeholder, always [0, 0, 0]
```

Coordinate system: OpenSim Y-up (X forward, Y up, Z lateral).
**Not** the same axes as H36M world space (Z-up).
Values are dimensionless (divided by body height).

Approximate marker-to-body-part correspondence:

| Index | Body part |
|-------|-----------|
| 0 | pelvis |
| 1 | thorax |
| 2 | neck/head |
| 3-6 | left arm (shoulder, elbow, wrist, hand) |
| 7-10 | right arm (shoulder, elbow, wrist, hand) |
| 11-14 | left leg (hip, knee, ankle, foot) |
| 15-18 | right leg (hip, knee, ankle, foot) |

### 3g. Contact criteria — touch and seat

`criterion_touch.npy` — shape (T, 2): `[left_foot_contact, right_foot_contact]`
`criterion_seat.npy` — shape (T, 1): seat contact flag

Raw values in the CSVs are **continuous** (e.g. normalized ground-reaction force or
a soft contact probability, range roughly 0–1). They are **not** pre-thresholded in
`processed_all/`.

BioModule's `dataset.py` applies the threshold at load time:
```python
chunk = (chunk > 0).astype(np.float32)   # converts to hard {0, 1}
```

`criterion_seat.npy` is derived from the same source CSV as
`criterion_dimensionless_seat_reaction.npy` — the seat reaction force is thresholded
to produce the binary contact label.

---

## 4. Dimensionless Normalization (OpenSim → stored values)

All physical quantities in H36M+ are normalized before storage using:

```
value_dimensionless = value_physical / (body_weight × body_height^n)
```

where `n` depends on the quantity:

| Quantity | Denominator | Typical stored range |
|----------|-------------|----------------------|
| Torque | BW · h | ≈ [−12, 12] N·m / (N · m) = dimensionless |
| Power | BW · h² / s | ≈ [−55, 34] |
| Force (GRF) | BW | ≈ [0, 1.1] |
| Speed (angular) | rad/s (already dimensionless) | ≈ [−11, 13] |
| Acceleration | rad/s² | ≈ [−89, 91] |
| Marker position | h | ≈ [−2.6, 2.6] |
| Scaling functions | — (already [0, 1]) | [0, 1.4] |

Body weight and body height are subject-specific constants from the OpenSim model.

---

## 5. Frame Count and Temporal Interpolation

Both H36M and OpenSim CSVs run at **50 Hz** (time step = 0.02 s). However, the OpenSim
simulation length is consistently **shorter** than the H36M capture for the same action —
typically by 1–48 frames, occasionally much more (e.g. WalkingDog for S1 is ~1400 frames short).

**The CSVs are temporally interpolated to match the H36M frame count exactly.**

The method, confirmed by reverse-engineering against the raw CSVs:

```python
import numpy as np
from scipy.interpolate import interp1d

T_csv = len(csv_data)         # e.g. 3101 for S1/Walking
T_h36m = pose_array.shape[0]  # e.g. 3134

t_src = np.arange(T_csv)
t_dst = np.linspace(0, T_csv - 1, T_h36m)   # evenly re-sample to H36M length

f = interp1d(t_src, csv_data, axis=0, kind='linear', fill_value='extrapolate')
resampled = f(t_dst)   # (T_h36m, n_cols)
```

This linearly stretches the simulation timeline to cover the full H36M recording duration.
For the typical 1–48 frame difference the change is negligible (<0.02 s). For the
anomalous WalkingDog cases, the simulation is extrapolated well beyond its original end —
which is why S1/WalkingDog and S1/WalkingDog1 are marked as corrupted in `criteria_loader.py`
and excluded from training.

**Frame counts confirmed for S1 (sample):**

| Action | CSV rows | H36M frames | Difference |
|--------|----------|-------------|------------|
| Walking | 3101 | 3134 | 33 |
| Directions | 1601 | 1612 | 11 |
| Greeting | 1101 | 1149 | 48 |
| Sitting1 | 3301 | 3304 | 3 |
| WalkingDog | 1751 | 3134 | **1383** ← corrupted |

Typical clip lengths after alignment: 1000–6000 frames (20 s – 2 min).

### NaN values in marker CSV

The raw `dimensionless_marker.csv` contains NaN values (primarily in the last 3 columns,
i.e. the 20th marker's x/y/z — the placeholder marker). After interpolation, NaN positions
are **filled with 0.0** (not interpolated, since neighboring values are also NaN).
This is consistent with what `dataset.py` does at load time (`nan_to_num(..., nan=0.0)`)
and explains why marker index 19 is always [0, 0, 0] in the stored `.npy`.

---

## 6. Pose File: `{Subject}_{Action}.npy`

Shape: `(T, 32, 3)`, float32, **metres**, H36M world-space (Z-up convention).

This is the ground-truth 3D skeleton from `data_3d_h36m.npz`, frame-aligned
with the OpenSim simulation.

At training/eval time `dataset.py` applies two further transforms:
1. **Joint selection:** 32 joints → 17 joints via `H36M_17_JOINTS = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]`
2. **Root centering:** pelvis (joint 0) subtracted from all joints

These two steps are done in memory; the stored `.npy` file always contains all 32 joints
in original world coordinates.

---

## 7. Z-score Normalization at Training Time

Criterion values are stored **raw** (physical dimensionless units, §4) in `processed_all/`.
Z-score normalization is applied at training time by `BioDataset._compute_norm_stats()`:

```python
mean = array_over_training_clips.mean()
std  = array_over_training_clips.std()
chunk = (chunk - mean) / max(std, 1e-6)
```

Statistics are computed from **training subjects only** (S1, S5, S6, S7, S8)
and saved to `checkpoint/bio_gt/norm_stats.json` via `save_norm_stats.py`.

Binary criteria (`touch`, `seat`) are **excluded** from z-score normalization.

At inference / visualization, predictions are denormalized:
```python
value = prediction * std + mean
```

---

## 8. Train / Test Split

| Split | Subjects | Pose input |
|-------|----------|-----------|
| Train | S1, S5, S6, S7, S8 | GT 3D keypoints (`data_3d_h36m.npz`) |
| Test  | S9, S11            | Pose estimator output (MHFormer / D3DP / etc.) |

The domain gap between GT poses (train) and estimated poses (test) is intentional —
it reflects real deployment conditions.

---

## 9. Window Construction

BioDataset extracts sliding windows of **27 frames** with **stride 1**:

```python
for start in range(0, T - 27 + 1, 1):
    window = poses[start : start + 27]   # (27, 17, 3)
```

Approximate window counts (stride=1):

| Split | Clips | Windows |
|-------|-------|---------|
| Train (S1,S5,S6,S7,S8) | ~160 | ~386 k |
| Test  (S9, S11)        | ~60  | ~134 k |

---

## 10. Reproducing the Preprocessing

The Colab notebooks that generated `processed_all/` are on Google Drive at:
```
My Drive / Colab Notebooks / MHFormer+h3.6plus /
```

To recompute z-score stats from an existing `processed_all/`:
```bash
python3 save_norm_stats.py \
    --pose-3d       dataset/data_3d_h36m/positions_3d.npy \
    --processed-dir dataset/processed_all/processed_all \
    --checkpoint-dir checkpoint/bio_gt
```

To build the aligned H36M ↔ H36M+ dataset (marker pixel coords + DoF values):
```bash
python3 build_aligned_dataset.py
```
(Requires access to the H36M+ database path on the local drive — see `H36M_PLUS` in that script.)
