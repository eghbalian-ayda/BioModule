# BioModuleV3 — Architecture & Training Documentation

**Total parameters: 3,824,073**
**Task:** Predict 17 biomechanical criteria from a 27-frame window of 3D skeleton poses.

---

## Input

```
(B, W=27, 51)
```
- B = batch size
- W = 27 frames (temporal window)
- 51 = 17 H36M joints × 3D coordinates (x, y, z), root-centred (pelvis = origin), metres

**Source at training time:** GT 3D keypoints from `aligned.npz → poses_3d` (H36M 17-joint subset of 32-joint array)
**Source at inference time:** Output of a 3D pose estimator (e.g. MHFormer) applied to video

**Input is NOT normalised.** Raw metre-scale root-centred coordinates are fed directly into the model. Only the criteria labels are z-score normalised.

---

## Architecture

```
INPUT  (B, 27, 51)
  │
  ▼
┌──────────────────────────────┐
│  Linear(51 → 256)            │  Input embedding        13,312 params
└──────────────────────────────┘
  │
  ▼  (+)
┌──────────────────────────────┐
│  Sinusoidal Positional Enc.  │  Added in-place, non-learnable
│  max_len=512, d=256          │  0 params
└──────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Transformer Encoder  ×4 layers (Pre-LN)            │  3,159,040 params
│                                                     │
│  Each layer:                                        │
│  ┌───────────────────────────────────────────────┐  │
│  │  LayerNorm(256)                               │  │
│  │  Multi-Head Self-Attention  (8 heads, d=32)   │  │
│  │  + residual                                   │  │
│  │  LayerNorm(256)                               │  │
│  │  FFN: Linear(256→1024) → GELU → Drop(0.1)    │  │
│  │       Linear(1024→256)                        │  │
│  │  + residual                                   │  │
│  └───────────────────────────────────────────────┘  │
│  Per layer: 789,760 params                          │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────┐
│  LayerNorm(256)              │  Final normalisation    512 params
│  shape: (B, 27, 256)         │
└──────────────────────────────┘
  │
  ▼  split into 17 independent heads

OUTPUT HEADS  (610,209 params total)
Each head:  Linear(256→128) → GELU → Dropout(0.1) → Linear(128→dim_out)
```

---

## Output Heads (17 criteria)

| Criterion | Output dim | Tier | Loss | Head params |
|---|---|---|---|---|
| `coordinate` | 40 | Kinematic | MSE | 38,056 |
| `speed` | 40 | Kinematic | MSE | 38,056 |
| `acceleration` | 40 | Kinematic | MSE | 38,056 |
| `active_torque` | 34 | Kinetic | MSE | 37,282 |
| `passive_torque` | 34 | Kinetic | MSE | 37,282 |
| `ideal_torque` | 34 | Kinetic | MSE | 37,282 |
| `instantaneous_power` | 34 | Kinetic | MSE | 37,282 |
| `instantaneous_power_raw` | 34 | Kinetic | MSE | 37,282 |
| `ground_reaction` | 12 | Kinetic | MSE | 34,444 |
| `seat_reaction` | 1 | Kinetic | MSE | 33,025 |
| `touch` | 2 | Kinetic | BCE logits | 33,154 |
| `activation_signal` | 68 | Neuromuscular | MSE | 41,668 |
| `excitation_signal` | 68 | Neuromuscular | MSE | 41,668 |
| `normalized_active_torque` | 68 | Neuromuscular | MSE | 41,668 |
| `angle_scaling` | 68 | Neuromuscular | MSE | 41,668 |
| `velocity_scaling` | 68 | Neuromuscular | MSE | 41,668 |
| `maximum_joint_torque` | 68 | Neuromuscular | MSE | 41,668 |

**Total output: 617 values per frame across all 17 criteria.**

---

## Criterion Descriptions

| Criterion | DOF range | Unit | Description |
|---|---|---|---|
| `coordinate` | J1–J40 | rad / m | Generalized joint coordinates. J1–J6: pelvis 6-DOF. J7–J40: 34 joint angles. |
| `speed` | J1–J40 | rad/s · m/s | First derivative of generalized coordinates (joint velocities). |
| `acceleration` | J1–J40 | rad/s² · m/s² | Second derivative of generalized coordinates (joint accelerations). |
| `active_torque` | J7–J40 | N·m | Net torque from active muscle force at each actuated DOF. |
| `passive_torque` | J7–J40 | N·m | Torque from passive elastic elements (ligaments, soft tissue). |
| `ideal_torque` | J7–J40 | N·m | Net inverse-dynamics torque required to reproduce the observed motion. |
| `instantaneous_power` | J7–J40 | W | Joint power (filtered): ideal_torque × angular_velocity. |
| `instantaneous_power_raw` | J7–J40 | W | Same as `instantaneous_power` but unfiltered. |
| `ground_reaction` | R+L feet | N / N·m | GRFs and torques: [RGRFx/y/z, RGRTx/y/z, LGRFx/y/z, LGRTx/y/z]. |
| `seat_reaction` | Pelvis | N | Vertical seat reaction force (RSRFy). ~body weight when seated. |
| `touch` | R/L ankle | — | Binary foot contact. Raw logits; sigmoid applied at inference. |
| `activation_signal` | J7–J40 ×2 | 0–1 | Muscle activation for 34 DOFs × 2 actuators (±). |
| `excitation_signal` | J7–J40 ×2 | 0–1 | Neural excitation (motor command); leads activation by ~10 ms. |
| `normalized_active_torque` | J7–J40 ×2 | — | Active torque divided by maximum isometric torque (0–1). |
| `angle_scaling` | J7–J40 ×2 | — | Force–length scaling factor at current joint angle. |
| `velocity_scaling` | J7–J40 ×2 | — | Force–velocity (Hill model) scaling; <1 concentric, >1 eccentric. |
| `maximum_joint_torque` | J7–J40 ×2 | N·m | Maximum isometric torque at current angle and velocity. |

---

## Block-by-Block Parameter Count

| Block | Type | Shape / Config | Params |
|---|---|---|---|
| Input embedding | `nn.Linear` | 51 → 256 | 13,312 |
| Positional encoding | `_SinPosEnc` | d=256, max_len=512 | 0 |
| Encoder — Self-attention ×4 | `nn.MultiheadAttention` | 8 heads, d_k=32 | 1,054,720 |
| Encoder — Feed-forward ×4 | 2-layer MLP + GELU | 256 → 1024 → 256 | 2,104,320 |
| Final LayerNorm | `nn.LayerNorm` | d=256 | 512 |
| Prediction heads ×17 | 2-layer MLP per criterion | 256 → 128 → dim_out | 610,209 |
| **Grand total** | | | **3,824,073** |

---

## Tiered Weighted Loss

Criteria are grouped into three tiers reflecting physical causality. The total loss is a weighted sum of per-tier mean losses.

```
Kinematic  (weight 1.0):  coordinate, speed, acceleration
Kinetic    (weight 0.5):  active_torque, passive_torque, ideal_torque,
                           instantaneous_power, instantaneous_power_raw,
                           ground_reaction, seat_reaction, touch
Neuromuscular (weight 0.3): activation_signal, excitation_signal,
                             normalized_active_torque, angle_scaling,
                             velocity_scaling, maximum_joint_torque

L_total = 1.0 × mean(MSE_kinematic)
        + 0.5 × mean(MSE_kinetic)
        + 0.3 × mean(MSE_neuromuscular)
```

- Continuous criteria: `F.mse_loss(pred, target)` on z-score normalised targets
- Binary criterion (`touch`): `F.binary_cross_entropy_with_logits(pred, target)`
- Loss is computed on **all 27 frames** in the window during training; only the **centre frame** (W//2 = 13) is used for evaluation metrics.

---

## Training Pipeline

Training proceeds in three sequential phases.

### Phase 1 — GT Training

| Hyperparameter | Value | Notes |
|---|---|---|
| Train subjects | S1, S5, S6, S7, S8 | GT 3D poses from `aligned.npz → poses_3d` |
| Val subjects | S9, S11 | Non-overlapping stride-27 windows |
| Window (W) | 27 frames | ~0.54 s of context at 50 fps |
| Stride (train / val) | 1 / 27 | Dense for train; non-overlapping for val |
| Optimizer | AdamW | β₁=0.9, β₂=0.999, ε=1e-8 |
| Learning rate | 1e-4 | Initial LR |
| Weight decay | 1e-4 | Applied to all parameters |
| LR schedule | CosineAnnealingLR | T_max=50, η_min=1e-6 (lr × 0.01) |
| Batch size | 64 | |
| Epochs | 50 | Best checkpoint saved at lowest val loss |
| Gradient clipping | 1.0 | Global norm clip |
| Dropout | 0.1 | Encoder FFN and prediction heads |
| Input normalisation | None | Raw metres, root-centred |
| Label normalisation | Per-dim z-score | μ, σ from train set → `norm_stats.json` |
| Best checkpoint | Epoch 2, val loss = 8.9503 | `checkpoint/bio_v3_gt_train/best.pth` |

### Phase 2 — Frozen Evaluation

GT-trained weights applied directly to poses from each of 7 pose estimators. No gradient updates. Establishes a cross-estimator baseline.

| Parameter | Value |
|---|---|
| Weights source | `checkpoint/bio_v3_gt_train/best.pth` |
| Gradient updates | None (`model.eval()`, `torch.no_grad()`) |
| Test subjects | S9, S11 |
| Window stride | 27 (non-overlapping) |

### Phase 3 — Fine-tuning × 7

One fine-tuned model per pose estimator. Starts from GT checkpoint; adapts to the noise/bias profile of each estimator using that estimator's predicted poses on training subjects.

| Hyperparameter | Value |
|---|---|
| Starting weights | `checkpoint/bio_v3_gt_train/best.pth` |
| Train subjects | S1, S5, S6, S7, S8 (estimator poses) |
| Epochs | 10 |
| Learning rate | 1e-5 (fixed, no scheduler) |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Checkpoint | `checkpoint/bio_v3_ft_{key}/best.pth` |

### Pose Estimator Models

| Key | Model | Year | Test poses file | Train poses file |
|---|---|---|---|---|
| `mhformer` | MHFormer | 2022 | `mhformer_poses_s9_s11.npy` | `poses_mhformer_train.npy` |
| `tcpformer` | TCPFormer | 2025 | `poses_tcpformer.npy` | `poses_tcpformer_train.npy` |
| `posemamba` | PoseMamba | 2024 | `poses_posemamba.npy` | `poses_posemamba_train.npy` |
| `videopose` | VideoPose3D | 2019 | `poses_videopose.npy` | `poses_videopose_train.npy` |
| `motionagformer` | MotionAGFormer | 2024 | `poses_motionagformer.npy` | `poses_motionagformer_train.npy` |
| `ktpformer` | KTPFormer | 2024 | `poses_ktpformer.npy` | `poses_ktpformer_train.npy` |
| `d3dp` | D3DP | 2023 | `poses_d3dp.npy` | `poses_d3dp_train.npy` |

---

## Data Loading & Preprocessing

### Input — 3D Keypoints

| Step | Operation | Result |
|---|---|---|
| 1. Joint selection | Extract H36M 17-joint subset from 32-joint `poses_3d` using `H36M_17_JOINTS = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]` | (T, 17, 3) |
| 2. Root-centring | Subtract pelvis position (joint 0) from all 17 joints every frame | (T, 17, 3), pelvis = [0,0,0] |
| 3. Flatten | Reshape to 51-dimensional vector per frame | (T, 51) float32 |
| 4. Windowing | Sliding window W=27, stride 1 (train) or 27 (val) | (B, 27, 51) |

### Labels — Criteria CSVs

| Property | Value |
|---|---|
| Path pattern | `{db_root}/{S}/{action}/{action}_{gender}_{suffix}.csv` |
| Gender fallback | Try `female` first, then `male` |
| Time column | First column (`T`) dropped; remaining columns are data |
| Temporal alignment | Linear interpolation from T_csv → T_npz (`scipy.interp1d`) |
| NaN handling | `np.nan_to_num(arr, nan=0.0)` before normalisation |
| Normalisation (continuous) | Z-score: `(arr − mean) / std` per dimension; std < 1e-6 → clamped to 1.0 |
| Normalisation (binary) | `touch`: threshold > 0 → {0,1} float32; no z-score |

### Norm Statistics

Computed once from training subjects (S1, S5, S6, S7, S8) before training. Saved to `checkpoint/bio_v3_gt_train/norm_stats.json` and reused across all phases without recomputation.

Format: `{"mean": {criterion: [float × dim]}, "std": {criterion: [float × dim]}}`.
Binary criterion (`touch`) is excluded — no normalisation stats stored.

---

## Evaluation Protocol

| Metric | Formula | Applied to |
|---|---|---|
| MAE | `mean(|pred − gt|)` de-normalised | All continuous criteria |
| RMSE | `sqrt(mean((pred − gt)²))` de-normalised | All continuous criteria |
| nMAE (%) | `MAE / (max(gt) − min(gt)) × 100` | Continuous criteria with range > 1e-6 |
| Accuracy | `mean(sigmoid(pred) > 0.5 == gt > 0.5)` | `touch` (binary) only |
| Tier MAE | `mean(per-criterion MAEs within tier)` | Kinematic / Kinetic / Neuromuscular |
| Val loss | Tiered weighted loss (same as training) | Best-checkpoint selection |

- All metrics use the **centre frame only** (W//2 = index 13) to avoid window boundary bias.
- Predictions are de-normalised (`raw × std + mean`) before computing MAE/RMSE/nMAE.

---

## Checkpoint Structure

### Phase 1 — GT Training

| File | Contents |
|---|---|
| `latest.pth` | `model`, `optimizer`, `epoch`, `best_val`, `args` — saved every epoch for resuming |
| `best.pth` | `model`, `epoch`, `val_loss` — saved when val loss improves; source for Phases 2 & 3 |
| `norm_stats.json` | `mean`, `std` dicts; computed once, never overwritten |
| `train_log.csv` | One row per epoch: `epoch, train_loss, train_kinematic, train_kinetic, train_neuromuscular, val_loss, val_kinematic, val_kinetic, val_neuromuscular, lr, elapsed_s` |

### Phase 3 — Fine-tuning

| File | Contents |
|---|---|
| `checkpoint/bio_v3_ft_{key}/best.pth` | `model`, `epoch`, `train_loss` — saved when train loss improves |

---

## Data Flow (Full Pipeline)

```
Video frames
    │
    ▼
3D Pose Estimator (e.g. MHFormer, TCPFormer, PoseMamba, ...)
    │
    ▼
3D keypoints  (T, 17, 3)  — root-centred, metres
    │
    ▼  sliding window W=27
BioModuleV3 Encoder
    │
    ▼
17 × biomechanical criterion predictions  (B, 27, dim)
    │
    ▼  centre frame  [:, 13, :]
Per-frame estimates: coordinate, speed, acceleration,
  active_torque, ground_reaction, seat_reaction, touch,
  activation_signal, excitation_signal, ...
```

---

## Key Design Decisions

- **Shared encoder, separate heads** — temporal context learned once, specialised per criterion at the output. No parameter sharing between criteria.
- **All 27 frames contribute to loss** during training; only the centre frame (frame 13) is used for evaluation metrics. This eliminates boundary-effect bias at evaluation time.
- **Pre-LN (norm_first=True)** — LayerNorm applied before attention/FFN rather than after, for stable gradient flow without warm-up.
- **Pose input unnormalised** — raw root-centred metres fed to the model; only criterion labels are z-scored. This keeps the input space physically interpretable.
- **`coordinate` ≠ input poses** — the `coordinate` head predicts OpenSim generalized coordinates (joint angles in an anatomically-constrained model), not the H36M Euclidean positions used as input. Non-zero error on `coordinate` during GT-input evaluation is expected: the model must learn the mapping from H36M joint positions to OpenSim DOF angles.
- **`default_marker.csv`** (3D OpenSim surface markers) is used for spatial alignment verification only — not a model input or prediction target.
- **Binary criterion `touch`**: raw logits predicted; sigmoid applied at inference; BCE loss during training.
- **Tiered loss weights** encode physical causality: kinematics are most directly observable (1.0), kinetics require inverse dynamics (0.5), neuromuscular quantities depend on the muscle model (0.3).
