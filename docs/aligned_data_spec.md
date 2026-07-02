# Aligned Data and Attribute Specifications

Reference tables for the `aligned.npz` per-clip archive, the BioModule V3
prediction attribute set, and the cross-dataset anatomical correspondence
between OpenSim degrees of freedom, Human3.6M joint indices, and H36M+
K-markers.

---

## Table 1 — `aligned.npz` array layout

Each `aligned.npz` file is produced by `scripts/build_aligned_dataset.py` for
one (subject, action) clip. All arrays share the leading temporal dimension `T`
(frame count per clip, roughly 600–5,000 frames at 50 fps).

Camera calibration (intrinsics K, rotation R, translation t) is applied
identically to H36M joints and OpenSim K-markers so both modalities project to
the same pixel space, enabling frame-accurate co-registration via the shared
pelvis anchor (K1 ≡ joint 0).

| Array | Shape | dtype | Purpose |
|-------|-------|-------|---------|
| `poses_2d` | `(T, 4, 17, 2)` | float32 | Raw H36M 2D joint annotations in pixel coordinates across 4 synchronised camera views. Native image-space detections; direct input for 2D-to-3D lifting estimators. |
| `poses_3d` | `(T, 32, 3)` | float32 | Triangulated 3D world coordinates (metres) of all 32 H36M joints in the camera-rig frame (z-up). Pelvis retains absolute world position; root-centring is applied at load time. |
| `poses_3d_px` | `(T, 4, 32, 2)` | float32 | `poses_3d` projected into each camera via the H36M calibration matrices. The 17-joint subset reproduces `poses_2d` to <0.28 px, validating 3D–2D consistency. |
| `markers_px` | `(T, 4, 19, 2)` | float32 | 19 OpenSim K-markers projected via the *same* calibration. K1 coincides with H36M joint 0 to machine precision, confirming the shared pelvis root anchor. |
| `dofs` | `(T, 40)` | float32 | OpenSim generalised coordinates (y-up body frame). Idx 0–2: pelvis translation; 3–5: pelvis rotation; 6–39: 34 actuated DOFs (J7–J40). |
| `cam_ids` | `(4,)` | str | Camera IDs in array order: `54138969`, `55011271`, `58860488`, `60457274`. |
| `n_frames` | scalar | int32 | Total frame count T for the clip; used for array-length validation at load time. |

---

## Table 2 — BioModule V3 prediction attributes

Output attributes predicted by BioModule V3, grouped by biomechanical tier.
`dim` is the output dimensionality per frame per attribute. Kinematic attributes
span all 40 generalised coordinates (J1–J40); actuated-DOF attributes cover the
34 driven joints (J7–J40); neuromuscular outputs are doubled (×2) for paired
agonist–antagonist actuators. The tier weight λ scales each group's contribution
to the combined training loss.

| Attribute | Tier (λ) | dim | DOF range | Loss | Unit |
|-----------|----------|-----|-----------|------|------|
| `coordinate` | Kinematic (1.0) | 40 | J1–J40 | MSE | rad / m |
| `speed` | Kinematic (1.0) | 40 | J1–J40 | MSE | rad/s, m/s |
| `acceleration` | Kinematic (1.0) | 40 | J1–J40 | MSE | rad/s² |
| `active_torque` | Kinetic (0.5) | 34 | J7–J40 | MSE | N·m |
| `passive_torque` | Kinetic (0.5) | 34 | J7–J40 | MSE | N·m |
| `ideal_torque` | Kinetic (0.5) | 34 | J7–J40 | MSE | N·m |
| `instantaneous_power` | Kinetic (0.5) | 34 | J7–J40 | MSE | W |
| `instantaneous_power_raw` | Kinetic (0.5) | 34 | J7–J40 | MSE | W |
| `ground_reaction` | Kinetic (0.5) | 12 | R+L feet | MSE | N / N·m |
| `seat_reaction` | Kinetic (0.5) | 1 | Pelvis | MSE | N |
| `touch` | Kinetic (0.5) | 2 | R/L ankles | BCE | — |
| `activation_signal` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | [0, 1] |
| `excitation_signal` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | [0, 1] |
| `normalized_active_torque` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | — |
| `angle_scaling` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | — |
| `velocity_scaling` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | — |
| `maximum_joint_torque` | Neuromuscular (0.3) | 68 | J7–J40 ×2 | MSE | N·m |

> **Note:** `activation_signal` and `excitation_signal` were not exported from
> OpenSim and are zero-filled in the stored data. They are excluded from
> training (not present in `checkpoint/norm_stats.json`).

---

## Table 3 — DOF / H36M joint / K-marker correspondence

Cross-dataset anatomical correspondence between OpenSim generalised coordinates
(DOF index), Human3.6M joint indices (0-indexed, 32-joint convention), and
H36M+ K-marker identifiers. DOFs J1–J6 encode the pelvis free-body and anchor
to joint 0 / K1. Dashes indicate landmarks present in one dataset but absent or
non-uniquely mappable in the other (fingertips, toe tips, and the undivided
thorax–abdomen segment).

| DOF | Anatomical segment | H36M joint (idx, name) | K-marker |
|-----|--------------------|------------------------|----------|
| J1–J3 | Pelvis translation (x, y, z) | 0 — Hip / Pelvis | K1 |
| J4–J6 | Pelvis rotation (rx, ry, rz) | 0 — Hip / Pelvis | K1 |
| J7–J9 | Right hip (flex / add / rot) | 1 — RHip | K2 |
| J10 | Right knee (flex) | 2 — RKnee | K3 |
| J11–J12 | Right ankle / subtalar | 3 — RFoot | K4, K5 |
| J13 | Right metatarsal (MTP) | 3 — RFoot | K6 |
| J14–J16 | Left hip (flex / add / rot) | 6 — LHip | K8 |
| J17 | Left knee (flex) | 7 — LKnee | K9 |
| J18–J19 | Left ankle / subtalar | 8 — LFoot | K10, K11 |
| J20 | Left metatarsal (MTP) | 8 — LFoot | K12 |
| J21–J23 | Lumbar (flex / ext / lat / rot) | 12 — Spine | — |
| J24 | Thorax / abdomen (unified) | 13 — Thorax | — |
| J25–J27 | Right shoulder (flex / add / rot) | 14 — RShoulder | K13 |
| J28 | Right elbow (flex) | 15 — RElbow | K14 |
| J29 | Right wrist / forearm | 16 — RWrist | K15 |
| J30 | Right 3rd finger MCP | — | K16 |
| J31–J33 | Left shoulder (flex / add / rot) | 11 — LShoulder | K17 |
| J34 | Left elbow (flex) | 12 — LElbow | K18 |
| J35 | Left wrist / forearm | 13 — LWrist | K19 |
| J36 | Left 3rd finger MCP | — | — |

**H36M+ markers with no DOF or H36M joint equivalent:**

| DOF | Anatomical segment | H36M joint | K-marker |
|-----|--------------------|------------|----------|
| — | Right 3rd fingertip | — | K7 |
| — | Left 3rd fingertip | — | K11 |
| — | Right 2nd toe tip | — | K15 |
| — | Left 2nd toe tip | — | K19 |
