# Data

This directory holds reference data for BioModule.

```
data/
└── sample/          # Tiny reference clip (S1 / Sitting1) — committed to this repo
    ├── S1_Sitting1.npy                      # (T, 32, 3) H36M 3D poses, metres
    ├── criterion_dimensionless_active_torque.npy
    ├── criterion_dimensionless_passive_torque.npy
    ├── criterion_dimensionless_ideal_torque.npy
    ├── criterion_dimensionless_instantaneous_power.npy
    ├── criterion_dimensionless_acceleration.npy
    ├── criterion_dimensionless_speed.npy
    ├── criterion_dimensionless_ground_reaction.npy
    ├── criterion_dimensionless_seat_reaction.npy
    ├── criterion_dimensionless_marker.npy
    ├── criterion_normalized_active_torque.npy
    ├── criterion_active_torque_angle_scaling_function.npy
    ├── criterion_active_torque_angular_velocity_scaling_function.npy
    ├── criterion_activation_signals.npy
    ├── criterion_excitation_signals.npy
    ├── criterion_touch.npy
    ├── criterion_seat.npy
    └── metadata.json
```

---

## Full Training & Evaluation Data

The full dataset is too large to store in this repository. Download the
components you need from the links below.

### 1 — H36M+ Biomechanical GT (`processed_all/`)

Pre-processed OpenSim biomechanical criteria for all subjects/actions,
in the `processed_all/` format consumed by BioModule v1 (`bio_module/`).

**Download (GitHub Release):**
```bash
wget https://github.com/UTSA-VIRLab/BioModule/releases/download/v1.0-data/processed_all.zip
unzip processed_all.zip
```
File: `processed_all.zip` (~948 MB compressed, ~2.4 GB unzipped)

Expected layout after extraction:
```
processed_all/
  processed_all/
    S1/ S5/ S6/ S7/ S8/ S9/ S11/
      {Action}/
        {Subject}_{Action}.npy       # (T, 32, 3) 3D poses
        criterion_*.npy              # one file per criterion
        metadata.json
```

### 2 — H36M+ Aligned Data (`aligned data/`)

Full aligned database used by BioModule v2/v3 (`bio_module_v2/`, `bio_module_v3/`).
Contains per-action `aligned/aligned.npz` files and the OpenSim criteria CSVs.

This dataset is hosted on Google Drive (contact the authors for access).

Expected layout:
```
h36m_database/aligned data/
  S1/ S5/ S6/ S7/ S8/ S9/ S11/
    {Action}/
      aligned/
        aligned.npz        # poses_3d, poses_2d, markers_px, dofs, cam_ids
      {Action}_{gender}_{criterion}.csv
```

Pass the root to training/eval scripts with `--db-root`.

### 3 — Human3.6M Pose Data

The original Human3.6M 3D/2D keypoint files (`data_3d_h36m.npz`,
`data_2d_h36m_cpn_ft_h36m_dbb.npz`, etc.) are subject to the
[Human3.6M license](http://vision.imar.ro/human3.6m/). Request access
from the dataset providers directly:

```
http://vision.imar.ro/human3.6m/
```

Expected location after download:
```
dataset/
  data_3d_h36m.npz
  data_2d_h36m_cpn_ft_h36m_dbb.npz
  data_2d_h36m_gt.npz
  data_3d_h36m/
    positions_3d.npy
```
