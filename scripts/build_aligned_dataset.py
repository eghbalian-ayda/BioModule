"""
build_aligned_dataset.py
------------------------
For every (subject, action) folder in the H36M+ database, produce an
`aligned/aligned.npz` file that frame-aligns:

  - H36M GT 2D keypoints      (data_2d_h36m_gt.npz)
  - H36M 3D world joints      (data_3d_h36m.npz)
  - H36M 3D joints projected  (cv2.projectPoints + full lens distortion)
  - H36M+ DoF values          ({action}_{gender}_default_coordinate.csv)
  - H36M+ marker pixel coords (derived from projected 3D via a fixed joint map)

Output layout — one file per (subject, action):
    {db_root}/{Subject}/{Action}/aligned/aligned.npz

Arrays inside each .npz
-----------------------
  poses_2d     (N, 4, 17, 2)  float32  GT 2D keypoints, 4 cameras
  poses_3d     (N, 32, 3)     float32  3D world joints, metres
  poses_3d_px  (N, 4, 32, 2)  float32  3D joints projected per camera
  markers_px   (N, 4, 19, 2)  float32  marker pixel coords per camera
  dofs         (N, 40)        float32  joint DoF values from coordinate CSV
  cam_ids      (4,)           str      camera ID strings
  n_frames     scalar         int      usable frames (min of H36M and H36M+)

Usage
-----
  python scripts/build_aligned_dataset.py \\
      --db-root  /path/to/h36m_database/aligned_data \\
      --pose-2d  /path/to/data_2d_h36m_gt.npz \\
      --pose-3d  /path/to/data_3d_h36m.npz \\
      --cam-json /path/to/human36m-camera-parameters/camera-parameters.json

Data sources
------------
  H36M poses     : http://vision.imar.ro/human3.6m/  (license required)
  Camera params  : https://github.com/karfly/human36m-camera-parameters
  H36M+ CSV data : contact the authors (OpenSim simulations aligned to H36M)
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── camera IDs (index 0-3 matches npz camera order) ──────────────────────────
CAM_IDS = ['54138969', '55011271', '58860488', '60457274']

# ── marker → H36M 32-joint index ─────────────────────────────────────────────
# M1=pelvis, M2=thorax, M3=neck, M4-M7=L arm, M8-M11=R arm,
# M12-M15=L leg, M16-M19=R leg
MARKER_TO_J32 = np.array([
     0,   # M1  pelvis
    12,   # M2  thorax
    13,   # M3  neck/head
    17,   # M4  L. shoulder
    18,   # M5  L. elbow
    19,   # M6  L. wrist
    20,   # M7  L. hand
    25,   # M8  R. shoulder
    26,   # M9  R. elbow
    27,   # M10 R. wrist
    28,   # M11 R. hand
     6,   # M12 L. hip
     7,   # M13 L. knee
     8,   # M14 L. ankle
     9,   # M15 L. foot
     1,   # M16 R. hip
     2,   # M17 R. knee
     3,   # M18 R. ankle
     4,   # M19 R. foot
])

# ── action-name normalisation ─────────────────────────────────────────────────
_SPECIAL = {
    'TakingPhoto':  'Photo',
    'TakingPhoto1': 'Photo 1',
    'WalkingDog':   'WalkDog',
    'WalkingDog1':  'WalkDog 1',
}

def folder_to_h36m_key(folder_name: str) -> str:
    if folder_name in _SPECIAL:
        return _SPECIAL[folder_name]
    m = re.match(r'^([A-Za-z]+)(\d+)$', folder_name)
    if m:
        return m.group(1) + ' ' + m.group(2)
    return folder_name


def project_all(joints3d_m: np.ndarray, rvec, t_vec, K_mat, dist) -> np.ndarray:
    """Project (N, J, 3) world joints [metres] → (N, J, 2) pixel coords."""
    N, J, _ = joints3d_m.shape
    pts_mm  = (joints3d_m * 1000.0).reshape(-1, 1, 3).astype(np.float64)
    proj, _ = cv2.projectPoints(pts_mm, rvec, t_vec, K_mat, dist)
    return proj.reshape(N, J, 2).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db-root',  required=True,
                   help='Root of the H36M+ database (contains S1/, S5/, …)')
    p.add_argument('--pose-2d',  required=True,
                   help='data_2d_h36m_gt.npz')
    p.add_argument('--pose-3d',  required=True,
                   help='data_3d_h36m.npz')
    p.add_argument('--cam-json', required=True,
                   help='human36m-camera-parameters/camera-parameters.json')
    p.add_argument('--subjects', nargs='+', default=None,
                   help='Subjects to process (default: all S* dirs found)')
    args = p.parse_args()

    db_root = Path(args.db_root)

    print('Loading H36M npz files …')
    data_2d = np.load(args.pose_2d, allow_pickle=True)['positions_2d'].item()
    data_3d = np.load(args.pose_3d, allow_pickle=True)['positions_3d'].item()

    print('Loading camera parameters …')
    with open(args.cam_json) as f:
        cam_data = json.load(f)

    cam_params: dict = {}
    for subj in data_2d:
        cam_params[subj] = {}
        for cam_id in CAM_IDS:
            if subj not in cam_data['extrinsics']:
                continue
            if cam_id not in cam_data['extrinsics'][subj]:
                continue
            R_mat = np.array(cam_data['extrinsics'][subj][cam_id]['R'], dtype=np.float64)
            t_vec = np.array(cam_data['extrinsics'][subj][cam_id]['t'], dtype=np.float64)
            K_mat = np.array(cam_data['intrinsics'][cam_id]['calibration_matrix'], dtype=np.float64)
            dist  = np.array(cam_data['intrinsics'][cam_id]['distortion'], dtype=np.float64)
            rvec, _ = cv2.Rodrigues(R_mat)
            cam_params[subj][cam_id] = (rvec, t_vec, K_mat, dist)

    subjects = sorted(
        d for d in db_root.iterdir()
        if d.is_dir() and re.match(r'^S\d+$', d.name)
        and (args.subjects is None or d.name in args.subjects)
    )

    total_ok = total_skip = 0

    for subj_dir in subjects:
        subj = subj_dir.name
        if subj not in data_2d:
            print(f'  [WARN] {subj} not in H36M npz — skipping')
            continue

        for action_dir in sorted(d for d in subj_dir.iterdir() if d.is_dir()):
            folder   = action_dir.name
            h36m_key = folder_to_h36m_key(folder)

            if h36m_key not in data_2d.get(subj, {}):
                print(f'  [SKIP] {subj}/{folder} → "{h36m_key}" not in H36M npz')
                total_skip += 1
                continue

            kp2d_all = data_2d[subj][h36m_key]   # list of 4 arrays (N, 17, 2)
            kp3d     = data_3d[subj][h36m_key]   # (N, 32, 3)
            N_h36m   = kp3d.shape[0]

            coord_path = action_dir / f'{folder}_female_default_coordinate.csv'
            if not coord_path.exists():
                coord_path = action_dir / f'{folder}_male_default_coordinate.csv'
            if not coord_path.exists():
                print(f'  [SKIP] {subj}/{folder}: no coordinate CSV found')
                total_skip += 1
                continue

            df_coord = pd.read_csv(coord_path)
            dof_cols = [f'J{i}' for i in range(1, 41)]
            N_csv    = len(df_coord)
            N        = min(N_h36m, N_csv)

            poses_2d    = np.stack([kp2d_all[c][:N] for c in range(4)], axis=1).astype(np.float32)
            poses_3d    = kp3d[:N].astype(np.float32)
            poses_3d_px = np.empty((N, 4, 32, 2), dtype=np.float32)

            for c_idx, cam_id in enumerate(CAM_IDS):
                if cam_id not in cam_params.get(subj, {}):
                    poses_3d_px[:, c_idx] = np.nan
                    continue
                rvec, t_vec, K_mat, dist = cam_params[subj][cam_id]
                poses_3d_px[:, c_idx]   = project_all(poses_3d, rvec, t_vec, K_mat, dist)

            markers_px = poses_3d_px[:, :, MARKER_TO_J32, :]
            dofs       = df_coord[dof_cols].values[:N].astype(np.float32)

            out_dir  = action_dir / 'aligned'
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / 'aligned.npz'
            np.savez_compressed(
                out_path,
                poses_2d    = poses_2d,
                poses_3d    = poses_3d,
                poses_3d_px = poses_3d_px,
                markers_px  = markers_px,
                dofs        = dofs,
                cam_ids     = np.array(CAM_IDS),
                n_frames    = np.int32(N),
            )

            total_ok += 1
            trim = f'(trimmed {abs(N_h36m - N_csv)} frames)' if N_h36m != N_csv else ''
            print(f'  OK  {subj}/{folder:20s} → "{h36m_key}"  N={N}  {trim}')

    print(f'\nDone. {total_ok} files written, {total_skip} skipped.')


if __name__ == '__main__':
    main()
