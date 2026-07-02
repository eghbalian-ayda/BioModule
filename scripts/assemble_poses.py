"""
Quick script: assemble poses_tcpformer.npy from cached .npy files.
The cache files are named: tcpformer_{subj}_{canon_action}_{cam_idx}.npy
We need to match them back to the original action names from the 2D keypoints file.
"""
import re
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

CACHE_DIR   = Path('/tmp/bio_pose_cache')
POSE_2D     = Path('/home/ayda/Documents/Githubs/MHFormer/dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz')
SUBJECTS    = ['S9', 'S11']
CACHE_TAG   = 'tcpformer'
CAMERA_IDX  = 0
OUT         = _ROOT / 'poses_tcpformer.npy'


def _canon(name: str) -> str:
    return re.sub(r'[\s_\-]+', '', name.strip().lower())


def main():
    print(f"Loading 2D keypoints to get action names ...")
    raw_2d = np.load(str(POSE_2D), allow_pickle=True)['positions_2d'].item()

    out_dict: dict[str, dict[str, np.ndarray]] = {}

    for subj in SUBJECTS:
        if subj not in raw_2d:
            print(f"  {subj} not in 2D keypoints, skipping")
            continue
        out_dict[subj] = {}
        for act, cam_list in raw_2d[subj].items():
            tag        = f"{CACHE_TAG}_{subj}_{_canon(act)}_{CAMERA_IDX}"
            cache_path = CACHE_DIR / f"{tag}.npy"
            if cache_path.exists():
                arr = np.load(str(cache_path))
                out_dict[subj][act] = arr
                print(f"  loaded {subj}/{act}  shape={arr.shape}")
            else:
                print(f"  MISSING: {cache_path.name}")

    n_clips = sum(len(v) for v in out_dict.values())
    np.save(str(OUT), out_dict)
    print(f"\nSaved {n_clips} clips → {OUT}")


if __name__ == '__main__':
    main()
