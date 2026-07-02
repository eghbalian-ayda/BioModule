"""
precompute_poses.py
-------------------
Run a frozen 3D pose estimator on Human3.6M CPN 2D detections and save the
resulting 3D poses as a .npy file for use with BioModule evaluation.

The output file is a nested dict saved with np.save:
    {subject: {action: np.ndarray (T, 17, 3)}}

This can be passed directly to the BioModule pipeline.

Built-in model types
--------------------
  mhformer   MHFormer (Multi-Hypothesis Transformer), 351-frame window.
             Pass --model-weights to the .pth file or its parent folder.

  custom     Placeholder for any other model. Edit the load_custom_model()
             function in this file to point to your model's repo and class,
             then pass --model-type custom.

Usage examples
--------------
MHFormer (test set S9+S11):
  python scripts/precompute_poses.py \\
      --pose-2d      dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz \\
      --model-type   mhformer \\
      --model-weights model/model_4294.pth \\
      --frames       351 \\
      --subjects     S9 S11 \\
      --out          poses_mhformer.npy

Custom model (243-frame window):
  python scripts/precompute_poses.py \\
      --pose-2d      dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz \\
      --model-type   custom \\
      --model-weights /path/to/weights.pth \\
      --frames       243 \\
      --subjects     S9 S11 \\
      --out          poses_mymodel.npy
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_module.dataset        import precompute_poses
from bio_module.pose_estimator import (make_mhformer_fn,
                                       make_center_frame_fn)


# ---------------------------------------------------------------------------
# Model loaders — edit load_custom_model() for your own model
# ---------------------------------------------------------------------------

def load_mhformer(weights_path: str, frames: int, device: torch.device):
    import types
    from model.mhformer import Model
    opt = types.SimpleNamespace(
        layers=3, channel=512, d_hid=1024, frames=frames,
        n_joints=17, out_joints=17, in_channels=2, out_channels=3,
        out_all=1, crop_uv=0, pad=(frames - 1) // 2,
    )
    model = Model(opt).to(device)
    p = weights_path
    if os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.pth')))
        p = next((x for x in paths if Path(x).name.startswith('model')), paths[0])
    ckpt       = torch.load(p, map_location=device, weights_only=False)
    model_dict = model.state_dict()
    state      = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(state)
    model.load_state_dict(model_dict)
    model.requires_grad_(False)
    model.eval()
    print(f'Loaded MHFormer from {p}')
    return model


def load_custom_model(weights_path: str, frames: int, device: torch.device):
    """
    Edit this function to load your own pose estimator.

    Expected interface: the model takes a batch of 2D keypoints
        (B, frames, 17, 2) → (B, 17, 3)   (centre-frame output)
    or
        (B, frames, 17, 2) → (B, frames, 17, 3)   (all-frames output)

    After loading, wrap with make_center_frame_fn or make_mhformer_fn
    as appropriate (see the main() function below).

    Example (PoseMamba):
        POSEMAMBA_ROOT = Path('/path/to/PoseMamba')
        sys.path.insert(0, str(POSEMAMBA_ROOT))
        from lib.model.PoseMamba import PoseMamba
        model = PoseMamba(num_frame=frames, num_joints=17, in_chans=2,
                          embed_dim_ratio=128, depth=20, mlp_ratio=2).to(device)
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_pos'])
        model.requires_grad_(False)
        model.eval()
        return model
    """
    raise NotImplementedError(
        'Edit load_custom_model() in this file to load your model.'
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pose-2d',       required=True,
                   help='data_2d_h36m_cpn_ft_h36m_dbb.npz')
    p.add_argument('--model-type',    required=True,
                   choices=['mhformer', 'custom'])
    p.add_argument('--model-weights', required=True,
                   help='.pth file or parent folder')
    p.add_argument('--frames',        type=int, required=True,
                   help='Temporal window size (e.g. 81, 243, 351)')
    p.add_argument('--subjects',      nargs='+', default=['S9', 'S11'])
    p.add_argument('--camera-idx',    type=int, default=0)
    p.add_argument('--batch-size',    type=int, default=256)
    p.add_argument('--cache-dir',     default='/tmp/bio_pose_cache')
    p.add_argument('--cache-tag',     default=None)
    p.add_argument('--out',           required=True,
                   help='Output .npy path')
    p.add_argument('--gpu',           default='0')
    return p.parse_args()


def main():
    args   = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cache_tag = args.cache_tag or args.model_type

    if args.model_type == 'mhformer':
        model    = load_mhformer(args.model_weights, args.frames, device)
        model_fn = make_mhformer_fn(model)
    else:
        model    = load_custom_model(args.model_weights, args.frames, device)
        model_fn = make_center_frame_fn(model, args.frames)

    print(f'\nLoading 2D keypoints from {args.pose_2d} …')
    raw_2d = np.load(args.pose_2d, allow_pickle=True)['positions_2d'].item()

    print(f'\nRunning {args.model_type} (frames={args.frames}) on {args.subjects} …\n')
    poses_dict = precompute_poses(
        model_fn   = model_fn,
        pos_2d     = raw_2d,
        subjects   = args.subjects,
        device     = device,
        cache_dir  = args.cache_dir,
        frames     = args.frames,
        batch_size = args.batch_size,
        camera_idx = args.camera_idx,
        cache_tag  = cache_tag,
    )

    out_dict: dict[str, dict[str, np.ndarray]] = {}
    for (subj, act), arr in poses_dict.items():
        out_dict.setdefault(subj, {})[act] = arr

    np.save(args.out, out_dict)
    n_clips = sum(len(v) for v in out_dict.values())
    print(f'\nSaved {n_clips} clips → {args.out}')


if __name__ == '__main__':
    main()
