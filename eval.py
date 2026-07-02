"""
Generic BioModule evaluation script.

Plug any 3D pose estimator into the BioModule by providing its output poses
in a standard .npy file, then run this script to get per-criterion results.

Input pose file format
----------------------
A .npy file saved with allow_pickle=True containing a nested dict:
    {subject: {action: np.ndarray (T, J, 3)}}

where
  subject  e.g. 'S9', 'S11'
  action   e.g. 'Walking 1', 'Directions'
  T        number of frames
  J        17 (already H36M 17-joint) or 32 (full H36M, joints auto-selected)
  3        x, y, z in the same coordinate system as H36M GT (metres, Y-up)

The poses are automatically:
  • subset to 17 joints if J == 32 (H36M_17_JOINTS selection)
  • root-centred (pelvis joint set to origin)

Saving poses from your model
-----------------------------
    import numpy as np
    poses = {}          # {subject: {action: (T, 17, 3)}}
    poses['S9'] = {'Walking 1': my_model_output_S9_walking, ...}
    np.save('my_model_poses.npy', poses)

Usage
-----
  python3.10 run_bio_eval.py \\
      --poses         my_model_poses.npy \\
      --processed-dir /path/to/processed_all/processed_all \\
      --norm-stats    checkpoint/bio_gt/norm_stats.json \\
      --checkpoint    checkpoint/bio_gt/bio_gt_best.pth \\
      --subjects      S9 S11

Optional flags
--------------
  --bio-win   sliding window size (default 27, must match training)
  --stride    window stride (default 1)
  --batch-size  (default 64)
  --out-json  where to save full results (default: bio_eval_results.json)
  --gpu       CUDA device index (default 0; use -1 for CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_module.model   import BioModule, CRITERIA_DIMS
from bio_module.dataset import BioDataset, collate_fn, H36M_17_JOINTS
from bio_module.loss    import BioLoss, evaluate, BINARY_CRITERIA


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate BioModule with any 3D pose estimator output.')
    p.add_argument('--poses',         required=True,
                   help='.npy file: {subject: {action: (T, J, 3)}}')
    p.add_argument('--processed-dir', required=True,
                   help='Path to processed_all/processed_all')
    p.add_argument('--norm-stats',    required=True,
                   help='JSON file with crit_mean / crit_std (from save_norm_stats.py)')
    p.add_argument('--checkpoint',    required=True,
                   help='BioModule weights (.pth)')
    p.add_argument('--subjects',      nargs='+', default=['S9', 'S11'],
                   help='Subjects to evaluate (default: S9 S11)')
    p.add_argument('--bio-win',       type=int,   default=27)
    p.add_argument('--stride',        type=int,   default=1)
    p.add_argument('--batch-size',    type=int,   default=64)
    p.add_argument('--num-workers',   type=int,   default=4)
    p.add_argument('--gpu',           type=str,   default='0')
    p.add_argument('--out-json',      default='bio_eval_results.json')
    p.add_argument('--d-model',       type=int,   default=256)
    p.add_argument('--nhead',         type=int,   default=8)
    p.add_argument('--nlayers',       type=int,   default=4)
    p.add_argument('--dropout',       type=float, default=0.1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pose loading helpers
# ---------------------------------------------------------------------------

def load_poses(path: str, subjects: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """
    Load poses from .npy file and return {(subj, action): (T, 17, 3)}.
    Handles both 17-joint and 32-joint inputs; always root-centres.
    """
    raw = np.load(path, allow_pickle=True).item()
    out: dict[tuple[str, str], np.ndarray] = {}

    for subj in subjects:
        if subj not in raw:
            print(f"  WARNING: subject {subj} not found in poses file, skipping.")
            continue
        for action, arr in raw[subj].items():
            arr = arr.astype(np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                print(f"  WARNING: {subj}/{action} has unexpected shape {arr.shape}, skipping.")
                continue
            J = arr.shape[1]
            if J == 32:
                arr = arr[:, H36M_17_JOINTS, :]   # select 17 joints
            elif J != 17:
                print(f"  WARNING: {subj}/{action} has {J} joints (expected 17 or 32), skipping.")
                continue
            arr -= arr[:, :1, :]   # root-centre (pelvis = 0)
            out[(subj, action)] = arr
            print(f"  Loaded: {subj} | {action}  →  {arr.shape}")

    return out


# ---------------------------------------------------------------------------
# Results table printer (same format as train.py)
# ---------------------------------------------------------------------------

def print_results_table(results: dict, criteria_names: list[str], ranges: dict | None = None):
    subjects = list(results.keys())
    col_w    = 14
    lbl_w    = 55

    subj_hdr = ' ' * lbl_w
    for s in subjects:
        subj_hdr += f"  {s:^{col_w * 3 + 2}}"
    print('\n' + subj_hdr)

    col_hdr = ' ' * lbl_w
    for _ in subjects:
        col_hdr += f"  {'MSE':>{col_w}}{'MAE':>{col_w}}{'RMSE':>{col_w}}"
    print(col_hdr)
    print('-' * (lbl_w + len(subjects) * (col_w * 3 + 2) + 4))

    for cname in criteria_names:
        is_bin = cname in BINARY_CRITERIA
        tag    = ' *' if is_bin else ''
        row    = f"{(cname + tag):<{lbl_w}}"
        for s in subjects:
            res  = results[s]
            if is_bin:
                mse  = res.get(f'{cname}_mse',       float('nan'))
                mae  = res.get(f'{cname}_mae',        float('nan'))
                rmse = res.get(f'{cname}_rmse',       float('nan'))
            else:
                mse  = res.get(f'{cname}_mse_orig',  float('nan'))
                mae  = res.get(f'{cname}_mae_orig',  float('nan'))
                rmse = res.get(f'{cname}_rmse_orig', float('nan'))
            row += f"  {mse:>{col_w}.6f}{mae:>{col_w}.6f}{rmse:>{col_w}.6f}"
        print(row)

    print('-' * (lbl_w + len(subjects) * (col_w * 3 + 2) + 4))
    loss_row = f"{'Total loss':<{lbl_w}}"
    for s in subjects:
        loss_row += f"  {results[s]['loss']:>{col_w}.6f}{'':>{col_w}}{'':>{col_w}}"
    print(loss_row)
    print('  * binary criteria: metrics in sigmoid probability space [0,1]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── load norm stats ───────────────────────────────────────────────
    with open(args.norm_stats) as f:
        norm = json.load(f)
    criteria_names: list[str] = norm['criteria']
    crit_mean: dict[str, float] = norm['crit_mean']
    crit_std:  dict[str, float] = norm['crit_std']
    print(f"Loaded norm stats for {len(criteria_names)} criteria from {args.norm_stats}")

    # ── load poses ────────────────────────────────────────────────────
    print(f"\nLoading poses from {args.poses} …")
    precomputed_poses = load_poses(args.poses, args.subjects)
    if not precomputed_poses:
        print("ERROR: no poses loaded. Check --poses file and --subjects.")
        sys.exit(1)
    print(f"  {len(precomputed_poses)} clips loaded.\n")

    # ── build dataset ─────────────────────────────────────────────────
    dataset = BioDataset(
        processed_dir     = args.processed_dir,
        bio_win           = args.bio_win,
        stride            = args.stride,
        subjects          = args.subjects,
        precomputed_poses = precomputed_poses,
    )
    # inject training norm stats so evaluation is in original units
    dataset.crit_mean = crit_mean
    dataset.crit_std  = crit_std

    # ── load model ────────────────────────────────────────────────────
    model = BioModule(
        win=args.bio_win, d_model=args.d_model,
        nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Checkpoint loaded from {args.checkpoint}\n")

    loss_fn = BioLoss()

    # ── evaluate per subject + combined ──────────────────────────────
    results = {}
    for s in args.subjects:
        s_idx = dataset.subject_window_indices(s)
        if not s_idx:
            print(f"  WARNING: no windows for subject {s}, skipping.")
            continue
        loader = DataLoader(
            Subset(dataset, s_idx),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        results[s] = evaluate(model, loader, loss_fn, device, criteria_names,
                               crit_std=crit_std)
        print(f"  {s} done  (loss={results[s]['loss']:.4f})")

    if len(args.subjects) > 1:
        combined_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        label = '+'.join(args.subjects)
        results[label] = evaluate(model, combined_loader, loss_fn, device,
                                   criteria_names, crit_std=crit_std)

    print_results_table(results, criteria_names)

    # ── save JSON ─────────────────────────────────────────────────────
    with open(args.out_json, 'w') as f:
        json.dump({s: {k: float(v) for k, v in r.items()} for s, r in results.items()},
                  f, indent=2)
    print(f"\nFull results saved to {args.out_json}")


if __name__ == '__main__':
    main()
