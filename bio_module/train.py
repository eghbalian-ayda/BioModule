"""
BioModule training script.

  Mode A – normalized directory (recommended):
    Training : GT 3D keypoints + pre-normalized criteria from normalized/
    Testing  : same source, S9 + S11

  Mode B – legacy (GT poses + raw processed_all criteria + frozen MHFormer test):
    Training : GT 3D keypoints (positions_3d.npy), subjects S1 S5 S6 S7 S8
    Testing  : Frozen MHFormer 3D estimates from CPN 2D keypoints, S9 + S11

Binary criteria (touch, seat) use BCE-with-logits loss.
All others use MSE on pre-normalized (Mode A) or z-score-normalized (Mode B) targets.

Usage – Mode A
--------------
  python -m bio_module.train \\
      --normalized-dir /path/to/processed_all/processed_all/normalized \\
      [options]

Usage – Mode B (legacy)
------------------------
  python -m bio_module.train \\
      --pose-3d        /path/to/positions_3d.npy \\
      --pose-2d        /path/to/data_2d_h36m_cpn_ft_h36m_dbb.npz \\
      --processed-dir  /path/to/processed_all/processed_all \\
      --model-weights  /path/to/model_dir_or_pth \\
      [options]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_module.model   import BioModule, CRITERIA_DIMS
from bio_module.dataset import (BioDataset, collate_fn,
                                SUBJECTS_TRAIN, SUBJECTS_TEST,
                                precompute_poses)
from bio_module.loss    import BioLoss, evaluate, BINARY_CRITERIA


# ---------------------------------------------------------------------------
# Frozen MHFormer loader (legacy mode)
# ---------------------------------------------------------------------------

def _make_mhformer_opts(frames: int = 351):
    import types
    return types.SimpleNamespace(
        layers=3, channel=512, d_hid=1024, frames=frames,
        n_joints=17, out_joints=17, in_channels=2, out_channels=3,
        out_all=1, crop_uv=0, pad=(frames - 1) // 2,
    )


def load_frozen_mhformer(weights_path: str, device: torch.device) -> nn.Module:
    from model.mhformer import Model
    opt   = _make_mhformer_opts()
    model = Model(opt).to(device)

    p = weights_path
    if os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.pth')))
        p = next((x for x in paths if Path(x).name.startswith('model')), paths[0])

    ckpt       = torch.load(p, map_location=device)
    model_dict = model.state_dict()
    state      = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(state)
    model.load_state_dict(model_dict)
    print(f"Loaded MHFormer weights from {p}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("MHFormer frozen.\n")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # ── Mode A: normalized directory ──────────────────────────────────────
    p.add_argument('--normalized-dir',
                   help='Pre-normalized directory (Mode A). '
                        'Each <Subject>/<Action>/ contains {S}_{Act}.npy and criterion_*.npy')

    # ── Mode B: legacy paths ──────────────────────────────────────────────
    p.add_argument('--pose-3d',
                   help='positions_3d.npy  (Mode B – GT poses for training)')
    p.add_argument('--pose-2d',
                   help='data_2d_h36m_cpn_ft_h36m_dbb.npz  (Mode B – MHFormer test)')
    p.add_argument('--processed-dir',
                   help='processed_all/processed_all  (Mode B – raw criteria)')
    p.add_argument('--model-weights',
                   help='MHFormer weights  (Mode B – model_4294.pth or folder)')
    p.add_argument('--cache-dir',      default='/tmp/mhformer_bio_cache')

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument('--checkpoint-dir', default='checkpoint/bio')

    # ── BioModule architecture ────────────────────────────────────────────
    p.add_argument('--bio-win',    type=int,   default=27)
    p.add_argument('--stride',     type=int,   default=1)
    p.add_argument('--d-model',    type=int,   default=256)
    p.add_argument('--nhead',      type=int,   default=8)
    p.add_argument('--nlayers',    type=int,   default=4)
    p.add_argument('--dropout',    type=float, default=0.1)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch-size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-ratio',    type=float, default=0.1)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--gpu',          type=str,   default='0')
    p.add_argument('--mhf-batch',    type=int,   default=256)
    p.add_argument('--num-workers',  type=int,   default=4)
    return p.parse_args()


def _validate_args(args):
    mode_a = args.normalized_dir is not None
    mode_b = args.pose_3d is not None or args.processed_dir is not None

    if mode_a and mode_b:
        raise ValueError(
            "--normalized-dir is mutually exclusive with "
            "--pose-3d / --processed-dir / --model-weights"
        )
    if not mode_a and not mode_b:
        raise ValueError(
            "Provide --normalized-dir (Mode A) OR "
            "--pose-3d + --processed-dir [+ --model-weights] (Mode B)"
        )
    if mode_b:
        if args.pose_3d is None:
            raise ValueError("Mode B requires --pose-3d")
        if args.processed_dir is None:
            raise ValueError("Mode B requires --processed-dir")

    return 'A' if mode_a else 'B'


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for batch in loader:
        poses   = batch['poses_3d'].to(device)
        preds   = model(poses)
        targets = {k: v.to(device) for k, v in batch.get('criteria', {}).items()}
        losses  = loss_fn(preds, targets)

        optimizer.zero_grad()
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            sums[k] += v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict], criteria_names: list[str]):
    subjects = list(results.keys())
    col_w    = 14

    print('\n' + '=' * (55 + len(subjects) * (col_w * 3 + 2) + 4))
    subj_hdr = ' ' * 55
    for s in subjects:
        subj_hdr += f"  {s:^{col_w * 3 + 2}}"
    print(subj_hdr)
    col_hdr = ' ' * 55
    for _ in subjects:
        col_hdr += f"  {'MSE':>{col_w}}{'MAE':>{col_w}}{'RMSE':>{col_w}}"
    print(col_hdr)
    print('-' * (55 + len(subjects) * (col_w * 3 + 2) + 4))

    for cname in criteria_names:
        is_bin = cname in BINARY_CRITERIA
        row    = f"{cname + (' *' if is_bin else ''):<55}"
        for s in subjects:
            res = results[s]
            if is_bin:
                mse  = res.get(f'{cname}_mse',  float('nan'))
                mae  = res.get(f'{cname}_mae',  float('nan'))
                rmse = res.get(f'{cname}_rmse', float('nan'))
            else:
                mse  = res.get(f'{cname}_mse_orig',  float('nan'))
                mae  = res.get(f'{cname}_mae_orig',  float('nan'))
                rmse = res.get(f'{cname}_rmse_orig', float('nan'))
            row += f"  {mse:>{col_w}.6f}{mae:>{col_w}.6f}{rmse:>{col_w}.6f}"
        print(row)

    print('-' * (55 + len(subjects) * (col_w * 3 + 2) + 4))
    loss_row = f"{'Total loss':<55}"
    for s in subjects:
        loss_row += f"  {results[s]['loss']:>{col_w}.6f}{'':>{col_w}}{'':>{col_w}}"
    print(loss_row)
    print('  * binary criteria: metrics in probability space [0,1]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    mode  = _validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Mode: {mode}\n")

    # =========================================================================
    # Mode A: normalized directory
    # =========================================================================
    if mode == 'A':
        norm_dir = args.normalized_dir

        # ── training dataset ─────────────────────────────────────────────
        print("Building training dataset (Mode A – normalized dir) …")
        train_dataset = BioDataset(
            normalized_dir = norm_dir,
            bio_win        = args.bio_win,
            stride         = args.stride,
            subjects       = SUBJECTS_TRAIN,
        )

        criteria_names = train_dataset.criteria_names
        print(f"\nCriteria ({len(criteria_names)}):")
        for c in criteria_names:
            tag = ' [BCE]' if c in BINARY_CRITERIA else ''
            print(f"  {c}  →  dim {CRITERIA_DIMS.get(c, '?')}{tag}")

        # ── train / val split ─────────────────────────────────────────────
        all_idx = np.arange(len(train_dataset))
        rng     = np.random.RandomState(args.seed)
        rng.shuffle(all_idx)
        n_val     = max(1, int(len(all_idx) * args.val_ratio))
        val_idx   = all_idx[:n_val]
        train_idx = all_idx[n_val:]

        print(f"\nSplit:")
        print(f"  Train : {SUBJECTS_TRAIN}  ({len(train_idx)} windows)")
        print(f"  Val   : {int(args.val_ratio*100)}% of train  ({len(val_idx)} windows)")
        print(f"  Test  : {SUBJECTS_TEST}  (GT poses from normalized dir)")

        train_loader = DataLoader(
            Subset(train_dataset, train_idx.tolist()),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_idx.tolist()),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )

        # ── model ─────────────────────────────────────────────────────────
        model = BioModule(
            win=args.bio_win, d_model=args.d_model,
            nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loss_fn   = BioLoss()

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ck_path    = os.path.join(args.checkpoint_dir, 'bio_best.pth')
        best_val   = float('inf')
        best_state = None

        # ── training loop ─────────────────────────────────────────────────
        print(f"\nTraining for {args.epochs} epochs …\n")
        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            vl = evaluate(model, val_loader, loss_fn, device, criteria_names,
                          crit_std=train_dataset.crit_std)
            scheduler.step(vl['loss'])

            flag = ''
            if vl['loss'] < best_val:
                best_val   = vl['loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ck_path)
                flag = ' *'

            print(f"  Ep {epoch:3d}/{args.epochs}  "
                  f"train={tr['total']:.4f}  val={vl['loss']:.4f}{flag}")

        print(f"\nBest val loss: {best_val:.4f}  →  {ck_path}")
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── test dataset (S9 + S11, GT poses from normalized dir) ─────────
        print("\n" + "=" * 72)
        print("  EVALUATING ON TEST SET (S9 + S11, GT poses from normalized dir)")
        print("=" * 72 + "\n")

        test_dataset = BioDataset(
            normalized_dir = norm_dir,
            bio_win        = args.bio_win,
            stride         = args.stride,
            subjects       = SUBJECTS_TEST,
        )
        # Share normalisation stats (identity in Mode A, but kept for evaluate())
        test_dataset.crit_mean = train_dataset.crit_mean
        test_dataset.crit_std  = train_dataset.crit_std

        results = {}
        for s in SUBJECTS_TEST:
            s_idx  = test_dataset.subject_window_indices(s)
            loader = DataLoader(
                Subset(test_dataset, s_idx),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_fn,
            )
            results[s] = evaluate(model, loader, loss_fn, device, criteria_names,
                                  crit_std=train_dataset.crit_std)

        combined_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        results['S9+S11'] = evaluate(model, combined_loader, loss_fn, device,
                                     criteria_names, crit_std=train_dataset.crit_std)

        print_results_table(results, criteria_names)

        import json
        out_json = os.path.join(args.checkpoint_dir, 'test_results.json')
        json.dump({s: {k: float(v) for k, v in r.items()} for s, r in results.items()},
                  open(out_json, 'w'), indent=2)
        print(f"\nFull results saved to {out_json}")

        norm_stats = {
            'criteria':  criteria_names,
            'crit_mean': train_dataset.crit_mean,
            'crit_std':  train_dataset.crit_std,
        }
        norm_json = os.path.join(args.checkpoint_dir, 'norm_stats.json')
        json.dump(norm_stats, open(norm_json, 'w'), indent=2)
        print(f"Norm stats saved to {norm_json}")

    # =========================================================================
    # Mode B: legacy (GT + raw processed_all + frozen MHFormer test)
    # =========================================================================
    else:
        from bio_module.pose_estimator import make_mhformer_fn

        # ── training dataset (GT 3D, S1–S8) ──────────────────────────────
        train_dataset = BioDataset(
            processed_dir = args.processed_dir,
            bio_win       = args.bio_win,
            stride        = args.stride,
            subjects      = SUBJECTS_TRAIN,
            pose_3d_path  = args.pose_3d,
        )

        criteria_names = train_dataset.criteria_names
        print(f"\nCriteria ({len(criteria_names)}):")
        for c in criteria_names:
            tag = ' [BCE]' if c in BINARY_CRITERIA else ''
            print(f"  {c}  →  dim {CRITERIA_DIMS.get(c, '?')}{tag}")

        all_idx = np.arange(len(train_dataset))
        rng     = np.random.RandomState(args.seed)
        rng.shuffle(all_idx)
        n_val     = max(1, int(len(all_idx) * args.val_ratio))
        val_idx   = all_idx[:n_val]
        train_idx = all_idx[n_val:]

        print(f"\nSplit:")
        print(f"  Train : {SUBJECTS_TRAIN}  ({len(train_idx)} windows)")
        print(f"  Val   : {int(args.val_ratio*100)}% of train  ({len(val_idx)} windows)")
        print(f"  Test  : {SUBJECTS_TEST}  (MHFormer poses, evaluated after training)")

        train_loader = DataLoader(
            Subset(train_dataset, train_idx.tolist()),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_idx.tolist()),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )

        model = BioModule(
            win=args.bio_win, d_model=args.d_model,
            nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loss_fn   = BioLoss()

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ck_path    = os.path.join(args.checkpoint_dir, 'bio_gt_best.pth')
        best_val   = float('inf')
        best_state = None

        print(f"\nTraining for {args.epochs} epochs …\n")
        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            vl = evaluate(model, val_loader, loss_fn, device, criteria_names,
                          crit_std=train_dataset.crit_std)
            scheduler.step(vl['loss'])

            flag = ''
            if vl['loss'] < best_val:
                best_val   = vl['loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ck_path)
                flag = ' *'

            print(f"  Ep {epoch:3d}/{args.epochs}  "
                  f"train={tr['total']:.4f}  val={vl['loss']:.4f}{flag}")

        print(f"\nBest val loss: {best_val:.4f}  →  {ck_path}")
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── build test dataset using frozen MHFormer ──────────────────────
        print("\n" + "=" * 72)
        print("  BUILDING TEST DATASET (MHFormer poses for S9 + S11)")
        print("=" * 72 + "\n")

        mhformer = load_frozen_mhformer(args.model_weights, device)
        raw_2d   = np.load(args.pose_2d, allow_pickle=True)['positions_2d'].item()
        mhf_poses = precompute_poses(
            model_fn   = make_mhformer_fn(mhformer),
            pos_2d     = raw_2d,
            subjects   = SUBJECTS_TEST,
            device     = device,
            cache_dir  = args.cache_dir,
            frames     = 351,
            batch_size = args.mhf_batch,
            camera_idx = 0,
            cache_tag  = 'mhf',
        )
        del mhformer
        torch.cuda.empty_cache()

        test_dataset = BioDataset(
            processed_dir     = args.processed_dir,
            bio_win           = args.bio_win,
            stride            = args.stride,
            subjects          = SUBJECTS_TEST,
            precomputed_poses = mhf_poses,
        )
        test_dataset.crit_mean = train_dataset.crit_mean
        test_dataset.crit_std  = train_dataset.crit_std

        results = {}
        for s in SUBJECTS_TEST:
            s_idx  = test_dataset.subject_window_indices(s)
            loader = DataLoader(
                Subset(test_dataset, s_idx),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_fn,
            )
            results[s] = evaluate(model, loader, loss_fn, device, criteria_names,
                                  crit_std=train_dataset.crit_std)

        combined_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )
        results['S9+S11'] = evaluate(model, combined_loader, loss_fn, device,
                                     criteria_names, crit_std=train_dataset.crit_std)

        print_results_table(results, criteria_names)

        import json
        out_json = os.path.join(args.checkpoint_dir, 'test_results.json')
        json.dump({s: {k: float(v) for k, v in r.items()} for s, r in results.items()},
                  open(out_json, 'w'), indent=2)
        print(f"\nFull results saved to {out_json}")

        norm_stats = {
            'criteria':  criteria_names,
            'crit_mean': train_dataset.crit_mean,
            'crit_std':  train_dataset.crit_std,
        }
        norm_json = os.path.join(args.checkpoint_dir, 'norm_stats.json')
        json.dump(norm_stats, open(norm_json, 'w'), indent=2)
        print(f"Norm stats saved to {norm_json}")


if __name__ == '__main__':
    main()
