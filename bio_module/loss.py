"""
Multi-task loss over all biomechanical criteria.
  - Continuous criteria : MSE loss (z-score normalised targets)
  - Binary criteria     : BCE with logits (raw 0/1 targets, no normalisation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict

# Criteria treated as binary classification (BCE with logits)
BINARY_CRITERIA = {'touch', 'seat'}


class BioLoss(nn.Module):
    """
    Mixed MSE / BCE loss.

    Parameters
    ----------
    weights : optional per-criterion weight dict.  Missing keys default to 1.0.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__()
        self.weights  = weights or {}
        self.mse      = nn.MSELoss()
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds:   dict[str, torch.Tensor],   # {name: (B, W, dim)}
        targets: dict[str, torch.Tensor],   # {name: (B, W, dim)}
    ) -> dict[str, torch.Tensor]:
        losses:  dict[str, torch.Tensor] = {}
        total    = torch.tensor(0.0, device=next(iter(preds.values())).device)
        n_active = 0

        for name, target in targets.items():
            if name not in preds:
                continue
            pred = preds[name]
            d    = min(pred.shape[-1], target.shape[-1])
            p, t = pred[..., :d], target[..., :d]

            if name in BINARY_CRITERIA:
                l = self.bce(p, t)
            else:
                l = self.mse(p, t)

            losses[name] = l
            total        = total + self.weights.get(name, 1.0) * l
            n_active    += 1

        losses['total'] = total / max(n_active, 1)
        return losses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn:        BioLoss,
    device:         torch.device,
    criteria_names: list[str],
    crit_std:       dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Evaluate BioModule on a DataLoader.

    Returns dict with per-criterion:
      '{name}_mse'       – MSE in normalised space (or binary prob space)
      '{name}_mae'       – MAE  (same space)
      '{name}_rmse'      – RMSE (same space)
      '{name}_mse_orig'  – MSE  in original units (continuous only)
      '{name}_mae_orig'  – MAE  in original units (continuous only)
      '{name}_rmse_orig' – RMSE in original units (continuous only)
    plus 'loss' (total).
    """
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    n = 0

    for batch in loader:
        poses   = batch['poses_3d'].to(device)
        preds   = model(poses)
        targets = {k: v.to(device) for k, v in batch.get('criteria', {}).items()}
        losses  = loss_fn(preds, targets)
        sums['loss'] += losses['total'].item()

        for cname in criteria_names:
            if cname not in preds or cname not in targets:
                continue
            p = preds[cname]
            t = targets[cname]
            d = min(p.shape[-1], t.shape[-1])
            p, t = p[..., :d], t[..., :d]

            if cname in BINARY_CRITERIA:
                p = torch.sigmoid(p)   # convert logits → probabilities

            err = p - t
            sums[f'{cname}_mse'] += (err ** 2).mean().item()
            sums[f'{cname}_mae'] += err.abs().mean().item()
        n += 1

    res = {k: v / max(n, 1) for k, v in sums.items()}

    # RMSE
    for cname in criteria_names:
        if f'{cname}_mse' in res:
            res[f'{cname}_rmse'] = res[f'{cname}_mse'] ** 0.5

    # Original-unit metrics for continuous criteria
    if crit_std:
        for cname in criteria_names:
            if cname in BINARY_CRITERIA:
                continue
            s = crit_std.get(cname, 1.0)
            if f'{cname}_mae' in res:
                res[f'{cname}_mae_orig']  = res[f'{cname}_mae']  * s
                res[f'{cname}_mse_orig']  = res[f'{cname}_mse']  * (s ** 2)
                res[f'{cname}_rmse_orig'] = res[f'{cname}_rmse'] * s

    return res
