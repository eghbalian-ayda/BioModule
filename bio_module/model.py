"""
BioModule: biomechanical-criteria predictor that plugs on top of frozen MHFormer.

Input  : 3D poses produced by MHFormer  -- (B, W, 17, 3)
Output : dict {criterion_name: (B, W, dim)}  -- per-frame predictions

MHFormer is never imported, instantiated, or modified here.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Criterion output dimensions (one entry per criterion_*.npy file)
# ---------------------------------------------------------------------------
CRITERIA_DIMS: dict[str, int] = {
    'active_torque_angle_scaling_function':            68,
    'active_torque_angular_velocity_scaling_function': 68,
    'dimensionless_acceleration':                      34,
    'dimensionless_active_torque':                     34,
    'dimensionless_ground_reaction':                   12,
    'dimensionless_ideal_torque':                      34,
    'dimensionless_instantaneous_power':               34,
    'dimensionless_marker':                            60,
    'dimensionless_passive_torque':                    34,
    'dimensionless_seat_reaction':                      1,
    'dimensionless_speed':                             40,
    'normalized_active_torque':                        68,
    'seat':                                             1,
    'touch':                                            2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe   = torch.zeros(max_len, d_model)
        pos  = torch.arange(max_len).unsqueeze(1).float()
        div  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, d)
        return self.drop(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# BioModule
# ---------------------------------------------------------------------------

class BioModule(nn.Module):
    """
    Plug-in biomechanical predictor.  Receives frozen MHFormer 3D-pose
    estimates and regresses 14 biomechanical criteria per frame.

    Parameters
    ----------
    win          : temporal window size (frames fed in at once)
    criteria_dims: {short_name: output_dim} – defaults to CRITERIA_DIMS
    d_model      : transformer hidden size
    nhead        : attention heads
    nlayers      : transformer encoder layers
    dropout      : dropout rate
    """

    def __init__(
        self,
        win:           int,
        criteria_dims: dict[str, int] | None = None,
        d_model:       int   = 256,
        nhead:         int   = 8,
        nlayers:       int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()

        if criteria_dims is None:
            criteria_dims = CRITERIA_DIMS
        self.criteria_dims = criteria_dims
        self.win = win

        IN_DIM = 17 * 3   # 51 – flattened (joints × coords)

        # ── input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(IN_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = _PosEnc(d_model, max_len=win + 8, dropout=dropout)

        # ── temporal encoder ──────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayers,
            norm=nn.LayerNorm(d_model),
        )

        # ── per-criterion output heads ────────────────────────────────────
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, dim),
            )
            for name, dim in criteria_dims.items()
        })

    # ------------------------------------------------------------------

    def forward(self, poses_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        poses_3d : (B, W, 17, 3) – 3D poses from (frozen) MHFormer

        Returns
        -------
        dict {criterion_short_name: (B, W, dim)}
        """
        B, W, J, C = poses_3d.shape
        x = poses_3d.reshape(B, W, J * C)   # (B, W, 51)
        x = self.input_proj(x)               # (B, W, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)              # (B, W, d_model)

        return {name: head(x) for name, head in self.heads.items()}
