"""
Pose estimator wrappers for BioModule.

Each wrapper converts a model's forward pass into the standard callable
interface expected by precompute_poses() in bio_module.dataset:

    model_fn(x: Tensor[B, frames, 17, 2]) -> Tensor[B, 17, 3]

where
  - x      : batch of sliding windows of normalised 2D keypoints
  - output : centre-frame 3D poses, root-centred (pelvis = 0)
  - frames : the temporal window the model was trained on (81, 243, 351, …)

Built-in wrappers
-----------------
  make_mhformer_fn(model)             MHFormer  (frames=351)
  make_center_frame_fn(model, frames) any model that outputs (B, T, J, 3)
  make_single_frame_fn(model)         any model that outputs (B, J, 3) directly

Adding a new model
------------------
Write a function that returns a callable matching the signature above.
Example for a hypothetical PoseFormer (81 frames, outputs (B, J, 3)):

    from my_model import PoseFormer
    model = PoseFormer(...).to(device)
    model.load_state_dict(torch.load('poseformer.pth'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def poseformer_fn(x):               # x: (B, 81, 17, 2)
        out = model(x)                  # (B, 17, 3)
        out = out.clone()
        out[:, 0, :] = 0.0              # root-centre
        return out

Then pass it to precompute_poses():

    from bio_module.dataset import precompute_poses
    poses = precompute_poses(
        model_fn   = poseformer_fn,
        pos_2d     = raw_2d,
        subjects   = ['S9', 'S11'],
        device     = device,
        cache_dir  = '/tmp/bio_cache',
        frames     = 81,
        cache_tag  = 'poseformer',
    )
"""

from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# MHFormer  (frames = 351)
# ---------------------------------------------------------------------------

def make_mhformer_fn(model) -> callable:
    """
    Wrap a frozen MHFormer model.

    MHFormer takes  (B, 351, 17, 2)  and returns  (B, 351, 17, 3).
    The centre frame (index 175) is extracted and the root joint is zeroed.
    """
    model.eval()
    center = 351 // 2   # 175

    def fn(x: torch.Tensor) -> torch.Tensor:
        pred = model(x)             # (B, 351, 17, 3)
        out  = pred[:, center].clone()   # (B, 17, 3)
        out[:, 0, :] = 0.0          # root-centre
        return out

    return fn


# ---------------------------------------------------------------------------
# Generic: model outputs (B, T, J, 3)  — extract centre frame
# ---------------------------------------------------------------------------

def make_center_frame_fn(model, frames: int) -> callable:
    """
    Wrap any model that outputs a full temporal sequence (B, T, J, 3).

    The centre frame (index frames // 2) is extracted and the root is zeroed.

    Parameters
    ----------
    model  : frozen nn.Module  (B, frames, 17, 2) → (B, frames, 17, 3)
    frames : temporal window size (e.g. 243 for MotionBERT, 81 for PoseFormer)
    """
    model.eval()
    center = frames // 2

    def fn(x: torch.Tensor) -> torch.Tensor:
        pred = model(x)
        out  = pred[:, center].clone()   # (B, 17, 3)
        out[:, 0, :] = 0.0
        return out

    return fn


# ---------------------------------------------------------------------------
# TCPFormer  (frames = 243, default; also supports 81)
# ---------------------------------------------------------------------------

def make_tcpformer_fn(model, frames: int = 243,
                      res_w: int = 1000, res_h: int = 1002) -> callable:
    """
    Wrap a frozen TCPFormer (MemoryInducedTransformer) model.

    TCPFormer takes  (B, T, J, 3)  where the 3rd channel is a confidence
    score.  We fill confidence = 1.0 since CPN 2D keypoints do not include
    scores.  The 2D x/y channels use the same normalisation as MHFormer
    (normalize_screen_coordinates: x = x/w*2-1, y = y/w*2-h/w).

    TCPFormer outputs  (B, T, J, 3)  in normalised image coordinate space.
    This wrapper:
      1. Appends the confidence channel to the input.
      2. Runs the model.
      3. Extracts the centre frame.
      4. Denormalises to pixel/mm scale using camera resolution.
      5. Root-centres the output (pelvis = 0).

    Parameters
    ----------
    model  : frozen MemoryInducedTransformer
    frames : temporal window (243 for H36M-243, 81 for H36M-81)
    res_w, res_h : camera image resolution (default: cam0 → 1000×1002)
    """
    model.eval()
    center = frames // 2

    def fn(x: torch.Tensor) -> torch.Tensor:
        # x: (B, frames, 17, 2)  normalised 2D keypoints
        B = x.shape[0]
        conf = torch.ones(B, frames, 17, 1, dtype=x.dtype, device=x.device)
        inp  = torch.cat([x, conf], dim=-1)          # (B, frames, 17, 3)

        pred = model(inp)                             # (B, frames, 17, 3)
        out  = pred[:, center].clone()                # (B, 17, 3)

        # denormalise to pixel/mm scale
        out[:, :, :2] = (out[:, :, :2]
                         + torch.tensor([1.0, res_h / res_w],
                                        device=out.device)) * (res_w / 2.0)
        out[:, :, 2:] = out[:, :, 2:] * (res_w / 2.0)

        # root-centre
        out = out - out[:, :1, :]
        return out

    return fn


# ---------------------------------------------------------------------------
# Generic: model outputs a single frame (B, J, 3) directly
# ---------------------------------------------------------------------------

def make_single_frame_fn(model) -> callable:
    """
    Wrap any model that already returns only the centre frame  (B, J, 3).

    The root joint is zeroed; no frame indexing is applied.

    Parameters
    ----------
    model : frozen nn.Module  (B, frames, 17, 2) → (B, 17, 3)
    """
    model.eval()

    def fn(x: torch.Tensor) -> torch.Tensor:
        out = model(x).clone()      # (B, 17, 3)
        out[:, 0, :] = 0.0
        return out

    return fn
