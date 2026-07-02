"""
BioDataset: feeds 3D keypoints → BioModule training/testing pipeline.

Two modes (provide exactly one):

  normalized_dir  – path to the pre-normalized directory
                    (<normalized_dir>/<Subject>/<Action>/)
                    Each action folder contains:
                      {Subject}_{Action}.npy   – 3D poses (T, 32, 3), GT
                      criterion_*.npy          – already-normalized criteria
                    Criteria are used as-is; no z-score is recomputed.

  pose_3d_path + processed_dir  – legacy mode: GT 3D keypoints from
                    positions_3d.npy and raw criteria from processed_all.

Binary criteria (touch, seat) are thresholded to {0,1} via > 0 and trained
with BCE-with-logits loss regardless of mode.
"""

from __future__ import annotations

import re
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from bio_module.loss import BINARY_CRITERIA

# ---------------------------------------------------------------------------
# H36M 17-joint subset (same joints MHFormer outputs)
# Removes {4,5,9,10,11,16,20,21,22,23,24,28,29,30,31} from 32-joint skeleton
# ---------------------------------------------------------------------------
H36M_17_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]


# ---------------------------------------------------------------------------
# Action-name normalisation & aliasing  (legacy mode helpers)
# ---------------------------------------------------------------------------

def _canon(name: str) -> str:
    return re.sub(r'[\s_\-]+', '', name.strip().lower())


_ALIASES: dict[str, list[str]] = {
    'photo':        ['takingphoto',  'photo'],
    'photo1':       ['takingphoto1', 'photo1'],
    'walkdog':      ['walkingdog',   'walkdog'],
    'walkdog1':     ['walkingdog1',  'walkdog1'],
    'sittingdown2': ['sittingdown2', 'sittingdown1'],
    'eating2':      ['eating2',      'eating1'],
}


def _build_proc_lookup(processed_dir: Path) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for sd in sorted(processed_dir.iterdir()):
        if not sd.is_dir() or not sd.name.startswith('S'):
            continue
        for ad in sorted(sd.iterdir()):
            if ad.is_dir():
                lookup[(sd.name, _canon(ad.name))] = ad.name
    return lookup


def _resolve_folder(subject, act, lookup):
    c = _canon(act)
    folder = lookup.get((subject, c))
    if folder:
        return folder
    for alias in _ALIASES.get(c, []):
        folder = lookup.get((subject, alias))
        if folder:
            return folder
    base     = re.sub(r'\d+$', '', c)
    suffix_n = int(re.search(r'\d+$', c).group()) if re.search(r'\d+$', c) else 0
    candidates = sorted(
        [cn for (s, cn) in lookup if s == subject and re.sub(r'\d+$', '', cn) == base],
        key=lambda cn: int(re.search(r'\d+$', cn).group()) if re.search(r'\d+$', cn) else 0
    )
    if candidates:
        return lookup.get((subject, candidates[min(suffix_n, len(candidates) - 1)]))
    return None


# ---------------------------------------------------------------------------
# MHFormer pre-computation (used at test time, legacy mode)
# ---------------------------------------------------------------------------

_CAM_RES = [(1000, 1002), (1000, 1000), (1000, 1000), (1000, 1002)]


def _normalize_2d(kps: np.ndarray, cam_idx: int) -> np.ndarray:
    _HERE = Path(__file__).resolve().parent.parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from common.camera import normalize_screen_coordinates
    w, h = _CAM_RES[cam_idx]
    out  = normalize_screen_coordinates(kps.copy().astype(np.float32), w=w, h=h)
    return out.astype(np.float32)


def _run_model_on_clip(
    model_fn,
    kps_2d:    np.ndarray,
    device,
    frames:    int,
    batch_size: int = 256,
) -> np.ndarray:
    T   = kps_2d.shape[0]
    pad = (frames - 1) // 2
    left   = kps_2d[:pad][::-1].copy()
    right  = kps_2d[-pad:][::-1].copy()
    padded = np.concatenate([left, kps_2d, right], axis=0)
    wins   = np.stack([padded[t:t + frames] for t in range(T)]).astype(np.float32)

    out_3d = []
    with torch.no_grad():
        for i in range(0, T, batch_size):
            chunk = torch.from_numpy(wins[i:i + batch_size]).to(device)
            pred  = model_fn(chunk)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            out_3d.append(pred)
    return np.concatenate(out_3d, axis=0)   # (T, 17, 3)


def precompute_poses(
    model_fn,
    pos_2d:     dict,
    subjects:   list[str],
    device,
    cache_dir:  str | Path,
    frames:     int = 351,
    batch_size: int = 256,
    camera_idx: int = 0,
    cache_tag:  str = 'pose',
) -> dict[tuple[str, str], np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[tuple[str, str], np.ndarray] = {}

    for subj in subjects:
        if subj not in pos_2d:
            continue
        for act, cam_list in pos_2d[subj].items():
            if camera_idx >= len(cam_list):
                continue
            tag        = f"{cache_tag}_{subj}_{_canon(act)}_{camera_idx}"
            cache_path = cache_dir / f"{tag}.npy"

            if cache_path.exists():
                result[(subj, act)] = np.load(str(cache_path))
                continue

            print(f"  [{cache_tag}] {subj} | {act} | cam{camera_idx}", flush=True)
            kps_norm = _normalize_2d(cam_list[camera_idx], camera_idx)
            poses_3d = _run_model_on_clip(model_fn, kps_norm, device, frames, batch_size)
            np.save(str(cache_path), poses_3d)
            result[(subj, act)] = poses_3d

    return result


def precompute_mhformer_poses(
    mhformer,
    pos_2d:    dict,
    subjects:  list[str],
    device,
    cache_dir: str | Path,
    batch_size: int = 256,
    camera_idx: int = 0,
) -> dict[tuple[str, str], np.ndarray]:
    """Backward-compatible wrapper: runs MHFormer (351 frames) via precompute_poses."""
    from bio_module.pose_estimator import make_mhformer_fn
    return precompute_poses(
        model_fn   = make_mhformer_fn(mhformer),
        pos_2d     = pos_2d,
        subjects   = subjects,
        device     = device,
        cache_dir  = cache_dir,
        frames     = 351,
        batch_size = batch_size,
        camera_idx = camera_idx,
        cache_tag  = 'mhf',
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SUBJECTS_ALL   = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
SUBJECTS_TRAIN = ['S1', 'S5', 'S6', 'S7', 'S8']
SUBJECTS_TEST  = ['S9', 'S11']


class BioDataset(Dataset):
    """
    Sliding-window dataset pairing 3D poses with biomechanical criteria.

    Provide exactly one of:
      normalized_dir              – pre-normalized directory (new mode)
      pose_3d_path + processed_dir – legacy GT + raw criteria mode
      precomputed_poses + processed_dir – legacy MHFormer test mode

    Binary criteria (touch, seat) are thresholded to {0,1} via > 0 and
    excluded from any z-score normalization.
    """

    def __init__(
        self,
        normalized_dir:   str | Path | None = None,
        processed_dir:    str | Path | None = None,
        bio_win:          int  = 27,
        stride:           int  = 1,
        subjects:         list[str] | None = None,
        pose_3d_path:     str | Path | None = None,
        precomputed_poses: dict | None      = None,
    ):
        # Validate mode
        using_normalized = normalized_dir is not None
        using_legacy_gt  = pose_3d_path is not None
        using_legacy_pre = precomputed_poses is not None

        if using_normalized:
            assert not using_legacy_gt and not using_legacy_pre, \
                "normalized_dir is mutually exclusive with pose_3d_path / precomputed_poses"
        else:
            assert using_legacy_gt != using_legacy_pre, \
                "In legacy mode provide exactly one of pose_3d_path or precomputed_poses"
            assert processed_dir is not None, \
                "Legacy mode requires processed_dir"

        self.bio_win = bio_win
        self.stride  = stride

        if subjects is None:
            subjects = SUBJECTS_ALL
        self.subjects = subjects

        # ── build clips & windows ────────────────────────────────────────
        self._clips:   list[dict] = []
        self._windows: list[tuple[int, int, int]] = []

        # Initialise normalisation dicts (will be populated below or left 0/1)
        self.crit_mean: dict[str, float] = {}
        self.crit_std:  dict[str, float] = {}

        if using_normalized:
            self._init_normalized(Path(normalized_dir), subjects, bio_win, stride)
        else:
            self._init_legacy(
                processed_dir    = Path(processed_dir),
                pose_3d_path     = pose_3d_path,
                precomputed_poses= precomputed_poses,
                subjects         = subjects,
                bio_win          = bio_win,
                stride           = stride,
            )

        print(
            f"BioDataset ready: {len(self._clips)} clips, "
            f"{len(self._windows)} windows, "
            f"{len(self.criteria_names)} criteria"
        )

    # ------------------------------------------------------------------
    # Normalized-directory mode
    # ------------------------------------------------------------------

    def _init_normalized(
        self,
        norm_dir: Path,
        subjects: list[str],
        bio_win:  int,
        stride:   int,
    ):
        """
        Load poses and pre-normalized criteria from the normalized directory.

        Structure: norm_dir/{Subject}/{Action}/
          {Subject}_{Action}.npy   – poses (T, 32, 3)
          criterion_*.npy          – already-normalized criteria
        """
        self._norm_dir   = norm_dir
        self._poses_3d: dict[tuple[str, str], np.ndarray] = {}
        _crit_set: set[str] | None = None

        for subj in subjects:
            subj_dir = norm_dir / subj
            if not subj_dir.is_dir():
                continue
            for act_dir in sorted(subj_dir.iterdir()):
                if not act_dir.is_dir():
                    continue
                action = act_dir.name
                pose_file = act_dir / f'{subj}_{action}.npy'
                if not pose_file.exists():
                    continue

                poses = np.load(str(pose_file)).astype(np.float32)  # (T, 32, 3)
                poses = poses[:, H36M_17_JOINTS, :]                  # (T, 17, 3)
                poses -= poses[:, :1, :]                              # root-centre

                crit_files = sorted(act_dir.glob('criterion_*.npy'))
                crit_here  = {f.stem.replace('criterion_', '') for f in crit_files}
                if _crit_set is None:
                    _crit_set = crit_here
                else:
                    _crit_set &= crit_here

                T = poses.shape[0]
                if T < bio_win:
                    continue

                self._poses_3d[(subj, action)] = poses
                clip_idx = len(self._clips)
                self._clips.append({
                    'subject':   subj,
                    'action':    action,
                    'act_dir':   act_dir,
                    'T':         T,
                    'pose_key':  (subj, action),
                    'mode':      'normalized',
                })
                for start in range(0, T - bio_win + 1, stride):
                    self._windows.append((clip_idx, start, start + bio_win))

        self.criteria_names: list[str] = sorted(_crit_set) if _crit_set else []

        # Criteria are pre-normalized; set identity stats for compatibility
        for cname in self.criteria_names:
            self.crit_mean[cname] = 0.0
            self.crit_std[cname]  = 1.0

        # Criterion cache
        self._crit_cache: dict[tuple[str, str, str], np.ndarray | None] = {}

    # ------------------------------------------------------------------
    # Legacy mode
    # ------------------------------------------------------------------

    def _init_legacy(
        self,
        processed_dir:     Path,
        pose_3d_path:      str | Path | None,
        precomputed_poses: dict | None,
        subjects:          list[str],
        bio_win:           int,
        stride:            int,
    ):
        self.proc_dir = processed_dir

        # ── load 3D poses ────────────────────────────────────────────────
        if pose_3d_path is not None:
            print("Loading GT 3D keypoints …")
            raw = np.load(str(pose_3d_path), allow_pickle=True).item()
            self._poses_3d: dict[tuple[str, str], np.ndarray] = {}
            for subj in subjects:
                for act, arr in raw.get(subj, {}).items():
                    poses = arr[:, H36M_17_JOINTS, :].astype(np.float32)
                    poses -= poses[:, :1, :]
                    self._poses_3d[(subj, act)] = poses
            print(f"  Loaded {len(self._poses_3d)} clips.\n")
        else:
            self._poses_3d = precomputed_poses

        # ── processed_all lookup ─────────────────────────────────────────
        self._proc_lookup = _build_proc_lookup(self.proc_dir)

        # ── discover criteria ────────────────────────────────────────────
        _crit_set: set[str] | None = None

        for subj in subjects:
            for (subj2, action), poses in sorted(self._poses_3d.items()):
                if subj2 != subj:
                    continue

                folder = _resolve_folder(subj, action, self._proc_lookup)
                if folder is None:
                    continue

                pose_file = self.proc_dir / subj / folder / f'{subj}_{folder}.npy'
                if not pose_file.exists():
                    continue

                T_crit = np.load(str(pose_file)).shape[0]
                T_pose = poses.shape[0]
                T      = min(T_crit, T_pose)

                if T < bio_win:
                    continue

                crit_files = sorted((self.proc_dir / subj / folder).glob('criterion_*.npy'))
                crit_here  = {f.stem.replace('criterion_', '') for f in crit_files}
                if _crit_set is None:
                    _crit_set = crit_here
                else:
                    _crit_set &= crit_here

                clip_idx = len(self._clips)
                self._clips.append({
                    'subject':   subj,
                    'action':    action,
                    'folder':    folder,
                    'T':         T,
                    'pose_key':  (subj, action),
                    'mode':      'legacy',
                })
                for start in range(0, T - bio_win + 1, stride):
                    self._windows.append((clip_idx, start, start + bio_win))

        self.criteria_names: list[str] = sorted(_crit_set) if _crit_set else []

        # ── z-score stats (continuous criteria only) ─────────────────────
        self._compute_norm_stats()

        # ── criterion cache ──────────────────────────────────────────────
        self._crit_cache: dict[tuple[str, str, str], np.ndarray | None] = {}

    # ------------------------------------------------------------------

    def _compute_norm_stats(self):
        """Compute z-score stats from raw criteria (legacy mode only)."""
        seen  = set()
        accum = {c: [] for c in self.criteria_names if c not in BINARY_CRITERIA}
        for clip in self._clips:
            k = (clip['subject'], clip['folder'])
            if k in seen:
                continue
            seen.add(k)
            for cname in accum:
                p = self.proc_dir / clip['subject'] / clip['folder'] / f'criterion_{cname}.npy'
                if p.exists():
                    arr = np.load(str(p)).astype(np.float32)
                    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    step = max(1, arr.shape[0] // 500)
                    accum[cname].append(arr[::step].reshape(-1))

        for cname in self.criteria_names:
            if cname in BINARY_CRITERIA:
                self.crit_mean[cname] = 0.0
                self.crit_std[cname]  = 1.0
            elif accum.get(cname):
                cat = np.concatenate(accum[cname])
                self.crit_mean[cname] = float(cat.mean())
                self.crit_std[cname]  = float(max(cat.std(), 1e-6))
            else:
                self.crit_mean[cname] = 0.0
                self.crit_std[cname]  = 1.0

    def _get_criterion(self, clip: dict, cname: str) -> np.ndarray | None:
        """Load and cache a criterion array for one clip."""
        if clip['mode'] == 'normalized':
            key = (clip['subject'], clip['action'], cname)
            if key not in self._crit_cache:
                p = clip['act_dir'] / f'criterion_{cname}.npy'
                if p.exists():
                    arr = np.load(str(p)).astype(np.float32)
                    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    self._crit_cache[key] = arr
                else:
                    self._crit_cache[key] = None
        else:
            key = (clip['subject'], clip['folder'], cname)
            if key not in self._crit_cache:
                p = self.proc_dir / clip['subject'] / clip['folder'] / f'criterion_{cname}.npy'
                if p.exists():
                    arr = np.load(str(p)).astype(np.float32)
                    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    self._crit_cache[key] = arr
                else:
                    self._crit_cache[key] = None
        return self._crit_cache[key]

    # ------------------------------------------------------------------

    def subject_window_indices(self, subject: str) -> list[int]:
        return [
            i for i, (ci, _, _) in enumerate(self._windows)
            if self._clips[ci]['subject'] == subject
        ]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, start, end = self._windows[idx]
        clip = self._clips[clip_idx]

        poses_3d = self._poses_3d[clip['pose_key']][start:end]

        criteria: dict[str, torch.Tensor] = {}
        for cname in self.criteria_names:
            arr = self._get_criterion(clip, cname)
            if arr is None or end > arr.shape[0]:
                continue
            chunk = arr[start:end]
            if chunk.ndim == 1:
                chunk = chunk[:, None]

            if cname in BINARY_CRITERIA:
                # Threshold to {0,1} — use > 0 to handle both raw force
                # values (e.g. 29–34 N) and probability-like values (0–1)
                chunk = (chunk > 0).astype(np.float32)
            elif clip['mode'] == 'legacy':
                # z-score normalise in legacy mode (criteria not pre-normalised)
                chunk = (chunk - self.crit_mean[cname]) / self.crit_std[cname]
            # normalized mode: criteria already normalised, use as-is

            criteria[cname] = torch.from_numpy(chunk)

        return {
            'poses_3d': torch.from_numpy(poses_3d),
            'criteria': criteria,
            'subject':  clip['subject'],
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    out = {
        'poses_3d': torch.stack([b['poses_3d'] for b in batch]),
        'subject':  [b['subject'] for b in batch],
    }
    if batch[0]['criteria']:
        out['criteria'] = {
            k: torch.stack([b['criteria'][k] for b in batch])
            for k in batch[0]['criteria']
        }
    return out
