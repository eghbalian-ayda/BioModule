#!/bin/bash
# Evaluate BioModule with any 3D pose estimator output.
#
# Usage:
#   ./scripts/eval.sh /path/to/my_model_poses.npy [S9 S11 ...]
#
# The poses file must be a .npy containing a nested dict:
#   {subject: {action: np.ndarray (T, J, 3)}}
# where J is 17 (H36M 17-joint) or 32 (full H36M, joints auto-selected).
# Poses are root-centred automatically.
#
# Defaults to evaluating S9 and S11 if no subjects are specified.

# ── User-configurable paths ────────────────────────────────────────────────
PROCESSED_DIR="/path/to/processed_all/processed_all"
CHECKPOINT="checkpoint/bio_gt/bio_gt_best.pth"
NORM_STATS="checkpoint/bio_gt/norm_stats.json"
OUT_JSON="results/eval_output.json"
BIO_WIN=27
STRIDE=1
BATCH_SIZE=64
NUM_WORKERS=4
GPU=0
# ──────────────────────────────────────────────────────────────────────────

POSES="${1:?Usage: $0 /path/to/poses.npy [S9 S11 ...]}"
shift
SUBJECTS="${@:-S9 S11}"

cd "$(dirname "$0")/.."

python eval.py \
    --poses         "$POSES"          \
    --processed-dir "$PROCESSED_DIR"  \
    --norm-stats    "$NORM_STATS"     \
    --checkpoint    "$CHECKPOINT"     \
    --subjects      $SUBJECTS         \
    --bio-win       $BIO_WIN          \
    --stride        $STRIDE           \
    --batch-size    $BATCH_SIZE       \
    --num-workers   $NUM_WORKERS      \
    --gpu           $GPU              \
    --out-json      "$OUT_JSON"
