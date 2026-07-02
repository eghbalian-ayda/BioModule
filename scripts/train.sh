#!/bin/bash
# Train BioModule on GT 3D keypoints (S1, S5, S6, S7, S8).
# Evaluates on S9 + S11 after training.
#
# Mode A (recommended) — pre-normalized directory:
#   ./scripts/train.sh --mode A --normalized-dir /path/to/normalized
#
# Mode B (legacy) — raw data + frozen pose estimator at test time:
#   ./scripts/train.sh --mode B \
#       --pose-3d       /path/to/positions_3d.npy \
#       --pose-2d       /path/to/data_2d_h36m_cpn_ft_h36m_dbb.npz \
#       --processed-dir /path/to/processed_all/processed_all \
#       --model-weights /path/to/mhformer_weights/

# ── User-configurable paths ────────────────────────────────────────────────
CHECKPOINT_DIR="checkpoint/bio_gt"
BIO_WIN=27
STRIDE=1
D_MODEL=256
NHEAD=8
NLAYERS=4
DROPOUT=0.1
EPOCHS=50
BATCH_SIZE=64
LR=3e-4
WEIGHT_DECAY=1e-4
VAL_RATIO=0.1
GPU=0
MHF_BATCH=256
NUM_WORKERS=4
# ──────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

# Parse --mode flag
MODE="A"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        *)      EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$MODE" == "A" ]]; then
    python -m bio_module.train \
        "${EXTRA_ARGS[@]}" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --bio-win        $BIO_WIN          \
        --stride         $STRIDE           \
        --d-model        $D_MODEL          \
        --nhead          $NHEAD            \
        --nlayers        $NLAYERS          \
        --dropout        $DROPOUT          \
        --epochs         $EPOCHS           \
        --batch-size     $BATCH_SIZE       \
        --lr             $LR               \
        --weight-decay   $WEIGHT_DECAY     \
        --val-ratio      $VAL_RATIO        \
        --gpu            $GPU              \
        --num-workers    $NUM_WORKERS
else
    python -m bio_module.train \
        "${EXTRA_ARGS[@]}" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --bio-win        $BIO_WIN          \
        --stride         $STRIDE           \
        --d-model        $D_MODEL          \
        --nhead          $NHEAD            \
        --nlayers        $NLAYERS          \
        --dropout        $DROPOUT          \
        --epochs         $EPOCHS           \
        --batch-size     $BATCH_SIZE       \
        --lr             $LR               \
        --weight-decay   $WEIGHT_DECAY     \
        --val-ratio      $VAL_RATIO        \
        --gpu            $GPU              \
        --mhf-batch      $MHF_BATCH        \
        --num-workers    $NUM_WORKERS
fi
