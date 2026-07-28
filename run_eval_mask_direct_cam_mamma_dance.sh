#!/usr/bin/env bash
# Standalone frame-by-frame evaluation of the direct_cam checkpoint on
# MAMMA_eval_dance/test. This does not import or launch the Gradio demo.
#
# Metrics:
#   - MPJPE: first 24 SMPL joints, pelvis/root aligned, millimetres
#   - PA-MPJPE: first 24 SMPL joints, similarity/Procrustes aligned, millimetres
#   - presence precision / recall / F1
#
# Override CONFIG, CHECKPOINT, DATASET_ROOT, DATASET_SPLIT, IMAGE_IDS,
# PRESENCE_THRESHOLD, MAX_FRAMES, OUTPUT, or CUDA_VISIBLE_DEVICES as needed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
  for candidate in \
    /train-data-3-hdd/yian/conda/etc/profile.d/conda.sh \
    /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh; do
    if [[ -f "$candidate" ]]; then
      CONDA_SH="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[run_eval] conda.sh not found. Set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate mamma

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/multi_smpl_matplotlib}"

CONFIG="${CONFIG:-mamma_mask_direct_cam}"
CHECKPOINT="${CHECKPOINT:-$REPO_DIR/model/direct_cam/checkpoint_17.pt}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/MAMMA_eval_dance}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
IMAGE_IDS="${IMAGE_IDS:-0 1 2 3}"
PRESENCE_THRESHOLD="${PRESENCE_THRESHOLD:-0.5}"
SELECTION_MODE="${SELECTION_MODE:-threshold}"
OUTPUT="${OUTPUT:-$REPO_DIR/eval/eval_results/direct_cam_checkpoint_17_mamma_dance_summary.json}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-1}"

cmd=(
  python eval_mamma_dance_mpjpe.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATASET_ROOT"
  --dataset-split "$DATASET_SPLIT"
  --image-ids "$IMAGE_IDS"
  --presence-threshold "$PRESENCE_THRESHOLD"
  --selection-mode "$SELECTION_MODE"
  --output "$OUTPUT"
  --device "$DEVICE"
  --log-every "$LOG_EVERY"
)
if [[ -n "${MAX_FRAMES:-}" ]]; then
  cmd+=(--max-frames "$MAX_FRAMES")
fi

echo "[run_eval] config=$CONFIG"
echo "[run_eval] checkpoint=$CHECKPOINT"
echo "[run_eval] dataset=$DATASET_ROOT/$DATASET_SPLIT"
echo "[run_eval] image_ids=$IMAGE_IDS selection_mode=$SELECTION_MODE presence_threshold=$PRESENCE_THRESHOLD"
echo "[run_eval] max_frames=${MAX_FRAMES:-all} device=$DEVICE output=$OUTPUT"

"${cmd[@]}"
