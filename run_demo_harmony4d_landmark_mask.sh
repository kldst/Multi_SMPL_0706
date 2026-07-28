#!/usr/bin/env bash
# Inference the raw-Mamma harmony4d_train_1_NC_200_00_contact scene with the
# DPT-mask / no-avg-scale checkpoint, using the per-person-mask viewer
# (demo_gradio_landmark_mask.py). This scene ships GT (.data.pyd cameras+SMPL,
# .mask.jpg per-person masks), so the demo shows predicted vs GT SMPL / mask /
# landmark + IoU metrics + a 3D .glb.
#
# scale_by_extrinsics is read from mamma_mask_dpt.yaml (False), so the predicted
# mesh is placed in the metric cam0 gauge (avg_scale == 1.0) that matches training.
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
  echo "[run_demo] conda.sh not found. Set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate mamma

export PYTHONPATH="$REPO_DIR:$REPO_DIR/training:$REPO_DIR/debug:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-mamma_mask_dpt}"
CHECKPOINT="${CHECKPOINT:-$REPO_DIR/model/checkpoint_47.pt}"
# A raw-Mamma scene root; the dataset walks it and picks the first sequence.
# Point SCENE at a single sequence dir to pin one (e.g. .../be_HsuS3iLSSWWZ_seq_000000).
SCENE="${SCENE:-$REPO_DIR/mamma/mamma/harmony4d_train_1_NC_200_00_contact}"
NUM_VIEWS="${NUM_VIEWS:-4}"       # matches img_nums:[4,4]
MAX_PEOPLE="${MAX_PEOPLE:-6}"     # matches dataset max_num_people:6
PORT="${PORT:-7869}"

echo "[run_demo] config=$CONFIG"
echo "[run_demo] checkpoint=$CHECKPOINT"
echo "[run_demo] scene=$SCENE views=$NUM_VIEWS people=$MAX_PEOPLE port=$PORT"

python demo_gradio_landmark_mask.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --scene "$SCENE" \
    --num-views "$NUM_VIEWS" \
    --max-people "$MAX_PEOPLE" \
    --device cuda:0 \
    --port "$PORT" \
    "$@"
