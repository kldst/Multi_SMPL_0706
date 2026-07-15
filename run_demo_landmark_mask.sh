#!/usr/bin/env bash
# Launch the per-person-mask (+ optional dense-landmark) Gradio demo.
# Default now targets the DPT-mask run (mamma_mask_dpt, landmark head OFF) on the
# raw Mamma_mv_split data. The demo auto-detects the missing landmark head and
# shows mask-only (pred vs GT mask + mask IoU); with a landmark-on checkpoint it
# also shows landmarks. Mirrors run_demo_mamma_dance.sh (conda env `mamma`).
set -e

REPO="/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle"
cd "$REPO"

source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma

export PYTHONPATH="$REPO:$REPO/training:$REPO/debug:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-mamma_mask_dpt}"
CKPT="${CKPT:-$REPO/training/logs/mamma_mask_dpt/checkpoint_23.pt}"
# Raw Mamma_mv_split scene root (contains tmp/<batch>/<dataset>/png/<seq>/IOI_*).
SCENE="${SCENE:-/mnt/train-data-4-hdd/yian/Mamma_mv_split/train}"
NUM_VIEWS="${NUM_VIEWS:-4}"       # matches img_nums:[4,4]
MAX_PEOPLE="${MAX_PEOPLE:-6}"     # matches dataset max_num_people:6
PORT="${PORT:-7869}"

python demo_gradio_landmark_mask.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --scene "$SCENE" \
    --num-views "$NUM_VIEWS" \
    --max-people "$MAX_PEOPLE" \
    --device cuda:0 \
    --port "$PORT" \
    "$@"
