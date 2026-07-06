#!/usr/bin/env bash
# Launch the dense-landmark + per-person-mask Gradio demo on the overfit checkpoint.
# Mirrors run_demo_mamma_dance.sh (conda env `mamma`), but for the new heads.
set -e

REPO="/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle"
cd "$REPO"

source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma

export PYTHONPATH="$REPO:$REPO/training:$REPO/debug:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # GPU 0 is busy with the overfit run

CKPT="${CKPT:-$REPO/training/logs/mamma_overfit_newlandmark/ckpts/checkpoint_600.pt}"
PORT="${PORT:-7869}"

python demo_gradio_landmark_mask.py \
    --config mamma_overfit \
    --checkpoint "$CKPT" \
    --num-views 4 \
    --max-people 2 \
    --device cuda:0 \
    --port "$PORT" \
    "$@"
