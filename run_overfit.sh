#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle
cd "$REPO/training"
source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma
# exp_name in mamma_overfit.yaml is mamma_overfit_newlandmark -> clear ITS ckpt dir
# so the trainer starts fresh (a stale/partial ckpt here would auto-resume).
rm -rf "$REPO/training/logs/mamma_overfit_newlandmark/ckpts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"   # pick a FREE gpu (demo is on 2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:$REPO/training"
echo "[run_overfit] start $(date) on GPU ${CUDA_VISIBLE_DEVICES}"
torchrun --standalone --nproc_per_node=3 launch.py --config mamma_overfit
echo "[run_overfit] exit=$? $(date)"
