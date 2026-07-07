#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle
cd "$REPO/training"
source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma
# exp_name in mamma_small.yaml is mamma_small -> clear ITS ckpt dir so the trainer
# starts fresh (a stale/partial ckpt here would auto-resume).
rm -rf "$REPO/training/logs/mamma_small/ckpts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # pick a FREE gpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vggt lives at the REPO root (not under training/); launch.py adds it to sys.path
# only AFTER `from trainer import Trainer`, so trainer's `import vggt` needs it here.
export PYTHONPATH="$REPO:$REPO/training"
echo "[run_small] start $(date) on GPU ${CUDA_VISIBLE_DEVICES}"
torchrun --standalone --nproc_per_node=1 launch.py --config mamma_small
echo "[run_small] exit=$? $(date)"
