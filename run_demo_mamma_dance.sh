#!/usr/bin/env bash
# Launch the SMPL multi-person Gradio demo.
#
# Default mode targets the current MAMMA overfit run:
#   config:     training/config/mamma_overfit.yaml
#   checkpoint: training/logs/mamma_overfit_newlandmark/ckpts/checkpoint_300.pt
#   data:       the single raw MAMMA scene used by mamma_overfit.yaml
#
# To recover the old dance-eval defaults:
#   DEMO_MODE=dance ./run_demo_mamma_dance.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
# render_mesh_projection_cpu.py only needs numpy/opencv/torch (all already in this
# env), so point the projection worker back at this same interpreter instead of the
# original (missing) pytorch3d env.
export PYTORCH3D_PROJECTION_PYTHON="${PYTORCH3D_PROJECTION_PYTHON:-$(command -v python)}"

DEMO_MODE="${DEMO_MODE:-overfit}"

OVERFIT_SCENE_DEFAULT="/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/interactions_couple_close_1_C_200_00_contact/be_0GcH1mWtRfKu_seq_000050/tmp/bedlam_lab_20251031_191434"

if [[ "$DEMO_MODE" == "dance" ]]; then
  CONFIG="${CONFIG:-0621_mamma}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/training/logs/mamma_overfit_newlandmark/ckpts/checkpoint_600.pt}"
  DATASET_ROOT="${DATASET_ROOT:-/mnt/train-data-4-hdd/yian/MAMMA_eval_dance}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3 4 5 6 7}"
else
  # mamma_small: trained on the raw Mamma_mv_split dataset (smpl_num_people:6,
  # img_nums:[4,4]). CONFIG must be mamma_small so the model architecture matches
  # the checkpoint (mamma_overfit uses smpl_num_people:2 -> shape mismatch).
  CONFIG="${CONFIG:-mamma_small}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/training/logs/mamma_small/checkpoint_450.pt}"
  # Raw Mamma_mv_split root; the demo globs */*/png under <root>/<split> to find
  # scene sequences (e.g. test/tmp/<batch>/<dataset>/png/<seq>).
  DATASET_ROOT="${DATASET_ROOT:-/mnt/train-data-4-hdd/yian/Mamma_mv_split}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  # mamma_small trained with img_nums:[4,4] (4 of 8 views per sample), so feed 4
  # views at inference to match the training distribution.
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"
fi

echo "[run_demo] mode=$DEMO_MODE config=$CONFIG"
echo "[run_demo] checkpoint=$CHECKPOINT"
echo "[run_demo] dataset_root=$DATASET_ROOT split=$DATASET_SPLIT"
echo "[run_demo] image_ids=$DEMO_IMAGE_IDS"

if [[ -n "${DEMO_SEQ_DIR:-}" ]]; then
  echo "[run_demo] demo_seq_dir=$DEMO_SEQ_DIR sequence=${DEMO_SEQUENCE:-<auto>} view=${DEMO_VIEW:-<auto>} max_frames=${DEMO_MAX_FRAMES:-10}"
  python demo_gradio_smpl_multi.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --dataset-split "$DATASET_SPLIT" \
    --demo-seq-dir "$DEMO_SEQ_DIR" \
    --demo-sequence "${DEMO_SEQUENCE:-}" \
    --demo-view "${DEMO_VIEW:-}" \
    --demo-max-frames "${DEMO_MAX_FRAMES:-10}" \
    --demo-fps "${DEMO_FPS:-2.0}"
else
  python demo_gradio_smpl_multi.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --dataset-split "$DATASET_SPLIT"
fi
