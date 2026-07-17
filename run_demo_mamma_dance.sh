#!/usr/bin/env bash
# Launch the SMPL multi-person Gradio demo.
#
# Default mode (maskdpt) targets the DPT-mask training run:
#   config:     training/config/mamma_mask_dpt.yaml  (smpl_num_people:20, DPT mask head)
#   checkpoint: training/logs/mamma_mask_dpt/checkpoint_16.pt
#   data:       raw Mamma_mv_split (test split)
#
# Other modes:
#   DEMO_MODE=maskdpt_eval_single ./run_demo_mamma_dance.sh # locally prepared eval-single frame
#   DEMO_MODE=pretrain_eval_single ./run_demo_mamma_dance.sh # checkpoint_step_10000 on same frame
#   DEMO_MODE=maskdpt_dance ./run_demo_mamma_dance.sh  # checkpoint_16 on MAMMA_eval_dance/test
#   DEMO_MODE=pretrain      ./run_demo_mamma_dance.sh  # SMPL-only pretrain checkpoint_step_10000
#   DEMO_MODE=overfit       ./run_demo_mamma_dance.sh  # mamma_small overfit run
#   DEMO_MODE=dance         ./run_demo_mamma_dance.sh  # old MAMMA_eval_dance defaults
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
# render_mesh_projection_cpu.py only needs numpy/opencv/torch (all already in this
# env), so point the projection worker back at this same interpreter instead of the
# original (missing) pytorch3d env.
export PYTORCH3D_PROJECTION_PYTHON="${PYTORCH3D_PROJECTION_PYTHON:-$(command -v python)}"

DEMO_MODE="${DEMO_MODE:-maskdpt}"

OVERFIT_SCENE_DEFAULT="/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/interactions_couple_close_1_C_200_00_contact/be_0GcH1mWtRfKu_seq_000050/tmp/bedlam_lab_20251031_191434"

if [[ "$DEMO_MODE" == "maskdpt" ]]; then
  # DPT-mask run: CONFIG must be mamma_mask_dpt so the model architecture matches
  # the checkpoint (smpl_num_people:20 + enable_person_mask dpt); mamma_small /
  # mamma_overfit have a different slot count -> shape mismatch on load.
  CONFIG="${CONFIG:-mamma_mask_dpt}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/training/logs/mamma_mask_dpt/checkpoint_23.pt}"
  DATASET_ROOT="${DATASET_ROOT:-/mnt/train-data-4-hdd/yian/Mamma_mv_split}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  # trained with img_nums:[4,4] -> feed 4 views at inference to match.
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"
elif [[ "$DEMO_MODE" == "maskdpt_eval_single" ]]; then
  # Prepared by tools/prepare_mamma_eval_demo.py. Each run is one synchronized
  # frame, with one IOI_XX.jpg per selected camera and a merged GT NPZ.
  CONFIG="${CONFIG:-mamma_mask_dpt}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/model/checkpoint_49.pt}"
  DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/mamma/demo_eval_single}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"
elif [[ "$DEMO_MODE" == "maskdpt_dance" ]]; then
  # Same DPT-mask checkpoint, but on the ORGANIZED MAMMA_eval_dance eval set
  # (out_image/out_data npz, 32 IOI_* views/run). GT cameras+SMPL come from the
  # npz; there is NO mask/landmark GT here, so the Landmark/Mask tab shows the
  # PREDICTED masks only (still fine -- masks are a model output, not GT-derived).
  CONFIG="${CONFIG:-mamma_mask_dpt}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/training/logs/mamma_mask_dpt/checkpoint_16.pt}"
  DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/MAMMA_eval_dance}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  # feed 4 of the 32 views to match the [4,4] training distribution.
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"
elif [[ "$DEMO_MODE" == "pretrain_eval_single" ]]; then
  # SMPL-only 20-slot pretrain evaluated on the locally prepared synchronized
  # four-camera frame. This checkpoint has no person-mask/landmark head.
  CONFIG="${CONFIG:-0621_mamma_demo}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/checkpoint_step_10000.pt}"
  DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/mamma/demo_eval_single}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"
elif [[ "$DEMO_MODE" == "pretrain" ]]; then
  # The SMPL-only pretrain (checkpoint_step_10000): aggregator + camera + 20-slot
  # SMPL head, NO mask/landmark head. CONFIG must be 0621_mamma_demo (num_people:20,
  # mask/landmark OFF) so the architecture matches -- 0621_mamma is now 5 slots and
  # would crash on the 20-slot person_queries. No mask tab here (head absent).
  CONFIG="${CONFIG:-0621_mamma_demo}"
  CHECKPOINT="${CHECKPOINT:-$REPO_DIR/training/logs/0621_mamma/ckpts/checkpoint_step_10000.pt}"
  DATASET_ROOT="${DATASET_ROOT:-/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle/MAMMA_eval_dance}"
  DATASET_SPLIT="${DATASET_SPLIT:-test}"
  # feed all 8 views per scene. NOTE: trained with img_nums:[4,4] (4 views); the
  # aggregator is view-count agnostic so 8 works, but it is off the training
  # distribution -- more views usually help, just isn't exactly what it saw.
  export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3 4 5 6 7}"
elif [[ "$DEMO_MODE" == "dance" ]]; then
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

if [[ -n "${DEMO_INPUT_DIR:-}" ]]; then
  echo "[run_demo] demo_input_dir=$DEMO_INPUT_DIR image_ids=${DEMO_IMAGE_IDS:-0 1 2 3}"
  python demo_gradio_smpl_multi.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --dataset-split "$DATASET_SPLIT" \
    --demo-input-dir "$DEMO_INPUT_DIR" \
    --demo-image-ids "${DEMO_IMAGE_IDS:-0 1 2 3}" \
    --demo-fps "${DEMO_FPS:-2.0}"
elif [[ -n "${DEMO_SEQ_DIR:-}" ]]; then
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
