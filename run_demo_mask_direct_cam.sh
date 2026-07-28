#!/usr/bin/env bash
# Launch the SMPL multi-person Gradio demo (demo_gradio_smpl_multi.py) for the
# direct_cam mask-head run. Mirrors run_demo_mamma_dance.sh but targets:
#   config:     training/config/mamma_mask_direct_cam.yaml
#               (smpl_num_people:20, person_mask_head_type: direct_cam)
#   checkpoint: model/direct_cam/checkpoint_17.pt
#   data:       organized MAMMA markerless multi-person eval set
#               (MAMMA_markerless_multiple_people/test)
#
# scale_by_extrinsics is NOT set in mamma_mask_direct_cam.yaml -> defaults to True.
# demo_gradio_smpl_multi.py reads it from the config, so the cam0 gauge divides by
# the mean camera baseline (as trained).
#
# Dataset presets:
#   DEMO_DATASET=markerless  -> MAMMA_markerless_multiple_people/test (default)
#   DEMO_DATASET=dance       -> MAMMA_eval_dance/test
#
# Both datasets use organized out_image/out_data directories under the test
# split, which the main demo discovers directly. DATASET_ROOT can still override
# either preset.
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
# render_mesh_projection_cpu.py only needs numpy/opencv/torch (all in this env),
# so point the projection worker back at this same interpreter.
export PYTORCH3D_PROJECTION_PYTHON="${PYTORCH3D_PROJECTION_PYTHON:-$(command -v python)}"

# config must be mamma_mask_direct_cam so the architecture (smpl_num_people:20 +
# direct_cam person-mask head) matches checkpoint_17.pt.
CONFIG="${CONFIG:-mamma_mask_direct_cam}"
CHECKPOINT="${CHECKPOINT:-$REPO_DIR/model/direct_cam/checkpoint_17.pt}"

DEMO_DATASET="${DEMO_DATASET:-markerless}"
case "$DEMO_DATASET" in
  markerless)
    DEFAULT_DATASET_ROOT="$REPO_DIR/MAMMA_markerless_multiple_people"
    ;;
  dance)
    DEFAULT_DATASET_ROOT="$REPO_DIR/MAMMA_eval_dance"
    ;;
  *)
    echo "[run_demo] unknown DEMO_DATASET=$DEMO_DATASET (expected: markerless or dance)" >&2
    exit 1
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
# trained with img_nums:[4,4] -> feed 4 views at inference to match.
export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 1 2 3}"

echo "[run_demo] config=$CONFIG"
echo "[run_demo] checkpoint=$CHECKPOINT"
echo "[run_demo] dataset=$DEMO_DATASET root=$DATASET_ROOT split=$DATASET_SPLIT"
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
