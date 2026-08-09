#!/usr/bin/env bash
# Launch the SMPL multi-person Gradio demo on the REAL Harmony4D capture.
#
# Same shape as run_demo_mask_dpt_noavgscale.sh, but targets:
#   config:     training/config/mamma_mask_dpt.yaml   (smpl_num_people:20, DPT
#               mask head, scale_by_extrinsics: False)
#   checkpoint: model/no_avg_8view/checkpoint_18.pt   (trained with img_nums:[8,8])
#   data:       Harmony4D 01_hugging/001_hugging, EXPORTED + FISHEYE-RECTIFIED by
#               debug/debug_harmony4d_real/export_harmony4d_for_demo.py
#
# Why an export step instead of pointing straight at the capture:
#   * Harmony4D stores per-camera folders of frames; the demo browses
#     <root>/<split>/out_image/<run>/*.jpg where one run = one time instant
#     holding every view.
#   * The exo cameras are OPENCV_FISHEYE (k1..k4).  Rectifying moves the
#     effective focal length 1754 -> 1566 (~11%); feeding raw fisheye frames to a
#     model trained on pinhole crops reprojects wrong.  The export does the
#     rectification once so the demo never sees a distorted pixel.
#
# GT is exported from Harmony4D's neutral-SMPL annotations.  The published
# aria_from_colmap similarity aligns the COLMAP exo cameras with the SMPL world;
# export_harmony4d_gt_for_demo.py applies it and verifies the result against the
# released 2D joints before writing out_data/*.npz and out_mesh/*.npz.
#
# 8 views are fed by default because this checkpoint was trained with
# img_nums:[8,8]; DEMO_IMAGE_IDS indexes the SORTED file list, i.e. cam01..cam22.
#
# Override CONFIG / CHECKPOINT / DATASET_ROOT / DATASET_SPLIT / DEMO_IMAGE_IDS /
# EXPORT_GT / CUDA_VISIBLE_DEVICES / PORT via env vars.
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
  echo "[run_demo_h4d] conda.sh not found. Set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate mamma

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH3D_PROJECTION_PYTHON="${PYTORCH3D_PROJECTION_PYTHON:-$(command -v python)}"

CONFIG="${CONFIG:-mamma_mask_dpt}"
CHECKPOINT="${CHECKPOINT:-$REPO_DIR/model/no_avg_8view/checkpoint_18.pt}"

H4D_SEQUENCE="${H4D_SEQUENCE:-/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D/01_hugging/001_hugging}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/Harmony4D_real_demo}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
FRAME_STEP="${FRAME_STEP:-10}"
# EXPORT_MODE=model  -> export what the model actually sees (default)
# EXPORT_MODE=full   -> export the whole rectified frame at EXPORT_WIDTH; the demo
#                       then centre-crops it on the way in and 44% of the
#                       horizontal FOV silently disappears.
EXPORT_MODE="${EXPORT_MODE:-model}"
EXPORT_WIDTH="${EXPORT_WIDTH:-1920}"
EXPORT_GT="${EXPORT_GT:-1}"
# 8 evenly spaced cameras around the ring: cam01 cam04 cam07 cam09 cam12 cam15 cam17 cam20
export DEMO_IMAGE_IDS="${DEMO_IMAGE_IDS:-0 3 6 8 11 14 16 19}"

# Export once; delete DATASET_ROOT (or set FORCE_EXPORT=1) to regenerate.
if [[ "${FORCE_EXPORT:-0}" == "1" || ! -d "$DATASET_ROOT/$DATASET_SPLIT/out_image" ]]; then
  echo "[run_demo_h4d] exporting + rectifying $H4D_SEQUENCE -> $DATASET_ROOT (mode=$EXPORT_MODE)"
  if [[ "$EXPORT_MODE" == "model" ]]; then
    python debug/debug_harmony4d_real/export_harmony4d_for_demo.py \
      --sequence "$H4D_SEQUENCE" \
      --output "$DATASET_ROOT" \
      --split "$DATASET_SPLIT" \
      --frame-step "$FRAME_STEP" \
      --config "$CONFIG" \
      --model-input
  else
    python debug/debug_harmony4d_real/export_harmony4d_for_demo.py \
      --sequence "$H4D_SEQUENCE" \
      --output "$DATASET_ROOT" \
      --split "$DATASET_SPLIT" \
      --frame-step "$FRAME_STEP" \
      --width "$EXPORT_WIDTH"
  fi
else
  echo "[run_demo_h4d] reusing existing export at $DATASET_ROOT (FORCE_EXPORT=1 to rebuild)"
fi

# GT conversion is cheap compared with image rectification and is independently
# resumable, so run it even when an existing image export is reused.  Existing
# complete archives are skipped unless FORCE_EXPORT=1.
if [[ "$EXPORT_GT" == "1" ]]; then
  GT_ARGS=(
    --sequence "$H4D_SEQUENCE"
    --dataset-root "$DATASET_ROOT"
    --split "$DATASET_SPLIT"
  )
  if [[ "${FORCE_EXPORT:-0}" == "1" ]]; then
    GT_ARGS+=(--overwrite)
  fi
  python debug/debug_harmony4d_real/export_harmony4d_gt_for_demo.py "${GT_ARGS[@]}"
fi

echo "[run_demo_h4d] config=$CONFIG"
echo "[run_demo_h4d] checkpoint=$CHECKPOINT"
echo "[run_demo_h4d] dataset=$DATASET_ROOT split=$DATASET_SPLIT"
DEMO_VIEW_COUNT="$(wc -w <<< "$DEMO_IMAGE_IDS")"
echo "[run_demo_h4d] image_ids=$DEMO_IMAGE_IDS  ($DEMO_VIEW_COUNT views)"
if [[ "$EXPORT_MODE" == "model" ]]; then
  echo "[run_demo_h4d] gallery images are pre-cropped 523x523; demo resizes them to 518x518"
  echo "[run_demo_h4d]       GT intrinsics follow the same resize (EXPORT_MODE=full for full rectified frames)."
fi
if [[ "$EXPORT_GT" == "1" ]]; then
  echo "[run_demo_h4d] GT enabled: 'Use GT' and predicted+GT mesh overlay are available."
else
  echo "[run_demo_h4d] GT disabled by EXPORT_GT=0; cameras are predicted."
fi
# demo_gradio_smpl_multi.py forces cfg.model.enable_point=True for its point-cloud
# view, but mamma_mask_dpt trains with enable_point:False -- so the demo always
# prints '(missing=62 ...)' and those 62 point_head.* tensors stay RANDOM. The SMPL
# / camera / mask outputs are unaffected; only the point cloud is meaningless.
echo "[run_demo_h4d] NOTE: expect '(missing=62, unexpected=0)' -- point_head is random"
echo "[run_demo_h4d]       (demo forces enable_point=True; this config never trained it)."
echo "[run_demo_h4d]       Ignore the point-cloud view; SMPL/camera/mask outputs are fine."

python demo_gradio_smpl_multi.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --dataset-split "$DATASET_SPLIT"
