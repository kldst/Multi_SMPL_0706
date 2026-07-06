#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source /mnt/train-data-4-hdd/yian/anaconda/etc/profile.d/conda.sh
conda activate mamma

MAMMA_ROOT="${MAMMA_ROOT:-/mnt/train-data-4-hdd/yian/Mamma_mv_split/train}"
SCENE_ROOT="${SCENE_ROOT:-tmp/bedlam_lab_20251031_191436/harmony4d_train_1_NC_200_00}"
SEQ_NAME="${SEQ_NAME:-}"
FRAME="${FRAME:-}"
NUM_VIEWS="${NUM_VIEWS:-4}"
SEED="${SEED:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/debug_outputs/mamma_pipeline}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
REQUIRE_VISIBLE_JOINTS="${REQUIRE_VISIBLE_JOINTS:-1}"
MIN_VISIBLE_JOINTS="${MIN_VISIBLE_JOINTS:-8}"

COMMON_ARGS=(
  --mamma-root "$MAMMA_ROOT"
  --scene-root "$SCENE_ROOT"
  --num-views "$NUM_VIEWS"
  --seed "$SEED"
  --output-root "$OUTPUT_ROOT"
  --image-size "$IMAGE_SIZE"
  --min-visible-joints "$MIN_VISIBLE_JOINTS"
)

if [[ "$REQUIRE_VISIBLE_JOINTS" == "1" || "$REQUIRE_VISIBLE_JOINTS" == "true" ]]; then
  COMMON_ARGS+=(--require-visible-joints)
fi

if [[ -n "$SEQ_NAME" ]]; then
  COMMON_ARGS+=(--seq-name "$SEQ_NAME")
fi
if [[ -n "$FRAME" ]]; then
  COMMON_ARGS+=(--frame "$FRAME")
fi

echo "[pipeline] output_root=$OUTPUT_ROOT"
python debug/debug_01_raw_dataloader_projection.py "${COMMON_ARGS[@]}"
python debug/debug_02_processed_batch_projection.py "${COMMON_ARGS[@]}"
python debug/debug_03_gt_as_pred_loss_zero.py "${COMMON_ARGS[@]}"

echo "[pipeline] done: $OUTPUT_ROOT"
