#!/usr/bin/env bash
set -euo pipefail

PREP_SESSION="${PREP_SESSION:-waymo_lidar_singleview_prep_20260505_2207}"
DATASET_DIR="${DATASET_DIR:-/data2/waymo_singleview_lidar_posttrain/training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/waymo_lidar_singleview_posttrain_output}"
LOG_DIR="${LOG_DIR:-/data2/waymo_singleview_lidar_posttrain/logs}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
TOTAL_ITER="${TOTAL_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-500}"
LOGGING_ITER="${LOGGING_ITER:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POLL_SECONDS="${POLL_SECONDS:-300}"
JOB_NAME="${JOB_NAME:-waymo_lidar_singleview_rangemap_layout_t8_g${TRAIN_GPUS}_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/train_after_prep_${JOB_NAME}.log}"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_PATH"
}

count_videos() {
  find "$DATASET_DIR/videos" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d " "
}

log "waiting for prep session: $PREP_SESSION"
while tmux has-session -t "$PREP_SESSION" 2>/dev/null; do
  log "prep running: videos=$(count_videos)"
  sleep "$POLL_SECONDS"
done

SUMMARY="$DATASET_DIR/prepare_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  log "ERROR: missing prepare summary: $SUMMARY"
  exit 1
fi

python - "$SUMMARY" <<'PY' 2>&1 | tee -a "$LOG_PATH"
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r") as f:
    summary = json.load(f)
print(summary)
if int(summary.get("num_prepared", 0)) <= 0:
    raise SystemExit(f"no prepared samples in {summary_path}")
PY

log "launching training: gpus=$TRAIN_GPUS job=$JOB_NAME"
exec ./train_waymo_lidar_singleview_chunked.sh \
  --no-tmux \
  --gpus "$TRAIN_GPUS" \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --job-name "$JOB_NAME" \
  --total-iter "$TOTAL_ITER" \
  --save-iter "$SAVE_ITER" \
  --logging-iter "$LOGGING_ITER" \
  --num-workers "$NUM_WORKERS" \
  2>&1 | tee -a "$LOG_PATH"
