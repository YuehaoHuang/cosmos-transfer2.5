#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${JOB_NAME:-waymo_lidar_singleview_rangemap_layout_i2v_t8_20260522}"
TOTAL_ITER="${TOTAL_ITER:-100000}"
CHUNK_ITER="${CHUNK_ITER:-10000}"
SAVE_ITER="${SAVE_ITER:-5000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/team/hyh/code/cosmos-transfer2.5/outputs/waymo_lidar_singleview_posttrain}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./train_waymo_lidar_singleview_posttrain.sh}"

NUM_WORKERS="${NUM_WORKERS:-1}"
PIN_MEMORY="${PIN_MEMORY:-false}"
DECORD_NUM_THREADS="${DECORD_NUM_THREADS:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
SAMPLE_ITER="${SAMPLE_ITER:-10000}"
SAMPLE_GENERATION_TYPES="${SAMPLE_GENERATION_TYPES:-i2v}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-100000}"
MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=... ./train_waymo_lidar_singleview_chunked.sh [options] [-- extra train args]

Options:
  --job-name NAME             Reuse this job directory for auto-resume.
  --total-iter N              Final absolute training iteration. Default: 100000.
  --chunk-iter N              Stop and relaunch every N iterations. Default: 10000.
  --save-iter N               Checkpoint interval. Must divide chunk-iter. Default: 5000.
  --sleep-seconds N           Sleep between chunks. Default: 30.
  --output-root DIR           Same OUTPUT_ROOT passed to the training script.
  --train-script PATH         Training script to call. Default: ./train_waymo_lidar_singleview_posttrain.sh.
  --num-workers N             DataLoader workers per rank. Default: 1.
  --pin-memory                Enable DataLoader pin_memory.
  --no-pin-memory             Disable DataLoader pin_memory. Default.
  -h, --help                  Show this help.

Any args after "--" are forwarded to train_waymo_lidar_singleview_posttrain.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --total-iter)
      TOTAL_ITER="$2"
      shift 2
      ;;
    --chunk-iter)
      CHUNK_ITER="$2"
      shift 2
      ;;
    --save-iter)
      SAVE_ITER="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --pin-memory)
      PIN_MEMORY=true
      shift
      ;;
    --no-pin-memory)
      PIN_MEMORY=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

require_non_negative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
}

require_positive_int TOTAL_ITER "$TOTAL_ITER"
require_positive_int CHUNK_ITER "$CHUNK_ITER"
require_positive_int SAVE_ITER "$SAVE_ITER"
require_non_negative_int SLEEP_SECONDS "$SLEEP_SECONDS"
require_non_negative_int NUM_WORKERS "$NUM_WORKERS"

if [[ "$PIN_MEMORY" != "true" && "$PIN_MEMORY" != "false" ]]; then
  echo "PIN_MEMORY must be true or false, got: $PIN_MEMORY" >&2
  exit 2
fi

if (( CHUNK_ITER % SAVE_ITER != 0 )); then
  echo "CHUNK_ITER=$CHUNK_ITER must be divisible by SAVE_ITER=$SAVE_ITER." >&2
  echo "Otherwise the next chunk may not resume exactly from the chunk boundary." >&2
  exit 2
fi

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "Training script is not executable: $TRAIN_SCRIPT" >&2
  exit 2
fi

run_dir="$OUTPUT_ROOT/cosmos_transfer2_posttrain/waymo_lidar_singleview/$JOB_NAME"
latest_file="$run_dir/checkpoints/latest_checkpoint.txt"

latest_iter=0
if [[ -f "$latest_file" ]]; then
  latest_checkpoint="$(tr -d '[:space:]' < "$latest_file")"
  if [[ "$latest_checkpoint" =~ iter_0*([0-9]+) ]]; then
    latest_iter="${BASH_REMATCH[1]}"
  else
    echo "[chunked-train] warning: could not parse latest checkpoint: $latest_checkpoint" >&2
  fi
fi

if (( latest_iter >= TOTAL_ITER )); then
  echo "[chunked-train] already finished: latest_iter=$latest_iter total_iter=$TOTAL_ITER"
  exit 0
fi

target_iter=$CHUNK_ITER
if (( latest_iter > 0 )); then
  target_iter=$(( ((latest_iter / CHUNK_ITER) + 1) * CHUNK_ITER ))
fi
if (( target_iter > TOTAL_ITER )); then
  target_iter=$TOTAL_ITER
fi

echo "[chunked-train] job: $JOB_NAME"
echo "[chunked-train] output root: $OUTPUT_ROOT"
echo "[chunked-train] latest iter: $latest_iter"
echo "[chunked-train] chunk iter: $CHUNK_ITER"
echo "[chunked-train] total iter: $TOTAL_ITER"
echo "[chunked-train] workers: num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY decord_num_threads=$DECORD_NUM_THREADS"

while (( target_iter <= TOTAL_ITER )); do
  echo "[chunked-train] starting chunk to MAX_ITER=$target_iter"
  JOB_NAME="$JOB_NAME" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MAX_ITER="$target_iter" \
    SAVE_ITER="$SAVE_ITER" \
    SAMPLE_ITER="$SAMPLE_ITER" \
    SAMPLE_GENERATION_TYPES="$SAMPLE_GENERATION_TYPES" \
    NUM_CONDITIONAL_FRAMES="$NUM_CONDITIONAL_FRAMES" \
    NUM_WORKERS="$NUM_WORKERS" \
    PIN_MEMORY="$PIN_MEMORY" \
    DECORD_NUM_THREADS="$DECORD_NUM_THREADS" \
    SCHEDULER_CYCLE_LENGTH="$SCHEDULER_CYCLE_LENGTH" \
    WANDB_MODE="$WANDB_MODE" \
    MALLOC_TRIM_THRESHOLD_="$MALLOC_TRIM_THRESHOLD_" \
    "$TRAIN_SCRIPT" --no-tmux "${EXTRA_ARGS[@]}"

  if [[ -f "$latest_file" ]]; then
    latest_checkpoint="$(tr -d '[:space:]' < "$latest_file")"
    echo "[chunked-train] latest checkpoint: $latest_checkpoint"
  else
    echo "[chunked-train] warning: latest checkpoint file not found: $latest_file" >&2
  fi

  if (( target_iter >= TOTAL_ITER )); then
    break
  fi

  target_iter=$(( target_iter + CHUNK_ITER ))
  if (( target_iter > TOTAL_ITER )); then
    target_iter=$TOTAL_ITER
  fi

  if (( SLEEP_SECONDS > 0 )); then
    echo "[chunked-train] sleeping ${SLEEP_SECONDS}s before next chunk"
    sleep "$SLEEP_SECONDS"
  fi
done

echo "[chunked-train] done"
