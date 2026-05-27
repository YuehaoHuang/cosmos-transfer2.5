#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-drivesync}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29731}"
RAW_LIDAR_ROOT="${RAW_LIDAR_ROOT:-/team/hyh/data/rds_hq_waymo}"
RAW_LIDAR_SPLIT="${RAW_LIDAR_SPLIT:-training}"
DATASET_DIR="${DATASET_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/team/hyh/code/cosmos-transfer2.5/outputs/waymo_lidar_singleview_posttrain}"
EXPERIMENT="${EXPERIMENT:-transfer2_singleview_posttrain_waymo_lidar_wan21_online_layout_fullfinetune}"
JOB_NAME="${JOB_NAME:-waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8}"
LOAD_PATH="${LOAD_PATH:-}"
LOAD_TRAINING_STATE="${LOAD_TRAINING_STATE:-false}"
LEARNING_RATE="${LEARNING_RATE:-}"
STATE_T=8
TOTAL_ITER="${TOTAL_ITER:-100000}"
CHUNK_ITER="${CHUNK_ITER:-10000}"
SAVE_ITER="${SAVE_ITER:-1000}"
LOGGING_ITER="${LOGGING_ITER:-500}"
SAMPLE_ITER="${SAMPLE_ITER:-5000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
SAMPLE_GENERATION_TYPES="${SAMPLE_GENERATION_TYPES:-i2v}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-100000}"
WANDB_MODE="${WANDB_MODE:-disabled}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-true}"
DECORD_NUM_THREADS="${DECORD_NUM_THREADS:-4}"
USE_TMUX="${USE_TMUX:-false}"
TMUX_SESSION="${TMUX_SESSION:-waymo_lidar_singleview_chunked_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
EXTRA_CONFIG_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  ./train_waymo_lidar_singleview_chunked.sh [options] [-- extra hydra overrides]

Options:
  --job-name NAME             Stable run name used for auto-resume.
  --total-iter N              Final absolute training iteration. Default: 100000.
  --max-iter N                Alias for --total-iter.
  --chunk-iter N              Stop and relaunch every N iterations. Default: 10000.
  --save-iter N               Checkpoint interval. Must divide chunk-iter. Default: 5000.
  --sleep-seconds N           Sleep between chunks. Default: 30.
  --gpus N                    GPUs for torchrun. state_t=8 requires N in 1,2,4,8.
  --master-port PORT          torchrun master port.
  --dataset-dir DIR           Optional root containing captions/ overrides; raw-online needs no MP4 dataset.
  --raw-lidar-root DIR        Waymo root containing <split>/lidar_raw/*.tar.
  --raw-lidar-split NAME      Raw LiDAR split, training or validation. Default: training.
  --output-root DIR           Output root for checkpoints and logs.
  --experiment NAME           Hydra experiment name.
  --load-path PATH            Initial checkpoint path when no latest checkpoint exists.
  --load-training-state       Resume optimizer/scheduler/trainer from --load-path.
  --no-load-training-state    Load model weights only from --load-path. Default.
  --learning-rate LR          Override optimizer.lr.
  --logging-iter N            Trainer logging interval. Default: 500.
  --sample-iter N             Sampling callback interval. Default: 10000.
  --num-conditional-frames N  One of 0, 1, 2. Default: 1.
  --sample-generation-types S Comma-separated t2v,i2v,v2v subset. Default: i2v.
  --scheduler-warmup-steps N  Scheduler warmup. Default: 1000.
  --scheduler-cycle-length N  Scheduler cycle length. Default: 100000.
  --num-workers N             DataLoader workers per rank. Default: 1.
  --decord-num-threads N      Decord decode threads per worker. Default: 1.
  --pin-memory                Enable DataLoader pin_memory.
  --no-pin-memory             Disable DataLoader pin_memory. Default.
  --wandb-mode MODE           WandB mode. Default: disabled.
  --tmux                      Launch this chunked run in a tmux session.
  --tmux-session NAME         tmux session name.
  --dry-run                   Generate config and exit through scripts.train dryrun.
  -h, --help                  Show this help.

Extra arguments after "--" are appended as Hydra overrides after script defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --raw-lidar-root)
      RAW_LIDAR_ROOT="$2"
      shift 2
      ;;
    --raw-lidar-split)
      RAW_LIDAR_SPLIT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --load-path)
      LOAD_PATH="$2"
      shift 2
      ;;
    --load-training-state)
      LOAD_TRAINING_STATE=true
      shift
      ;;
    --no-load-training-state)
      LOAD_TRAINING_STATE=false
      shift
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --total-iter|--max-iter)
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
    --logging-iter)
      LOGGING_ITER="$2"
      shift 2
      ;;
    --sample-iter)
      SAMPLE_ITER="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --num-conditional-frames)
      NUM_CONDITIONAL_FRAMES="$2"
      shift 2
      ;;
    --sample-generation-types)
      SAMPLE_GENERATION_TYPES="$2"
      shift 2
      ;;
    --scheduler-warmup-steps)
      SCHEDULER_WARMUP_STEPS="$2"
      shift 2
      ;;
    --scheduler-cycle-length)
      SCHEDULER_CYCLE_LENGTH="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --decord-num-threads)
      DECORD_NUM_THREADS="$2"
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
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --tmux-session)
      TMUX_SESSION="$2"
      shift 2
      ;;
    --tmux)
      USE_TMUX=true
      shift
      ;;
    --no-tmux)
      USE_TMUX=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CONFIG_OVERRIDES+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="$RAW_LIDAR_ROOT/$RAW_LIDAR_SPLIT"
fi

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

require_positive_int NUM_GPUS "$NUM_GPUS"
require_positive_int TOTAL_ITER "$TOTAL_ITER"
require_positive_int CHUNK_ITER "$CHUNK_ITER"
require_positive_int SAVE_ITER "$SAVE_ITER"
require_positive_int LOGGING_ITER "$LOGGING_ITER"
require_positive_int SAMPLE_ITER "$SAMPLE_ITER"
require_positive_int SCHEDULER_WARMUP_STEPS "$SCHEDULER_WARMUP_STEPS"
require_positive_int SCHEDULER_CYCLE_LENGTH "$SCHEDULER_CYCLE_LENGTH"
require_positive_int DECORD_NUM_THREADS "$DECORD_NUM_THREADS"
require_non_negative_int SLEEP_SECONDS "$SLEEP_SECONDS"
require_non_negative_int NUM_WORKERS "$NUM_WORKERS"

if [[ "$PIN_MEMORY" != "true" && "$PIN_MEMORY" != "false" ]]; then
  echo "PIN_MEMORY must be true or false, got: $PIN_MEMORY" >&2
  exit 2
fi

if [[ "$LOAD_TRAINING_STATE" != "true" && "$LOAD_TRAINING_STATE" != "false" ]]; then
  echo "LOAD_TRAINING_STATE must be true or false, got: $LOAD_TRAINING_STATE" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
  echo "DRY_RUN must be true or false, got: $DRY_RUN" >&2
  exit 2
fi

if ! [[ "$NUM_CONDITIONAL_FRAMES" =~ ^[0-2]$ ]]; then
  echo "NUM_CONDITIONAL_FRAMES must be one of 0, 1, or 2, got: $NUM_CONDITIONAL_FRAMES" >&2
  exit 2
fi

if (( STATE_T % NUM_GPUS != 0 )); then
  echo "Invalid NUM_GPUS=$NUM_GPUS for this experiment: state_t=$STATE_T must be divisible by context_parallel_size." >&2
  echo "Use one of: 1, 2, 4, 8." >&2
  exit 2
fi

if (( CHUNK_ITER % SAVE_ITER != 0 )); then
  echo "CHUNK_ITER=$CHUNK_ITER must be divisible by SAVE_ITER=$SAVE_ITER." >&2
  echo "Otherwise the next chunk may not resume exactly from the chunk boundary." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$visible_gpus" =~ ^[0-9]+$ ]] && (( visible_gpus > 0 && NUM_GPUS > visible_gpus )); then
    echo "Requested NUM_GPUS=$NUM_GPUS, but nvidia-smi reports only $visible_gpus visible GPU(s)." >&2
    exit 2
  fi
fi

if [[ "$RAW_LIDAR_SPLIT" != "training" && "$RAW_LIDAR_SPLIT" != "validation" ]]; then
  echo "RAW_LIDAR_SPLIT must be training or validation, got: $RAW_LIDAR_SPLIT" >&2
  exit 2
fi

if [[ ! -d "$RAW_LIDAR_ROOT/$RAW_LIDAR_SPLIT/lidar_raw" ]]; then
  echo "Raw LiDAR tar folder does not exist: $RAW_LIDAR_ROOT/$RAW_LIDAR_SPLIT/lidar_raw" >&2
  exit 1
fi

if [[ "$USE_TMUX" == "true" && -z "${WAYMO_LIDAR_SINGLEVIEW_INSIDE_TMUX:-}" ]]; then
  env_prefix=(
    "WAYMO_LIDAR_SINGLEVIEW_INSIDE_TMUX=1"
    "USE_TMUX=false"
    "CONDA_ENV=$(printf "%q" "$CONDA_ENV")"
    "CONDA_SH=$(printf "%q" "$CONDA_SH")"
    "NUM_GPUS=$(printf "%q" "$NUM_GPUS")"
    "MASTER_PORT=$(printf "%q" "$MASTER_PORT")"
    "DATASET_DIR=$(printf "%q" "$DATASET_DIR")"
    "RAW_LIDAR_ROOT=$(printf "%q" "$RAW_LIDAR_ROOT")"
    "RAW_LIDAR_SPLIT=$(printf "%q" "$RAW_LIDAR_SPLIT")"
    "OUTPUT_ROOT=$(printf "%q" "$OUTPUT_ROOT")"
    "EXPERIMENT=$(printf "%q" "$EXPERIMENT")"
    "JOB_NAME=$(printf "%q" "$JOB_NAME")"
    "LOAD_PATH=$(printf "%q" "$LOAD_PATH")"
    "LOAD_TRAINING_STATE=$(printf "%q" "$LOAD_TRAINING_STATE")"
    "LEARNING_RATE=$(printf "%q" "$LEARNING_RATE")"
    "TOTAL_ITER=$(printf "%q" "$TOTAL_ITER")"
    "CHUNK_ITER=$(printf "%q" "$CHUNK_ITER")"
    "SAVE_ITER=$(printf "%q" "$SAVE_ITER")"
    "LOGGING_ITER=$(printf "%q" "$LOGGING_ITER")"
    "SAMPLE_ITER=$(printf "%q" "$SAMPLE_ITER")"
    "SLEEP_SECONDS=$(printf "%q" "$SLEEP_SECONDS")"
    "NUM_CONDITIONAL_FRAMES=$(printf "%q" "$NUM_CONDITIONAL_FRAMES")"
    "SAMPLE_GENERATION_TYPES=$(printf "%q" "$SAMPLE_GENERATION_TYPES")"
    "SCHEDULER_WARMUP_STEPS=$(printf "%q" "$SCHEDULER_WARMUP_STEPS")"
    "SCHEDULER_CYCLE_LENGTH=$(printf "%q" "$SCHEDULER_CYCLE_LENGTH")"
    "WANDB_MODE=$(printf "%q" "$WANDB_MODE")"
    "NUM_WORKERS=$(printf "%q" "$NUM_WORKERS")"
    "PIN_MEMORY=$(printf "%q" "$PIN_MEMORY")"
    "DECORD_NUM_THREADS=$(printf "%q" "$DECORD_NUM_THREADS")"
    "DRY_RUN=$(printf "%q" "$DRY_RUN")"
  )
  tmux_cmd="cd $(printf "%q" "$PWD") && ${env_prefix[*]} bash $(printf "%q" "$0") --no-tmux"
  if (( ${#EXTRA_CONFIG_OVERRIDES[@]} > 0 )); then
    tmux_cmd+=" --"
    for override in "${EXTRA_CONFIG_OVERRIDES[@]}"; do
      tmux_cmd+=" $(printf "%q" "$override")"
    done
  fi
  tmux new-session -d -s "$TMUX_SESSION" "$tmux_cmd"
  echo "[launch] tmux session: $TMUX_SESSION"
  echo "[launch] attach: tmux attach -t $TMUX_SESSION"
  exit 0
fi

export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

run_dir="$OUTPUT_ROOT/cosmos_transfer2_posttrain/waymo_lidar_singleview/$JOB_NAME"
latest_file="$run_dir/checkpoints/latest_checkpoint.txt"
latest_checkpoint=""
latest_iter=0

parse_latest_iter() {
  latest_checkpoint=""
  latest_iter=0
  if [[ -f "$latest_file" ]]; then
    latest_checkpoint="$(tr -d '[:space:]' < "$latest_file")"
    if [[ "$latest_checkpoint" =~ iter_0*([0-9]+) ]]; then
      latest_iter="${BASH_REMATCH[1]}"
    else
      echo "[chunked-train] warning: could not parse latest checkpoint: $latest_checkpoint" >&2
    fi
  fi
}

next_target_iter() {
  if (( latest_iter > 0 )); then
    target_iter=$(( ((latest_iter / CHUNK_ITER) + 1) * CHUNK_ITER ))
  else
    target_iter=$CHUNK_ITER
  fi
  if (( target_iter > TOTAL_ITER )); then
    target_iter=$TOTAL_ITER
  fi
}

run_chunk() {
  local target_iter="$1"
  local train_args=()
  if [[ "$DRY_RUN" == "true" ]]; then
    train_args+=(--dryrun)
  fi

  local cmd=(
    torchrun
    --nproc_per_node="$NUM_GPUS"
    --master_port="$MASTER_PORT"
    -m scripts.train
    "${train_args[@]}"
    --config=cosmos_transfer2/singleview_config.py
    --
    "experiment=$EXPERIMENT"
    "dataloader_train.dataset.dataset_dir=$DATASET_DIR"
    "dataloader_train.dataset.raw_lidar_root=$RAW_LIDAR_ROOT"
    "dataloader_train.dataset.raw_lidar_split=$RAW_LIDAR_SPLIT"
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}'
    "dataloader_train.num_workers=$NUM_WORKERS"
    "dataloader_train.pin_memory=$PIN_MEMORY"
    "dataloader_train.dataset.decord_num_threads=$DECORD_NUM_THREADS"
    "trainer.max_iter=$target_iter"
    "trainer.logging_iter=$LOGGING_ITER"
    "checkpoint.save_iter=$SAVE_ITER"
    "model.config.min_num_conditional_frames=$NUM_CONDITIONAL_FRAMES"
    "model.config.max_num_conditional_frames=$NUM_CONDITIONAL_FRAMES"
    "model.config.conditional_frames_probs=null"
    "trainer.callbacks.every_n_sample_reg.every_n=$SAMPLE_ITER"
    "trainer.callbacks.every_n_sample_ema.every_n=$SAMPLE_ITER"
    "trainer.callbacks.every_n_sample_reg.generation_types=$SAMPLE_GENERATION_TYPES"
    "trainer.callbacks.every_n_sample_ema.generation_types=$SAMPLE_GENERATION_TYPES"
    "scheduler.warm_up_steps=[$SCHEDULER_WARMUP_STEPS]"
    "scheduler.cycle_lengths=[$SCHEDULER_CYCLE_LENGTH]"
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
  )

  if [[ -n "$LOAD_PATH" ]]; then
    cmd+=("checkpoint.load_path=$LOAD_PATH")
    cmd+=("checkpoint.load_training_state=$LOAD_TRAINING_STATE")
  fi

  if [[ -n "$LEARNING_RATE" ]]; then
    cmd+=("optimizer.lr=$LEARNING_RATE")
  fi

  if (( ${#EXTRA_CONFIG_OVERRIDES[@]} > 0 )); then
    cmd+=("${EXTRA_CONFIG_OVERRIDES[@]}")
  fi

  echo "[train] command: ${cmd[*]}"
  "${cmd[@]}"
}

parse_latest_iter
if (( latest_iter >= TOTAL_ITER )); then
  echo "[chunked-train] already finished: latest_iter=$latest_iter total_iter=$TOTAL_ITER"
  exit 0
fi
next_target_iter

echo "[chunked-train] output root: $IMAGINAIRE_OUTPUT_ROOT"
echo "[chunked-train] dataset: $DATASET_DIR"
echo "[chunked-train] raw lidar source: $RAW_LIDAR_ROOT/$RAW_LIDAR_SPLIT/lidar_raw"
echo "[chunked-train] experiment: $EXPERIMENT"
echo "[chunked-train] job: $JOB_NAME"
echo "[chunked-train] latest iter: $latest_iter"
echo "[chunked-train] chunk iter: $CHUNK_ITER"
echo "[chunked-train] total iter: $TOTAL_ITER"
echo "[chunked-train] save/log/sample: save_iter=$SAVE_ITER logging_iter=$LOGGING_ITER sample_iter=$SAMPLE_ITER"
echo "[chunked-train] conditioning: num_conditional_frames=$NUM_CONDITIONAL_FRAMES generation_types=$SAMPLE_GENERATION_TYPES"
echo "[chunked-train] workers: num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY decord_num_threads=$DECORD_NUM_THREADS"
echo "[chunked-train] scheduler: warmup=$SCHEDULER_WARMUP_STEPS cycle_length=$SCHEDULER_CYCLE_LENGTH"
if [[ -n "$LOAD_PATH" ]]; then
  echo "[chunked-train] load path: $LOAD_PATH"
  echo "[chunked-train] load training state: $LOAD_TRAINING_STATE"
fi
if [[ -n "$LEARNING_RATE" ]]; then
  echo "[chunked-train] learning rate: $LEARNING_RATE"
fi

while (( target_iter <= TOTAL_ITER )); do
  echo "[chunked-train] starting chunk to MAX_ITER=$target_iter"
  run_chunk "$target_iter"

  parse_latest_iter
  if [[ -n "$latest_checkpoint" ]]; then
    echo "[chunked-train] latest checkpoint: $latest_checkpoint"
  elif [[ "$DRY_RUN" == "true" ]]; then
    echo "[chunked-train] dry run completed without checkpoint"
  else
    echo "[chunked-train] latest checkpoint file not found: $latest_file" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" == "false" && "$latest_iter" =~ ^[0-9]+$ ]] && (( latest_iter < target_iter )); then
    echo "[chunked-train] latest checkpoint iter $latest_iter is behind target iter $target_iter" >&2
    exit 1
  fi

  if (( target_iter >= TOTAL_ITER )); then
    break
  fi

  next_target_iter
  if (( SLEEP_SECONDS > 0 )); then
    echo "[chunked-train] sleeping ${SLEEP_SECONDS}s before next chunk"
    sleep "$SLEEP_SECONDS"
  fi
done

echo "[chunked-train] done"
