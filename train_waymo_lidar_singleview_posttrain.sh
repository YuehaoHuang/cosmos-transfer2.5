#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-cosmos-transfer2.5-merge}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29731}"
DATASET_DIR="${DATASET_DIR:-/data2/waymo_singleview_lidar_posttrain/training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/waymo_lidar_singleview_posttrain_output}"
EXPERIMENT="${EXPERIMENT:-transfer2_singleview_posttrain_waymo_lidar_rangemap_layout}"
JOB_NAME="${JOB_NAME:-waymo_lidar_singleview_rangemap_layout_t8_$(date +%Y%m%d_%H%M%S)}"
LOAD_PATH="${LOAD_PATH:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
STATE_T=8
MAX_ITER="${MAX_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-500}"
LOGGING_ITER="${LOGGING_ITER:-50}"
WANDB_MODE="${WANDB_MODE:-disabled}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_TMUX="${USE_TMUX:-true}"
TMUX_SESSION="${TMUX_SESSION:-waymo_lidar_singleview_posttrain_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"

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
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --max-iter)
      MAX_ITER="$2"
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
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --tmux-session)
      TMUX_SESSION="$2"
      shift 2
      ;;
    --no-tmux)
      USE_TMUX=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$USE_TMUX" == "true" && -z "${WAYMO_LIDAR_SINGLEVIEW_INSIDE_TMUX:-}" ]]; then
  env_prefix=(
    "WAYMO_LIDAR_SINGLEVIEW_INSIDE_TMUX=1"
    "USE_TMUX=false"
    "CONDA_ENV=$(printf "%q" "$CONDA_ENV")"
    "NUM_GPUS=$(printf "%q" "$NUM_GPUS")"
    "MASTER_PORT=$(printf "%q" "$MASTER_PORT")"
    "DATASET_DIR=$(printf "%q" "$DATASET_DIR")"
    "OUTPUT_ROOT=$(printf "%q" "$OUTPUT_ROOT")"
    "EXPERIMENT=$(printf "%q" "$EXPERIMENT")"
    "JOB_NAME=$(printf "%q" "$JOB_NAME")"
    "LOAD_PATH=$(printf "%q" "$LOAD_PATH")"
    "LEARNING_RATE=$(printf "%q" "$LEARNING_RATE")"
    "MAX_ITER=$(printf "%q" "$MAX_ITER")"
    "SAVE_ITER=$(printf "%q" "$SAVE_ITER")"
    "LOGGING_ITER=$(printf "%q" "$LOGGING_ITER")"
    "WANDB_MODE=$(printf "%q" "$WANDB_MODE")"
    "NUM_WORKERS=$(printf "%q" "$NUM_WORKERS")"
    "DRY_RUN=$(printf "%q" "$DRY_RUN")"
  )
  tmux new-session -d -s "$TMUX_SESSION" \
    "cd $(printf "%q" "$PWD") && ${env_prefix[*]} bash $(printf "%q" "$0") --no-tmux"
  echo "[launch] tmux session: $TMUX_SESSION"
  echo "[launch] attach: tmux attach -t $TMUX_SESSION"
  exit 0
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi

if (( STATE_T % NUM_GPUS != 0 )); then
  echo "Invalid NUM_GPUS=$NUM_GPUS for this experiment: state_t=$STATE_T must be divisible by context_parallel_size." >&2
  echo "Use one of: 1, 2, 4, 8." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$visible_gpus" =~ ^[0-9]+$ ]] && (( visible_gpus > 0 && NUM_GPUS > visible_gpus )); then
    echo "Requested NUM_GPUS=$NUM_GPUS, but nvidia-smi reports only $visible_gpus visible GPU(s)." >&2
    exit 2
  fi
fi

if [[ ! -d "$DATASET_DIR/videos" || ! -d "$DATASET_DIR/rangemap_layout" || ! -d "$DATASET_DIR/captions" ]]; then
  echo "Dataset is missing videos/, rangemap_layout/, or captions/: $DATASET_DIR" >&2
  echo "Run scripts/prepare_waymo_lidar_singleview_posttrain_dataset.py first." >&2
  exit 1
fi

export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

train_args=()
if [[ "$DRY_RUN" == "true" ]]; then
  train_args+=(--dryrun)
fi

cmd=(
  torchrun
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  -m scripts.train
  "${train_args[@]}"
  --config=cosmos_transfer2/singleview_config.py
  --
  "experiment=$EXPERIMENT"
  "dataloader_train.dataset.dataset_dir=$DATASET_DIR"
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}'
  "dataloader_train.num_workers=$NUM_WORKERS"
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=$LOGGING_ITER"
  "checkpoint.save_iter=$SAVE_ITER"
  "job.name=$JOB_NAME"
  "job.wandb_mode=$WANDB_MODE"
)

if [[ -n "$LOAD_PATH" ]]; then
  cmd+=("checkpoint.load_path=$LOAD_PATH")
fi

if [[ -n "$LEARNING_RATE" ]]; then
  cmd+=("optimizer.lr=$LEARNING_RATE")
fi

echo "[train] output root: $IMAGINAIRE_OUTPUT_ROOT"
echo "[train] dataset: $DATASET_DIR"
echo "[train] experiment: $EXPERIMENT"
if [[ -n "$LOAD_PATH" ]]; then
  echo "[train] load path: $LOAD_PATH"
fi
if [[ -n "$LEARNING_RATE" ]]; then
  echo "[train] learning rate: $LEARNING_RATE"
fi
echo "[train] command: ${cmd[*]}"
exec "${cmd[@]}"
