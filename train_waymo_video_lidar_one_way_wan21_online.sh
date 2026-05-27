#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-7}"
MASTER_PORT="${MASTER_PORT:-29682}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LR="${LR:-1e-4}"
LIDAR_NUM_BLOCKS="${LIDAR_NUM_BLOCKS:-28}"
CHECKPOINT_LIDAR_BLOCKS="${CHECKPOINT_LIDAR_BLOCKS:-true}"
VIDEO_KV_EVERY_N_LAYERS="${VIDEO_KV_EVERY_N_LAYERS:-4}"
INIT_LIDAR_FROM_VIDEO="${INIT_LIDAR_FROM_VIDEO:-true}"
RF_CONVENTION="${RF_CONVENTION:-predict2}"
RF_TRAIN_TIME_DISTRIBUTION="${RF_TRAIN_TIME_DISTRIBUTION:-logitnormal}"
RF_SHIFT="${RF_SHIFT:-5.0}"
VIDEO_LATENT_DIR="${VIDEO_LATENT_DIR:-/data/waymo/chunk/training/samples}"
LIDAR_LATENT_CONTRACT="${LIDAR_LATENT_CONTRACT:-wan21_native64x1280_repeatrow11_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/waymo_video_lidar_one_way_expert}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PROFILE_STEP="${PROFILE_STEP:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
DRY_RUN=false

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit-samples)
      LIMIT_SAMPLES="$2"
      shift 2
      ;;
    --profile-step)
      PROFILE_STEP="$2"
      shift 2
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

if [[ ! -d "$VIDEO_LATENT_DIR" ]]; then
  echo "Missing real video latent dir: $VIDEO_LATENT_DIR" >&2
  exit 1
fi

if [[ "$CHECKPOINT_LIDAR_BLOCKS" == "true" ]]; then
  CHECKPOINT_LABEL="cp"
elif [[ "$CHECKPOINT_LIDAR_BLOCKS" == "false" ]]; then
  CHECKPOINT_LABEL="nocp"
else
  echo "CHECKPOINT_LIDAR_BLOCKS must be true or false, got: $CHECKPOINT_LIDAR_BLOCKS" >&2
  exit 2
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="real_video_wan21_online_lidar${LIDAR_NUM_BLOCKS}_vkv${VIDEO_KV_EVERY_N_LAYERS}_${CHECKPOINT_LABEL}_b${BATCH_SIZE}_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  scripts/train_waymo_video_lidar_one_way_expert_baseline.py
  --output-dir "$OUTPUT_DIR"
  --precomputed-video-latent-dir "$VIDEO_LATENT_DIR"
  --lidar-tokenizer wan21
  --lidar-latent-contract "$LIDAR_LATENT_CONTRACT"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --max-steps "$MAX_STEPS"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --lr "$LR"
  --lidar-num-blocks "$LIDAR_NUM_BLOCKS"
  --video-kv-every-n-layers "$VIDEO_KV_EVERY_N_LAYERS"
  --train-height 88
  --train-width 160
  --native-n-rows 64
  --native-n-cols 1280
  --downsample-factor-row 1
  --downsample-factor-col 1
  --repeat-row 11
  --repeat-col 1
  --input-channel-mode repeat_depth
  --decode-channel-mode mean
  --wan-spatial-align 8
  --sdpa-backends flash_only
  --cross-frame-rule all
  --rf-convention "$RF_CONVENTION"
  --rf-train-time-distribution "$RF_TRAIN_TIME_DISTRIBUTION"
  --rf-shift "$RF_SHIFT"
  --cached-latent-device-dtype auto
  --no-empty-cache-after-encode
  --offload-lidar-encoder
)

if [[ "$INIT_LIDAR_FROM_VIDEO" == "true" ]]; then
  cmd+=(--init-lidar-from-video)
elif [[ "$INIT_LIDAR_FROM_VIDEO" == "false" ]]; then
  cmd+=(--no-init-lidar-from-video)
else
  echo "INIT_LIDAR_FROM_VIDEO must be true or false, got: $INIT_LIDAR_FROM_VIDEO" >&2
  exit 2
fi

if [[ "$CHECKPOINT_LIDAR_BLOCKS" == "true" ]]; then
  cmd+=(--checkpoint-lidar-blocks)
elif [[ "$CHECKPOINT_LIDAR_BLOCKS" == "false" ]]; then
  cmd+=(--no-checkpoint-lidar-blocks)
fi

if [[ "$LIMIT_SAMPLES" != "0" ]]; then
  cmd+=(--limit-samples "$LIMIT_SAMPLES")
fi
if [[ "$PROFILE_STEP" != "0" ]]; then
  cmd+=(--profile-step "$PROFILE_STEP" --profile-dir "$OUTPUT_DIR/profiles")
fi
if [[ "$DRY_RUN" == true ]]; then
  cmd+=(--dry-run)
fi

echo "output_dir=$OUTPUT_DIR"
echo "video_latent_dir=$VIDEO_LATENT_DIR"
echo "num_gpus=$NUM_GPUS batch_size_per_rank=$BATCH_SIZE lidar_num_blocks=$LIDAR_NUM_BLOCKS checkpoint_lidar_blocks=$CHECKPOINT_LIDAR_BLOCKS video_kv_every_n_layers=$VIDEO_KV_EVERY_N_LAYERS init_lidar_from_video=$INIT_LIDAR_FROM_VIDEO"
echo "lidar_tokenizer=wan21-online lidar_latent_contract=$LIDAR_LATENT_CONTRACT train_hw=88x160"
echo "rf_convention=$RF_CONVENTION rf_train_time_distribution=$RF_TRAIN_TIME_DISTRIBUTION rf_shift=$RF_SHIFT"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
