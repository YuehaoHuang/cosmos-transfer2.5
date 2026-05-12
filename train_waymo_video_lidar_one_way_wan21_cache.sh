#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29681}"
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
DISTRIBUTED_PARALLELISM="${DISTRIBUTED_PARALLELISM:-ddp}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-0}"
RF_CONVENTION="${RF_CONVENTION:-predict2}"
RF_TRAIN_TIME_DISTRIBUTION="${RF_TRAIN_TIME_DISTRIBUTION:-logitnormal}"
RF_SHIFT="${RF_SHIFT:-5.0}"
LIDAR_LATENT_CONTRACT="${LIDAR_LATENT_CONTRACT:-wan21_native64x1312_repeatrow11_v1}"
CACHE_DIR="${CACHE_DIR:-/data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/waymo_video_lidar_one_way_expert}"
RUN_NAME="${RUN_NAME:-real_video_wan21_lidar_lidar${LIDAR_NUM_BLOCKS}_vkv${VIDEO_KV_EVERY_N_LAYERS}_ckpt${CHECKPOINT_LIDAR_BLOCKS}_b${BATCH_SIZE}_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
PROFILE_STEP="${PROFILE_STEP:-0}"
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
    --profile-step)
      PROFILE_STEP="$2"
      shift 2
      ;;
    --distributed-parallelism)
      DISTRIBUTED_PARALLELISM="$2"
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

if [[ ! -d "$CACHE_DIR/video" || ! -d "$CACHE_DIR/lidar" ]]; then
  echo "Missing linked Wan2.1 cache layout under $CACHE_DIR; expected video/ and lidar/." >&2
  echo "Generate LiDAR latents with scripts/cache_waymo_lidar_wan21_latents.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  scripts/train_waymo_video_lidar_one_way_expert_baseline.py
  --paired-latent-cache-dir "$CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --max-steps "$MAX_STEPS"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --lr "$LR"
  --lidar-num-blocks "$LIDAR_NUM_BLOCKS"
  --video-kv-every-n-layers "$VIDEO_KV_EVERY_N_LAYERS"
  --distributed-parallelism "$DISTRIBUTED_PARALLELISM"
  --fsdp-shard-size "$FSDP_SHARD_SIZE"
  --lidar-latent-contract "$LIDAR_LATENT_CONTRACT"
  --train-height 88
  --train-width 164
  --sdpa-backends flash_only
  --cross-frame-rule all
  --rf-convention "$RF_CONVENTION"
  --rf-train-time-distribution "$RF_TRAIN_TIME_DISTRIBUTION"
  --rf-shift "$RF_SHIFT"
  --cached-latent-device-dtype auto
  --no-empty-cache-after-encode
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
else
  echo "CHECKPOINT_LIDAR_BLOCKS must be true or false, got: $CHECKPOINT_LIDAR_BLOCKS" >&2
  exit 2
fi

if [[ "$PROFILE_STEP" != "0" ]]; then
  cmd+=(--profile-step "$PROFILE_STEP" --profile-dir "$OUTPUT_DIR/profiles")
fi
if [[ "$DRY_RUN" == true ]]; then
  cmd+=(--dry-run)
fi

echo "output_dir=$OUTPUT_DIR"
echo "cache_dir=$CACHE_DIR"
echo "num_gpus=$NUM_GPUS batch_size_per_rank=$BATCH_SIZE lidar_num_blocks=$LIDAR_NUM_BLOCKS checkpoint_lidar_blocks=$CHECKPOINT_LIDAR_BLOCKS video_kv_every_n_layers=$VIDEO_KV_EVERY_N_LAYERS init_lidar_from_video=$INIT_LIDAR_FROM_VIDEO distributed_parallelism=$DISTRIBUTED_PARALLELISM fsdp_shard_size=$FSDP_SHARD_SIZE"
echo "lidar_latent_contract=$LIDAR_LATENT_CONTRACT train_hw=88x164"
echo "rf_convention=$RF_CONVENTION rf_train_time_distribution=$RF_TRAIN_TIME_DISTRIBUTION rf_shift=$RF_SHIFT"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
