#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-drivesync}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_FSDP="${USE_FSDP:-false}"
USE_DDP="${USE_DDP:-false}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
LR="${LR:-1e-5}"
TRAIN_SCOPE="${TRAIN_SCOPE:-full}"
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-none}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-none}"
LOSS_IN_FORWARD="${LOSS_IN_FORWARD:-false}"
DETACH_TEMPORAL_CACHE_GRADIENT="${DETACH_TEMPORAL_CACHE_GRADIENT:-false}"
NUM_FRAMES="${NUM_FRAMES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
POLYPHASE_ROLL="${POLYPHASE_ROLL:-640}"
RAW_LIDAR_ROOT="${RAW_LIDAR_ROOT:-/team/hyh/data/rds_hq_waymo}"
WAN_REPO="${WAN_REPO:-/team/hyh/code/Wan2.1}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/team/hyh/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/waymo_lidar_vae_finetune}"
RUN_NAME="${RUN_NAME:-polyphase3_repeat_roll${POLYPHASE_ROLL}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$USE_DDP" == "true" && "$USE_FSDP" == "true" ]]; then
  echo "USE_DDP and USE_FSDP cannot both be true" >&2
  exit 2
fi

cmd=(
  python scripts/train_waymo_lidar_wan21_vae_polyphase.py
  --wan-repo "$WAN_REPO"
  --wan-vae-path "$WAN_VAE_PATH"
  --raw-lidar-root "$RAW_LIDAR_ROOT"
  --output-dir "$OUTPUT_DIR"
  --range-map-layout polyphase3_repeat
  --polyphase-roll "$POLYPHASE_ROLL"
  --num-frames "$NUM_FRAMES"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --lr "$LR"
  --train-scope "$TRAIN_SCOPE"
  --activation-checkpoint "$ACTIVATION_CHECKPOINT"
  --activation-offload "$ACTIVATION_OFFLOAD"
)
if [[ "$LOSS_IN_FORWARD" == "true" ]]; then
  cmd+=(--loss-in-forward)
fi
if [[ "$DETACH_TEMPORAL_CACHE_GRADIENT" == "true" ]]; then
  cmd+=(--detach-temporal-cache-gradient)
fi

if [[ "$USE_DDP" == "true" ]]; then
  cmd=(
    torchrun
    --standalone
    --nproc_per_node "$NUM_GPUS"
    scripts/train_waymo_lidar_wan21_vae_polyphase.py
    --wan-repo "$WAN_REPO"
    --wan-vae-path "$WAN_VAE_PATH"
    --raw-lidar-root "$RAW_LIDAR_ROOT"
    --output-dir "$OUTPUT_DIR"
    --range-map-layout polyphase3_repeat
    --polyphase-roll "$POLYPHASE_ROLL"
    --num-frames "$NUM_FRAMES"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --save-every "$SAVE_EVERY"
    --log-every "$LOG_EVERY"
    --lr "$LR"
    --train-scope "$TRAIN_SCOPE"
    --activation-checkpoint "$ACTIVATION_CHECKPOINT"
    --activation-offload "$ACTIVATION_OFFLOAD"
  )
  if [[ "$LOSS_IN_FORWARD" == "true" ]]; then
    cmd+=(--loss-in-forward)
  fi
  if [[ "$DETACH_TEMPORAL_CACHE_GRADIENT" == "true" ]]; then
    cmd+=(--detach-temporal-cache-gradient)
  fi
elif [[ "$USE_FSDP" == "true" ]]; then
  cmd=(
    torchrun
    --standalone
    --nproc_per_node "$NUM_GPUS"
    scripts/train_waymo_lidar_wan21_vae_polyphase.py
    --fsdp
    --wan-repo "$WAN_REPO"
    --wan-vae-path "$WAN_VAE_PATH"
    --raw-lidar-root "$RAW_LIDAR_ROOT"
    --output-dir "$OUTPUT_DIR"
    --range-map-layout polyphase3_repeat
    --polyphase-roll "$POLYPHASE_ROLL"
    --num-frames "$NUM_FRAMES"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --save-every "$SAVE_EVERY"
    --log-every "$LOG_EVERY"
    --lr "$LR"
    --train-scope "$TRAIN_SCOPE"
    --activation-checkpoint "$ACTIVATION_CHECKPOINT"
    --activation-offload "$ACTIVATION_OFFLOAD"
  )
  if [[ "$LOSS_IN_FORWARD" == "true" ]]; then
    cmd+=(--loss-in-forward)
  fi
  if [[ "$DETACH_TEMPORAL_CACHE_GRADIENT" == "true" ]]; then
    cmd+=(--detach-temporal-cache-gradient)
  fi
fi

cmd+=("$@")

echo "output_dir=$OUTPUT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES use_ddp=$USE_DDP use_fsdp=$USE_FSDP num_gpus=$NUM_GPUS num_frames=$NUM_FRAMES batch_size=$BATCH_SIZE train_scope=$TRAIN_SCOPE activation_checkpoint=$ACTIVATION_CHECKPOINT activation_offload=$ACTIVATION_OFFLOAD loss_in_forward=$LOSS_IN_FORWARD detach_temporal_cache_gradient=$DETACH_TEMPORAL_CACHE_GRADIENT lr=$LR max_steps=$MAX_STEPS save_every=$SAVE_EVERY log_every=$LOG_EVERY"
printf command:
printf  %q "${cmd[@]}"
printf n

"${cmd[@]}"
