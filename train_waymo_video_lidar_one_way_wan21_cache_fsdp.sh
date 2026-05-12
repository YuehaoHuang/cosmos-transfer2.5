#!/usr/bin/env bash
set -euo pipefail

export DISTRIBUTED_PARALLELISM="${DISTRIBUTED_PARALLELISM:-fsdp}"
export CHECKPOINT_LIDAR_BLOCKS="${CHECKPOINT_LIDAR_BLOCKS:-false}"
export LIDAR_NUM_BLOCKS="${LIDAR_NUM_BLOCKS:-28}"
export VIDEO_KV_EVERY_N_LAYERS="${VIDEO_KV_EVERY_N_LAYERS:-4}"
export INIT_LIDAR_FROM_VIDEO="${INIT_LIDAR_FROM_VIDEO:-true}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-0}"
export RUN_NAME="${RUN_NAME:-real_video_wan21_lidar_lidar${LIDAR_NUM_BLOCKS}_vkv${VIDEO_KV_EVERY_N_LAYERS}_fsdp_nocp_b${BATCH_SIZE:-1}_${NUM_GPUS:-8}gpu_$(date +%Y%m%d_%H%M%S)}"

exec ./train_waymo_video_lidar_one_way_wan21_cache.sh "$@"
