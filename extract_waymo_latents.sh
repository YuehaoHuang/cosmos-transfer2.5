#!/bin/bash

set -e

# ==================== Configuration Parameters ====================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12371}
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
EXPERIMENT="waymo_multiview_post_train"
SPLIT_ROOTS=${SPLIT_ROOTS:-"/data/waymo/chunk/training,/data/waymo/chunk/validation"}
SAMPLES_SUBDIR=${SAMPLES_SUBDIR:-"samples"}
MAX_BATCHES=${MAX_BATCHES:-0}
SKIP_TRAIN_SAMPLES=${SKIP_TRAIN_SAMPLES:-0}
SKIP_VAL_SAMPLES=${SKIP_VAL_SAMPLES:-0}
SAVE_EVERY=${SAVE_EVERY:-10}
LOADER=${LOADER:-both}
SEPARATE_SPLITS=false
OVERWRITE=false
DRYRUN_MODE=false
OFFLINE_MODE=${OFFLINE_MODE:-true}

# ==================== Parse Command Line Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --split-roots)
            SPLIT_ROOTS="$2"
            shift 2
            ;;
        --samples-subdir)
            SAMPLES_SUBDIR="$2"
            shift 2
            ;;
        --max-batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        --skip-train-samples)
            SKIP_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --skip-val-samples)
            SKIP_VAL_SAMPLES="$2"
            shift 2
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --loader)
            LOADER="$2"
            shift 2
            ;;
        --separate-splits)
            SEPARATE_SPLITS=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --dryrun)
            DRYRUN_MODE=true
            shift
            ;;
        --offline)
            OFFLINE_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--gpus N] [--master-port PORT] [--config PATH] [--experiment NAME]"
            echo "          [--split-roots CSV] [--samples-subdir NAME] [--max-batches N]"
            echo "          [--skip-train-samples N] [--skip-val-samples N] [--save-every N]"
            echo "          [--loader train|val|both] [--separate-splits] [--overwrite] [--dryrun] [--offline true|false]"
            exit 1
            ;;
    esac
done

if [[ "$LOADER" != "train" && "$LOADER" != "val" && "$LOADER" != "both" ]]; then
    echo "Error: --loader must be one of train|val|both"
    exit 1
fi

if [[ "$OFFLINE_MODE" != "true" && "$OFFLINE_MODE" != "false" ]]; then
    echo "Error: --offline must be either true or false"
    exit 1
fi

# ==================== Check Environment ====================
echo "========================================"
echo "Cosmos-Transfer2.5 Waymo Latent Extractor"
echo "========================================"
echo "Config file:       $CONFIG_FILE"
echo "Experiment:        $EXPERIMENT"
echo "GPUs:              $NUM_GPUS"
echo "Master port:       $MASTER_PORT"
echo "Split roots:       $SPLIT_ROOTS"
echo "Samples subdir:    $SAMPLES_SUBDIR"
echo "Max batches:       $MAX_BATCHES"
echo "Skip train:        $SKIP_TRAIN_SAMPLES"
echo "Skip val:          $SKIP_VAL_SAMPLES"
echo "Save every:        $SAVE_EVERY"
echo "Loader:            $LOADER"
echo "Separate splits:   $SEPARATE_SPLITS"
echo "Overwrite:         $OVERWRITE"
echo "Dryrun:            $DRYRUN_MODE"
echo "Offline mode:      $OFFLINE_MODE"
echo "========================================"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

if [ "$OFFLINE_MODE" = true ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export UV_OFFLINE=1
    export UV_NO_PROGRESS=1
    echo "Setting offline cache environment variables: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 UV_OFFLINE=1 UV_NO_PROGRESS=1"
fi

run_extract() {
    local loader="$1"
    local port="$2"

    local cmd=(
        torchrun
        --nproc_per_node="$NUM_GPUS"
        --master_port="$port"
        packages/cosmos-oss/scripts/extract_waymo_latents.py
        --config "$CONFIG_FILE"
        --split-roots "$SPLIT_ROOTS"
        --samples-subdir "$SAMPLES_SUBDIR"
        --max-batches "$MAX_BATCHES"
        --skip-train-samples "$SKIP_TRAIN_SAMPLES"
        --skip-val-samples "$SKIP_VAL_SAMPLES"
        --save-every "$SAVE_EVERY"
        --loader "$loader"
    )

    if [ "$OVERWRITE" = true ]; then
        cmd+=(--overwrite)
    fi

    if [ "$DRYRUN_MODE" = true ]; then
        cmd+=(--dryrun)
    fi

    cmd+=(-- "experiment=$EXPERIMENT")

    echo "Executing command: ${cmd[*]}"
    "${cmd[@]}"
}

if [ "$SEPARATE_SPLITS" = true ] && [ "$LOADER" = "both" ]; then
    run_extract "train" "$MASTER_PORT"
    run_extract "val" "$((MASTER_PORT + 1))"
else
    run_extract "$LOADER" "$MASTER_PORT"
fi

echo "========================================"
echo "Latent extraction completed."
echo "========================================"
