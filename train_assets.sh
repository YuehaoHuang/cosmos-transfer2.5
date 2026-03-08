#!/bin/bash

set -e  # Exit immediately on error

# ==================== Configuration Parameters ====================
# Number of GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Training configuration
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
EXPERIMENT="transfer2_auto_multiview_post_train_example"

# Port number (for distributed training)
MASTER_PORT=${MASTER_PORT:-12341}

# ==================== Parse Command Line Arguments ====================
DEBUG_MODE=false
DRYRUN_MODE=false
PROFILE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            echo "🐛 Debug mode enabled (debugpy will listen on port 5678)"
            shift
            ;;
        --dryrun)
            DRYRUN_MODE=true
            echo "🏃 Dry-run mode enabled (only prints configuration)"
            shift
            ;;
        --profile)
            PROFILE_MODE=true
            echo "📊 Profiling enabled"
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            echo "🎮 Requested $NUM_GPUS GPUs"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--debug] [--dryrun] [--profile] [--gpus N]"
            exit 1
            ;;
    esac
done

# Force single GPU for debug mode (to avoid multi-process port conflicts)
if [ "$DEBUG_MODE" = true ]; then
    if [ "$NUM_GPUS" -ne 1 ]; then
        echo "⚠️  Debug mode automatically set to single GPU (original setting: $NUM_GPUS)"
        NUM_GPUS=1
    fi
fi

# ==================== Check Environment =