#!/bin/bash

set -e

# ==================== Configuration Parameters ====================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12341}
EXPERIMENT="waymo_multiview_post_train"
CHECKPOINT_PATH="/data/cosmos-transfer2.5/output/20260304_220742/cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/checkpoints/iter_000004400/model_ema_bf16.pt"
INPUT_FILE="${WAYMO_INPUT_FILE:-}"
OUTPUT_DIR="outputs/waymo-autoregressive-mv"
WAYMO_SPLIT="training"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
WAYMO_DATA_ROOT="/data/waymo/inference"
OFFLINE_MODE=${OFFLINE_MODE:-true}

# ==================== Parse Command Line Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"            
            shift 2
            ;;
        --split)
            WAYMO_SPLIT="$2"
            shift 2
            ;;
        --data-root)
            WAYMO_DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--input PATH] [--output DIR] [--split training|validation] [--data-root DIR] [--checkpoint PT] [--gpus N] [--offline true|false]"
            exit 1
            ;;
    esac
done

OUTPUT_DIR="outputs/waymo-mv-${WAYMO_SPLIT}-${TIMESTAMP}"

# ==================== Check Parameters ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 Inference Script (Assets Autoregressive)"
echo "========================================"
echo "📁 Input File:   ${INPUT_FILE:-<Not specified, will iterate through specs directory>}"
echo "📂 Output Directory:   $OUTPUT_DIR"
echo "🧪 Experiment Name:   $EXPERIMENT"
echo "🎮 Number of GPUs:   $NUM_GPUS"
echo "🔌 Master Port:     $MASTER_PORT"
echo "💾 Checkpoint: $CHECKPOINT_PATH"
echo "📦 Offline Mode:   $OFFLINE_MODE"
echo "========================================"

if [[ "$OFFLINE_MODE" != "true" && "$OFFLINE_MODE" != "false" ]]; then
    echo "❌ Error: --offline must be either true or false"
    exit 1
fi

if [ -z "$INPUT_FILE" ]; then
    if [ "$WAYMO_SPLIT" != "training" ] && [ "$WAYMO_SPLIT" != "validation" ]; then
        echo "❌ Error: --split must be either training or validation"
        exit 1
    fi

    SPEC_DIR="$WAYMO_DATA_ROOT/$WAYMO_SPLIT/specs"
    if [ ! -d "$SPEC_DIR" ]; then
        echo "❌ Error: specs directory does not exist: $SPEC_DIR"
        exit 1
    fi
elif [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

if [ "$OFFLINE_MODE" = true ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export UV_OFFLINE=1
    export UV_NO_PROGRESS=1
    echo "📦 Setting offline cache environment variables: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 UV_OFFLINE=1 UV_NO_PROGRESS=1"
fi

# ==================== Start Inference ====================
if [ -n "$INPUT_FILE" ]; then
    echo "🔧 Starting distributed inference with torchrun ($NUM_GPUS GPUs)..."
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m examples.multiview -i $INPUT_FILE -o $OUTPUT_DIR --checkpoint_path $CHECKPOINT_PATH --experiment $EXPERIMENT --disable-guardrails"
    echo "📝 Executing command: $CMD"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m examples.multiview \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_DIR" \
        --experiment "$EXPERIMENT" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --disable-guardrails
else
    echo "🔧 Starting distributed inference with torchrun ($NUM_GPUS GPUs)..."
    if [ -n "$ZSH_VERSION" ]; then
        setopt null_glob
    elif [ -n "$BASH_VERSION" ]; then
        shopt -s nullglob
    fi
    SPECS=("$SPEC_DIR"/*.json)
    echo "📝 Number of spec: ${#SPECS[@]}"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m examples.multiview \
        -i "${SPECS[@]}" \
        -o "$OUTPUT_DIR" \
        --experiment "$EXPERIMENT" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --disable-guardrails
fi

# ==================== Inference Completed ====================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Inference completed!"
    echo "📂 Output saved to: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ Inference failed, exit code: $?"
    echo "========================================"
    exit 1
fi