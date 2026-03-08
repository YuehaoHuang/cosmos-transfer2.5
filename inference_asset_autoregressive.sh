#!/bin/bash

set -e

# ==================== Configuration Parameters ====================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12341}
EXPERIMENT="transfer2_auto_multiview_post_train_example"
CHECKPOINT_PATH="/data/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/00c591edab119e8a6ca06e6e091351a04ce0ecc9/auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt"
INPUT_FILE="assets/multiview_example/multiview_autoregressive_spec.json"
OUTPUT_DIR="outputs/postrained-auto-autoregressive-mv"
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
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--input PATH] [--output DIR] [--gpus N] [--checkpoint PT] [--offline true|false]"
            exit 1
            ;;
    esac
done

# ==================== Check Parameters ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 Inference Script (Assets Autoregressive)"
echo "========================================"
echo "📁 Input File:       $INPUT_FILE"
echo "📂 Output Directory: $OUTPUT_DIR"
echo "🧪 Experiment Name:  $EXPERIMENT"
echo "🎮 Number of GPUs:   $NUM_GPUS"
echo "🔌 Master Port:      $MASTER_PORT"
echo "💾 Checkpoint: $CHECKPOINT_PATH"
echo "📦 Offline Mode:     $OFFLINE_MODE"
echo "========================================"

if [[ "$OFFLINE_MODE" != "true" && "$OFFLINE_MODE" != "false" ]]; then
    echo "❌ Error: --offline must be either true or false"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
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
echo "🔧 Starting distributed inference with torchrun ($NUM_GPUS GPUs)..."
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m examples.multiview -i $INPUT_FILE -o $OUTPUT_DIR --experiment $EXPERIMENT --checkpoint_path $CHECKPOINT_PATH --disable-guardrails"
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