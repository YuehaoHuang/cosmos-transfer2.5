#!/bin/bash

set -e

# ==================== Configuration Parameters ====================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12341}
EXPERIMENT="waymo_multiview_post_train"
CHECKPOINT_PATH="/data/cosmos-transfer2.5/output/20260313_225508/cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/checkpoints/iter_000011000/model_ema_bf16.pt"
INPUT_FILE="${WAYMO_INPUT_FILE:-}"
OUTPUT_DIR=""
OUTPUT_DIR_SPECIFIED=false
WAYMO_SPLIT="training"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
WAYMO_DATA_ROOT="/data/waymo/inference"
OFFLINE_MODE=${OFFLINE_MODE:-true}
PREDICT2_TOKENIZER_REPO="nvidia/Cosmos-Predict2.5-2B"
PREDICT2_TOKENIZER_REVISION="6787e176dce74a101d922174a95dba29fa5f0c55"
PREDICT2_TOKENIZER_FILE="tokenizer.pth"

# Prefer visible GPU count from CUDA_VISIBLE_DEVICES. Fall back to nvidia-smi when available.
VISIBLE_GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _CUDA_VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES// /}"
    VISIBLE_GPU_COUNT=${#_CUDA_VISIBLE_GPU_ARRAY[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
fi

# ==================== Parse Command Line Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            OUTPUT_DIR_SPECIFIED=true
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

if [ "$OUTPUT_DIR_SPECIFIED" = false ]; then
    OUTPUT_DIR="outputs/waymo-mv-${WAYMO_SPLIT}-${TIMESTAMP}"
fi

get_spec_name_from_filename() {
    local spec_path="$1"
    local spec_file
    spec_file="$(basename "$spec_path")"
    echo "${spec_file%.json}"
}

check_hf_file_in_offline_cache() {
    local repo="$1"
    local revision="$2"
    local filename="$3"
    if ! command -v hf >/dev/null 2>&1; then
        echo "❌ Error: offline mode requires HuggingFace CLI ('hf') to verify local cache, but it was not found."
        echo "   Install once: uv tool install -U \"huggingface_hub[cli]\""
        return 1
    fi
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 hf download \
        "$repo" \
        "$filename" \
        --repo-type model \
        --revision "$revision" \
        --quiet >/dev/null 2>&1
}

# ==================== Check Parameters ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 Inference Script (Waymo Autoregressive)"
echo "========================================"
echo "📁 Input File:   ${INPUT_FILE:-<Not specified, will iterate through specs directory>}"
echo "📂 Output Directory:   $OUTPUT_DIR"
echo "🧪 Experiment Name:   $EXPERIMENT"
echo "🎮 Number of GPUs:   $NUM_GPUS"
if [ -n "$VISIBLE_GPU_COUNT" ]; then
    echo "🖥️  Visible GPUs:     $VISIBLE_GPU_COUNT"
    echo "🎯 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
echo "🔌 Master Port:     $MASTER_PORT"
echo "💾 Checkpoint: $CHECKPOINT_PATH"
echo "📦 Offline Mode:   $OFFLINE_MODE"
echo "========================================"

if [ -n "$VISIBLE_GPU_COUNT" ] && [ "$NUM_GPUS" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "❌ Error: requested --gpus=$NUM_GPUS but only $VISIBLE_GPU_COUNT GPU(s) are visible to this process."
    echo "   Hint: set --gpus <= visible GPU count, or adjust CUDA_VISIBLE_DEVICES."
    exit 1
fi

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

mkdir -p "$OUTPUT_DIR"

if [ "$OFFLINE_MODE" = true ]; then
    # Prefer shared HF cache when available (commonly pre-populated on training/inference servers).
    if [ -d "/data/huggingface/hub" ]; then
        export HF_HOME="${HF_HOME:-/data/huggingface}"
        export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
        echo "📦 Using HuggingFace cache: HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE"
    fi

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export UV_OFFLINE=1
    export UV_NO_PROGRESS=1
    echo "📦 Setting offline cache environment variables: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 UV_OFFLINE=1 UV_NO_PROGRESS=1"

    echo "🔍 Checking required local cache files for offline mode..."
    if ! check_hf_file_in_offline_cache "$PREDICT2_TOKENIZER_REPO" "$PREDICT2_TOKENIZER_REVISION" "$PREDICT2_TOKENIZER_FILE"; then
        echo "❌ Error: missing offline cache file: $PREDICT2_TOKENIZER_REPO/$PREDICT2_TOKENIZER_FILE@$PREDICT2_TOKENIZER_REVISION"
        echo "   Fix A (recommended): run once with --offline false so cache can be populated."
        echo "   Fix B (manual pre-download):"
        echo "     hf download \"$PREDICT2_TOKENIZER_REPO\" \"$PREDICT2_TOKENIZER_FILE\" --repo-type model --revision \"$PREDICT2_TOKENIZER_REVISION\""
        exit 1
    fi
    echo "✅ Offline cache check passed: $PREDICT2_TOKENIZER_FILE"
fi

# ==================== Start Inference ====================
if [ -n "$INPUT_FILE" ]; then
    SAMPLE_NAME="$(get_spec_name_from_filename "$INPUT_FILE")"
    if [ -d "$OUTPUT_DIR/$SAMPLE_NAME" ]; then
        echo "⏭️  Found existing output folder, skipping spec: $SAMPLE_NAME"
        echo "📌 Existing folder: $OUTPUT_DIR/$SAMPLE_NAME"
        exit 0
    fi

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
    if [ -n "$ZSH_VERSION" ]; then
        setopt null_glob
    elif [ -n "$BASH_VERSION" ]; then
        shopt -s nullglob
    fi

    SPECS=("$SPEC_DIR"/*.json)
    if [ ${#SPECS[@]} -eq 0 ]; then
        echo "❌ Error: no spec files found in $SPEC_DIR"
        exit 1
    fi

    PENDING_SPECS=()
    COMPLETED_COUNT=0
    for spec in "${SPECS[@]}"; do
        SAMPLE_NAME="$(get_spec_name_from_filename "$spec")"
        if [ -d "$OUTPUT_DIR/$SAMPLE_NAME" ]; then
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            continue
        fi
        PENDING_SPECS+=("$spec")
    done

    echo "📝 Number of spec (total/completed/pending): ${#SPECS[@]}/$COMPLETED_COUNT/${#PENDING_SPECS[@]}"
    if [ ${#PENDING_SPECS[@]} -eq 0 ]; then
        echo "✅ All specs are already completed in '$OUTPUT_DIR'. Nothing to run."
        exit 0
    fi

    echo "🔧 Starting distributed inference with torchrun ($NUM_GPUS GPUs)..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m examples.multiview \
        -i "${PENDING_SPECS[@]}" \
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
