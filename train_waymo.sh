#!/bin/bash

set -e

# ==================== Configuration Parameters ====================
NUM_GPUS=${NUM_GPUS:-8}
WANDB_MODE=${WANDB_MODE:-offline}
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
EXPERIMENT="waymo_multiview_post_train"
JOB_PROJECT="cosmos_transfer_v2p5"
JOB_GROUP="waymo_multiview"
JOB_NAME="waymo_5cam_post_train"
MASTER_PORT=${MASTER_PORT:-12351}
DEBUG_MODE=false
DRYRUN_MODE=false
PROFILE_MODE=false
CHECKPOINT_LOAD_PATH=""
LOAD_TRAINING_STATE="false"
RUN_VALIDATION=""
JOB_NAME_EFFECTIVE="$JOB_NAME"

# ==================== Parse Command Line Arguments ====================


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
        --wandb-mode)
            WANDB_MODE="$2"
            echo "🧪 Set wandb mode to: $WANDB_MODE"
            shift 2
            ;;
        --checkpoint-load-path)
            CHECKPOINT_LOAD_PATH="$2"
            echo "🔁 Set checkpoint.load_path: $CHECKPOINT_LOAD_PATH"
            shift 2
            ;;
        --load-training-state)
            LOAD_TRAINING_STATE="$2"
            echo "🧠 Set checkpoint.load_training_state: $LOAD_TRAINING_STATE"
            shift 2
            ;;
        --run-validation)
            RUN_VALIDATION="$2"
            echo "🧪 Set trainer.run_validation: $RUN_VALIDATION"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--debug] [--dryrun] [--profile] [--gpus N] [--wandb-mode MODE] [--checkpoint-load-path PATH] [--load-training-state BOOL] [--run-validation BOOL]"
            exit 1
            ;;
    esac
done

# Derive the output root directory from the checkpoint path to avoid creating a new timestamp directory when resuming training.
derive_output_root_from_checkpoint() {
    local checkpoint_path="$1"
    local job_dir=""

    # Supports two formats: .../checkpoints/iter_xxx or .../checkpoints
    if [[ "$checkpoint_path" == *"/checkpoints/"* ]]; then
        job_dir="${checkpoint_path%/checkpoints/*}"
    elif [[ "$checkpoint_path" == *"/checkpoints" ]]; then
        job_dir="${checkpoint_path%/checkpoints}"
    else
        return 1
    fi

    if [[ -z "$job_dir" ]]; then
        return 1
    fi

    # job_dir format: <output_root>/<project>/<group>/<name>
    local output_root
    output_root="$(dirname "$(dirname "$(dirname "$job_dir")")")"

    if [[ -z "$output_root" || "$output_root" == "." || "$output_root" == "/" ]]; then
        return 1
    fi

    echo "$output_root"
    return 0
}

# Derive job name from checkpoint path: <output_root>/<project>/<group>/<name>/checkpoints[/iter_xxx]
derive_job_name_from_checkpoint() {
    local checkpoint_path="$1"
    local job_dir=""

    if [[ "$checkpoint_path" == *"/checkpoints/"* ]]; then
        job_dir="${checkpoint_path%/checkpoints/*}"
    elif [[ "$checkpoint_path" == *"/checkpoints" ]]; then
        job_dir="${checkpoint_path%/checkpoints}"
    else
        return 1
    fi

    if [[ -z "$job_dir" ]]; then
        return 1
    fi

    basename "$job_dir"
    return 0
}

# Output directory strategy:
# 1) If --checkpoint-load-path is provided AND --load-training-state=true, reuse historical output root directory.
# 2) Otherwise, create a new timestamped output root directory.
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -n "$CHECKPOINT_LOAD_PATH" && "$LOAD_TRAINING_STATE" == "true" ]]; then
    DERIVED_OUTPUT_ROOT="$(derive_output_root_from_checkpoint "$CHECKPOINT_LOAD_PATH" || true)"
    DERIVED_JOB_NAME="$(derive_job_name_from_checkpoint "$CHECKPOINT_LOAD_PATH" || true)"
    if [[ -n "$DERIVED_OUTPUT_ROOT" ]]; then
        export IMAGINAIRE_OUTPUT_ROOT="$DERIVED_OUTPUT_ROOT"
        echo "♻️  Checkpoint resumption detected, reusing output root directory: $IMAGINAIRE_OUTPUT_ROOT"
    else
        export IMAGINAIRE_OUTPUT_ROOT="/data/cosmos-transfer2.5/output/${RUN_TIMESTAMP}"
        echo "⚠️  Failed to derive output root directory from checkpoint path, falling back to a new directory: $IMAGINAIRE_OUTPUT_ROOT"
    fi

    if [[ -n "$DERIVED_JOB_NAME" ]]; then
        JOB_NAME_EFFECTIVE="$DERIVED_JOB_NAME"
        echo "♻️  Checkpoint resumption detected, reusing job name from checkpoint path: $JOB_NAME_EFFECTIVE"
    else
        echo "⚠️  Failed to derive job name from checkpoint path, keeping configured job name: $JOB_NAME_EFFECTIVE"
    fi
else
    export IMAGINAIRE_OUTPUT_ROOT="/data/cosmos-transfer2.5/output/${RUN_TIMESTAMP}"
fi

# Non-resume runs should keep a stable job name and use timestamped output root for uniqueness.
if [[ -z "$CHECKPOINT_LOAD_PATH" || "$LOAD_TRAINING_STATE" != "true" ]]; then
    JOB_NAME_EFFECTIVE="$JOB_NAME"
    echo "🆕 Non-resume run: creating a new run directory under timestamped output root with stable job name: $JOB_NAME_EFFECTIVE"
fi

# Validate wandb mode
if [[ "$WANDB_MODE" != "online" && "$WANDB_MODE" != "offline" && "$WANDB_MODE" != "disabled" ]]; then
    echo "❌ Error: Invalid wandb mode: $WANDB_MODE"
    echo "Valid options: online | offline | disabled"
    exit 1
fi

# Validate load_training_state
if [[ "$LOAD_TRAINING_STATE" != "true" && "$LOAD_TRAINING_STATE" != "false" ]]; then
    echo "❌ Error: Invalid load_training_state: $LOAD_TRAINING_STATE"
    echo "Valid options: true | false"
    exit 1
fi

# Validate run_validation (only when explicitly provided by the user)
if [[ -n "$RUN_VALIDATION" && "$RUN_VALIDATION" != "true" && "$RUN_VALIDATION" != "false" ]]; then
    echo "❌ Error: Invalid run_validation: $RUN_VALIDATION"
    echo "Valid options: true | false"
    exit 1
fi

# Force single GPU for debug mode (to avoid multi-process port conflicts)
if [ "$DEBUG_MODE" = true ]; then
    if [ "$NUM_GPUS" -ne 1 ]; then
        echo "⚠️  Debug mode automatically set to single GPU (original setting: $NUM_GPUS)"
        NUM_GPUS=1
    fi
fi

# ==================== Check Environment ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 Training Script"
echo "========================================"
echo "📁 Config file: $CONFIG_FILE"
echo "🧪 Experiment: $EXPERIMENT"
echo "🏷️  Job name: $JOB_NAME_EFFECTIVE"
echo "🎮 Number of GPUs: $NUM_GPUS"
echo "🛰️  wandb mode: $WANDB_MODE"
echo "📤 Output root dir: $IMAGINAIRE_OUTPUT_ROOT"
echo "🔌 Master port: $MASTER_PORT"
echo "========================================"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

# Assemble optional overrides (only added when explicitly provided by the user)
EXTRA_OVERRIDES=()
if [ -n "$CHECKPOINT_LOAD_PATH" ]; then
    EXTRA_OVERRIDES+=("checkpoint.load_path=$CHECKPOINT_LOAD_PATH")
fi
EXTRA_OVERRIDES+=("checkpoint.load_training_state=$LOAD_TRAINING_STATE")
if [ -n "$RUN_VALIDATION" ]; then
    EXTRA_OVERRIDES+=("trainer.run_validation=$RUN_VALIDATION")
fi

# ==================== Start Training ====================
# Build arguments based on mode flags
TRAIN_ARGS=""
[ "$DRYRUN_MODE" = true ] && TRAIN_ARGS="$TRAIN_ARGS --dryrun"
[ "$PROFILE_MODE" = true ] && TRAIN_ARGS="$TRAIN_ARGS --profile"

# Debug mode is controlled via environment variables (to avoid argparse parameter ordering issues)
if [ "$DEBUG_MODE" = true ]; then
    export COSMOS_DEBUG=1
    echo "🔍 Setting environment variable: COSMOS_DEBUG=1"
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "🔧 Training with a single GPU..."
    echo "📝 Executing command: python -m scripts.train$TRAIN_ARGS --config=\"$CONFIG_FILE\" -- experiment=\"$EXPERIMENT\" job.name=\"$JOB_NAME_EFFECTIVE\" job.wandb_mode=$WANDB_MODE ${EXTRA_OVERRIDES[*]}"
    python -m scripts.train \
        $TRAIN_ARGS \
        --config="$CONFIG_FILE" \
        -- \
        experiment="$EXPERIMENT" \
        job.name="$JOB_NAME_EFFECTIVE" \
        job.wandb_mode="$WANDB_MODE" \
        "${EXTRA_OVERRIDES[@]}"
else
    # Multi-GPU distributed training
    echo "🔧 Launching distributed training via torchrun ($NUM_GPUS GPUs)..."
    echo "📝 Executing command: torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m scripts.train$TRAIN_ARGS --config=\"$CONFIG_FILE\" -- experiment=\"$EXPERIMENT\" job.name=\"$JOB_NAME_EFFECTIVE\" job.wandb_mode=$WANDB_MODE ${EXTRA_OVERRIDES[*]}"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m scripts.train \
        $TRAIN_ARGS \
        --config="$CONFIG_FILE" \
        -- \
        experiment="$EXPERIMENT" \
        job.name="$JOB_NAME_EFFECTIVE" \
        job.wandb_mode="$WANDB_MODE" \
        "${EXTRA_OVERRIDES[@]}"
fi

# Clean up environment variables
unset COSMOS_DEBUG

# ==================== Training Completed ====================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Training completed!"
    echo "========================================"
    
    if [ "$DRYRUN_MODE" = false ]; then
        CHECKPOINT_DIR="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/${JOB_PROJECT}/${JOB_GROUP}/${JOB_NAME_EFFECTIVE}/checkpoints"
        echo "📦 Checkpoint save location: $CHECKPOINT_DIR"
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo "📂 Checkpoint list:"
            ls -lh "$CHECKPOINT_DIR" | tail -n 10
        fi
    fi
else
    echo ""
    echo "========================================"
    echo "❌ Training failed with exit code: $?"
    echo "========================================"
    exit 1
fi