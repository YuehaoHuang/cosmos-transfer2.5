#!/bin/bash
# Waymo 多视角推理脚本
# 使用方法: ./inference_waymo.sh [选项]
# 选项:
#   --input PATH     输入 spec JSON 路径 (必填，或设置 WAYMO_INPUT_FILE 环境变量)
#   --output DIR     输出目录 (默认: outputs/postrained-waymo-mv)
#   --checkpoint PT  checkpoint 文件路径 (必填，或通过 CHECKPOINT_DIR 环境变量指定)
#   --gpus N         使用 N 个 GPU (默认: 8)
#   --debug          启用调试模式 (等待 debugger 连接到端口 5678)

set -e

# ==================== 配置参数 ====================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12341}
EXPERIMENT="waymo_multiview_post_train"

INPUT_FILE="${WAYMO_INPUT_FILE:-}"
OUTPUT_DIR="outputs/postrained-waymo-mv"
CHECKPOINT_PATH="${CHECKPOINT_DIR:+${CHECKPOINT_DIR}/model_ema_bf16.pt}"

# ==================== 解析命令行参数 ====================
DEBUG_MODE=false

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
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            echo "🐛 调试模式已启用 (debugpy 将监听端口 5678)"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 --input PATH [--output DIR] [--checkpoint PT] [--gpus N] [--debug]"
            exit 1
            ;;
    esac
done



# ==================== 检查参数 ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 推理脚本 (Waymo)"
echo "========================================"
echo "📁 输入文件:   ${INPUT_FILE:-<未指定>}"
echo "📂 输出目录:   $OUTPUT_DIR"
echo "🧪 实验名称:   $EXPERIMENT"
echo "🎮 GPU 数量:   $NUM_GPUS"
echo "🔌 主端口:     $MASTER_PORT"
echo "========================================"

if [ -z "$INPUT_FILE" ]; then
    echo "❌ 错误: 未指定输入文件"
    echo "   请通过 --input /path/to/waymo_spec.json 或环境变量 WAYMO_INPUT_FILE 指定"
    exit 1
fi
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误: 未指定 checkpoint 路径"
    echo "   请通过 --checkpoint /path/to/model_ema_bf16.pt 或环境变量 CHECKPOINT_DIR 指定"
    exit 1
fi
echo "💾 Checkpoint:  $CHECKPOINT_PATH"
echo "========================================"

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "✅ Python 版本: $(python --version)"
echo ""

# ==================== 启动推理 ====================
if [ "$DEBUG_MODE" = true ]; then
    export COSMOS_DEBUG=1
    echo "🔍 设置环境变量: COSMOS_DEBUG=1"
fi

echo "🔧 使用 torchrun 启动分布式推理 ($NUM_GPUS GPUs)..."
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m examples.multiview -i $INPUT_FILE -o $OUTPUT_DIR --checkpoint_path $CHECKPOINT_PATH --experiment $EXPERIMENT"
echo "📝 执行命令: $CMD"
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m examples.multiview \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --experiment "$EXPERIMENT"

unset COSMOS_DEBUG

# ==================== 推理完成 ====================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 推理完成！"
    echo "📂 输出保存至: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ 推理失败，退出码: $?"
    echo "========================================"
    exit 1
fi
