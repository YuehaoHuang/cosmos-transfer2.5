#!/bin/bash
# Waymo 多视角后训练脚本
# 使用方法: ./train_waymo.sh [选项]
# 选项:
#   --debug     启用调试模式 (等待 debugger 连接到端口 5678)
#   --dryrun    空运行模式 (只打印配置，不实际训练)
#   --profile   启用性能分析
#   --gpus N    使用 N 个 GPU (默认: 8)

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
# GPU 数量
NUM_GPUS=${NUM_GPUS:-8}

# 训练配置
CONFIG_FILE="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
EXPERIMENT="waymo_multiview_post_train"

# 端口号（用于分布式训练）
MASTER_PORT=${MASTER_PORT:-12341}

# ==================== 解析命令行参数 ====================
DEBUG_MODE=false
DRYRUN_MODE=false
PROFILE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            echo "🐛 调试模式已启用 (debugpy 将监听端口 5678)"
            shift
            ;;
        --dryrun)
            DRYRUN_MODE=true
            echo "🏃 空运行模式已启用 (只打印配置)"
            shift
            ;;
        --profile)
            PROFILE_MODE=true
            echo "📊 性能分析已启用"
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            echo "🎮 请求使用 $NUM_GPUS 个 GPU"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--debug] [--dryrun] [--profile] [--gpus N]"
            exit 1
            ;;
    esac
done

# 调试模式强制使用单GPU（避免多进程端口冲突）
if [ "$DEBUG_MODE" = true ]; then
    if [ "$NUM_GPUS" -ne 1 ]; then
        echo "⚠️  调试模式自动设置为单GPU（原设置: $NUM_GPUS）"
        NUM_GPUS=1
    fi
fi

# ==================== 检查环境 ====================
echo "========================================"
echo "🚀 Cosmos-Transfer2.5 训练脚本"
echo "========================================"
echo "📁 配置文件: $CONFIG_FILE"
echo "🧪 实验名称: $EXPERIMENT"
echo "🎮 GPU 数量: $NUM_GPUS"
echo "🔌 主端口: $MASTER_PORT"
echo "========================================"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 找不到 Python"
    exit 1
fi

echo "✅ Python 版本: $(python --version)"
echo ""

# ==================== 启动训练 ====================
# 根据模式标志构建参数
TRAIN_ARGS=""
[ "$DRYRUN_MODE" = true ] && TRAIN_ARGS="$TRAIN_ARGS --dryrun"
[ "$PROFILE_MODE" = true ] && TRAIN_ARGS="$TRAIN_ARGS --profile"

# 调试模式通过环境变量控制（避免 argparse 参数顺序问题）
if [ "$DEBUG_MODE" = true ]; then
    export COSMOS_DEBUG=1
    echo "🔍 设置环境变量: COSMOS_DEBUG=1"
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单 GPU 训练
    echo "🔧 使用单 GPU 训练..."
    echo "📝 执行命令: python -m scripts.train$TRAIN_ARGS --config=\"$CONFIG_FILE\" -- experiment=\"$EXPERIMENT\" job.wandb_mode=disabled"
    python -m scripts.train \
        $TRAIN_ARGS \
        --config="$CONFIG_FILE" \
        -- \
        experiment="$EXPERIMENT" \
        job.wandb_mode=disabled
else
    # 多 GPU 分布式训练
    echo "🔧 使用 torchrun 启动分布式训练 ($NUM_GPUS GPUs)..."
    echo "📝 执行命令: torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m scripts.train$TRAIN_ARGS --config=\"$CONFIG_FILE\" -- experiment=\"$EXPERIMENT\" job.wandb_mode=disabled"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m scripts.train \
        $TRAIN_ARGS \
        --config="$CONFIG_FILE" \
        -- \
        experiment="$EXPERIMENT" \
        job.wandb_mode=disabled
fi

# 清理环境变量
unset COSMOS_DEBUG

# ==================== 训练完成 ====================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 训练完成！"
    echo "========================================"
    
    # 如果不是 dryrun，显示检查点位置
    if [ "$DRYRUN_MODE" = false ]; then
        CHECKPOINT_DIR="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints"
        echo "📦 检查点保存位置: $CHECKPOINT_DIR"
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo "📂 检查点列表:"
            ls -lh "$CHECKPOINT_DIR" | tail -n 10
        fi
    fi
else
    echo ""
    echo "========================================"
    echo "❌ 训练失败，退出码: $?"
    echo "========================================"
    exit 1
fi
