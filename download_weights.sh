#!/bin/bash
# 下载 Cosmos-Transfer2.5 全套权重文件
# 包含：Guardrail1、Transfer2.5-2B、Predict2.5-2B、Reason1-7B、Qwen3Guard、SigLIP

set -e

# 确保环境变量设置
export HF_HOME=/team/hyh/huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "📥 下载 Cosmos-Transfer2.5 全套权重"
echo "=========================================="
echo "🗂️  缓存目录: ${HF_HOME:-<default ~/.cache/huggingface>}"
# echo "🌐 镜像站点: $HF_ENDPOINT"
echo ""

# 检查是否在 conda 环境中
# if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
#     echo "⚠️  警告: 未检测到 conda 环境，请先激活环境"
#     echo "   运行: conda activate cosmos-transfer2.5"
#     exit 1
# fi

# echo "✅ 当前环境: $CONDA_DEFAULT_ENV"
echo ""

# ==================== 1. Cosmos-Guardrail1 ====================
echo "=========================================="
echo "📦 [1/6] nvidia/Cosmos-Guardrail1"
echo "=========================================="
hf download "nvidia/Cosmos-Guardrail1" \
    --repo-type model \
    --revision "d6d4bfa899a71454a700907664f3e88f503950cf"
echo "✅ Cosmos-Guardrail1 完成"
echo ""

# ==================== 2. Cosmos-Transfer2.5-2B ====================
echo "=========================================="
echo "📦 [2/6] nvidia/Cosmos-Transfer2.5-2B"
echo "=========================================="

echo "  → Multiview (auto/multiview)"
hf download "nvidia/Cosmos-Transfer2.5-2B" \
    --repo-type model \
    --revision "00c591edab119e8a6ca06e6e091351a04ce0ecc9" \
    --include "auto/multiview/*.pt"

echo "  → Edge (general/edge)"
hf download "nvidia/Cosmos-Transfer2.5-2B" \
    --repo-type model \
    --revision "b67b64abda3801a9aceddbff2bdb86126c06db74" \
    --include "general/edge/*.pt"

echo "  → Depth (general/depth)"
hf download "nvidia/Cosmos-Transfer2.5-2B" \
    --repo-type model \
    --revision "dea7737ca29dd8d9086413c6dc5724b8250a0bb4" \
    --include "general/depth/*.pt"

echo "  → Segmentation (general/seg)"
hf download "nvidia/Cosmos-Transfer2.5-2B" \
    --repo-type model \
    --revision "23057a4167b89de89a4a397fdbf3887994d115eb" \
    --include "general/seg/*.pt"

echo "  → Blur (general/blur)"
hf download "nvidia/Cosmos-Transfer2.5-2B" \
    --repo-type model \
    --revision "eb5325b77d358944da58a690157dd2b8071bbf85" \
    --include "general/blur/*.pt"

echo "✅ Cosmos-Transfer2.5-2B 完成"
echo ""

# ==================== 3. Cosmos-Predict2.5-2B ====================
echo "=========================================="
echo "📦 [3/6] nvidia/Cosmos-Predict2.5-2B"
echo "=========================================="

echo "  → Tokenizer"
hf download "nvidia/Cosmos-Predict2.5-2B" \
    --repo-type model \
    --revision "f176dc95b4a70f53ce01c4b302851595e7322b00" \
    "tokenizer.pth"

echo "  → Multiview backbone (auto/multiview)"
hf download "nvidia/Cosmos-Predict2.5-2B" \
    --repo-type model \
    --revision "865baf084d4c9e850eac59a021277d5a9b9e8b63" \
    --include "auto/multiview/*.pt"

echo "✅ Cosmos-Predict2.5-2B 完成"
echo ""

# ==================== 4. Cosmos-Reason1-7B ====================
echo "=========================================="
echo "📦 [4/6] nvidia/Cosmos-Reason1-7B"
echo "=========================================="
hf download "nvidia/Cosmos-Reason1-7B" \
    --repo-type model \
    --revision "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
echo "✅ Cosmos-Reason1-7B 完成"
echo ""

# ==================== 5. Qwen3Guard-Gen-0.6B ====================
echo "=========================================="
echo "📦 [5/6] Qwen/Qwen3Guard-Gen-0.6B"
echo "=========================================="
hf download "Qwen/Qwen3Guard-Gen-0.6B" \
    --repo-type model \
    --revision "fada3b2f655b89601929198343c94cd2f64d93cc"
echo "✅ Qwen3Guard-Gen-0.6B 完成"
echo ""

# ==================== 6. SigLIP ====================
echo "=========================================="
echo "📦 [6/6] google/siglip-so400m-patch14-384"
echo "=========================================="
hf download "google/siglip-so400m-patch14-384" \
    --repo-type model \
    --revision "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
echo "✅ SigLIP 完成"
echo ""

# ==================== 7. 离线可用性校验 ====================
echo "=========================================="
echo "🔎 [7/7] 关键文件离线校验"
echo "=========================================="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 hf download \
    "nvidia/Cosmos-Predict2.5-2B" \
    "tokenizer.pth" \
    --repo-type model \
    --revision "f176dc95b4a70f53ce01c4b302851595e7322b00" \
    --quiet >/dev/null
echo "✅ 离线校验通过: nvidia/Cosmos-Predict2.5-2B/tokenizer.pth"
echo ""

# ==================== 完成 ====================
echo "=========================================="
echo "🎉 所有权重文件下载完成！"
echo "=========================================="
echo ""
echo "📂 缓存位置: $HF_HOME/hub/"
echo ""
