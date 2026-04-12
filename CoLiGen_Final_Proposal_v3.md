# CoLiGen: Consistency-First Joint Video and LiDAR Generation on World Foundation Models

> **Target Venue**: NeurIPS 2026 Main Track
> **Official Submission Milestones**: Abstract due May 4, 2026 (AOE); Full paper due May 6, 2026 (AOE)
> **Today / Remaining Time**: March 26, 2026 -> 39 days to abstract, 41 days to full paper
> **Backbone**: Cosmos Transfer 2.5 - 2B Rectified Flow DiT
> **Hardware Budget**: 8 x NVIDIA A100-80GB (single node)
> **Proposal Status**: Progress-updated revision v4.1

### 已完成里程碑

| 模块 | 状态 | 关键指标 |
|---|---|---|
| Waymo multiview video 后训练 | **训练中** — 11k/100k iter，loss 稳定 ~0.05 | 5-cam, 720p, 29 frames, HDMap control |
| Waymo LiDAR tokenizer 后训练 | **已完成** — 5k iter 收敛 | RMSE 0.72m, MAE 0.39m, RelErr 2% |
| Waymo latent 提取脚本 | **已就绪** | `extract_waymo_latents.py` 可用 |
| Waymo 数据 pipeline | **已就绪** | `WaymoMultiviewDataset`, 798 train / 202 val clips |
| Joint training 基础设施 | **已就绪** | `joint_training.py` 概率采样多 dataloader |

工程锚点：
- Video / multiview 主干：`cosmos_transfer2/multiview.py`
- Waymo 后训练配置：`cosmos_transfer2/experiments/multiview/waymo_posttrain.py`
- Waymo 数据加载：`cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/defaults/dataloader_local.py`
- Rectified-flow control model：`cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py`
- LiDAR tokenizer 后训练（已完成）：`/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/`
- LiDAR tokenizer checkpoint：`/data2/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/`
- Video 后训练 checkpoint：`/data/cosmos-transfer2.5/output/20260313_225508/`
- Joint training 框架：`cosmos_transfer2/_src/imaginaire/datasets/joint_training.py`
- Action + video 联合去噪参考：`third_party_ref_repo/dreamzero`

---

## 一、Executive Summary

### 1.1 One-Sentence Pitch

**CoLiGen 的目标不是单纯“同时生成 video 和 LiDAR”，而是让两者在几何、运动和时序上保持高一致性。** 我们将 Cosmos Transfer 2.5 的预训练 video world foundation model 改造成一个 consistency-first 的联合生成器，通过几何锚定 token 对齐、模态解耦噪声调度和跨模态一致性训练，在有限算力内实现单阶段 joint denoising，并把驾驶场景作为高精度标定的多模态试验台。

### 1.2 这篇论文必须回答的三个问题

1. 预训练 video WFM 能否在不明显牺牲视频质量的前提下，扩展到 video + LiDAR 联合生成？
2. 显式几何锚定的跨模态 token 对齐，是否能显著优于串联流水线和 naive latent 拼接？
3. 更高的跨模态一致性，是否会转化为更好的下游 3D 感知增益？

### 1.3 投稿导向的主张

这篇工作最适合的 NeurIPS 叙事不是“又一个自动驾驶生成系统”，而是：

**我们提出了一种将预训练 world foundation model 扩展到 geometry-consistent multimodal generation 的通用范式，自动驾驶只是因为具备高质量标定、多视角和 LiDAR 同步而成为理想验证场景。**

---

## 二、Problem Statement and Gap

### 2.1 任务定义

给定条件信号

$$
C = \{\text{BEV layout}, \text{3D boxes}, \text{HD map}, \text{camera poses}, \text{text prompt}\},
$$

目标是联合生成多视角视频序列 $V$ 与 LiDAR range-map / point-cloud 序列 $L$，同时优化：

$$
\text{Video Quality}, \quad \text{LiDAR Quality}, \quad \text{Cross-Modal Consistency}.
$$

真正困难的不是把两个模态都“做出来”，而是在以下三个层面同时成立：

- 同一时刻，LiDAR 投影到图像后与视觉几何一致。
- 连续时刻，Video 的运动与 LiDAR 的运动同步。
- 多视角视频之间一致，且与同一帧 LiDAR 保持统一世界状态。

### 2.2 现有方法为什么还不够

| 方法族 | 优点 | 对高一致性的关键短板 |
|---|---|---|
| 从头训练的联合生成 | 目标定义上是 joint | 缺少大规模视觉先验，训练代价高，视频质量和长时稳定性容易不足 |
| 串联式 Video -> LiDAR 流水线 | 视频质量通常较强 | LiDAR 误差被 video 误差绑定，无法在联合去噪中双向纠正 |
| 只做 multiview video 的 WFM | 强视觉和动态先验 | 没有 LiDAR 分支，无法优化跨模态一致性 |
| 独立训练的 LiDAR 生成器 | 几何可控 | 与 video 的联合状态并非端到端共享，容易出现“各自逼真但互相对不上” |

### 2.3 我们真正要填的空白

当前最有价值的空白不是“再造一个更大的 joint model”，而是：

**如何以小改动、低训练成本，把强大的预训练 video WFM 转化为一个几何一致的 joint video-LiDAR generator。**

这也是更符合当前 NeurIPS 审稿口味的点：不是纯工程堆砌，而是一个具有可迁移意义的模型适配范式。

---

## 三、Central Hypothesis and Paper Scope

### 3.1 Central Hypothesis

如果联合生成失败，根本原因通常来自两件事：

1. **缺少可靠的跨模态对应关系**：video token 与 LiDAR token 的语义和坐标系不同，直接拼接会把“纹理密集模态”和“几何稀疏模态”混在一起。
2. **训练过程对两个模态一视同仁**：统一噪声调度会让 LiDAR 的几何细节在高噪声阶段被过度破坏，也让 video 与 LiDAR 在不同难度区间里无法互补。

因此我们提出如下中心假设：

**只要在预训练 WFM 上补上“几何锚定的 token 对齐”与“模态感知的联合去噪训练”，就可以显著提升 video-LiDAR 几何一致性，而不需要从头训练一个全新的大模型。**

### 3.2 本文只押注三件核心创新

为了在剩余 41 天内把论文打磨到可投稿状态（基础设施已就绪），proposal 收束为三个必须成立的创新点：

1. **GALA**: Geometry-Anchored Latent Alignment
   用标定信息把 video token 与 LiDAR token 建立稀疏几何对应，而不是全局盲对齐。
2. **MDNS**: Modality-Decoupled Noise Scheduling
   在 rectified flow 中为 video 与 LiDAR 使用不同噪声难度分布，从训练一开始就让两个模态形成互补（不延后引入）。
3. **Consistency-Centric Evaluation**
   不再只报告 FVD 和 Chamfer Distance，而是把 depth reprojection (DAS)、跨模态运动同步 (CME) 和下游 3D 感知增益作为主结果的一部分。

### 3.3 不是本文主线的内容

以下内容如果时间足够可以做，但不应成为主叙事：

- 重新设计 LiDAR tokenizer
- 大规模从头联合预训练
- 7-camera 长时序全量训练
- 复杂的新位置编码家族作为核心 novelty

这些都可以保留为附加实验或工程增强，但不应稀释主线。

---

## 四、Method

### 4.1 整体思路

```
Condition C
  = BEV / 3D boxes / HDMap / camera poses / text
                     |
                     v
      Frozen Cosmos video VAE + frozen LiDAR tokenizer
                     |
                     v
       video latents z_v     lidar latents z_l
                |                  |
                +------ GALA ------+
                          |
                          v
      Cosmos Transfer 2.5 joint rectified-flow DiT
      with shared denoiser + gated cross-modal blocks
                          |
                          v
          video decode                lidar decode
                          |
                          v
       multiview video sequence + lidar point-cloud sequence
```

核心原则是：**尽量复用 Cosmos Transfer 2.5 的视觉与时序先验，只在必须的位置插入跨模态结构。**

### 4.2 Joint Latent Formulation

- **Video 分支**：使用 Cosmos 原生 video VAE 编码多视角视频得到 $z_v$。
- **LiDAR 分支**：使用 Cosmos LiDAR tokenizer (CI8x8) 编码 range map 序列得到 $z_l$。
- **共享主干**：将两个模态送入同一个 rectified-flow DiT，而不是两个独立生成器级联。
- **最小化改造策略**：冻结 VAEs，主干以 LoRA + 小模块增量为主，避免 2B 主干全量微调失控。

#### 4.2.1 Latent Space 兼容性问题（关键工程难点）

Video VAE 和 LiDAR tokenizer 的 latent 空间在**通道维度、空间分辨率、时序维度和分布**上均不同。直接拼接到同一个 DiT 是不可行的。

**两个 tokenizer 的参数对比：**

| | Video VAE (WAN2.1) | LiDAR Tokenizer (CI8x8) |
|---|---|---|
| 时间压缩率 | **4x** | **1x / 无时间压缩** |
| 空间压缩率 | 8x8 | 8x8 |
| 29 像素帧 → latent 时间维度 | `state_t = (29-1)/4+1 = 8` | `T_l = 29`（逐帧 latent） |
| 通道数 | 16 | 16 (待确认) |

**核心难点不是“29 帧无法整除”，而是两个模态的 latent 时钟粒度不同**：video 分支在 latent 空间里是 8-step 低频时钟，LiDAR 分支保留 29-step 高频时钟。解决方案应避免去改 video backbone 的 chunk 长度，而是在架构里显式处理双时钟。

- **方案 1：固定 video 主时钟 + LiDAR temporal adapter**
  保持 video `29 frames / state_t=8` 不变。先用 CI8x8 对 29 帧 LiDAR 逐帧编码，再用轻量 1D temporal adapter（例如 `kernel=5, stride=4, padding=2` 的 depthwise conv 或 attention pooling）将 29 个 LiDAR latent 时间步压到 8 个对齐时槽，供 shared DiT 使用。

- **方案 2：双时钟交互**
  Video 保持 8 步，LiDAR 保持 29 步，只在 GALA 中做带时间索引的稀疏 cross-attention，而不要求进入主干前先压成同长。几何上更细，但实现和显存都更重。

- **方案 3：改 backbone chunk 长度**
  把 video 改为 33 帧 / `state_t=9` 或 25 帧 / `state_t=7`，让两个模态在时间长度上“看起来更整齐”。代价是偏离现有 Waymo post-train checkpoint 的 `29f/t8` 分布，需要重新验证位置编码、采样、稳定性和已有 checkpoint 的可迁移性。

**推荐方案 1**：固定 `29f/t8` 作为主干时钟，LiDAR 用原生 29 帧提取 latent，再通过 temporal adapter 对齐到 8。这样既保留现有 checkpoint 的时序先验，也不会把 joint training 变成二次 backbone 时长适配。方案 2 作为精度优先 fallback；方案 3 不建议作为首选。

**通道与时间维度对齐：**

在 LiDAR latent 进入 DiT 前，用一个轻量 LiDAR adapter 先做 `29 -> 8` 的时间对齐，再用 1x1 Conv 或小 MLP 将 $z_l$ 投影到与 $z_v$ 相同的通道维度。进入 shared DiT 的 LiDAR token 序列与 video 一样保持 8 个时间槽，通过 modality embedding 区分。

#### 4.2.2 时序维度约束

Cosmos multiview DiT 沿时序维度拼接多视角：latent shape 为 `B C (V*state_t) H W`，其中 `state_t=8` 对应 29 像素帧（temporal compression = 4）。DiT 内部的 3D temporal attention 期望看到完整序列。因此：

- **不能用单帧训练** — temporal attention 在 length=1 上运行会严重偏离预训练分布
- **最小训练单元是 `state_t` 个 latent 帧** — 这是 DiT 的原子操作粒度
- 当前 Waymo video checkpoint 已在 `29f/t8` 上后训练，**joint training 应把 video latent grid 视为主时钟**
- LiDAR 不需要把 raw frame 数改成“公共倍数”，而是通过 temporal adapter / 时间索引映射对齐到这 8 个 video latent 时槽

### 4.3 GALA: Geometry-Anchored Latent Alignment

#### 4.3.1 为什么 GALA 是第一优先级

直接拼接 latent 的问题不在于“维度不一样”，而在于**世界坐标没有对齐**。  
Video token 关注纹理与外观，LiDAR token 关注深度和几何；如果没有显式对应关系，联合注意力更容易学到统计共现，而不是同一物体、同一表面的跨模态约束。

#### 4.3.2 核心做法

利用已知相机内外参与 LiDAR 标定，将 video latent 的空间位置投影到 LiDAR range image，建立稀疏对应：

$$
\hat{z}_l = z_l + \alpha \cdot \mathrm{CrossAttn}(Q=f_q(z_l), K=f_k(\Pi_{v \rightarrow l}(z_v)), V=f_v(\Pi_{v \rightarrow l}(z_v)))
$$

反向也做一次轻量路由：

$$
\hat{z}_v = z_v + \beta \cdot \mathrm{CrossAttn}(Q=g_q(z_v), K=g_k(\Pi_{l \rightarrow v}(z_l)), V=g_v(\Pi_{l \rightarrow v}(z_l)))
$$

其中：

- $\Pi_{v \rightarrow l}$ / $\Pi_{l \rightarrow v}$ 为基于标定的投影算子。
- $\alpha, \beta$ 采用 zero-init gate，确保训练初期不破坏预训练主干。
- 只在有效投影区域做稀疏 cross-attention，显存更可控，也更符合几何先验。
- 时间上只在匹配的局部时间窗口内做路由：每个 video latent step 只与其对应的 LiDAR temporal adapter 输出或邻域帧交互，而不是对全 29 帧全局注意。

#### 4.3.3 Fallback 机制

现实场景中总会存在无法精确对齐的区域，例如：

- 天空与远处弱回波区域
- 运动物体边界处的标定误差
- 遮挡和 LiDAR 稀疏采样导致的空洞

因此 GALA 采用双通道机制：

- 有效区域：几何锚定 cross-attention
- 无效区域：弱全局 attention fallback

这使得模型既不完全依赖 noisy projection，也不会退化为纯统计对齐。

### 4.4 MDNS: Modality-Decoupled Noise Scheduling

#### 4.4.1 动机

Video 与 LiDAR 的学习难度分布并不相同：

- Video 更依赖外观、纹理和长时外观稳定性
- LiDAR 更依赖稀疏几何、深度边界和距离结构

统一噪声调度会导致一个常见失败模式：

- 当 video 仍可恢复时，LiDAR 已经被破坏得过深；
- 当 LiDAR 刚开始收敛时，video 分支却开始过拟合低频外观。

#### 4.4.2 做法

在 rectified-flow 训练中，为两个模态独立采样噪声级别：

$$
z_v^{t_v} = (1 - t_v) z_v^0 + t_v \epsilon_v, \quad t_v \sim p_v(t)
$$

$$
z_l^{t_l} = (1 - t_l) z_l^0 + t_l \epsilon_l, \quad t_l \sim p_l(t)
$$

其中：

- $p_v(t)$ 保持与 Cosmos 预训练兼容的分布；
- $p_l(t)$ 使用向低噪声偏移的可学习 logit-normal，以保留 LiDAR 几何细节。

#### 4.4.3 直觉

MDNS 的本质不是“给 LiDAR 少加点噪声”，而是让模型在训练中反复看到这样的状态：

- video 比较模糊，但 LiDAR 仍保留可用几何
- LiDAR 较稀疏，但 video 仍保留外观上下文

这样 joint denoiser 才有机会学到真正的跨模态互补恢复。

### 4.5 Shared Denoiser with Lightweight Cross-Modal Blocks

基于现有 Cosmos Transfer 2.5 rectified-flow 主干，采用最小改造：

- 共享 DiT 主干，不复制一个完整 second backbone
- 每隔 $k$ 层插入轻量 gated cross-modal block
- 使用 modality-aware AdaLN 或等价条件缩放，避免 video 与 LiDAR 完全共享归一化统计
- 新增模块尽量参数高效，以 LoRA 与小 adapter 为主

这样的方法更适合当前代码基础，也更符合 8 x A100 的资源约束。

### 4.6 Consistency Losses and Curriculum

#### 4.6.1 几何一致性损失

把生成的 LiDAR 投影回相机平面，得到投影深度 $D_{\mathrm{proj}}$，再与视频分支估计的深度 $D_{\mathrm{img}}$ 对齐：

$$
\mathcal{L}_{\mathrm{geo}} = \| D_{\mathrm{proj}} - \mathrm{scale\_align}(D_{\mathrm{img}}) \|_1
$$

这项 loss 直接约束“同一时刻同一位置”的跨模态几何一致性。

#### 4.6.2 时序一致性损失

对相邻帧计算 video optical flow 与 LiDAR 投影运动场的一致性：

$$
\mathcal{L}_{\mathrm{temp}} = \| \mathrm{Flow}_{v}(t, t+1) - \Pi(\mathrm{Flow}_{l}(t, t+1)) \|_2
$$

这项损失用于抑制最常见的错误：  
静态背景在 video 中看起来稳定，但在 LiDAR 中抖动；或者动态物体在两个模态里位移不一致。

#### 4.6.3 训练课程（两阶段）

由于 Cosmos DiT 的最小操作粒度是 `state_t=8`（29 帧），不能用单帧训练。采用两阶段 curriculum：

1. **Stage 1: Joint Alignment（几何对齐 + 联合去噪）**
   - 使用完整序列（video 固定为 `state_t=8` / 29 帧），5 cameras + 29 帧 LiDAR；LiDAR 经 temporal adapter 对齐到 8 个 joint 时槽
   - GALA + MDNS **从一开始就启用**（MDNS 仅改 noise schedule，无额外参数开销）
   - 仅使用 $\mathcal{L}_{\mathrm{flow}}^{v} + \mathcal{L}_{\mathrm{flow}}^{l} + \lambda_1 \mathcal{L}_{\mathrm{geo}}$
   - 不引入 temporal loss（避免 noisy motion supervision 干扰早期收敛）

2. **Stage 2: Temporal Consistency（时序一致性精调）**
   - 在 Stage 1 checkpoint 上继续训练
   - 加入 $\mathcal{L}_{\mathrm{temp}}$，使用 cosine LR decay
   - 训练到收敛

**为什么不做三阶段：**
- 原 Stage A “单帧”与 video DiT 的 3D temporal attention 不兼容，会导致分布漂移
- 原 Stage C “低学习率收尾”本质上只是 LR decay，不是独立的 curriculum 阶段，用 cosine schedule 在 Stage 2 内部即可实现
- 两阶段减少了 checkpoint 切换的工程开销和调参复杂度

#### 4.6.4 总目标

$$
\mathcal{L} =
\mathcal{L}_{\mathrm{flow}}^{v}
+ \mathcal{L}_{\mathrm{flow}}^{l}
+ \lambda_1 \mathcal{L}_{\mathrm{geo}}
+ \lambda_2 \mathcal{L}_{\mathrm{temp}}
$$

其中 $\mathcal{L}_{\mathrm{temp}}$ 仅在 Stage 2 开启。MDNS 在两个阶段均生效。

---

## 五、Current Progress and Remaining Engineering

### 5.1 已完成的基础设施（不再需要投入时间）

| 模块 | 状态 | 具体成果 |
|---|---|---|
| Waymo multiview video 后训练 | **进行中，预计 4 月上旬完成** | 5-cam 720p 29f, iter 11k/100k, loss ~0.05 稳定下降，checkpoints 每 1k iter 保存 |
| Waymo LiDAR tokenizer | **已完成** | RMSE 0.72m, MAE 0.39m, RelErr 2%；JIT 编译的 encoder/decoder 可直接调用 |
| `WaymoMultiviewDataset` | **已完成** | 支持 5 视角帧对齐、caption 加载、HDMap control input |
| Joint training 框架 | **已完成** | `create_dataloader_dict()` + 概率采样，可直接用于 video+LiDAR 混合 batch |
| Latent 提取 | **已就绪** | `extract_waymo_latents.py` 支持分布式提取，输出 `.pt` 格式 |
| 推理 pipeline | **已完成** | `inference_waymo_autoregressive.sh` 支持自回归多 chunk 生成 |
| 数据质量检测 | **已完成** | `check_waymo_samples.py` 并行验证视频完整性 |

### 5.2 剩余需要实现的核心模块（按优先级排序）

由于基础设施已基本就绪，剩余工作可以聚焦在**纯方法层**：

1. **LiDAR latent 提取与时间对齐**（~2 天）
   用已训好的 LiDAR tokenizer（CI8x8，逐帧编码）对 Waymo 训练集提取 latent。
   **关键决策**：固定 video `29f/t8` 不变，并确定 LiDAR `29 -> 8` 的 temporal adapter / 时间映射方案。

2. **LiDAR adapter（latent space 桥接）**（~2 天）
   Video VAE（temporal 4x）和 LiDAR tokenizer（逐帧 CI8x8）的 latent 在通道/分辨率/时间粒度上均不同。需要：
   - 通道维度：轻量 adapter（1x1 Conv）投影到 DiT token 维度
   - 时间维度：用 temporal adapter 将 LiDAR 从 29 个逐帧 latent 对齐到 8 个 video latent 时槽
   这是 joint training 能跑通的前提。

3. **GALA 模块 + MDNS 实现**（~5 天，合并实现）
   GALA：稀疏投影索引（利用 Waymo 标定）、validity mask、gated cross-attention。
   MDNS：在 rectified-flow noise schedule 中为 video/LiDAR 分别采样 $t_v, t_l$。改动量小，与 GALA 一起实现。
   起点：在 `MultiviewControlVideo2WorldModelRectifiedFlow` 的训练 step 中插入。

4. **Joint rectified-flow model**（~4 天）
   扩展现有 model 接收 video + LiDAR 双 latent token stream。
   关键约束：必须保持 `state_t=8` 的时序粒度，LiDAR token 拼入 attention window。
   利用现有 `joint_training.py` 框架组织 batch。

5. **评测脚本**（~3 天）
   DAS（depth reprojection）、CME（跨模态运动一致性）、下游感知评测。
   可复用 `packages/cosmos-oss/vqa/` 中的 LPIPS/PSNR/SSIM 框架。

### 5.3 关键优势：Waymo 后训练为 joint training 铺路

与原方案相比，Waymo video 后训练的完成带来三个关键优势：

1. **DiT 主干已适配 Waymo 域**：joint training 时不再需要同时完成域迁移和跨模态对齐两件事，降低训练难度。
2. **LiDAR tokenizer 已在同一数据域验证**：video 和 LiDAR 共享相同的场景分布和标定体系，避免跨数据集对齐噪声。
3. **数据 pipeline 已打通**：`WaymoMultiviewDataset` + LiDAR range map 加载只需要在 joint dataloader 层面组合，不需要从头构建。

### 5.4 不建议在第一轮就做的工程冒险

- 不建议一开始就 full fine-tune 2B backbone（Waymo 后训练 checkpoint 已是很好的起点）
- 不建议同时引入过多新损失和新编码器
- 不建议第一轮就做长序列自回归联合生成（先验证单 chunk 29 帧）

第一轮目标是尽快得到一个**明确优于串联 baseline 的一致性曲线**，而不是一次性做完所有 fancy 设计。

---

## 六、Training Plan Under 8 x A100

### 6.1 前置条件（已完成 / 进行中）

| 前置项 | 状态 | 对 joint training 的影响 |
|---|---|---|
| Waymo video 后训练 | 进行中，~11k iter | 完成后提供域适配的 DiT checkpoint，作为 joint training 的初始化权重 |
| Waymo LiDAR tokenizer | 已完成 | 直接用于 LiDAR latent 编解码，冻结不动 |
| Waymo LiDAR latent 提取 | 待执行（~1 天） | 用已训好 tokenizer 离线提取所有训练集 latent |

### 6.2 两阶段联合训练 + 轻量 ablation（从 Waymo video checkpoint 出发）

#### 主训练

| Stage | 目标 | 起始权重 | 冻结策略 | 训练模块 | 数据规模 | Loss | 预估成本 |
|---|---|---|---|---|---|---|---|
| **1** | Joint Alignment | Waymo video ckpt + frozen LiDAR tokenizer | 冻结 VAEs + DiT 主干 | GALA + LiDAR adapter + LoRA + MDNS | 5-cam x 29f paired clips，LiDAR 对齐到 8 时槽 | $\mathcal{L}_{\mathrm{flow}}^{v,l} + \mathcal{L}_{\mathrm{geo}}$ | ~500-700 A100-h |
| **2** | Temporal Consistency | Stage 1 ckpt | 同 Stage 1 | 同 Stage 1 | 同 Stage 1 | 加入 $\mathcal{L}_{\mathrm{temp}}$, cosine LR decay | ~400-500 A100-h |

#### Ablation 预算（关键问题：原方案未计入）

5 个完整 ablation × 完整训练 = ~5000+ A100-h，远超预算。采用**缩短版 ablation**策略：

| Ablation | 做法 | 预估成本 |
|---|---|---|
| w/o GALA | 仅跑 Stage 1 的 50% iter，对比 DAS 趋势 | ~250 A100-h |
| w/o MDNS | 仅跑 Stage 1 的 50% iter，对比 DAS 趋势 | ~250 A100-h |
| w/o $\mathcal{L}_{\mathrm{geo}}$ | 仅跑 Stage 1 的 50% iter，对比 DAS | ~250 A100-h |

**总预算：主训练 ~900-1200 + ablation ~750 = ~1650-1950 A100-h**（8×A100 约 8-10 天）。

> 注：5-cam 比 3-cam 增加约 40% 计算量，但与 Waymo 后训练配置一致，避免了视角子集选择带来的额外实验变量。

#### 为什么不再做三阶段

| 原方案 | 问题 | 修改 |
|---|---|---|
| Stage A: 单帧 | DiT 的 `state_t=8` 要求最少 29 帧输入，单帧导致 temporal attention 分布漂移 | 直接用 29 帧 |
| Stage B: 短序列 | MDNS 不需要延后到 Stage B，它只是 noise schedule 的改动 | MDNS 从 Stage 1 开始 |
| Stage C: 低 LR 收尾 | 不是独立 curriculum，cosine schedule 在 Stage 2 内部即可 | 合并到 Stage 2 |

### 6.3 显存与算力控制策略

- Gradient checkpointing 默认开启（现有训练已使用）
- BF16 主干 + FP32 loss 计算（与现有 Waymo 后训练一致）
- ZeRO-2 或等价优化器分片
- 直接使用 5-camera（与 Waymo 后训练一致），如果显存紧张再降帧数
- **LiDAR latent 离线预提取**，不在训练 loop 中做 tokenize，节省显存
- 投影索引离线预计算，避免每 step 动态构图
- 主干尽量 LoRA 化，控制新增参数规模

### 6.4 Stage 1 最小可验证设置

- `5 cameras x 29 frames`（与 Waymo 后训练完全一致：front / front_left / front_right / side_left / side_right，video `state_t=8`）
- `29-frame LiDAR sequence -> 8 aligned LiDAR slots`（通过 temporal adapter 对齐到 video latent 时钟）
- `720p`（与 Waymo 后训练分辨率一致，避免分辨率迁移开销）
- 冻结 Waymo 后训练的 DiT 主干，仅训 GALA + LiDAR adapter + LoRA
- MDNS 从一开始启用
- 仅使用 $\mathcal{L}_{\mathrm{geo}}$

这个设置足以验证最关键的结论：
**几何锚定对齐是否真的提升一致性。**

如果 Stage 1 中 DAS 明确优于 cascade 和 naive joint baseline，则方法核心成立，可以进入 Stage 2 和 ablation。

---

## 七、Experiment Plan: Consistency Is the Main Result

### 7.1 数据集与使用原则

| 数据集 | 角色 | 原因 |
|---|---|---|
| **Waymo Open** | **主实验** | 已完成 video 后训练 + LiDAR tokenizer 后训练，数据 pipeline 完全打通，798 train / 202 val clips，5-cam 720p 高分辨率，标定精度高 |
| nuScenes | 泛化验证 | 公开标准 benchmark，6-cam + LiDAR，可作为跨数据集泛化实验 |
| Cosmos-Drive-Dreams 配对数据 | 训练补充（可选） | 与 Cosmos 生态一致，可作为合成数据增强对比 |

原则更新：

- **主论文以 Waymo 为主战场** — 因为 video 后训练和 LiDAR tokenizer 都已在 Waymo 上完成，切换到 nuScenes 意味着两套基础设施都要重做，时间不允许
- **nuScenes 作为泛化验证** — 如果时间允许，在 nuScenes 上做少量实验证明方法可迁移
- Waymo 的标定精度和分辨率实际上更适合验证 geometry consistency 这一核心主张

### 7.2 Baselines 只保留最能说明问题的三类

1. **Cascade baseline**  
   `Cosmos multiview video -> LiDAR generator`
2. **Naive joint baseline**  
   同一主干、直接 token concat、无几何对齐
3. **CoLiGen**

如果公开可复现实验足够顺利，可再补充：

- 从头训练 joint model 的参考结果
- 公共工作中最强可复现方法

但论文成败不能依赖这些高风险外部 baseline。

### 7.3 评估指标

#### Video quality

- FVD
- FID
- CLIP-SIM

#### LiDAR quality

- Chamfer Distance
- JSD
- MMD

#### Cross-modal consistency

- **DAS**: Depth Alignment Score  
  LiDAR 投影深度与 video depth 的一致性
- **CME**: Cross-Modal Motion Error  
  相邻帧 video flow 与 LiDAR 投影运动场差异
- **Reproj-Edge Precision**  
  LiDAR 深度边界投影到图像边缘的对齐精度

#### Downstream utility

- 3D Detection mAP
- Tracking AMOTA

### 7.4 主实验必须长什么样

#### Exp 1: Main comparison on Waymo

主结果表应该同时包含三类指标：

- Video 质量
- LiDAR 质量
- Consistency 指标

论文不能只证明“生成得更清楚”，而必须证明：

**在 video / LiDAR 各自质量基本不退步的前提下，一致性显著变强。**

#### Exp 2: Consistency ablation

在算力约束（~600 A100-h ablation 预算）下，采用**缩短版 ablation**：每个 ablation 只跑 Stage 1 的 50% iter，比较 DAS 趋势。

必须做的三个 ablation（各 ~200 A100-h）：

| 变体 | 作用 | 预期信号 |
|---|---|---|
| w/o GALA（= naive joint） | 验证几何锚定是否必要 | DAS 应明显下降 |
| w/o MDNS | 验证模态解耦训练是否必要 | DAS 有下降但可能不如去 GALA 明显 |
| w/o $\mathcal{L}_{\mathrm{geo}}$ | 验证单靠 joint denoising 是否足够 | DAS 下降；如果不下降则说明 $\mathcal{L}_{\mathrm{geo}}$ 不重要 |

可选 ablation（如有余力）：

| 变体 | 作用 |
|---|---|
| w/o temporal loss | 对比 Stage 2 vs Stage 1 的 CME 差异即可说明，不需要独立 ablation run |
| GALA 单向 vs 双向 | 如果双向 GALA 显存紧张，这个 ablation 可证明单向也够用 |

#### Exp 3: Data efficiency

比较在 25%、50%、100% 训练数据下：

- cascade baseline
- naive joint
- CoLiGen

如果 CoLiGen 的优势在小数据 regime 更明显，这会很符合“预训练 WFM 适配”的叙事。

#### Exp 4: Downstream augmentation

比较三种合成数据增强：

- cascade 生成数据
- independent 生成数据
- CoLiGen 联合生成数据

如果在真实验证集上带来稳定的检测或跟踪增益，这将极大提高 paper 的说服力。

### 7.5 成功标准

为了避免项目目标失焦，建议用以下 success bar 做 go / no-go：

1. 相比 cascade，DAS 明显提升，CME 明显下降。
2. Video FVD 与 LiDAR CD 至少不显著退化。
3. 至少一个下游任务出现稳定增益。

换句话说，本项目不是要在每个单项指标都 SOTA，而是要证明：

**联合一致性可以被显式优化，并且这种优化有实际收益。**

---

## 八、41-Day Submission Backward Plan（Progress-Adjusted）

### 8.1 关键日期

- `2026-05-04 AOE`: NeurIPS 2026 abstract deadline
- `2026-05-06 AOE`: NeurIPS 2026 full paper deadline

从 `2026-03-26` 开始，实际只有约 41 天。但**基础设施已基本就绪**，比原计划节省了约一周。

### 8.2 当前状态与剩余依赖

```
[已完成] Waymo LiDAR tokenizer 后训练 (RMSE 0.72m)
[进行中] Waymo video 后训练 (11k/100k iter) ──> 预计 ~Apr 5 完成
[待做]   LiDAR latent 离线提取 ──> 依赖 LiDAR tokenizer (已完成)，可立即开始
[待做]   GALA 实现 ──> 不依赖 video 训练完成，可并行开发
[待做]   Joint model ──> 依赖 video checkpoint + GALA
```

### 8.3 倒排计划（两阶段训练版）

| 日期 | 工程目标 | 论文目标 | Exit Criterion |
|---|---|---|---|
| **Mar 26 - Apr 2** | (1) **确定时序对齐方案**（固定 `29f/t8`，敲定 LiDAR `29 -> 8` temporal adapter 与时间映射）; (2) 离线提取全部 Waymo LiDAR latents; (3) 实现 LiDAR adapter（通道 + 时间步对齐）; (4) 实现标定投影算子 $\Pi_{v \leftrightarrow l}$; (5) 构建 5-cam joint dataloader; (6) 跑通 cascade baseline 推理 | 定稿 paper story 与图表模板 | joint batch 可可视化且 video/LiDAR latent shape 对齐通过 |
| **Apr 3 - Apr 10** | (1) 实现 GALA 模块 + MDNS; (2) 启动 Stage 1 训练（GALA + MDNS + $\mathcal{L}_{\mathrm{geo}}$，5-cam x 29f）; (3) 同时启动 naive joint baseline 训练; (4) 实现 DAS/CME 评测脚本 | 写 Method 初稿 | Stage 1 训练启动，naive joint baseline 有初步 DAS |
| **Apr 11 - Apr 18** | (1) Stage 1 训练完成，评估 DAS/CME; (2) 启动 Stage 2（加 $\mathcal{L}_{\mathrm{temp}}$）; (3) 同时启动 3 个缩短版 ablation（各 50% iter） | 整理主表格字段与 ablation 模板 | CoLiGen DAS 优于 cascade 和 naive joint |
| **Apr 19 - Apr 25** | (1) Stage 2 训练完成; (2) 3 个 ablation 完成; (3) 启动下游 3D detection 评测; (4) 生成全部定量结果 | 写 Experiments 和 Intro | 主表 + ablation 表 + DAS/CME 曲线图已成型 |
| **Apr 26 - May 1** | (1) 全量评测（FVD/FID/CD/JSD + DAS/CME）; (2) qualitative 可视化（depth reprojection overlay, point cloud 对比）; (3) 如有余力做 nuScenes 泛化 | 完成全文初稿、摘要、supp 草稿 | 论文主体完整可审 |
| **May 2 - May 3** | 补 failure case 分析、整理 supp 视频 | 润色全文、确认格式 | 论文可提交 abstract |
| **May 4** | 提交 abstract | 固定标题、作者、摘要 | abstract submitted |
| **May 5 - May 6** | 最后补实验、查缺补漏、整理补充材料 | 提交 final PDF + supp | paper submitted |

### 8.4 并行化策略与 GPU 调度

```
Week 1 (Mar 26 - Apr 2): 时序对齐决策 + 基础设施收尾
  Day 1:  确定时序对齐方案（固定 `29f/t8`，确定 LiDAR `29 -> 8` temporal adapter）
  GPU:    Waymo video 后训练继续跑（8xA100 后台不中断）
  GPU:    LiDAR latent 提取（按 29 帧逐帧 latent 方案）
  CPU:    LiDAR adapter + 投影算子 + 5-cam joint dataloader 实现

Week 2 (Apr 3 - Apr 10): 方法实现 + 首轮训练
  GPU (8xA100): Stage 1 训练（CoLiGen full, 5-cam）
  GPU (间隙):   naive joint baseline 训练
  CPU:          GALA/MDNS 调试、评测脚本实现

Week 3 (Apr 11 - Apr 18): 主训练 + ablation 并行
  GPU-A (4xA100): Stage 2 训练
  GPU-B (4xA100): 3 个缩短版 ablation 串行跑（各 ~2 天）

Week 4 (Apr 19 - Apr 25): 评测 + 补实验
  GPU:  下游 3D detection 训练/评测
  GPU:  最终模型 5-cam 全量推理生成
  CPU:  写论文、做图表
```

**关键路径**：`时序对齐决策(Day 1) → video 后训练完成 → Stage 1 → Stage 2 → 全量评测`，约 19 天（Apr 3 - Apr 22）。Ablation 与 Stage 2 并行，不增加关键路径长度。

### 8.5 必须冻结的范围

- **Apr 10**：冻结主架构（GALA + MDNS + joint denoiser），不再新增核心模块
- **Apr 25**：冻结实验范围，不再增加新 benchmark 或 baseline

---

## 九、Risk Register and Fallback Strategy

### 9.1 风险一：Waymo video 后训练未收敛或质量不足

**风险**：当前 11k/100k iter，如果后续 loss 停滞或生成质量不够，joint training 的起点就不够好。
**概率**：低（loss 已稳定下降至 ~0.05，与预期一致）
**兜底**：

- 即使不到 100k iter，只要生成质量可接受就可提前停止（例如 30-50k iter）
- 保留用更早 checkpoint（如 iter 11k）做快速 joint training 原型验证的选项
- 必要时降低分辨率到 384p 减少后训练时间

### 9.2 风险二：联合训练导致 video 质量掉得太多

**风险**：跨模态交互太强，破坏了 Waymo 后训练的视觉先验。
**概率**：中（因为从域适配 checkpoint 出发，比从通用预训练出发更稳健）
**兜底**：

- 更强冻结主干（只允许 LoRA + GALA 可训练）
- 减少 cross-modal block 插入频率
- 先把 GALA 只做单向 `video -> LiDAR`，再尝试双向

### 9.3 风险三：时序 loss 噪声太大

**风险**：光流或投影运动场本身不稳定，训练发散。
**概率**：中
**兜底**：

- 将 temporal loss 从训练项降级为评测项
- 先确保 geometry consistency 明显提升
- 保留短序列可视化证明时序收益

### 9.4 风险四：5-camera + LiDAR 联合训练显存不够

**风险**：Waymo 5-cam x 29 frames 的 video latent + 全帧 LiDAR latent 同时进入 DiT，超出 8xA100 显存。
**概率**：中（Waymo video 后训练已验证 5-cam x 29f 可在 8xA100 上跑通，新增的 LiDAR latent + GALA 模块是增量）
**兜底**：

- 降低 batch size（从 1 降到 gradient accumulation）
- LiDAR latent 不拼入主 attention，改用轻量 cross-attention（路径 B fallback）
- 减少 GALA cross-modal block 的插入频率

### 9.5 风险五：Waymo 上缺乏公开可比 baseline

**风险**：Waymo 上没有现成的 video-LiDAR joint generation baseline 可直接对比。
**概率**：高（这是事实）
**兜底**：

- 以 **self-contained ablation** 为主论证策略：cascade vs naive joint vs CoLiGen 三者均由我们自己实现
- 重点放在 consistency 指标的 ablation 上，而不是与外部方法的绝对数值比较
- 如果时间允许，在 nuScenes 上补充与公开方法的对比

### 9.6 风险六：下游增益不稳定

**风险**：一致性变好了，但 detection / tracking 提升不够稳定。
**概率**：中
**兜底**：

- 把下游增益作为 “supporting evidence”，不是唯一结论
- 强化主论文对 consistency metrics 的定义与可视化
- 如果任务增益有限，也要证明 calibration-consistent generation 的质量价值

---

## 十、Paper Positioning for NeurIPS

### 10.1 论文应该怎么讲

最推荐的叙事顺序：

1. 预训练 world foundation models 很强，但今天几乎仍停留在单模态或弱联合。
2. Video-LiDAR 联合生成真正缺的是一致性，而不是单模态样本质量。
3. 一致性的关键在于几何锚定对齐与模态感知去噪训练。
4. 在一个预训练 WFM 上做 parameter-efficient 适配，就足以显著改善 joint consistency。

### 10.2 论文不应该怎么讲

不建议把主线写成：

- “我们堆了很多模块，所以效果更好”
- “自动驾驶数据集上又多了一个生成系统”
- “我们可能是首个做这个的工作”

这些说法都不如以下表述稳健：

**我们提出了一种低成本、可复用、对一致性有显式归因的 WFM joint-generation 适配方法。**

### 10.3 可选标题方向

可作为后续标题候选：

1. `CoLiGen: Consistency-First Joint Video-LiDAR Generation on World Foundation Models`
2. `Geometry-Anchored Joint Denoising for Video-LiDAR Generation`
3. `Adapting World Foundation Models to Geometry-Consistent Multimodal Generation`

其中第 1 个最适合作为项目名，第 3 个最适合强化 NeurIPS 的通用性叙事。

---

## 十一、Final Recommendation（Updated）

### 11.1 战略优势回顾

当前项目状态比原计划**好于预期**：

- Waymo video 后训练和 LiDAR tokenizer 后训练的完成意味着**基础设施阶段已经结束**，剩余 41 天可以全部投入在核心方法（GALA + MDNS）和实验上。
- 主数据集从 nuScenes 切换到 Waymo 是自然选择 — 不是因为 Waymo 更好，而是因为**所有前置工作已经在 Waymo 上完成**，切换到 nuScenes 会浪费 2-3 周重做基础设施。
- video 和 LiDAR 在**同一数据域**完成后训练，使得 joint training 只需要解决跨模态对齐问题，而不需要同时做域迁移。

### 11.2 必须坚持的原则

1. **只围绕”高一致性联合生成”做主线，不再扩充支线 novelty。**
2. **以 self-contained ablation（cascade / naive joint / CoLiGen）为主论证方式**，不依赖外部 baseline 复现。
3. **把 driving 写成验证场景，把 WFM adaptation 写成核心 scientific message。**
4. **把 geometry consistency 作为第一结果，而不是附属指标。**
5. **Waymo 为主，nuScenes 为可选泛化验证。**

### 11.3 本周最紧急的四件事

1. **确定时序对齐方案**（~0.5 天）— 固定 video `29f/t8` 不变，确认 LiDAR tokenizer 为逐帧 CI8x8，并敲定 `29 -> 8` temporal adapter 与时间映射细节。
2. **启动 LiDAR latent 离线提取**（利用已完成的 tokenizer，~1 天）— 按 29 帧逐帧 latent 方案提取。
3. **确认 latent space 维度并实现 LiDAR adapter**（video VAE vs LiDAR tokenizer 的通道/时间粒度对齐，~2 天）— 这是 joint training 能跑通的前提。
4. **实现标定投影算子 $\Pi_{v \leftrightarrow l}$ + 搭建 5-cam joint dataloader 原型**（~2 天）

这四件事均不依赖 video 后训练完成，可以立即并行推进。其中 #1 是最高优先级——时序对齐方案决定了后续所有 latent 提取和 dataloader 的设计。

一句话总结：

**基础设施已就绪，剩余 41 天全部聚焦在核心方法实现和实验闭环。这篇工作的胜出条件是”在 Waymo 域适配的 WFM 上，以几何锚定对齐和模态解耦训练两个关键改动，显著提高 video 与 LiDAR 的几何和时序一致性”。**

---

## Submission References

- NeurIPS 2026 dates: <https://neurips.cc/Conferences/2026/Dates>
- NeurIPS 2026 call for papers: <https://neurips.cc/Conferences/2026/CallForPapers>
