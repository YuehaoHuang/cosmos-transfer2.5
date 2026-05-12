# CoLiGen: Consistency-First Joint Video and LiDAR Generation on World Foundation Models

> **Target Venue**: NeurIPS 2026 Main Track
> **Official Submission Milestones**: Abstract due May 4, 2026 (AOE); Full paper due May 6, 2026 (AOE)
> **Today / Remaining Time**: May 5, 2026 -> abstract deadline passed; 1 day to full paper deadline
> **Backbone**: Cosmos Transfer 2.5 - 2B Rectified Flow DiT
> **Hardware Budget**: 8 x NVIDIA A100-80GB (single node)
> **Proposal Status**: Progress-updated revision v4.7（full 28-block one-way baseline 已停训；当前 best candidate 为 step_005000）

### 已完成里程碑（2026-04-18 状态快照）

| 模块 | 状态 | 关键指标 |
|---|---|---|
| Waymo multiview video 后训练 | **已有可用 checkpoint** — 最远 run 达 `iter_33000`（20260310 run），另一条 run 至 `iter_17000`（20260326 run） | 5-cam, 720p, 29 frames, HDMap control；loss 平稳 ~0.05 |
| Waymo LiDAR tokenizer Stage 1（`CI8x8-Waymo`，2D 单帧） | **已完成 20k iter** | RMSE 0.20m, MAE 0.09m, Rel 0.00（单帧空间重建） |
| Waymo LiDAR tokenizer Stage 2（`OpenSora-S1/S2/S3`，3D `29→8→29`） | **三阶段已完成** — S1 30k / S2 40k / S3 40k iter | 聚合 val：mae 1.133 / rmse 3.761 / rel 0.0433；S1→S3 mae **-49%**, rmse **-36%**, rel **-48%**；单条 clip S3 RMSE 5.80m / MAE 1.58m / Rel 0.05 |
| Waymo latent / cache 提取 | **video 已就绪，LiDAR 改同采样器在线提取** | `extract_waymo_latents.sh` 可继续抽 video latent；LiDAR 以 `WaymoMultiviewDataset` 的 window 在线 encode，可懒加载缓存 |
| Waymo 数据 pipeline | **已就绪** | `WaymoMultiviewDataset`, 798 train / 202 val clips |
| Joint training 基础设施 | **已就绪** | `joint_training.py` 概率采样多 dataloader |

工程锚点：
- Video / multiview 主干：`cosmos_transfer2/multiview.py`
- Waymo 后训练配置：`cosmos_transfer2/experiments/multiview/waymo_posttrain.py`
- Waymo 数据加载：`cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/defaults/dataloader_local.py`
- Rectified-flow control model：`cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py`
- LiDAR tokenizer 后训练（已完成）：`/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/`
- **LiDAR tokenizer 部署 checkpoint（推荐）**：`/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/checkpoints/iter_000035500.pt`
- LiDAR tokenizer config：`.../Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/config.yaml`
- Video 后训练 checkpoint（主）：`/data/cosmos-transfer2.5/output/20260310_104528/cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/checkpoints/iter_000033000/`
- Video 后训练 checkpoint（副本）：`/data/cosmos-transfer2.5/output/20260326_195622/cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/checkpoints/iter_000017000/`
- Joint training 框架：`cosmos_transfer2/_src/imaginaire/datasets/joint_training.py`
- Action + video 联合去噪参考：`third_party_ref_repo/dreamzero`
- LiDAR tokenizer 后训练报告：[waymo_lidar_tokenizer_post_training.md](/root/workspace/Cosmos-Drive-Dreams/waymo_lidar_tokenizer_post_training.md)

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

#### 4.2.1 Latent Space 兼容性（工程难点已被 S3 大幅简化）

**重要更新（2026-04-18）**：LiDAR tokenizer 已经从逐帧 `CI8x8-Waymo` 升级为 3D `OpenSora-S3`，原生实现 `29→8→29` 的 temporal 4x 压缩，与 video VAE 的 `state_t=8` 时钟**天然对齐**。原 proposal 中的"双时钟"问题因此被消除，不再需要 LiDAR temporal adapter。

**更新后的两个 tokenizer 参数对比：**

| | Video VAE (WAN2.1) | LiDAR Tokenizer (OpenSora-S3) |
|---|---|---|
| 结构 | 3D causal VAE | 冻结 `CI8x8-Waymo` encoder + `opensora_temporal_vae` |
| 时间压缩率 | **4x** (`29→8`) | **4x** (`29→8`)，首帧 bypass 保留 |
| 空间压缩率 | 8x8 | 8x8 |
| 29 像素帧 → latent 时间维度 | `state_t = (29-1)/4+1 = 8` | **`state_t = 8`（同时钟）** |
| 单帧空间输入 | 720p (~`720×1280`) | `512 x 1800`（Waymo TOP range map，raw `3600` 先按列 `scatter_min` 下采样 2x） |
| 通道数 | 16 | 16 |
| 部署 checkpoint | `waymo_5cam_post_train/iter_000033000` | `OpenSora-S3/iter_000035500.pt` |

**新的关键设计约束：**

1. **时间维度**：两个模态都是 `state_t=8`，无需 temporal adapter；但不把 LiDAR 当作第 6 个 camera 直接塞进现有 video grid。
2. **空间维度**：video 是 `(H_v/8, W_v/8)`（720p → 90×160），LiDAR range map 走 tokenizer 正式预处理后得到 `128×3600 -> 512×1800`。2026-04-21 已按官方 `lidar_cli.py` / LTCV 路径核对：真正送入 tokenizer 的输入会先做 spatial pad，`512×1800 -> 512×1808`，压到 **`latent (64,226)`**；decode 时还必须带 `exact_context_latent` 再按 `crop_region=[0,0,4,29,512,1804]` 裁回 `1800`。不能再像之前那样在 tokenizer encode 前裁成 `1792` 或 `896`，否则 target latent decode 会显著劣化。空间网格不同，因此采用 FastWAM-inspired 的独立 expert / token stream，再用 attention policy 和 GALA sparse mask 建立跨模态对应。
3. **通道维度**：假设两者都是 16 通道（需验证 S3 config），若不一致，用 1x1 Conv 做轻量通道投影；如果一致则零参数对齐。

**方案**：**FastWAM-inspired 双 expert 同步去噪，无 temporal adapter**。
- Video expert：保留现有 `MultiViewDiT / MultiViewControlDiT`，video token 序列仍是 `B × C × (V × 8) × H_v/8 × W_v/8`，`V=5` 视角
- LiDAR expert：新增独立 LiDAR token stream，输入 `B × C × 8 × 64 × 226`
- 默认 baseline 采用 `video -> LiDAR` one-way attention：LiDAR query 可以看 video K/V，video query 不看 LiDAR，保证旧 video 生成路径不被 LiDAR 改写
- 主方法在 selected layers 上加入 GALA sparse gated interaction；双向分支使用 zero-init gate，只在 joint config 显式开启

**被淘汰的旧方案**：
- ~~方案 1：LiDAR 29 帧逐帧 latent + temporal adapter 压到 8~~（已被 S3 原生 4x 时间压缩取代）
- ~~方案 2：双时钟交互~~（不再需要）
- ~~方案 3：改 backbone chunk 长度~~（本就不建议，现已完全无必要）

#### 4.2.2 FastWAM-inspired joint policy（新增实现约束）

参考 FastWAM 的 MoT 思路，但不整体替换 Cosmos multiview video backbone：

1. **默认 one-way baseline**：`video_to_lidar`，LiDAR denoiser 读 video tokens，video denoiser 不读 LiDAR tokens。该 baseline 用于验证“完全不影响视频生成”。
2. **GALA 主方法**：在 one-way baseline 上加入几何稀疏 mask；先做 `video -> LiDAR`，再尝试 gated `LiDAR -> video`。
3. **独立 timestep / scheduler**：复用 FastWAM 的思想，为 video 与 LiDAR 分别采样 $t_v, t_l$，这就是 MDNS 的工程落点。
4. **实现边界**：所有 joint policy 代码放在新模块和新 config 下，默认 inference / video post-training 不 import、不初始化。

当前已新增 opt-in helper：`cosmos_transfer2/_src/transfer2_multiview/networks/video_lidar_joint_policy.py`，提供 `[video, lidar]` token mask、one-way / bidirectional policy 和后续 GALA sparse mask 接口。

#### 4.2.2.1 Online-first baseline 入口（2026-04-18 已跑通）

项目环境：

```bash
conda activate cosmos-transfer2.5-merge
```

当前 baseline 不再默认依赖离线 video latent / LiDAR latent cache，而是以 `WaymoMultiviewDataset` 为主时钟在线取样：

1. `WaymoMultiviewDataset(include_lidar_alignment_metadata=True)` 在线读取 5-camera mp4，并输出 `waymo_segment_key / waymo_lidar_frame_indices`
2. 训练 loop 按这些 index 在线从 LiDAR tar 恢复同窗 29 帧 range map
3. frozen `OpenSora-S3 iter_000035500.pt` 在线 encode LiDAR latent
4. 视频侧默认可用 `raw_downsample` 做在线 smoke / baseline starter；如果本机具备 Wan2.1 VAE 权重或 S3 credential，可切到 `--video-tokenizer wan2pt1` 做真正 frozen video VAE 在线 encode
5. 只训练独立 LiDAR denoiser；video branch 不反向、不改权重、不被 LiDAR 读写

入口脚本：

```bash
python scripts/train_waymo_video_to_lidar_baseline.py \
  --video-tokenizer raw_downsample \
  --limit-samples 1 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1 \
  --dry-run \
  --hidden-channels 8 \
  --device cuda
```

LiDAR baseline 必须使用官方在线提取核对过的 S3 latent 尺寸 `64x226`；主线做法不是 pre-tokenizer 裁剪，而是保留 full-width `1800` 输入，让 tokenizer 自己 pad 到 `1808`，保存 `compressed_latent_from_encoder`，并额外保存 `exact_context_latent=(16,1,64,226)` 供 decode 使用。脚本默认会拒绝非 `64x226` 的 `--train-height/--train-width`，除非显式加 `--allow-lidar-resize-for-smoke` 做本地非 baseline smoke。当前应验证输出：`video_latent=(1,16,40,90,160)`，`lidar_latent=(1,16,8,64,226)`。可选 cache 工具 `scripts/cache_waymo_lidar_s3_latents.py` 只作为加速/复现实验的写回路径，不是 baseline 的默认依赖。

2026-04-26 更新：完整 one-way expert 主线已切到真实 video latent + 在线 Wan2.1 LiDAR VAE。video latent 读取 `/data/waymo/chunk/training/samples` 的真实 Wan latent；LiDAR 仍由同一个 Waymo dataloader 的 `waymo_lidar_frame_indices` 在线读取 raw tar，并用 `docs/waymo_lidar_wan21_vae_preprocessing.md` 的置顶方案 encode 成 `lidar_latent=(B,16,8,88,164)`。入口脚本为 `train_waymo_video_lidar_one_way_wan21_online.sh`。2026-04-29 起默认改为 `LIDAR_NUM_BLOCKS=28`、`VIDEO_KV_EVERY_N_LAYERS=4`、`CHECKPOINT_LIDAR_BLOCKS=true`、`INIT_LIDAR_FROM_VIDEO=true`。

#### 4.2.2.2 One-way expert baseline（2026-04-18 已启动 8-GPU 训练）

当前已新增独立主线代码：

- one-way expert 主模块：`cosmos_transfer2/_src/transfer2_multiview/networks/video_lidar_one_way_expert.py`
- 训练入口：`scripts/train_waymo_video_lidar_one_way_expert_baseline.py`

实现约束：

1. 默认仍然是 opt-in，新模块不被旧 video generation / video post-training 路径 import。
2. frozen Waymo video DiT 保持原路径；LiDAR expert 逐层读取 frozen video K/V。
3. video query 不读 LiDAR，保证 one-way baseline 下 video branch 行为不被 LiDAR 改写。
4. Waymo paired batch 继续使用在线 video / LiDAR 对齐采样；LiDAR tokenizer 支持按 batch 上卡、编码后下卡（`--offload-lidar-encoder`），避免长期占满显存。
5. 默认 frozen video checkpoint 已对齐 proposal 推荐：`/data/cosmos-transfer2.5/output/20260310_104528/.../iter_000033000/model_ema_bf16.pt`（即 `DEFAULT_WAYMO_VIDEO_CHECKPOINT`）。
6. `cross_frame_rule` 默认主线暂用 `all`，以保证 v0 baseline 吞吐；严格帧对齐的 `same_step` 已改成向量化 SDPA/FlashAttention 路径，可作为 temporal anchoring ablation 直接打开。

2026-04-18 当天状态：

- 当前 one-way expert 主线默认尺寸已切到：`video_pred=(1,16,40,90,160)`，`lidar_pred=(1,16,8,64,226)`
- 单卡 1-step smoke 已通过并保存 checkpoint：`/data2/waymo_video_lidar_one_way_expert/smoke_20260418_201246/checkpoints/step_000001.pt`，loss=5.03
- 正式 8-GPU baseline 已通过 `tmux` 启动（首次 bs=1 显存仅 ~37GB，已重启为 bs=2 → ~51GB / 80GB / GPU）：
  - session：`one_way_expert_v2l_20260418_202505`
  - output：`/data2/waymo_video_lidar_one_way_expert/one_way_expert_v2l_20260418_202505`
  - log：`/data2/waymo_video_lidar_one_way_expert/logs/one_way_expert_v2l_20260418_202505.log`
  - 配置：`torchrun --standalone --nproc_per_node=8`，`batch_size_per_rank=2`，`num_workers=4`，`global_batch=16`，`max_steps=30000`，`save_every=500`，`log_every=10`，`--offload-lidar-encoder`
  - dataloader 已确认 `dataset_size=13900`（798 train clips × 平均 chunk 数）

##### 4.2.2.2.1 LiDAR expert 模型结构与参数（实测）

LiDAR expert 不是轻量 LoRA / 小 adapter，而是与 frozen video DiT 主干**同尺寸的独立 2B DiT clone**，输入输出空间换成 LiDAR latent；和 frozen Waymo video DiT 的网络配置完全镜像，便于逐层 mixed attention 对齐。

| 项 | 值 | 备注 |
|---|---|---|
| 类 | `VideoConditionedLidarExpert(MinimalV1LVGDiT)` | single-view 版本的 Cosmos DiT，对应 `video_lidar_one_way_expert.py:317` |
| Backbone | DiT，28 个 block | 与 frozen video DiT `num_blocks` 一致 |
| `model_channels` | 2048 | 同 video DiT |
| `num_heads` | 16 | 同 video DiT |
| `mlp_ratio` | 4.0 | 同 video DiT |
| `patch_spatial / patch_temporal` | 2 / 1 | 同 video DiT |
| `in_channels / out_channels` | 16 / 16 | S3 LiDAR latent 通道，已和 video VAE 一致 |
| `adaln_lora_dim` | 256 | AdaLN-LoRA 调制 |
| `pos_emb_cls` | rope3d | 同 video DiT |
| `crossattn_proj_in_channels → crossattn_emb_channels` | 100352 → 1024 | T5 文本投影；当前训练注入 null embedding，待接 caption |
| 输入 latent | `(B, 16, 8, 64, 226)` | state_t=8，与 video latent 时钟对齐；full-width `1800` 输入经官方 spatial pad 到 `1808` 后得到 |
| decode 额外上下文 | `(B, 16, 1, 64, 226)` | `exact_context_latent`；LTCV decode 必须一并传入，之后再按 `crop_region` 裁回 `1800` |
| `timestep_scale / use_wan_fp32_strategy` | 0.001 / True | 与 video DiT 一致 |
| Attention policy | `mode=video_to_lidar`，默认 `cross_frame_rule=all`；可切 `same_step` | `all` 让 LiDAR Q 读全部 video K/V，主线吞吐更稳；`same_step` 已向量化为单层 1 次 SDPA，恢复严格逐 step 对齐且可走 FlashAttention |
| 每层 forward | `_forward_one_way_block`：mixed self-attn → cross-attn(text) → MLP | 训练时整层包 `torch.utils.checkpoint`，节省 activation 显存 |
| 可训参数规模 | ~1.7B（与 video DiT 主干同量级） | 仅 `lidar_expert.parameters()` 进 optimizer |

训练面参数：

- 优化器：`AdamW(lidar_expert.parameters(), lr=1e-4, weight_decay=1e-4)`，对应 `train_waymo_video_lidar_one_way_expert_baseline.py:225-226`
- 噪声采样：rectified flow，video / LiDAR 默认共享 timestep（可加 `--independent-video-timesteps` 切到模态独立采样，作为 MDNS 前置开关）
- loss：`MSE(lidar_pred, clean_lidar - lidar_noise)`（v-pred）；video branch 全程 `no_grad`、不参与反向
- 全程 frozen：video DiT 全权重、LiDAR S3 tokenizer、video tokenizer（当前 raw_downsample 占位，待切 wan2pt1 真 VAE）
- DDP：`find_unused_parameters=False`，仅广播 LiDAR expert 梯度
- FlashAttention：训练入口显式开启 PyTorch SDPA flash/mem-efficient/math backend；`same_step` 不再用 Python `for step` 小 attention 循环，而是把 step 维 stack 到 batch 维后一次 SDPA。
- 速度排查结论：`--sdpa-backends flash_only` 已验证可跑且 `sdpa_math_enabled=False`，但 7-GPU 稳态仍约 `100s/step`，说明瓶颈不是 SDPA fallback；单卡 profiler 显示 top CUDA kernel 已是 `pytorch_flash::flash_fwd_kernel`。
- deadline baseline 加速开关：新增 `--lidar-num-blocks N`。`N=14` 单卡 smoke 已通过，step time 从 full 28-block 的约 `49.15s` 降到 `28.55s`；下一步优先跑 14-block 8-GPU/可用 GPU 短训，而不是继续只调 Flash backend。
- 进一步用显存换时间：新增 `--no-checkpoint-lidar-blocks` 与 `--no-empty-cache-after-encode`。`N=14 + flash_only + no checkpoint + no empty_cache` 单卡 smoke 为 `25.31s/step`，比 14-block checkpoint 版再快约 11%。若 7/8-GPU 显存允许，deadline 版优先使用这组。
- 多卡实测：机器重启后当前只枚举到 7 张 A100；`N=14 + batch_size_per_rank=2 + checkpoint_lidar_blocks=True + no_empty_cache + no_offload` 可稳定跑，显存约 `42.5GB/GPU`，step 2-5 稳态约 `53.7-55.6s/step`，loss `5.35 -> 2.93` 健康下降。`N=14 + no_checkpoint + batch_size_per_rank=2` 会 OOM（~79GB/GPU），不作为稳定配置。
- Wan2.1 在线 VAE 实测：`N=14 + video_kv_every_n_layers=4 + no_checkpoint + batch_size_per_rank=1` 单卡 smoke 通过，但 7-GPU DDP 首个 forward 峰值约 `77GB/GPU` 后再申请 `2.2GB` 导致 OOM；当前稳定运行改为 `CHECKPOINT_LIDAR_BLOCKS=true`，显存约 `30.6GB/GPU`，tmux session `wan21_online_train_20260426_004355`。

后续可调维度（若要压参数量或显存）：缩 `num_blocks` / `model_channels`；或把 LiDAR expert 改为 LoRA-only 训练 + 全冻结 backbone clone。**当前默认按"2B 级 LiDAR expert"路线训，与 proposal §4.2.2 FastWAM-inspired 双 expert 设计一致。**

#### 4.2.3 时序维度约束（沿用）

Cosmos multiview DiT 沿时序维度拼接多视角：latent shape 为 `B C (V*state_t) H W`。DiT 内部的 3D temporal attention 期望看到完整序列，仍然有：

- **不能用单帧训练** — temporal attention 在 length=1 上严重偏离预训练分布
- **最小训练单元是 `state_t=8` 个 latent 帧**
- 当前 Waymo video checkpoint 已在 `29f/t8` 上后训练；LiDAR S3 也已在 `29→8→29` 上训练。两者时钟天然一致，joint training 复用同一 `state_t=8` 时轴

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
   - 使用完整序列（video 固定为 `state_t=8` / 29 帧），5 cameras + 29 帧 LiDAR；LiDAR 经 S3 tokenizer 原生 `29→8` 压缩，与 video latent 时钟**天然对齐**
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
| Latent / cache 提取 | **video 已就绪，LiDAR 改为同采样器在线提取** | `extract_waymo_latents.py` 可继续用于 video latent；LiDAR 必须复用 `WaymoMultiviewDataset` 的 sample/window，再在线 no-grad encode 或写入 cache |
| 推理 pipeline | **已完成** | `inference_waymo_autoregressive.sh` 支持自回归多 chunk 生成 |
| 数据质量检测 | **已完成** | `check_waymo_samples.py` 并行验证视频完整性 |

### 5.2 剩余需要实现的核心模块（按优先级排序，2026-04-18 更新）

因为 LiDAR tokenizer 已完成 3D `29→8→29` 升级、video 后训练已有可用 checkpoint，剩余工作再次收窄：

1. **LiDAR latent 同步在线提取 / 缓存**（~1 天）
   不能再用独立 LiDAR sampler 随机抽窗后离线落盘；必须以 Waymo video 后训练 dataloader 的 `sample_key` 和窗口为准。当前视频样本名是 `<segment_key>_<chunk_idx>`，5 个 camera 与 world_scenario control 都读取该 29-frame chunk 的本地 `[0, 29)`；对应 LiDAR tar / metadata 则按 `<segment_key>` 命名。LiDAR 输入窗口应由 `chunk_idx` 反推：默认 `lidar_start = chunk_idx * 10 + local_frame_start`，取连续 29 帧后用 OpenSora-S3 `iter_000035500.pt` 在线 no-grad encode；主线保留 full-width `1800` 输入，让 tokenizer 官方 pad 到 `1808`，保存 `compressed_latent_from_encoder -> [B, 16, 8, 64, 226]`，并同步写下 `exact_context_latent -> [B, 16, 1, 64, 226]` 与 `tokenizer_crop_region`。稳定后可以把结果按完整 `sample_key=<segment_key>_<chunk_idx>` 写 cache，但 cache 只能由同一个 Waymo dataloader 生成。

2. **FastWAM-inspired joint policy / LiDAR expert scaffold**（~1 天）
   新增独立 LiDAR token stream 与 attention policy，默认 `video_to_lidar` one-way：LiDAR query 读 video K/V，video query 不读 LiDAR。当前 helper 已放在 `video_lidar_joint_policy.py`，后续 joint config 显式 import；默认 video generation / video post-training 不受影响。

3. **Channel-level latent 适配**（~0.5 天）
   假设两者通道都是 16，只需做 identity；若 S3 通道数与 video VAE 不一致，插入 1x1 Conv 通道 projector + modality embedding（尾部零初始化）。无需时间对齐步骤。

4. **GALA 模块 + MDNS 实现**（~5 天，合并实现）
   GALA：稀疏投影索引（利用 Waymo 标定）、validity mask、gated cross-attention（α, β 零初始化）。投影在每个 `state_t` slot 内对应到 LiDAR 同 slot 的 latent。
   MDNS：在 rectified-flow noise schedule 中为 video/LiDAR 分别采样 $t_v, t_l$。
   起点：在 `MultiviewControlVideo2WorldModelRectifiedFlow` 的训练 step 中插入。

5. **Joint rectified-flow model**（~3 天）
   扩展现有 model 接收 video + LiDAR 双 latent token stream。
   约束：`state_t=8` 时序粒度；LiDAR 不作为第 6 个 camera，而是独立 expert/token stream。
   用现有 `joint_training.py` 组织 batch（概率采样 video-only / video+LiDAR 混合）。

5. **评测脚本**（~3 天）
   DAS（depth reprojection）、CME（跨模态运动一致性）、下游感知评测。
   可复用 `packages/cosmos-oss/vqa/` 中的 LPIPS/PSNR/SSIM 框架。

时间总计：约 12.5 天工程工作量。与剩余 18 天的 deadline 比较，仍然非常紧张——但比上个版本（预计 16 天工程 + 16 天训练）缓解了约 3.5 天。

### 5.3 关键优势：Waymo 后训练为 joint training 铺路

与原方案相比，Waymo video 后训练的完成带来三个关键优势：

1. **DiT 主干已适配 Waymo 域**：joint training 时不再需要同时完成域迁移和跨模态对齐两件事，降低训练难度。
2. **LiDAR tokenizer 已在同一数据域验证**：video 和 LiDAR 共享相同的场景分布和标定体系，避免跨数据集对齐噪声。
3. **数据 pipeline 已打通且对齐点明确**：`WaymoMultiviewDataset` 可在显式开启 `include_lidar_alignment_metadata=True` 时输出 `waymo_segment_key / waymo_chunk_index / waymo_lidar_frame_indices`；默认 video dataloader 行为不变。LiDAR range map 应从这些字段在线恢复并 encode，避免 video chunk 与 LiDAR clip 各自采样导致错位。

### 5.4 不建议在第一轮就做的工程冒险

- 不建议一开始就 full fine-tune 2B backbone（Waymo 后训练 checkpoint 已是很好的起点）
- 不建议同时引入过多新损失和新编码器
- 不建议第一轮就做长序列自回归联合生成（先验证单 chunk 29 帧）

第一轮目标是尽快得到一个**明确优于串联 baseline 的一致性曲线**，而不是一次性做完所有 fancy 设计。

---

## 六、Training Plan Under 8 x A100

### 6.1 前置条件（2026-04-18 状态）

| 前置项 | 状态 | 对 joint training 的影响 |
|---|---|---|
| Waymo video 后训练 | **已有可用 checkpoint**：`iter_000033000`（20260310 run）和 `iter_000017000`（20260326 run） | 直接作为 joint training 的初始化权重；不再阻塞 |
| Waymo LiDAR tokenizer（3D `OpenSora-S3`） | **已完成**，S1→S3 mae -49% | 部署 `iter_000035500.pt`，冻结不动 |
| Waymo LiDAR latent 提取 | 待实现（~1 天） | 默认在线 no-grad 提取并可写 cache；必须复用 video dataloader 的 `<segment_key>_<chunk_idx>` 和 `waymo_lidar_frame_indices` |

### 6.2 两阶段联合训练 + 轻量 ablation（从 Waymo video checkpoint 出发）

#### 主训练

| Stage | 目标 | 起始权重 | 冻结策略 | 训练模块 | 数据规模 | Loss | 预估成本 |
|---|---|---|---|---|---|---|---|
| **1** | Joint Alignment | Waymo video `iter_000033000` + frozen `OpenSora-S3` LiDAR tokenizer | 冻结 VAEs + video DiT 主干 | FastWAM-inspired LiDAR expert + one-way policy + GALA + channel projector + LoRA + MDNS | 5-cam x 29f paired clips，LiDAR 同步 `state_t=8` | $\mathcal{L}_{\mathrm{flow}}^{v,l} + \mathcal{L}_{\mathrm{geo}}$ | ~500-700 A100-h |
| **2** | Temporal Consistency | Stage 1 ckpt | 同 Stage 1 | 同 Stage 1 | 同 Stage 1 | 加入 $\mathcal{L}_{\mathrm{temp}}$, cosine LR decay | ~400-500 A100-h |

#### Ablation 预算（关键问题：原方案未计入）

5 个完整 ablation × 完整训练 = ~5000+ A100-h，远超预算。采用**缩短版 ablation**策略：

| Ablation | 做法 | 预估成本 |
|---|---|---|
| w/o GALA | 仅跑 Stage 1 的 50% iter，对比 DAS 趋势 | ~250 A100-h |
| w/o MDNS | 仅跑 Stage 1 的 50% iter，对比 DAS 趋势 | ~250 A100-h |
| w/o $\mathcal{L}_{\mathrm{geo}}$ | 仅跑 Stage 1 的 50% iter，对比 DAS | ~250 A100-h |
| one-way MoT baseline | 只开 `video_to_lidar`，不开 GALA reverse gate | ~250 A100-h |

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
- **LiDAR latent 采用在线 no-grad encode + lazy cache**：第一版先保证和 video dataloader 完全同窗；确认无错位后再用 cache 降低训练 loop tokenize 成本
- 投影索引离线预计算，避免每 step 动态构图
- 主干尽量 LoRA 化，控制新增参数规模

### 6.4 Stage 1 最小可验证设置

- `5 cameras x 29 frames`（与 Waymo 后训练完全一致：front / front_left / front_right / side_left / side_right，video `state_t=8`）
- LiDAR `waymo_lidar_frame_indices` 对应的 29-frame range map -> OpenSora-S3 encode -> `state_t=8` latent（与 video chunk 同窗、同时钟，**无需 temporal adapter**）
- `720p`（与 Waymo 后训练分辨率一致，避免分辨率迁移开销）
- 冻结 Waymo 后训练的 video DiT 主干；先训 LiDAR expert + `video_to_lidar` one-way policy，再开 GALA sparse gate + channel projector + LoRA
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
| FastWAM-style one-way vs GALA 双向 | one-way 用于确认 video 完全不被影响；双向 gated GALA 用于验证是否能进一步提升 video/LiDAR 几何一致性 |

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

## 八、18-Day Submission Backward Plan（2026-04-18 重排版）

### 8.1 关键日期与剩余时间

- `2026-05-04 AOE`: NeurIPS 2026 abstract deadline（**16 天**）
- `2026-05-06 AOE`: NeurIPS 2026 full paper deadline（**18 天**）

从 `2026-04-18` 开始，实际只剩 18 天。与 v4.1 相比的节省：
- **OpenSora-S3 3D LiDAR tokenizer 已完成** → 节省 LiDAR temporal adapter 设计/调试 ~2 天
- **Video 后训练已有可用 checkpoint（iter_33000）** → 不再需要等待

剩余时间极其紧张，必须**并行工程 + 尽早启动 Stage 1**。

### 8.2 当前状态（2026-04-19 凌晨更新）

```
[已完成] Waymo LiDAR tokenizer 三阶段（CI8x8 → S1 → S2 → S3），best iter_000035500.pt
[已完成] Waymo video 后训练 iter_000033000（20260310 run），iter_000017000（20260326 run）
[已完成] FastWAM-inspired joint policy helper（opt-in，默认不影响 video）
[已完成] Online-first Waymo video/LiDAR paired baseline 入口：mp4 在线取视频，LiDAR tar 同窗在线 S3 encode
[已完成] One-way expert baseline 模块 + 训练入口（video_to_lidar，frozen video DiT + 可训 LiDAR expert）
[已完成] 单卡 1-step smoke run（loss=5.03，checkpoint 保存正常）
[已完成] 多卡 one-way expert 速度排查：当前机器只枚举 7×A100；`lidar_num_blocks=14 + batch_size_per_rank=1 + --no-checkpoint-lidar-blocks + flash_only` 稳态 24.3-25.0s/step，显存约 74GB/80GB，满足 May 4 前完成 30k step 的速度要求。`--no-checkpoint-lidar-blocks` 现已同时关闭 LiDAR 外层 block checkpoint 和 LiDAR expert 初始化时继承的 mm-only SAC；frozen video expert 仍保持原 selective checkpoint。
[已检查] one-way expert baseline 主训练 run：session `one_way_expert_lidar14_b1_no_lidar_ckpt_main_7gpu_20260419_004808`，输出 `/data2/waymo_video_lidar_one_way_expert/one_way_expert_lidar14_b1_no_lidar_ckpt_main_7gpu_20260419_004808`。日志到 step 1830 后停止，末尾无 traceback、无 `[train] done`；最后文件时间为 `2026-04-19 13:07:59 CST`。loss 从 `5.37` 降到 `0.48`，最后 50 个 log 点均值 `0.509`，稳态 `~24.1s/step`。当前唯一可用保存点是 `checkpoints/step_001000.pt`。
[已完成] step1000 推理 smoke：新增 `scripts/infer_waymo_video_lidar_one_way_expert.py`。validation sample `10203656353524179475_7625_000_7645_000_0` 上，teacher-forced 单步 denoise 已贴近 GT tokenizer recon（latent MSE `0.123`，MAE `0.248`）；完整 RF 采样仍弱，4-step sample MSE `1.347`，16-step `teacher_forced_flow` sample MSE `1.635`，16-step `clean video` sample MSE `1.596`。结论：step1000 已学到局部 video-conditioned denoise，但完整从噪声生成未收敛，下一步应优先继续训练/保存更高 step，而不是先调采样器。
[进行中] 8-GPU resume 主训练：训练脚本已新增 `--resume-checkpoint` / `--resume-load-optimizer`，当前 session `one_way_expert_lidar14_resume8gpu_to5k_20260419_221914` 从 `step_001000.pt` 续训到 `max_steps=5000`，`save_every=500`。配置为 `world_size=8`、`global_batch=8`、`lidar_num_blocks=14`、`--no-checkpoint-lidar-blocks`、`flash_only`。已确认 `step=1010 loss=0.4888`、`step=1020 loss=0.5961`，稳态约 `24.7s/step`，显存约 `74GB/GPU`。该 tmux job 在训练结束后会自动运行 step5000 的 validation inference：4-step preview、16-step `teacher_forced_flow`、16-step `clean video`。
[已完成] Paired latent cache 入口：`scripts/cache_waymo_paired_video_lidar_latents.py`，训练脚本新增 `--paired-latent-cache-dir`。canonical validation smoke `10203656353524179475_7625_000_7645_000_0` 已保存 paired payload、raw 五视图视频（row + 3x2 grid）、raw-vs-latent 视频对比和 LiDAR decode；LiDAR 可视化使用 `prediction_key=paired_cache_decode`、raw LiDAR GT、`front_view + vehicle + plotly`，指标 `RMSE 5.798 / MAE 1.576 / Rel 0.0501`。Plotly/Kaleido 点云渲染新增 `--pcd-workers`，本机标准用 `--pcd-workers 12`。截至 `2026-04-21 23:03 CST`，training paired cache 已写 `10296 / ~13900` 个 `.pt`，7 个 tmux shard 仍在运行。
[待做]   GALA sparse 投影 + MDNS noise schedule 实装（~5-6 天，可与上面训练并行开发）
[待做]   双向 gated GALA 接入（仅在 one-way 收敛后试探打开）
[待做]   评测脚本：DAS / CME / Reproj-Edge / 下游 3D detection（~3 天）
```

### 8.3 倒排计划（紧凑两阶段训练版）

| 日期 | 工程目标 | 论文目标 | Exit Criterion |
|---|---|---|---|
| **Apr 18 - Apr 20 (Sat-Mon)** | (1) 跑通 S3 tokenizer 的 encode-only 路径，并接到 `WaymoMultiviewDataset` 的同窗在线提取 / cache; (2) 验证 video VAE 与 S3 latent 的通道/shape；(3) 实装 `video_to_lidar` one-way joint policy + LiDAR expert scaffold；(4) 实现标定投影算子 $\Pi_{v \leftrightarrow l}$（先写 dense，后 GALA 里转稀疏）；(5) 构建 5-cam joint dataloader 原型 | 定稿 paper story 与图表模板（Intro + Method 骨架） | paired batch 可可视化；one-way baseline 能跑且 video branch 输出不变 |
| **Apr 21 - Apr 24 (Tue-Fri)** | (1) 实现 GALA（稀疏投影索引 + validity mask + gated cross-attn，α/β 零初始化）; (2) 实现 MDNS（rectified-flow noise schedule 双模态独立采样）; (3) 启动 Stage 1 训练（5-cam × 29f，$\mathcal{L}_{\mathrm{flow}}^{v,l} + \mathcal{L}_{\mathrm{geo}}$）; (4) 同时启动 one-way MoT baseline 训练; (5) 实现 DAS/CME 评测脚本 | Method 初稿完成 | Stage 1 持续训练；one-way baseline 有初步 DAS；曲线对比出现第一条 go/no-go 信号 |
| **Apr 25 - Apr 28 (Sat-Tue)** | (1) Stage 1 收敛评估 DAS/CME; (2) 启动 Stage 2（加 $\mathcal{L}_{\mathrm{temp}}$，cosine LR decay）; (3) 3 个缩短版 ablation 并行（w/o GALA / w/o MDNS / w/o $\mathcal{L}_{\mathrm{geo}}$，各 50% iter） | 主表格字段与 ablation 模板；Experiments 初稿 | CoLiGen DAS 优于 cascade 和 naive joint；ablation 表首轮可填 |
| **Apr 29 - May 2 (Wed-Sat)** | (1) Stage 2 训练收尾; (2) ablation 全部完成; (3) 下游 3D detection 评测（选 1 个）; (4) qualitative 可视化（depth reprojection overlay, point cloud 对比） | 全文初稿 + 摘要 + supp 骨架 | 论文主体完整可审 |
| **May 3 (Sun)** | failure case 与最终数字校对 | 润色全文、格式检查 | 可提交 abstract |
| **May 4** | 提交 abstract | 固定标题、作者、摘要 | abstract submitted |
| **May 5 - May 6** | 最后补实验、补 supp 视频 | 提交 final PDF + supp | paper submitted |

### 8.4 并行化策略与 GPU 调度（8×A100 单节点）

```
Apr 18 - Apr 20: 基础设施收尾
  GPU 1 (单卡): S3 LiDAR tokenizer encode-only 接入同窗在线提取 / cache（覆盖 798+202 clips）
  CPU:          GALA 投影算子、channel projector、5-cam joint dataloader 开发
  CPU:          Cascade baseline 推理脚本串起

Apr 21 - Apr 24: 方法实现 + 首轮训练
  GPU (8xA100):  Stage 1 训练（CoLiGen full, 5-cam × 29f）
  GPU (错峰):    naive joint baseline 训练（短 run，只取早期 DAS 信号）
  CPU:           DAS/CME/Reproj-Edge 评测脚本实现

Apr 25 - Apr 28: 主训练 + ablation 并行
  GPU-A (4xA100): Stage 2 训练
  GPU-B (4xA100): w/o GALA / w/o MDNS / w/o L_geo 三条缩短版 ablation 串行（各 ~1.5 天）

Apr 29 - May 2: 评测 + 补实验
  GPU:  下游 3D detection 训练/评测（选 1 个 detector）
  GPU:  最终模型 5-cam 全量推理与定性可视化
  CPU:  写论文、做图表
```

**关键路径**：`LiDAR latent 提取(Apr 18-19) → GALA/MDNS 实装(Apr 21) → Stage 1 启动(Apr 22) → Stage 2(Apr 26) → 评测收尾(May 2)`，约 14 天（Apr 18 - May 2）。Ablation 与 Stage 2 并行，不延长关键路径。

### 8.5 必须冻结的范围

- **Apr 24**：冻结主架构（GALA + MDNS + joint denoiser），不再新增核心模块
- **Apr 30**：冻结实验范围，不再增加新 benchmark 或 baseline

---

## 九、Risk Register and Fallback Strategy

### 9.1 风险一：Waymo video checkpoint 质量不足以支撑 joint training

**风险**：video 后训练已停在 `iter_33000`（20260310 run，2026-03-23 完成）。若该 checkpoint 生成质量在 Waymo 5-cam 上不够强，joint training 起点就不稳。
**概率**：低（loss 早已稳定在 ~0.05 量级，5-cam × 29f × 720p 后训练分布未漂移）
**兜底**：

- 默认使用 `iter_000033000`；若发现质量退化，退回 `20260326_195622/iter_000017000`（更新的 run，但 iter 更短）
- 若两者均不达标，重启最后 ~5-10k iter 的短 finetune（EMA 已存），再投入 joint training

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

当前项目状态（2026-04-18）：

- Waymo video 后训练（`iter_33000`）和 LiDAR tokenizer 三阶段（`OpenSora-S3`）均已完成，**基础设施阶段结束**，剩余 18 天全部投入在核心方法（GALA + MDNS）和实验上。
- **LiDAR tokenizer 从逐帧 CI8x8 升级为 3D OpenSora-S3（`29→8→29`）**，与 video VAE 的 `state_t=8` 时钟天然对齐，消除了之前 proposal 里的"双时钟"问题，不再需要 temporal adapter。
- 主数据集保持 Waymo — 所有前置工作已在 Waymo 上完成，切换到 nuScenes 会浪费 2-3 周重做基础设施。
- video 和 LiDAR 在**同一数据域**完成后训练，使得 joint training 只需解决跨模态对齐，不需要同时做域迁移。

### 11.2 必须坚持的原则

1. **只围绕”高一致性联合生成”做主线，不再扩充支线 novelty。**
2. **以 self-contained ablation（cascade / naive joint / CoLiGen）为主论证方式**，不依赖外部 baseline 复现。
3. **把 driving 写成验证场景，把 WFM adaptation 写成核心 scientific message。**
4. **把 geometry consistency 作为第一结果，而不是附属指标。**
5. **Waymo 为主，nuScenes 为可选泛化验证。**
6. **旧视频生成路径默认不变**：所有 video/LiDAR joint 逻辑必须通过新 config / 新 flag opt-in；默认 inference 与 video post-training 不 import、不初始化 LiDAR 模块。

### 11.3 本周最紧急的四件事（2026-04-18 晚 20:20 重排）

1. **监控当前 7-GPU one-way expert baseline 主训练**（已启动）— session `one_way_expert_lidar14_b1_no_lidar_ckpt_main_7gpu_20260419_004808`，前 100 step 看 loss 曲线 / OOM / NaN；前 1000 step 出第一个 checkpoint 后做一次 forward 一致性检查（确认 frozen video branch 输出未被 LiDAR 写入污染）。
2. **实现标定投影算子 $\Pi_{v \leftrightarrow l}$ + 5-cam joint dataloader 原型**（~2 天，CPU 侧并行进行）— 投影先 dense 实现、后在 GALA 里转稀疏；dataloader 用 `joint_training.py` 做 video-only / video+LiDAR 概率采样。
3. **GALA sparse 投影 + MDNS noise schedule 实装**（~3 天，目标 Apr 22 写完，Apr 24 启动 Stage 1 训练）— GALA 双向 gate 默认 zero-init，先确保 one-way 不改变 video branch；MDNS 在 rectified-flow noise sampling 处为 video / LiDAR 分别采样 $t_v, t_l$。
4. **DAS / CME 评测脚本起头**（~1 天）— 复用现有 video / LiDAR latent decode 路径，先把 single-clip 的 depth reprojection / cross-modal motion error 跑通，后续加 batch 评测。

这四件事均不依赖 video 后训练进一步训练（iter_33000 已足够作为起点）。其中 #1 已在 GPU 上跑，#2-#4 可在 CPU + 单卡 dev 环境并行推进。

一句话总结：

**LiDAR tokenizer 已升级到 3D OpenSora-S3，与 video VAE 时钟天然一致；video 后训练已有 iter_33000 可用 checkpoint；one-way expert baseline 已通过 1-step smoke，并在当前可见 7×A100 上以 `max_steps=30000` 主训练中。剩余时间聚焦 GALA + MDNS 接入 + ablation 实验闭环，且默认视频生成路径必须保持不变。**

2026-04-26 晚更新：

- 已停止 online Wan2.1 LiDAR VAE 训练和 step_2000 eval；online 训练最后可靠 checkpoint 为 `step_002500.pt`。
- 训练主线切到 paired cache：`video -> /data/waymo/chunk/training/samples`，LiDAR cache root 为 `/data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11`，目标补齐 `13,900` 个 sample。
- 当前 cache 补齐任务已加到 42 shards：`wan21_train_cache_shards42_20260426_234620_s{0..41}`，日志在 `/data2/waymo_paired_latents/logs/wan21_train_cache_shards42_20260426_234620`。
- watcher 为 `wan21_cache42_then_train_fallback_20260426_234638`，cache 完成后会先启动 no-checkpoint DDP；如果 DDP 非 0 退出，则自动切到 FSDP2 fallback。
- 当时 cache 完成后优先启动 `14 blocks + VIDEO_KV_EVERY_N_LAYERS=4 + --no-checkpoint-lidar-blocks` 做速度边界测试；2026-04-29 已切回 full 28-block + video-weight init 主线。同时新增 FSDP2 opt-in 入口 `train_waymo_video_lidar_one_way_wan21_cache_fsdp.sh`，只 shard trainable LiDAR expert，不影响默认 video generation。

2026-04-27 凌晨更新：

- paired cache 已补齐 `13,900/13,900`。旧 raw-downsample paired cache 已废弃；当前 cache video 侧软链接真实 Wan video latent，LiDAR 侧为 Wan2.1 native `64x1312 repeat_row=11` latent。
- no-checkpoint DDP 的实际内存边界已确认：`14 blocks + vkv=4 + batch/rank=1` 在释放 frozen-video Q/K/V/cross 中间张量后可以过 `step=1`，但下一轮仍在 frozen video MLP 处 OOM，峰值约 `78.1GB/GPU`，还差一次 `2.20GB` 分配。因此该配置不能作为稳定主训练。
- FSDP2 fallback 当前不是即插即用：one-way expert 直接调用 LiDAR block 内部模块，绕过 FSDP2 wrapper forward，导致 Tensor/DTensor 混用。要继续 FSDP2，需要重构 LiDAR block 的 forward 边界。
- 当前稳定训练已切到 checkpoint DDP：session `wan21_cp_b2_train_20260427_002810`，输出 `/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_lidar14_vkv4_cp_b2_7gpu_after_cache42_20260427_002810`。配置 `batch/rank=2`、`world_size=7`、`global_batch=14`、`lidar_num_blocks=14`、`VIDEO_KV_EVERY_N_LAYERS=4`、`CHECKPOINT_LIDAR_BLOCKS=true`。已到 `step=10`，loss `3.3592 -> 1.5709`，稳态 `42.15s/step`，显存约 `40.8-41.9GB/GPU`，训练继续运行中。

2026-04-29 检查更新：

- 已停止 `wan21_cp_b2_train_20260427_002810`。最后可靠 checkpoint 为 `step_004000.pt`；训练日志停在 `step=4020` 是人工 kill 后的正常退出。
- `step_001500 -> step_004000` 的 validation sample 基本无改善：`sample_mse 1.3619 -> 1.3629`，`single_step_denoised_mse 0.9227 -> 0.9257`。继续原配置训练收益很低。
- RF 目标与官方 Predict2 对齐：`shift=5`、`x_t=sigma*noise+(1-sigma)*clean`、target=`noise-clean`，UniPC `flow_prediction` 方向也正确；当前问题不是 sampler 符号反了。
- 主要代码问题：LiDAR expert 之前是同尺寸 14-block DiT clone，但从随机初始化开始训练，没有继承 frozen video DiT 的同形状权重。已新增 `--init-lidar-from-video`，只拷贝 same-shape tensor；Wan2.1 cache 训练入口默认开启。该改动仍是 opt-in one-way 训练路径，不影响既有 video generation。
- 按最新方案，主线改回 full 28-block LiDAR expert，默认从完整 frozen video DiT 初始化 28 个 block 的同形状权重。默认启动配置先用 `LIDAR_NUM_BLOCKS=28`、`CHECKPOINT_LIDAR_BLOCKS=true`、`BATCH_SIZE=1`，先测显存和首轮收敛，再决定是否放大 batch。

2026-05-05 检查更新：

- 已人工停止 full 28-block one-way 主训练：session `wan21_full28_init_cp_b2_7gpu_formal_20260429_005200`。最后保存 checkpoint 为 `step_007000.pt`，训练实际日志到 `step=7025` 左右。
- 该 run 配置：`LIDAR_NUM_BLOCKS=28`、`INIT_LIDAR_FROM_VIDEO=true`、`VIDEO_KV_EVERY_N_LAYERS=4`、`CHECKPOINT_LIDAR_BLOCKS=true`、`batch_size_per_rank=2`、`world_size=7`、`global_batch=14`，paired cache 为 `/data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11`。
- 训练性能稳定：`~83.8-84.2s/step`，显存约 `54.1GB/GPU`，7 张 A100 utilization 100%。checkpoint 每 500 step 保存，已有 `step_000500.pt` 到 `step_007000.pt`。
- loss 在 `5k -> 7k` 进入平台期，常见区间 `0.32-0.38`，后段偶有 `0.5-0.8` spike；继续训练的收益不明确。
- 同一 validation sample、4-step latent 推理显示 `step_007000` 不优于 `step_005000`：
  - `step_005000`: `sample_mse=0.733270`、`sample_mae=0.645834`、`single_step_denoised_mse=0.250332`
  - `step_007000`: `sample_mse=0.824434`、`sample_mae=0.691515`、`single_step_denoised_mse=0.266189`
- 因此当前 checkpoint 选择：`step_005000.pt` 作为 best candidate，`step_007000.pt` 只保留做 ablation/overfit 对照。
- 已按要求用 `step_003000.pt` 跑 4-step decode 可视化，不继续扩展 16-sample 指标。输出目录：`/tmp/waymo_infer_step003000_val0_s4_decode_20260505_2250`。
  - sample：`10203656353524179475_7625_000_7645_000_0`
  - preview：`10203656353524179475_7625_000_7645_000_0_preview.png`
  - intermediate preview：`10203656353524179475_7625_000_7645_000_0_intermediate_preview.png`
  - 4-step sampled latent metric：`mse=0.860675`、`mae=0.718463`、`rmse=0.927726`
  - single-step denoised metric：`mse=0.248937`、`mae=0.378692`
  - intermediate 4-step MSE 单调下降：`2.044374 -> 1.765859 -> 1.363221 -> 0.860675`
- 已补跑 `step_005000.pt` 的同配置 4-step decode 可视化。输出目录：`/tmp/waymo_infer_step005000_val0_s4_decode_20260505_2308`。
  - preview：`10203656353524179475_7625_000_7645_000_0_preview.png`
  - intermediate preview：`10203656353524179475_7625_000_7645_000_0_intermediate_preview.png`
  - 4-step sampled latent metric：`mse=0.733270`、`mae=0.645834`、`rmse=0.856312`
  - single-step denoised metric：`mse=0.250332`、`mae=0.374351`
  - intermediate 4-step MSE 单调下降：`2.026494 -> 1.718947 -> 1.257420 -> 0.733270`

---

## Submission References

- NeurIPS 2026 dates: <https://neurips.cc/Conferences/2026/Dates>
- NeurIPS 2026 call for papers: <https://neurips.cc/Conferences/2026/CallForPapers>
