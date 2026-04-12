# Waymo Wan LiDAR VAE Post-Training Plan

## 1. 目标

本文档给出一条可以直接开工的 `Waymo -> Wan2.1 LiDAR VAE` 微调方案，整体设计目标是尽量复用现有 `Wan2.1` video tokenizer 的接口、权重和 latent geometry，让 LiDAR latent 能尽快接入和 video latent 的联合去噪：

- 使用 `Wan2.1` 的 3D causal VAE 作为 LiDAR VAE backbone
- 保持原生 `3-channel in / 3-channel out` 结构，不改首尾层
- 使用 `range image + row repeat x4` 表征 LiDAR
- 将单通道 rangemap 在通道维 `repeat` 为 3 通道输入
- 保持 `temporal / spatial stride = (4, 8, 8)`
- 让 `29` 帧 LiDAR 经过编码后直接得到 `8` 个 latent 帧
- 训练后再做 `ULA`，将 LiDAR latent 分布对齐到 RGB Wan latent 分布

目标不是替代现有 `CI8x8` baseline，而是得到一条更适合 CoLiGen joint training 的 LiDAR tokenizer 路线：

- 输入时钟与 video 分支天然一致：`29 raw frames -> 8 latent frames`
- 不再需要额外的 `29 -> 8` temporal adapter 才能进入 shared DiT
- 后续可直接复用 `state_t=8` 的 multiview 主干

在开工前需要先澄清一个关键背景：`third_party_ref_repo/Cosmos-Drive-Dreams/cosmos-transfer-lidargen` 里官方的 Waymo LiDAR tokenizer，并不是“从 image tokenizer 重新发明一套 lidar tokenizer”，而是：

- 保留 `Cosmos-Tokenizer-CI8x8` 的连续 latent autoencoder 骨架
- 继续使用 `continuous_image` network 和 `image` loss
- 将 dataloader 换成 `lidar_range_map_rRow4_waymo`
- 将输入数据换成 LiDAR range map
- 从已有 LiDAR tokenizer checkpoint `autoencoder.pt` 继续微调

因此，参考实现的本质是“最大化复用 image-style tokenizer 框架，最小化修改数据入口与训练对象”；本文档中的 `Wan2.1 LiDAR VAE` 则是在这个思路之上，再往前走一步，把 LiDAR tokenizer 从 2D 逐帧 AE 升级到 3D video-style VAE，同时尽量不改动 Wan2.1 原生 VAE 结构和 loss。

---

## 2. 结论先行

推荐执行策略：

1. 先做一个 **参考基线对齐版**：
   - 沿用现有 LiDAR sampler / rangemap 恢复逻辑
   - 明确复现实验假设：`row repeat x4`、列下采样到 `1800`、随机列 crop 到 `896`
   - 用它作为后续 Wan 路线的 apples-to-apples 对照
2. 再做一个 **可跑通的 Wan pilot 版本**：
   - `29` 帧
   - `512 x 896` LiDAR crop
   - `4` GPU
   - 目标是验证 `Wan backbone + 3-channel repeated rangemap + row repeat` 是否稳定收敛
3. pilot 收敛后，再做 **主版本**：
   - `29` 帧
   - 优先 `512 x 896` 跑满
   - 如果显存和速度允许，再继续做 `512 x 1792` refinement
4. 只有当以下条件同时满足时，才在 CoLiGen 主线上替换当前 `CI8x8`：
   - reconstruction 不明显退化
   - latent shape 符合 `B x 16 x 8 x H/8 x W/8`
   - joint training 前 2k iter 稳定、无 NaN、loss 曲线正常

换句话说，这条路线值得试，但要保留当前 `CI8x8 + 29 -> 8 temporal adapter` 作为回退方案。

---

## 3. 和当前代码库的关系

### 3.1 当前基线

当前仓库里的 LiDAR tokenizer 仍然是逐帧 `CI8x8` 路线：

- LiDAR 分支 `T == 1`
- `temporal_compression_factor = 1`
- `29` 帧 LiDAR 会得到 `29` 个 latent 时间步

而当前 video 分支是 Wan2.1：

- `temporal_compression_factor = 4`
- `29` 帧 video 会得到 `8` 个 latent 时间步

这也是当前 proposal 里需要显式写 `29 -> 8 temporal adapter` 的根本原因。

### 3.1.1 参考仓库实际上做了什么

`cosmos-transfer-lidargen` 中 Waymo LiDAR tokenizer 的实验配置是：

- `network = continuous_image`
- `loss = image`
- `data_train/data_val = lidar_range_map_rRow4_waymo`
- `precision = float32`
- `ema = disabled`
- `load_path = checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/.../autoencoder.pt`

这说明官方路径并没有直接把 image tokenizer 训练脚本改造成 3D video tokenizer，而是：

- 复用 image-style continuous AE 训练框架
- 用 LiDAR rangemap dataloader 提供 `images` 键
- 通过 LiDAR 数据分布微调现有 tokenizer

所以本文档的新方案最好表述为：

- 参考基线：`2D image-style CI8x8 LiDAR tokenizer`
- 本文新增路线：`3D video-style Wan LiDAR VAE`

这样后续所有比较都会更清楚，也更方便判断收益究竟来自“LiDAR domain adaptation”还是“3D temporal tokenizer”。

### 3.2 本方案要改变什么

本方案只替换 LiDAR tokenizer，不动 video 主干：

- video 分支继续用当前 `Wan2.1` tokenizer
- multiview 主干继续保持 `29f / state_t = 8`
- LiDAR tokenizer 改成 `Wan-style 3D VAE`
- joint training 前不再做外部时间压缩，而是直接提取 `t = 8` LiDAR latents
- LiDAR 输入接口优先保持为 `3-channel`，以最大化复用 `Wan2.1_VAE.pth`

---

## 4. 数据定义与输入输出

## 4.1 Waymo LiDAR 输入

这里改为 **直接从 Waymo LiDAR tar 数据在线恢复 range map**，而不是依赖预先落盘的 `.npy` / `.pt` range-map 文件。

默认假设沿用当前 LiDAR repo 已经在用的数据组织方式：

```text
datasets/waymo_lidar_training/
├── lidar/
│   ├── <sample_key>.tar
│   └── ...
├── metadata/
│   ├── <sample_key>.npz
│   └── ...
```

其中每个 `lidar/<sample_key>.tar` 内部按帧保存：

```text
0.lidar_row.npz
0.lidar_col.npz
0.lidar_range.npz
1.lidar_row.npz
1.lidar_col.npz
1.lidar_range.npz
...
```

训练时在线读取这三个原始字段并恢复单帧 range map：

1. 从 tar 中读取 `lidar_row / lidar_col / lidar_range`
2. 将点填回到 `128 x 3600` 的原始 range image 网格
3. 对选中的 `29` 帧堆叠为 `T x 128 x 3600`
4. 列下采样：`3600 -> 1800`
5. 行重复 `x4`：`128 -> 512`
6. 宽度处理：
   - pilot：裁成 `896`
   - optional refine：裁成 `1792`
7. 归一化得到单通道 rangemap
8. 通道维复制三次，得到 3-channel 输入

也就是说，v1 的数据读取入口是“tar 内 row/col/range 三元组 -> 在线恢复 range map -> 预处理 -> 训练”，不是“先离线生成 range map 再训练”。

这里建议尽量贴近参考仓库现有做法，不要从第一天就把 dataloader 完全重写。更稳的做法是：

- 优先复用 `imagetolidar_dataloader.py` 中已经验证过的 tar 解析、顺序采样、crop 和 range-map 归一化逻辑
- 在此基础上新增一个 `is_video_tokenizer=True` 的 Waymo 配置
- 优先复用其 `to_three_channels=\"repeat\"` 这一路径，把 LiDAR rangemap 组织成 Wan VAE 直接可吃的 3 通道 video tensor

这样可以把“数据恢复是否正确”和“Wan 3D tokenizer 是否有效”这两个问题拆开，同时避免通道 surgery 带来的额外变量。

最终训练输入：

- pilot：`B x 3 x 29 x 512 x 896`
- refine：`B x 3 x 29 x 512 x 1792`

### 4.2 配套 metadata

为了和现有 Waymo LiDAR 数据采样方式兼容，建议继续使用同一套 `metadata/<sample_key>.npz`：

- `timestamps_list`
- `frame_indices`
- `pose_list`

在本方案里它们的作用是：

- 负责稳定采样 `29` 帧 clip
- 保持和 camera clip 的时间索引可对齐
- 为后续 joint training / ULA / 可视化保留 pose 和 timestamp

如果 v1 只训练 LiDAR VAE，本阶段并不强依赖 camera 图像本身，但仍建议保留 metadata 读取逻辑，不要把 LiDAR VAE 训练做成完全脱离多模态时间轴的独立脚本。

### 4.3 latent 输出

因为采用 `stride = (4, 8, 8)`：

- `29` 帧 -> `8` 个 latent 帧
- `512 -> 64`
- `896 -> 112`
- `1792 -> 224`

因此输出 shape 为：

- pilot：`B x 16 x 8 x 64 x 112`
- refine：`B x 16 x 8 x 64 x 224`

这正好和 CoLiGen 当前 `state_t=8` 对齐。

---

## 5. 模型设计

## 5.1 Backbone

直接复用 `third_party_ref_repo/Wan2.1/wan/modules/vae.py` 中的 `WanVAE_` 结构，保留：

- causal 3D conv
- chunked encode / decode
- time-aware cache
- temporal downsample / upsample 逻辑

不改动的部分：

- `3-channel in / 3-channel out`
- `z_dim = 16`
- `dim_mult`
- `temporal downsample pattern`
- overall encoder / decoder depth

v1 不改首尾层，不做结构 surgery。这样做的原因是：

1. 可以完整加载 `Wan2.1_VAE.pth`
2. 输入接口与 video tokenizer 完全一致
3. latent 空间几何最接近现有 video 分支
4. 更适合后续和 video latent 做联合去噪

## 5.2 从 Wan2.1 权重初始化

为避免从头训练，v1 直接完整加载官方 `Wan2.1_VAE.pth`。

由于输入和输出仍然保持 `3-channel`：

- 不需要修改 encoder 输入层
- 不需要修改 decoder 输出层
- 不需要做 weight surgery
- 直接继承 Wan2.1 原生预训练权重即可

## 5.3 为什么主方案先用 3-channel repeated rangemap

主方案先用 `3-channel repeated rangemap`，原因有四点：

1. 可以最大化复用 `Wan2.1` 原始 VAE 权重
2. 可以把“LiDAR 模态迁移”和“通道结构改造”两个问题分开
3. 可以让 LiDAR tokenizer 与现有 video tokenizer 的接口完全一致
4. 更适合当前目标：让 LiDAR latent 尽快接入和 video latent 的联合去噪

需要注意的是，3 通道只是接口兼容策略，不意味着 LiDAR latent 会天然和 RGB latent 对齐。真正决定联合去噪稳定性的仍然是：

- latent shape
- temporal length
- normalization
- ULA / affine alignment

可以保留一个后续 ablation：

- `3-channel repeated rangemap Wan LiDAR VAE`（主方案）
- `1-channel surgery Wan LiDAR VAE`（对照）

---

## 6. 训练目标

v1 的训练目标不是重新设计一套 LiDAR 专属 VAE loss，而是尽量复用现有 Wan / Cosmos 的训练范式，只改变输入数据分布。

因此，loss 建议先 **完全复用现有 Wan2.1 VAE / Cosmos 侧的原始配置**：

- 不新增 LiDAR 特化损失
- 不修改原有 loss 组合
- 不单独为 LiDAR 改权重调度

这样做的目标是：

1. 先得到一个尽量接近现有 video tokenizer 的 LiDAR tokenizer
2. 把变量收敛到“输入模态变化”本身
3. 让后续和 video latent 的联合去噪更容易归因和调试

### 6.1 v1 不做的事

第一版不引入：

- discriminator / GAN loss
- point-cloud Chamfer loss 反传
- occupancy supervision
- joint RGB-LiDAR co-training

原因很简单：先把 `Wan LiDAR VAE` 训稳、导出、接入 joint training，再考虑更复杂的 loss。

---

## 7. 训练阶段

## 7.1 Phase 0: 数据与 shape sanity

目标：

- 确认 dataloader 输出 shape 正确
- 确认 `29 -> 8` 的时间压缩成立
- 确认 encoder / decoder forward 无 shape mismatch

验收条件：

- 输入 `B x 3 x 29 x 512 x 896`
- latent `B x 16 x 8 x 64 x 112`
- decode 后回到 `B x 3 x 29 x 512 x 896`

建议只跑：

- 1 GPU
- 2 batch
- forward only / 10 iter

## 7.2 Phase 1: 8-sample overfit smoke test

目标：

- 验证模型和 loss 能否快速过拟合小数据
- 发现最早期 bug：梯度爆炸、KL 异常、decode 模糊、cache bug

配置建议：

- 数据：8 个 clips
- 输入：`29 x 512 x 896`
- batch size：1
- precision：bf16 mixed precision，KL 路径转 fp32
- iter：500 ~ 1000

验收条件：

- reconstruction loss 明显下降
- 可视化上能重建出静态结构、近处车辆轮廓和道路边界

## 7.3 Phase 2: pilot run

目标：

- 在可控显存下确认这条路线值得继续

配置建议：

- 输入：`29 x 512 x 896`
- GPU：4 x A100-80GB
- global batch：4
- lr：`5e-5`
- grad clip：`1.0`
- warmup：`1k iter`
- max_iter：`10k`
- save / val：每 `500` iter

验收条件：

- 无 NaN / 无持续性发散
- val reconstruction 指标明显优于 random init
- temporal reconstruction 明显比逐帧 `CI8x8` 更自然

## 7.4 Phase 3: main run

目标：

- 获得可替换现有 `CI8x8` 的主版本 checkpoint

推荐两种路线：

### Route A: 先把 896 版本做扎实

- 输入：`29 x 512 x 896`
- GPU：8 x A100-80GB
- global batch：8
- lr：`5e-5`
- max_iter：`20k ~ 30k`

适合：

- 快速产出可用 tokenizer
- 尽快接入 CoLiGen joint training

### Route B: 宽度 refinement

- 以 Route A checkpoint 为初始化
- 输入：`29 x 512 x 1792`
- GPU：8 x A100-80GB
- 更小 global batch
- max_iter：`5k ~ 10k`

适合：

- 显存允许
- 希望后续 joint training 使用全宽 LiDAR latent

建议默认先走 Route A，再决定是否做 Route B。

---

## 8. ULA 统计量导出

LiDAR VAE 训练完成后，不要直接把原生 latent 扔进 joint model；需要先做 `ULA`。

## 8.1 需要的统计量

记：

- `mu_C, sigma_C`：当前 RGB Wan tokenizer 的 normalization 参数
- `mu_C^D, sigma_C^D`：Wan RGB tokenizer 在 Waymo camera 数据上的经验统计
- `mu_L^D, sigma_L^D`：LiDAR Wan tokenizer 在 Waymo LiDAR 数据上的经验统计

然后按 `UniDriveDreamer` 的 affine calibration 计算：

```text
sigma_L = sigma_L^D * sigma_C / sigma_C^D
mu_L = mu_L^D - mu_C^D * sigma_L^D / sigma_C^D + mu_C * sigma_L^D / sigma_C^D
```

最终导出：

```text
waymo_wan_lidar_ula_stats.pt
{
  "mu_l": ...,
  "sigma_l": ...,
  "mu_l_dataset": ...,
  "sigma_l_dataset": ...,
  "mu_c_dataset": ...,
  "sigma_c_dataset": ...,
}
```

## 8.2 ULA 验收

接入 joint training 前检查：

- LiDAR latent 归一化后均值 / 方差量级和 RGB latent 接近
- 融合前后没有极端 channel 爆炸
- joint model 前 1k iter 的 loss 不因为 LiDAR 分支异常抖动

---

## 9. 代码改动清单

为了避免改动 `third_party_ref_repo/Wan2.1` 原始参考代码，建议在本仓库新增一个自包含的训练工具目录；但数据层尽量复用 `cosmos-transfer-lidargen` 已有组件，而不是把 tar 解析和采样逻辑整份复制出来。

## 9.1 新增目录

```text
tools/waymo_wan_vae/
```

建议新增文件：

```text
tools/waymo_wan_vae/
├── __init__.py
├── dataset.py
├── model.py
├── init_from_wan.py
├── losses.py
├── train.py
├── eval_recon.py
├── export_ula_stats.py
└── configs/
    ├── waymo_wan_lidar_smoke.yaml
    ├── waymo_wan_lidar_pilot.yaml
    ├── waymo_wan_lidar_full_896.yaml
    └── waymo_wan_lidar_full_1792.yaml
```

## 9.2 每个文件负责什么

### `dataset.py`

负责：

- 直接读取 `datasets/waymo_lidar_{training,validation}/lidar/<sample_key>.tar`
- 在线解析 `lidar_row / lidar_col / lidar_range`
- 恢复原始 `128 x 3600` range map
- 读取 `metadata/<sample_key>.npz`
- 采样 `29` 帧 clip
- 列下采样 `3600 -> 1800`
- 行重复 `128 -> 512`
- 输出 3-channel repeated rangemap tensor
- pilot / refine 两种 crop

说明：

更推荐的实现顺序是：

- v1：薄封装现有 sampler，只新增 `29` 帧顺序采样、`is_video_tokenizer=True`、`3-channel repeat` 输出和 crop 策略开关
- v2：等训练稳定后，再决定是否把关键逻辑收敛到本仓库自有 dataloader

只有当跨 repo 依赖确实成为维护负担时，再把以下最小逻辑内聚到本仓库：

- `load_each_frame_from_tar_data`
- `parse_range_map_tar`
- `sample_frame_indices`
- `normalize_range_map`
- `RangeMapDownsampler`

### `model.py`

负责：

- 基于 Wan VAE 构造 `WanLidarVAE`
- 保持原生 3-channel IO
- 提供 `encode / decode / forward`
- 保留 Wan 的 chunked causal forward 逻辑

### `init_from_wan.py`

负责：

- 加载官方 `Wan2.1_VAE.pth`
- 直接导出可训练初始化权重
- 不做通道适配

### `losses.py`

负责：

- 尽量薄封装现有 Wan / Cosmos loss 调用
- 不引入 LiDAR 特化 loss
- 保持和 video tokenizer 训练范式一致

### `train.py`

负责：

- DDP / torchrun 训练入口
- checkpoint 保存
- validation 可视化
- wandb / tensorboard logging

### `eval_recon.py`

负责：

- 输出 range-map RMSE / MAE / Relative Error
- 可选 point-cloud Chamfer / F-score
- 导出重建视频和点云可视化

### `export_ula_stats.py`

负责：

- 扫 Waymo RGB / LiDAR latent
- 计算 `mu / sigma`
- 导出 `ULA` 所需统计量

## 9.3 接入主仓库 tokenizer

训练完后再新增：

```text
cosmos_transfer2/_src/predict2/tokenizers/wan2pt1_lidar.py
```

职责：

- 封装训练好的 `WanLidarVAE`
- 对外暴露与当前 tokenizer 一致的接口：
  - `encode`
  - `decode`
  - `get_latent_num_frames`
  - `get_pixel_num_frames`
  - `temporal_compression_factor`
  - `spatial_compression_factor`

这一步完成后，`extract_waymo_latents.py` 和后续 joint training 就可以直接切到新 LiDAR tokenizer。

---

## 10. 训练命令

以下命令是本文档的标准执行入口。假设上述脚本已经按计划落地。

## 10.1 初始化 LiDAR Wan 权重

```bash
cd /root/workspace/cosmos-transfer2.5

python -m tools.waymo_wan_vae.init_from_wan \
  --wan_ckpt /data/checkpoints/Wan2.1/Wan2.1_VAE.pth \
  --output /data/checkpoints/waymo_wan_lidar/init_from_wan.pt
```

## 10.2 Smoke test

```bash
cd /root/workspace/cosmos-transfer2.5

CUDA_VISIBLE_DEVICES=0 python -m tools.waymo_wan_vae.train \
  --config tools/waymo_wan_vae/configs/waymo_wan_lidar_smoke.yaml \
  model.init_ckpt=/data/checkpoints/waymo_wan_lidar/init_from_wan.pt \
  data.root=/data/waymo_lidar_training
```

## 10.3 Pilot run

```bash
cd /root/workspace/cosmos-transfer2.5

torchrun --nproc_per_node=4 --master_port=12371 \
  -m tools.waymo_wan_vae.train \
  --config tools/waymo_wan_vae/configs/waymo_wan_lidar_pilot.yaml \
  model.init_ckpt=/data/checkpoints/waymo_wan_lidar/init_from_wan.pt \
  data.root=/data/waymo_lidar_training
```

## 10.4 Main run

```bash
cd /root/workspace/cosmos-transfer2.5

torchrun --nproc_per_node=8 --master_port=12372 \
  -m tools.waymo_wan_vae.train \
  --config tools/waymo_wan_vae/configs/waymo_wan_lidar_full_896.yaml \
  model.init_ckpt=/data/checkpoints/waymo_wan_lidar/init_from_wan.pt \
  data.root=/data/waymo_lidar_training
```

## 10.5 Optional 1792 refinement

```bash
cd /root/workspace/cosmos-transfer2.5

torchrun --nproc_per_node=8 --master_port=12373 \
  -m tools.waymo_wan_vae.train \
  --config tools/waymo_wan_vae/configs/waymo_wan_lidar_full_1792.yaml \
  model.init_ckpt=/data/checkpoints/waymo_wan_lidar/full_896_last.pt \
  data.root=/data/waymo_lidar_training
```

## 10.6 Reconstruction evaluation

```bash
cd /root/workspace/cosmos-transfer2.5

python -m tools.waymo_wan_vae.eval_recon \
  --ckpt /data/checkpoints/waymo_wan_lidar/full_896_best.pt \
  --split val \
  --data_root /data/waymo_lidar_validation \
  --max_samples 100
```

## 10.7 导出 ULA 统计量

```bash
cd /root/workspace/cosmos-transfer2.5

python -m tools.waymo_wan_vae.export_ula_stats \
  --lidar_ckpt /data/checkpoints/waymo_wan_lidar/full_896_best.pt \
  --output /data/checkpoints/waymo_wan_lidar/waymo_wan_lidar_ula_stats.pt \
  --lidar_data_root /data/waymo_lidar_training \
  --max_samples 2000
```

---

## 11. 验收标准

## 11.1 tokenizer 本身

最小验收：

- `29` 帧输入可稳定编码为 `8` 帧 latent
- decode 无明显 temporal flicker
- loss 曲线稳定，无持续发散
- reconstruction 在静态结构上明显优于 random init

推荐验收：

- range-map RMSE / MAE 不显著劣于当前 `CI8x8` baseline
- point-cloud Chamfer / F-score 达到可接受水平
- 896 版本已足够稳定，可以导出并接入主线

## 11.2 joint training readiness

只有满足以下条件才切主线：

1. `encode/decode` 接口已封装成 tokenizer
2. `extract_waymo_latents.py` 已能抽取 `t = 8` LiDAR latents
3. `ULA` 统计量已导出
4. joint training 试跑 `1k ~ 2k iter` 无 NaN / 无异常 loss 爆炸

---

## 12. 风险与应对

## 12.1 风险：29 帧全宽太吃显存

应对：

- 主版本先锁 `512 x 896`
- 1792 只做 refinement

## 12.2 风险：3-channel repeat 虽然稳定，但重建收益不明显

应对：

- 不强行替换 `CI8x8`
- 先把它作为“面向联合去噪的兼容 tokenizer”
- 后续再用 `1-channel surgery` 做重建向 ablation

## 12.3 风险：reconstruction 提升不明显

应对：

- 不强行替换 `CI8x8`
- 回退到 `CI8x8 + 29 -> 8 temporal adapter`

## 12.4 风险：ULA 后 joint training 仍不稳

应对：

- 先只接入新的 `LiDAR VAE`
- `ULA` 单独 ablate
- 必要时在 LiDAR 分支前再加一个轻量 `1x1` affine adapter

## 12.5 风险：你说的“Waymo 原始数据”其实是官方 TFRecord

本文档当前默认的“原始数据”指的是：

- 未预先生成 range map 文件
- 但已经整理成 tar 格式的 `lidar_row / lidar_col / lidar_range` 原始三元组

如果你后面想直接从 Waymo 官方 TFRecord 解码，本方案仍然成立，但需要额外实现：

- TFRecord 解析
- range image 投影
- 与当前 `metadata/*.npz` 对齐的 clip 采样逻辑

这不建议放进 v1。

---

## 13. 建议执行顺序

建议严格按下面顺序推进：

1. 先复核并固化参考基线配置：`CI8x8 + lidar_range_map_rRow4_waymo`
2. 完成 `dataset.py + model.py + init_from_wan.py`
3. 跑通 Phase 0 shape sanity
4. 跑 Phase 1 overfit smoke test
5. 跑 Phase 2 pilot
6. pilot 指标同时对比参考基线后，再跑 main run
7. 导出 ULA stats
8. 新增 `wan2pt1_lidar.py`
9. 修改 `extract_waymo_latents.py`
10. 用新 LiDAR latents 做 joint training 小规模试跑

不要一开始就同时做：

- 1792 全宽训练
- joint training
- ULA
- 新评估指标

这样会把问题耦在一起，不利于排查。

---

## 14. 预计产出

完成后应至少得到以下产物：

```text
/data/checkpoints/waymo_wan_lidar/
├── init_from_wan.pt
├── full_896_best.pt
├── full_896_last.pt
├── full_1792_best.pt            # optional
├── waymo_wan_lidar_ula_stats.pt
└── recon_eval/
```

以及代码产物：

```text
tools/waymo_wan_vae/*
cosmos_transfer2/_src/predict2/tokenizers/wan2pt1_lidar.py
```

---

## 15. 最终建议

这条路线适合作为 CoLiGen 的 **增强主线**，但不适合作为唯一赌注。

最稳的推进方式是：

- 短期主线：继续维护 `CI8x8 + 29 -> 8 temporal adapter`
- 对齐参考：先把官方 `CI8x8 + Waymo rangemap` 基线吃透并固定成对照
- 并行探索：推进本文档的 `UniDriveDreamer-style Wan LiDAR VAE`

如果新 tokenizer 在 reconstruction 和 joint training 稳定性上都通过验收，就切换到新方案；否则保留为论文中的一条有价值 ablation / future extension。

---

## 16. 参考

- UniDriveDreamer: A Single-Stage Multimodal World Model for Autonomous Driving
  https://arxiv.org/pdf/2602.02002
- Wan2.1 VAE 参考实现：
  `/root/workspace/cosmos-transfer2.5/third_party_ref_repo/Wan2.1/wan/modules/vae.py`
- 当前 Waymo video / multiview 主线：
  `/root/workspace/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/waymo_posttrain.py`
- 当前 LiDAR tokenizer 基线：
  `/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/configs/experiments/cosmos_lidar_tokenizer_waymo.py`
