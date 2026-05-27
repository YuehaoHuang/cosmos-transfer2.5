# Waymo LiDAR 原生 Rangemap 到 Wan2.1 VAE 实验记录

更新日期：2026-04-26

## 目标

本文记录当前使用图像/视频 VAE 对 Waymo TOP LiDAR range map 做 encode/decode 的 smoke-test 路径。
2026-05-26 起，29 帧 LiDAR 训练主线在官方 single-view post-training 框架中读取无损 normalized rangemap，并在线执行本文验证过的 Wan2.1 encode：`[1,3,29,704,1280] -> [1,16,8,88,160]`。训练不再将 rangemap MP4 二次 encode 为 target，见 `docs/waymo_lidar_singleview_posttraining.md`。

当前测试流程：

1. 读取 Waymo raw TOP LiDAR tar。
2. 将点云投影到 Waymo TOP 原生 range map，尺寸为 `64x2650`。
3. 将 range map 转成 Wan 兼容的 3 通道视频张量。
4. 使用冻结的 Wan2.1 VAE 做 encode 和 decode。
5. 将 decode 后的 range map 反归一化回米制距离，并输出 rangemap 视频和点云视频。

点云可视化使用的是 Wan2.1 VAE encode/decode 后的结果，不是 Cosmos LiDAR tokenizer/S3 decode 的输出。

## 代码

主脚本：

```bash
scripts/smoke_waymo_lidar_wan21_vae.py
```

参考代码路径：

- Waymo raw 点云投影：`/root/workspace/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py`
- Wan2.1 VAE latent 提取参考：`extract_waymo_latents.sh`
- 点云相机视角/可视化参考：`cosmos_predict1.tokenizer.inference.lidar_cli`

环境：

```bash
conda activate drivesync
```

## 固定样本

下面所有指标都使用同一个 validation segment：

```text
split: validation
segment_key: 10203656353524179475_7625_000_7645_000
frames: 29
raw lidar tar: /team/hyh/data/rds_hq_waymo/validation/lidar_raw/10203656353524179475_7625_000_7645_000.tar
```

## 当前置顶方案

使用 Waymo TOP raw 点云直接投影出的压缩宽度 range map，而不是已经转换给 LiDAR tokenizer 使用的 `128x3600` 表示，也不是先生成 `64x2650` 后再 resize。

推荐预处理：

- 投影 range map：`64x1280`
- 不做下采样：`downsample_factor_row=1`，`downsample_factor_col=1`
- 通道模式：`repeat_depth`
- 行重复：`repeat_row=11`
- spatial padding 前的 VAE 逻辑输入：`[1, 3, 29, 704, 1280]`
- VAE 输入：`[1, 3, 29, 704, 1280]`
- latent shape：`[1, 16, 8, 88, 160]`
- spatial tokens：`88 x 160 = 14080`，和当前 Transfer2 single-view `704x1280` latent 网格完全对齐
- decode 还原：对 3 个 decoded channels 求均值
- 当前 canonical clip 指标：Range MAE/RMSE 为 `0.9030m/3.3258m`，normalized MSE 为 `0.006233`
- 点云渲染结果：`/team/hyh/data/waymo_wan_lidar_vae_smoke/validation/exp_native64x1280_repeatrow11_mean_full/point_cloud_front_view_vehicle.mp4`

原因：

- Wan2.1 VAE 面向图像/视频输入，原始 64 条 scan lines 的垂直像素支撑太少；`repeat_row=11` 可以把输入高度提升到 `704`。
- `1280x11` 的 latent spatial grid 是 `88x160`，和当前 LiDAR single-view post-training 的 `704x1280` target/control 直接对齐，不需要宽度 padding 或额外位置编码改动。
- `1312x11` 的重建指标略好，但 latent width 是 `164`，会偏离官方 single-view 的 `88x160` 主线；当前选择 `1280x11` 是为了训练形状一致性。
- 对 decode 后的 3 个通道求均值，比只取第 0 通道更好。
- `concat_inv_depth` 通道策略已经测试过，效果差于直接把 depth 重复到 3 个通道。

## 推荐命令

高质量运行，包含 rangemap 视频和 Plotly 点云可视化：

```bash
conda activate drivesync

CUDA_VISIBLE_DEVICES=0 python scripts/smoke_waymo_lidar_wan21_vae.py \
  --preprocess-mode waymo_top_64x2650 \
  --raw-lidar-root /team/hyh/data/rds_hq_waymo \
  --lidar-utils-repo /team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen \
  --wan-vae-path /team/hyh/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth \
  --split validation \
  --segment-key 10203656353524179475_7625_000_7645_000 \
  --frame-start 0 \
  --num-frames 29 \
  --device cuda \
  --native-n-cols 1280 \
  --repeat-row 11 \
  --vis-pcd \
  --pcd-renderer plotly \
  --pcd-camera-view front_view \
  --pcd-display-frame vehicle \
  --pcd-workers 12 \
  --output-dir /team/hyh/data/waymo_wan_lidar_vae_smoke/validation/exp_native64x1280_repeatrow11_mean_full
```

置顶方案参数：

```text
native_n_cols = 1280
downsample_factor_row = 1
downsample_factor_col = 1
repeat_row = 11
repeat_col = 1
input_channel_mode = repeat_depth
decode_channel_mode = mean
wan_spatial_align = 8
```

## 生成训练主线：Single-View Layout Control 与 Online Wan Encode

当前独立生成 29 帧 LiDAR rangemap video 的训练路径不使用 depth 作为条件，也不和 camera video 联动。默认 dataloader 直接枚举 raw tar，在线投影 target 和 layout，再由 Wan VAE 在线 encode target。训练必需输入为：

```text
/team/hyh/data/rds_hq_waymo/<split>/lidar_raw/
└── <segment_key>.tar
```

以下数据准备命令仅用于生成调试视频或自定义 caption，不是默认训练前置步骤：

```bash
conda activate drivesync

python scripts/prepare_waymo_lidar_singleview_posttrain_dataset.py \
  --split training \
  --output-root /team/hyh/data/waymo_singleview_lidar_posttrain_raw1280 \
  --raw-waymo-root /team/hyh/data/rds_hq_waymo \
  --sample-list-source lidar \
  --range-map-source raw \
  --native-n-cols 1280 \
  --repeat-row 11 \
  --caption-mode fixed
```

训练：

```bash
./train_waymo_lidar_singleview_chunked.sh
```

训练时默认读取 `/team/hyh/data/rds_hq_waymo/training/lidar_raw/*.tar` 并使用固定 LiDAR caption；`.mp4` 和 `.npz` 不作为默认输入。验证集 smoke 需附加 `--raw-lidar-split validation`。

评估：

```bash
python scripts/evaluate_waymo_lidar_rangemap_generation.py \
  --generated-video /path/to/generated.mp4 \
  --gt-video /team/hyh/data/waymo_singleview_lidar_posttrain_raw1280/validation/videos/<sample>.mp4 \
  --layout-video /team/hyh/data/waymo_singleview_lidar_posttrain_raw1280/validation/rangemap_layout/<sample>.mp4
```

## 历史记录：在线 Wan2.1 LiDAR VAE

以下内容仅保留早期实验记录，不是当前 29 帧 rangemap video 生成训练入口。该历史路径当时直接以 Waymo video dataloader 为时钟：

1. video latent 从真实 Wan latent cache 读取：`/data/waymo/chunk/training/samples`
2. LiDAR 从 `/data2/rds_hq_waymo/training/lidar_raw/<segment_key>.tar` 按同一个 `waymo_lidar_frame_indices` 在线读取 29 帧
3. 使用本页置顶方案在线 Wan2.1 VAE encode，得到 `16 x 8 x 88 x 160`
4. 冻结 video DiT，只训练 full 28-layer LiDAR expert；默认从 frozen video DiT 拷贝同名同形状权重初始化

推荐入口：

```bash
conda activate drivesync

NUM_GPUS=7 \
BATCH_SIZE=1 \
LIDAR_NUM_BLOCKS=28 \
VIDEO_KV_EVERY_N_LAYERS=4 \
CHECKPOINT_LIDAR_BLOCKS=true \
INIT_LIDAR_FROM_VIDEO=true \
./train_waymo_video_lidar_one_way_wan21_online.sh
```

当前实测约束：

- `CHECKPOINT_LIDAR_BLOCKS=true` 是 7-GPU 在线 VAE 的稳定默认。
- `CHECKPOINT_LIDAR_BLOCKS=false` 单卡 smoke 可以通过，但 7-GPU DDP 会在首个 forward 接近 `77GB/GPU` 后再申请约 `2.2GB`，导致 OOM。
- 在线 Wan2.1 VAE encode 期间会临时开启 CUDA SDPA `flash+mem+math`，encode 结束后恢复训练主干的 `flash_only`；这是因为 Wan VAE attention head_dim 大于 FlashAttention fast path 支持范围。

当前运行记录：

```text
tmux session: wan21_online_train_20260426_004355
output_dir: /data2/waymo_video_lidar_one_way_expert/real_video_wan21_online_lidar14_vkv4_cp_b1_7gpu_20260426_004355
log: /data2/waymo_video_lidar_one_way_expert/real_video_wan21_online_lidar14_vkv4_cp_b1_7gpu_20260426_004355/train.log
```

## 历史记录：训练用 Wan2.1 LiDAR latent cache

早期实验曾生成配对 LiDAR latent cache。该流程不属于当前官方 single-view rangemap layout 训练主线，仅作为历史记录保留。当前 cache writer 默认已切到 `wan21_native64x1280_repeatrow11_v1`；旧 `1312` cache 仍可通过 payload contract 读取。

默认输入：

```text
video latent: /data/waymo/chunk/training/samples
raw lidar: /data2/rds_hq_waymo/training/lidar_raw
paired cache root: /data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1280_repeatrow11
```

单样本 smoke：

```bash
conda activate drivesync

CUDA_VISIBLE_DEVICES=0 python scripts/cache_waymo_lidar_wan21_latents.py \
  --split validation \
  --video-latent-dir /data/waymo/chunk/validation/samples \
  --paired-cache-root /tmp/waymo_wan21_lidar_cache_smoke \
  --sample-key 10203656353524179475_7625_000_7645_000_0 \
  --device cuda \
  --max-samples 1 \
  --overwrite
```

训练集分片生成：

```bash
conda activate drivesync

CUDA_VISIBLE_DEVICES=0 python scripts/cache_waymo_lidar_wan21_latents.py \
  --split training \
  --num-shards 16 \
  --shard-index 0 \
  --device cuda
```

2026-04-26 旧 `1312` cache 补齐任务：

```text
tmux sessions: wan21_train_cache_shards42_20260426_234620_s{0..41}
log dir: /data2/waymo_paired_latents/logs/wan21_train_cache_shards42_20260426_234620
cache root: /data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11
target count: 13,900
watcher: wan21_cache42_then_train_fallback_20260426_234638
watcher log: /data2/waymo_paired_latents/logs/wan21_cache42_then_train_fallback_20260426_234638.log
```

当前机器只暴露 7 张 A100，因此使用 42 shards，每张 GPU 6 个 cache 进程。cache writer 默认不 `--overwrite`，已有样本会跳过。
watcher 会在所有 cache shard 结束后先启动 no-checkpoint DDP；如果 DDP 非 0 退出，则自动切到 FSDP2 入口。

2026-04-27 状态：

- cache 已补齐：`lidar/*.pt = 13,900 / 13,900`。
- no-checkpoint DDP 内存边界已确认：`14 blocks + VIDEO_KV_EVERY_N_LAYERS=4 + batch_size_per_rank=1` 在显式释放 frozen-video 中间张量后可过 `step=1`，但下一轮仍在 frozen video MLP 处 OOM；峰值约 `78.1GB/GPU`，再次申请 `2.20GB` 失败。因此该配置不作为稳定主训练。
- FSDP2 fallback 当前不可直接用于主线：one-way forward 直接调用 LiDAR block 内部模块，绕过 FSDP2-wrapped module `forward`，会触发 Tensor/DTensor 混用错误。继续 FSDP2 需要把 LiDAR block 调用重构成 wrapper-owned forward。
- 当前稳定主训练改为 checkpoint DDP：session `wan21_cp_b2_train_20260427_002810`，输出 `/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_lidar14_vkv4_cp_b2_7gpu_after_cache42_20260427_002810`；配置 `batch_size_per_rank=2`、`global_batch=14`、`checkpoint_lidar_blocks=True`、`video_kv_every_n_layers=4`。已到 `step=10`，loss `3.3592 -> 1.5709`，稳态 `42.15s/step`，显存约 `40.8-41.9GB/GPU`。

2026-04-29 检查状态：

- 已停止上述 checkpoint DDP run。最后可靠 checkpoint 为 `step_004000.pt`。
- `step_001500` 到 `step_004000` 的同一样本推理指标基本不变，说明原配置继续训练收益低。
- RF/scheduler 方向与官方 Predict2 一致；主要问题是 LiDAR expert 之前从随机初始化训练。训练脚本新增 `--init-lidar-from-video`，按 key 和 shape 拷贝 frozen video DiT 的同形状权重到 LiDAR expert，跳过 LiDAR 特有或 shape 不匹配 tensor。
- `train_waymo_video_lidar_one_way_wan21_cache.sh` 默认 `INIT_LIDAR_FROM_VIDEO=true`。需要复现实验旧随机初始化时，显式设置 `INIT_LIDAR_FROM_VIDEO=false`。

2026-04-29 方案调整：

- 主线改回 full 28-block LiDAR expert，并默认 `--init-lidar-from-video`。这会拷贝完整 28 个 video block 的同形状权重到 LiDAR expert，保证 LiDAR 分支从 video WFM prior 出发，而不是 14-block 随机初始化。
- full 28-block 的默认稳定配置先用 `CHECKPOINT_LIDAR_BLOCKS=true`、`BATCH_SIZE=1` 起测；确认显存后再考虑提高 batch 或关 checkpoint。

2026-05-05 训练状态：

- full 28-block 主训练已人工停止。run 目录：
  `/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_full28_vkv4_initvideo_cp_b2_7gpu_formal_20260429_005200`
- 稳定配置为 `BATCH_SIZE=2`、`NUM_GPUS=7`、`global_batch=14`、`CHECKPOINT_LIDAR_BLOCKS=true`、`INIT_LIDAR_FROM_VIDEO=true`、`VIDEO_KV_EVERY_N_LAYERS=4`。
- 最后保存 checkpoint 为 `checkpoints/step_007000.pt`；训练日志实际到约 `step=7025`。稳态速度约 `84s/step`，显存约 `54GB/GPU`。
- 4-step latent 指标显示 `step_007000` 未优于 `step_005000`，因此当前选择 `step_005000.pt` 为 best candidate；`step_003000.pt` 用于补早期 4-step decode 可视化。

已记录的同一 validation sample 4-step 指标：

| checkpoint | sample_mse | sample_mae | single_step_denoised_mse |
|---|---:|---:|---:|
| `step_005000.pt` | 0.733270 | 0.645834 | 0.250332 |
| `step_007000.pt` | 0.824434 | 0.691515 | 0.266189 |

2026-05-05 `step_003000.pt` 4-step decode 可视化：

- checkpoint：`/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_full28_vkv4_initvideo_cp_b2_7gpu_formal_20260429_005200/checkpoints/step_003000.pt`
- validation sample：`10203656353524179475_7625_000_7645_000_0`
- 输出目录：`/tmp/waymo_infer_step003000_val0_s4_decode_20260505_2250`
- 可视化文件：
  - `10203656353524179475_7625_000_7645_000_0_preview.png`
  - `10203656353524179475_7625_000_7645_000_0_intermediate_preview.png`
- 指标：4-step sampled latent `mse=0.860675`、`mae=0.718463`；single-step denoised latent `mse=0.248937`、`mae=0.378692`。
- 4-step intermediate MSE：`2.044374 -> 1.765859 -> 1.363221 -> 0.860675`，说明采样过程本身在收敛，但该 3k checkpoint 的最终 4-step latent MSE 差于当前 best candidate `step_005000.pt`。

2026-05-05 `step_005000.pt` 4-step decode 可视化：

- checkpoint：`/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_full28_vkv4_initvideo_cp_b2_7gpu_formal_20260429_005200/checkpoints/step_005000.pt`
- validation sample：`10203656353524179475_7625_000_7645_000_0`
- 输出目录：`/tmp/waymo_infer_step005000_val0_s4_decode_20260505_2308`
- 可视化文件：
  - `10203656353524179475_7625_000_7645_000_0_preview.png`
  - `10203656353524179475_7625_000_7645_000_0_intermediate_preview.png`
- 指标：4-step sampled latent `mse=0.733270`、`mae=0.645834`；single-step denoised latent `mse=0.250332`、`mae=0.374351`。
- 4-step intermediate MSE：`2.026494 -> 1.718947 -> 1.257420 -> 0.733270`。同样 sample、同样 4-step 配置下，`step_005000.pt` 的最终 sampled MSE 明显优于 `step_003000.pt`。

输出 payload 使用 contract：

```text
wan21_native64x1312_repeatrow11_v1
latent shape: [16, 8, 88, 164]
video shape: [16, 40, 90, 160]
```

对应训练入口：

```bash
conda activate drivesync

LIDAR_NUM_BLOCKS=28 \
VIDEO_KV_EVERY_N_LAYERS=4 \
CHECKPOINT_LIDAR_BLOCKS=true \
INIT_LIDAR_FROM_VIDEO=true \
BATCH_SIZE=1 \
NUM_GPUS=7 \
./train_waymo_video_lidar_one_way_wan21_cache.sh
```

FSDP2 opt-in 入口：

```bash
conda activate drivesync

NUM_GPUS=7 \
BATCH_SIZE=1 \
CHECKPOINT_LIDAR_BLOCKS=false \
LIDAR_NUM_BLOCKS=28 \
VIDEO_KV_EVERY_N_LAYERS=4 \
INIT_LIDAR_FROM_VIDEO=true \
./train_waymo_video_lidar_one_way_wan21_cache_fsdp.sh
```

FSDP 入口只 shard trainable LiDAR expert；frozen video DiT 仍 replicated，避免影响既有 video generation。DDP/cache 入口默认行为不变。

cache 路径只作为加速/复现实验使用。2026-04-26 已停止旧 cache 生成任务，停止时 `/data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11/lidar` 已写入 `625` 个样本。

全宽参考运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_waymo_lidar_wan21_vae.py \
  --preprocess-mode waymo_top_64x2650 \
  --raw-lidar-root /team/hyh/data/rds_hq_waymo \
  --lidar-utils-repo /team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen \
  --wan-vae-path /team/hyh/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth \
  --split validation \
  --segment-key 10203656353524179475_7625_000_7645_000 \
  --frame-start 0 \
  --num-frames 29 \
  --device cuda \
  --native-n-cols 2650 \
  --repeat-row 16 \
  --vis-pcd \
  --pcd-renderer plotly \
  --pcd-camera-view front_view \
  --pcd-display-frame vehicle \
  --pcd-workers 12
```

## 输出路径

当前置顶点云渲染输出：

```text
/data2/waymo_wan_lidar_vae_smoke/validation/exp_native64x1312_repeatrow11_mean_vispcd/
```

重要文件：

```text
metrics.json
point_cloud_front_view_vehicle.mp4
```

`repeat_row=16` 输出目录约 `1.8G`，主要原因是保存的 tensor 较大。

## 指标

指标在 native/evaluation range map 的有效像素上计算。输入 range 会先裁剪到 `[5m, 100m]`。

| 配置 | 源 range map | VAE 逻辑输入 | Wan 输入 | Latent | Range MAE | Range RMSE | Normalized MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 tokenizer 路径，`128x3600 -> 512x1800` | converted | `[1,3,29,512,1800]` | same | `[1,16,8,64,225]` | 3.1946m | 6.6153m | 0.013479 |
| native，不重复行，channel 0 | `64x2650` | `[1,3,29,64,2650]` | `[1,3,29,64,2656]` | `[1,16,8,8,332]` | 2.9569m | 6.5403m | 0.023508 |
| native，`repeat_row=4`，channel mean | `64x2650` | `[1,3,29,256,2650]` | `[1,3,29,256,2656]` | `[1,16,8,32,332]` | 1.8267m | 4.9355m | 0.014772 |
| native，`repeat_row=6`，channel mean | `64x2650` | `[1,3,29,384,2650]` | `[1,3,29,384,2656]` | `[1,16,8,48,332]` | 1.6151m | 4.5502m | 0.012630 |
| native，`repeat_row=8`，channel mean | `64x2650` | `[1,3,29,512,2650]` | `[1,3,29,512,2656]` | `[1,16,8,64,332]` | 1.4549m | 4.2458m | 0.011430 |
| native，`repeat_row=10`，channel mean | `64x2650` | `[1,3,29,640,2650]` | `[1,3,29,640,2656]` | `[1,16,8,80,332]` | 1.3880m | 4.0983m | 0.010552 |
| native，`repeat_row=12`，channel mean | `64x2650` | `[1,3,29,768,2650]` | `[1,3,29,768,2656]` | `[1,16,8,96,332]` | 1.3359m | 3.9324m | 0.009821 |
| native，`repeat_row=16`，channel mean | `64x2650` | `[1,3,29,1024,2650]` | `[1,3,29,1024,2656]` | `[1,16,8,128,332]` | 1.2433m | 3.8242m | 0.009458 |
| native，`repeat_row=8`，`concat_inv_depth` | `64x2650` | `[1,3,29,512,2650]` | `[1,3,29,512,2656]` | `[1,16,8,64,332]` | 1.7843m | 5.1347m | 0.014177 |

## 观察

- native `64x2650` 投影比旧的 converted-tokenizer 输入更适合 Wan2.1 VAE。
- 在这个样本上，增加垂直方向 row repeat 会提升指标，当前测试到 `repeat_row=16` 仍是最优。
- `repeat_row=16` 相比 `repeat_row=8` 大约会让保存 tensor 的体积翻倍。
- 对 decode 后的 3 个通道求均值，可以把 `repeat_row=8` 的 MAE 从 `1.5547m` 降到 `1.4549m`，且不需要重新跑 VAE。
- 不推荐在 Wan2.1 VAE 路径使用 `concat_inv_depth`。它可能引入 RGB 训练 VAE 不能稳定保留的跨通道语义。
- 宽度 `2650` 只需要右侧 padding 到 `2656`，用于满足 Wan spatial compression 的 8 对齐；指标和可视化都会裁回 `2650`。

## 验证

当前最佳点云视频：

```text
/data2/waymo_wan_lidar_vae_smoke/validation/10203656353524179475_7625_000_7645_000_waymo_top_64x2650_repeatrow16/point_cloud_front_view_vehicle.mp4
```

视频元数据：

```text
width=1280
height=1444
duration=2.900000
nb_frames=29
```

中间帧非空检查：

```text
point cloud frame std: 18.1995
rangemap frame std: 74.5188
```

## CogVideoX1.5 VAE 对比

更新日期：2026-04-25

同一个 Waymo TOP native `64x2650` rangemap 输入也使用 CogVideoX1.5 VAE 做了测试。

模型路径：

```text
/data2/CogvideoX1.5-5B-T2V/vae
```

脚本：

```bash
scripts/smoke_waymo_lidar_cogvideo_vae.py
```

CogVideo 脚本参考 `/root/workspace/CogVideo/inference/cli_vae_demo.py`：

- 加载 `AutoencoderKLCogVideoX.from_pretrained(...)`
- 开启 VAE slicing 和 tiling
- 调用 `vae.encode(video).latent_dist.mode()`
- 调用 `vae.decode(latent).sample`

重要的时间维行为：

- 输入 `29` 帧会 encode 成 `8` 帧 latent。
- CogVideo 会从这 `8` 帧 latent decode 出 `32` 帧。
- 为了和输入公平对齐，脚本在计算指标和可视化前裁取 decode 输出的前 `29` 帧。

### CogVideo 命令

```bash
conda activate drivesync

CUDA_VISIBLE_DEVICES=0 python scripts/smoke_waymo_lidar_cogvideo_vae.py \
  --split validation \
  --segment-key 10203656353524179475_7625_000_7645_000 \
  --frame-start 0 \
  --num-frames 29 \
  --device cuda \
  --repeat-row 8 \
  --vis-pcd \
  --pcd-renderer plotly \
  --pcd-camera-view front_view \
  --pcd-display-frame vehicle \
  --pcd-workers 12
```

输出：

```text
/data2/waymo_cogvideo_vae_smoke/validation/10203656353524179475_7625_000_7645_000_waymo_top_64x2650_cogvideo_repeatrow8/
```

重要文件：

```text
metrics.json
input_rangemap.mp4
reconstruction_rangemap.mp4
abs_error_rangemap.mp4
point_cloud_front_view_vehicle.mp4
input_tensor.pt
cogvideo_input_tensor.pt
latent.pt
reconstruction.pt
cogvideo_reconstruction.pt
```

点云视频元数据：

```text
width=1280
height=1444
duration=2.900000
nb_frames=29
```

中间帧非空检查：

```text
point cloud frame std: 18.1199
rangemap frame std: 72.8120
```

### CogVideo 指标

| VAE | 配置 | 输入归一化 | VAE 逻辑输入 | 裁剪前 VAE 输出 | Latent | Range MAE | Range RMSE | Normalized MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CogVideoX1.5 | native，`repeat_row=8`，channel mean | `[0,1]` | `[1,3,29,512,2650]` | `[1,3,32,512,2656]` | `[1,16,8,64,332]` | 5.0279m | 11.5737m | 0.017240 |
| CogVideoX1.5 | native，`repeat_row=8`，channel mean | `[-1,1]` | `[1,3,29,512,2650]` | `[1,3,32,512,2656]` | `[1,16,8,64,332]` | 4.9995m | 11.6861m | 0.069060 |
| CogVideoX1.5 | native，`repeat_row=16`，channel mean | `[0,1]` | `[1,3,29,1024,2650]` | `[1,3,32,1024,2656]` | `[1,16,8,128,332]` | 5.0272m | 11.5583m | 0.017347 |
| Wan2.1 | native，`repeat_row=8`，channel mean | `[-1,1]` | `[1,3,29,512,2650]` | `[1,3,29,512,2656]` | `[1,16,8,64,332]` | 1.4549m | 4.2458m | 0.011430 |
| Wan2.1 | native，`repeat_row=16`，channel mean | `[-1,1]` | `[1,3,29,1024,2650]` | `[1,3,29,1024,2656]` | `[1,16,8,128,332]` | 1.2433m | 3.8242m | 0.009458 |

## OpenSora-S3 / Sora S3 Tokenizer 对比

参考文档：

```text
/root/workspace/cosmos-transfer2.5/third_party_ref_repo/Cosmos-Drive-Dreams/waymo_lidar_tokenizer_post_training.md
```

该文档里的 `OpenSora-S3` 是 Cosmos LiDAR tokenizer 后训练路线的第三阶段：

- `S1`：冻结 2D tokenizer，只训练 Open-Sora 风格 temporal VAE。
- `S2`：放开 `post_quant_conv + decoder`，以 latent reconstruction 为主。
- `S3`：从 `S2 iter_000035500.pt` 接力，切到 pixel reconstruction 主导，`color=1.0`，`latent_recon=0.25 -> 0.05`。

这里有两个指标口径：

- 单条 canonical clip 推理指标：和本文当前 Wan/CogVideo smoke 使用同一个 `10203656353524179475_7625_000_7645_000` clip，因此更适合做横向参考。
- validation 聚合选模指标：用于选择 S3 checkpoint，不能和本文单条 smoke 指标直接等价比较。

### 单条 Canonical Clip 对比

| 方法 | 输入/表示 | 时间契约 | Range MAE | Range RMSE | Rel | 备注 |
|---|---|---:|---:|---:|---:|---|
| Wan2.1 VAE smoke | native `64x2650`，`repeat_row=16`，channel mean | `29 -> 8 -> 29` | **1.2433m** | **3.8242m** | - | 当前本文最佳单条 smoke 结果 |
| Wan2.1 VAE smoke | native `64x2650`，`repeat_row=8`，channel mean | `29 -> 8 -> 29` | 1.4549m | 4.2458m | - | 低成本配置 |
| OpenSora-S3 tokenizer | converted Waymo tokenizer range map | `29 -> 8 -> 29` | 1.58m | 5.80m | 0.05 | `OpenSora-S3/iter_000035500.pt`，源文档 canonical clip 推理 |
| CogVideoX1.5 VAE smoke | native `64x2650`，`repeat_row=8`，channel mean | `29 -> 8 -> 32`，裁前 29 帧 | 5.0279m | 11.5737m | - | CogVideo decode 原生返回 32 帧 |

注意：

- OpenSora-S3 使用的是 Cosmos LiDAR tokenizer 路线，不是通用 RGB 视频 VAE；训练目标和本文的“直接拿 RGB/视频 VAE 做 LiDAR rangemap roundtrip”不同。
- OpenSora-S3 的输入表示来自 Waymo LiDAR tokenizer 数据转换路径，和本文 native `64x2650` 输入不同，因此该表只能作为同 clip 的工程参考，不是严格同分布 benchmark。
- 从单条 clip 的数值看，当前 native `64x2650 + Wan2.1 repeat_row=16` smoke 在 MAE/RMSE 上优于源文档里的 OpenSora-S3 单条推理结果；但 OpenSora-S3 是专门后训练过的 LiDAR tokenizer，其优势可能需要在多场景、时序一致性和完整 tokenizer 使用场景中再评估。

### OpenSora-S3 聚合选模指标

源文档记录的 S3 综合最优 checkpoint：

```text
OpenSora-S3/iter_000035500.pt
depth_mae = 1.133
depth_rmse = 3.761
depth_rel = 0.0433
```

补充节点：

| Checkpoint / iter | depth_mae | depth_rmse | depth_rel | 说明 |
|---|---:|---:|---:|---|
| S2 `iter_000035500.pt` | 1.305 | 4.090 | 0.0499 | S3 接力起点，S2 综合最优 |
| S3 `iter_000002500.pt` | 1.156 | **3.546** | 0.0480 | S3 rmse 全程最低 |
| S3 `iter_000035500.pt` | **1.133** | 3.761 | 0.0433 | S3 mae 最低，源文档推荐部署 checkpoint |

源文档结论：S3 相对 S2 的聚合指标继续改善，`mae 1.305 -> 1.133`，`rmse 4.041/4.090 -> 3.546`，`rel 0.0499 -> 0.0430/0.0433`。这些是 validation 聚合口径，和上面的单条 clip 推理口径不同。

## 宽度压缩实验

更新日期：2026-04-25

目标是让 LiDAR latent spatial token 数尽量接近视频 latent `90 x 160 = 14400`。

固定配置：

```text
VAE: Wan2.1
source rows: 64
repeat_row: 6
downsample_factor_row = 1
downsample_factor_col = 1
input_channel_mode = repeat_depth
decode_channel_mode = mean
```

做法：不对已经生成的 rangemap 做 resize，而是在 raw Waymo TOP 点云投影阶段直接改变 `native_n_cols`，例如 `2400` 列。这样可以避免图像插值对 sparse range map 造成额外模糊。

命令示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_waymo_lidar_wan21_vae.py \
  --preprocess-mode waymo_top_64x2650 \
  --raw-lidar-root /team/hyh/data/rds_hq_waymo \
  --lidar-utils-repo /team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen \
  --wan-vae-path /team/hyh/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth \
  --split validation \
  --segment-key 10203656353524179475_7625_000_7645_000 \
  --frame-start 0 \
  --num-frames 29 \
  --device cuda \
  --native-n-cols 2400 \
  --repeat-row 6 \
  --skip-videos \
  --skip-tensors \
  --no-vis-pcd \
  --output-dir /data2/waymo_wan_lidar_vae_smoke/validation/exp_native64x2400_repeatrow6_mean
```

实验结果：

| native_n_cols | Latent shape | Spatial tokens | 相对 `14400` | Range MAE | Range RMSE | Normalized MSE | Valid pixels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2650 | `[1,16,8,48,332]` | 15936 | +10.67% | 1.6151m | 4.5502m | 0.012630 | 3027028 |
| 2560 | `[1,16,8,48,320]` | 15360 | +6.67% | 1.5051m | 4.4047m | 0.011837 | 2955407 |
| 2480 | `[1,16,8,48,310]` | 14880 | +3.33% | 1.4744m | 4.3824m | 0.011755 | 2875313 |
| 2400 | `[1,16,8,48,300]` | 14400 | 0.00% | 1.4557m | 4.3663m | 0.011616 | 2790238 |
| 2304 | `[1,16,8,48,288]` | 13824 | -4.00% | 1.4377m | 4.3244m | 0.011521 | 2686808 |
| 2048 | `[1,16,8,48,256]` | 12288 | -14.67% | 1.3652m | 4.2392m | 0.011097 | 2408773 |
| 1800 | `[1,16,8,48,225]` | 10800 | -25.00% | 1.2669m | 4.1121m | 0.010364 | 2138345 |
| 1600 | `[1,16,8,48,200]` | 9600 | -33.33% | 1.1871m | 3.9832m | 0.009718 | 1919975 |
| 1280 | `[1,16,8,48,160]` | 7680 | -46.67% | **1.0287m** | **3.7359m** | **0.008430** | 1566118 |
| 1024 | `[1,16,8,48,128]` | 6144 | -57.33% | 1.0918m | 3.8650m | 0.009045 | 1261856 |
| 768 | `[1,16,8,48,96]` | 4608 | -68.00% | 1.1839m | 4.0351m | 0.009748 | 956274 |
| 688 | `[1,16,8,48,86]` | 4128 | -71.33% | 1.2138m | 4.0842m | 0.009988 | 860283 |
| 680 | `[1,16,8,48,85]` | 4080 | -71.67% | 1.2165m | 4.0637m | 0.009910 | 850682 |
| 640 | `[1,16,8,48,80]` | 3840 | -73.33% | 1.2372m | 4.0964m | 0.010112 | 802682 |
| 576 | `[1,16,8,48,72]` | 3456 | -76.00% | 1.2793m | 4.1871m | 0.010506 | 725498 |
| 512 | `[1,16,8,48,64]` | 3072 | -78.67% | 1.3272m | 4.2601m | 0.010909 | 648421 |
| 448 | `[1,16,8,48,56]` | 2688 | -81.33% | 1.3837m | 4.3511m | 0.011345 | 570750 |
| 384 | `[1,16,8,48,48]` | 2304 | -84.00% | 1.4523m | 4.4713m | 0.011940 | 493040 |
| 320 | `[1,16,8,48,40]` | 1920 | -86.67% | 1.5865m | 4.7612m | 0.013329 | 414738 |

### 低宽度 repeat_row 增强实验

上一张表固定 `repeat_row=6`。对其中指标较好的低宽度配置，继续提高 `repeat_row`，观察在 spatial tokens 接近 `14400` 时的变化。

| native_n_cols | repeat_row | Input HxW | Latent shape | Spatial tokens | 相对 `14400` | Range MAE | Range RMSE | Normalized MSE | Valid pixels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1800 | 6 | `384x1800` | `[1,16,8,48,225]` | 10800 | -25.00% | 1.2669m | 4.1121m | 0.010364 | 2138345 |
| 1800 | 8 | `512x1800` | `[1,16,8,64,225]` | 14400 | +0.00% | 1.1662m | 3.8931m | 0.009210 | 2138345 |
| 1600 | 6 | `384x1600` | `[1,16,8,48,200]` | 9600 | -33.33% | 1.1871m | 3.9832m | 0.009718 | 1919975 |
| 1600 | 8 | `512x1600` | `[1,16,8,64,200]` | 12800 | -11.11% | 1.0908m | 3.7486m | 0.008547 | 1919975 |
| 1600 | 10 | `640x1600` | `[1,16,8,80,200]` | 16000 | +11.11% | 1.0406m | 3.6264m | 0.007734 | 1919975 |
| 1280 | 6 | `384x1280` | `[1,16,8,48,160]` | 7680 | -46.67% | 1.0287m | 3.7359m | 0.008430 | 1566118 |
| 1280 | 8 | `512x1280` | `[1,16,8,64,160]` | 10240 | -28.89% | 0.9612m | 3.4978m | 0.007260 | 1566118 |
| 1280 | 10 | `640x1280` | `[1,16,8,80,160]` | 12800 | -11.11% | 0.9183m | 3.3817m | 0.006548 | 1566118 |
| 1280 | 11 | `704x1280` | `[1,16,8,88,160]` | 14080 | -2.22% | **0.9030m** | **3.3258m** | **0.006233** | 1566118 |
| 1280 | 12 | `768x1280` | `[1,16,8,96,160]` | 15360 | +6.67% | 0.8912m | 3.2751m | 0.006025 | 1566118 |
| 1024 | 6 | `384x1024` | `[1,16,8,48,128]` | 6144 | -57.33% | 1.0918m | 3.8650m | 0.009045 | 1261856 |
| 1024 | 12 | `768x1024` | `[1,16,8,96,128]` | 12288 | -14.67% | 0.9493m | 3.3810m | 0.006447 | 1261856 |
| 1024 | 14 | `896x1024` | `[1,16,8,112,128]` | 14336 | -0.44% | 0.9249m | 3.3477m | 0.006278 | 1261856 |
| 1024 | 16 | `1024x1024` | `[1,16,8,128,128]` | 16384 | +13.78% | 0.9131m | 3.3247m | 0.006201 | 1261856 |

### 14400 token 附近等量 sweep

本轮继续补充 spatial tokens 接近 `14400` 的不同 `native_n_cols` 和 `repeat_row` 组合。表中组合不重复上一张表里已经跑过的配置。

| native_n_cols | repeat_row | Input HxW | Latent shape | Spatial tokens | 相对 `14400` | Range MAE | Range RMSE | Normalized MSE | Valid pixels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 7 | `448x2048` | `[1,16,8,56,256]` | 14336 | -0.44% | 1.3009m | 4.1246m | 0.010417 | 2408773 |
| 1600 | 9 | `576x1600` | `[1,16,8,72,200]` | 14400 | +0.00% | 1.0610m | 3.6968m | 0.008133 | 1919975 |
| 1440 | 10 | `640x1440` | `[1,16,8,80,180]` | 14400 | +0.00% | 0.9643m | 3.4645m | 0.006918 | 1745350 |
| 1312 | 11 | `704x1312` | `[1,16,8,88,164]` | 14432 | +0.22% | **0.8988m** | **3.3089m** | **0.006171** | 1603553 |
| 1200 | 12 | `768x1200` | `[1,16,8,96,150]` | 14400 | +0.00% | 0.9136m | 3.3311m | 0.006215 | 1471147 |
| 1104 | 13 | `832x1104` | `[1,16,8,104,138]` | 14352 | -0.33% | 0.9200m | 3.3408m | 0.006238 | 1357126 |
| 1032 | 14 | `896x1032` | `[1,16,8,112,129]` | 14448 | +0.33% | 0.9284m | 3.3661m | 0.006315 | 1271406 |
| 960 | 15 | `960x960` | `[1,16,8,120,120]` | 14400 | +0.00% | 0.9475m | 3.4132m | 0.006520 | 1185406 |
| 896 | 16 | `1024x896` | `[1,16,8,128,112]` | 14336 | -0.44% | 0.9533m | 3.4126m | 0.006470 | 1109335 |
| 800 | 18 | `1152x800` | `[1,16,8,144,100]` | 14400 | +0.00% | 0.9932m | 3.5061m | 0.006711 | 994620 |
| 720 | 20 | `1280x720` | `[1,16,8,160,90]` | 14400 | +0.00% | 1.0296m | 3.6004m | 0.006991 | 898785 |

结论：

- 当前训练主线是 `native_n_cols=1280, repeat_row=11`：latent spatial 是 `88 x 160 = 14080`，和 Transfer2 single-view `704x1280` 完全对齐，MAE/RMSE 为 `0.9030m/3.3258m`。
- 纯 VAE 重建上 `native_n_cols=1312, repeat_row=11` 略好，latent spatial 是 `88 x 164 = 14432`，MAE/RMSE 为 `0.8988m/3.3089m`；它现在作为历史/对照方案保留，不作为默认训练 shape。
- 如果必须严格等于 `14400` tokens，当前最好的是 `native_n_cols=1200, repeat_row=12`，latent spatial 是 `96 x 150 = 14400`，MAE/RMSE 为 `0.9136m/3.3311m`。
- 如果更看重 tokens 几乎严格贴近 `14400`，`native_n_cols=1024, repeat_row=14` 是 `112 x 128 = 14336`，只少 `0.44%`，MAE/RMSE 为 `0.9249m/3.3477m`。
- 如果固定 `repeat_row=6`，必须和视频 latent spatial token 数严格对齐时选择 `native_n_cols=2400`，latent spatial 正好是 `48 x 300 = 14400`。
- 如果固定 `repeat_row=6` 且允许比 `14400` 少 `4%`，`native_n_cols=2304` 在这条 canonical clip 上指标最好，latent spatial 是 `48 x 288 = 13824`。
- 沿着 `14400` 等量线继续压缩宽度时，`1200~1312` 附近表现最好；继续压到 `960/896/800/720` 后 MAE/RMSE 开始回升。
- `2304~2400` 比原始 `2650` 更适合当前 `repeat_row=6` 设置：token 更接近视频 latent，同时 MAE/RMSE 也更低。
- 早期固定 `repeat_row=6` 的 sweep 里，`native_n_cols=1280` 指标最低，但它只有 `7680` spatial tokens，比 `14400` 少 `46.67%`；这类低高度配置不作为当前主线。
- `repeat_row=6` 时输入高度是 `384`。接近 `9:16` 的宽度是 `680/688`，对应 `4080/4128` tokens，比 `14400` 少约 `71%`；它们不适合“latent token 数接近视频”的目标。
- `512` 是严格 `3:4` 输入比例，即 `384x512`，latent spatial 是 `48 x 64 = 3072`，MAE/RMSE 为 `1.3272m/4.2601m`；低于 `640` 后指标开始持续变差。
- 这些指标是在不同投影列数各自的 evaluation grid 上计算的；因为有效像素数量不同，它们适合作为工程选型参考，不是严格逐像素同网格比较。

### CogVideo 观察

- CogVideoX1.5 VAE 可以在 GPU 上对 native Waymo rangemap tensor 正常 encode/decode，开启 tiling 后可跑通。
- latent shape 符合预期的时间/空间压缩规律，但 `29` 帧输入会 decode 出 `32` 帧，因此这个 smoke comparison 需要做时间维裁剪。
- CogVideoX1.5 在这个 LiDAR rangemap 分布上的重建明显差于 Wan2.1。
- 将输入归一化从 `[0,1]` 换成 `[-1,1]` 没有实质改善 range error。
- 将 `repeat_row` 从 `8` 增加到 `16` 也没有改善 CogVideo range error，这一点和 Wan2.1 不同。
- 当前这个 rangemap encode/decode 实验仍推荐使用 Wan2.1 VAE。

## 后续实验

- 只有在显存和磁盘成本可接受时，再测试 `repeat_row=20` 或 `repeat_row=24`。
- 默认不保存完整的 `wan_input_tensor.pt` 和 `wan_reconstruction.pt`，以降低输出体积。
- 在更多 validation segments 上评估 `native_n_cols=1280, repeat_row=11`，确认数据集级稳定性。
- 增加 P50/P90/P99 absolute range error 等分位数指标。
- 增加 masked point-cloud Chamfer 或 nearest-neighbor distance，用于几何层面的比较。
