# Waymo LiDAR Single-View Post-Training

更新日期：2026-05-30

本文记录新的 Waymo TOP LiDAR rangemap 生成主线。训练代码走官方 single-view post-training 框架；dataloader 每个 batch 从 raw LiDAR tar 在线投影 target 和 `rangemap_layout`，训练步再在线调用 Wan2.1 VAE `encode` 得到 diffusion target。MP4 与 `.npz` 都不进入默认监督路径。

## 当前服务器环境

H100-DriveSync 上的工作目录和环境：

```bash
cd /team/hyh/code/cosmos-transfer2.5
source /opt/conda/etc/profile.d/conda.sh
conda activate drivesync
```

当前训练和后续推理评估主线均使用 raw Waymo TOP 投影的 `704x1280` contract。旧 tokenizer-converted `704x1200` checkpoint、指标和点云结果只作为历史记录，不可用于判断当前 online encode 方案。

## 目标

- 训练目标：raw TOP LiDAR 在线投影的 normalized rangemap `[3,29,704,1280]`，在线 encode 为 `[16,8,88,160]` latent。
- 条件视频：由同一 raw 帧窗在线生成的稀疏 `rangemap_layout`，表达 occupancy 和 range discontinuity edge。
- 默认训练只要求 raw LiDAR tar：

```text
/team/hyh/data/rds_hq_waymo/<split>/
└── lidar_raw/
    └── <segment_key>.tar
```

如需覆盖默认固定 caption，可额外传入包含 `captions/<sample_key>.json` 的 `--dataset-dir`。调试视频或 `.npz` cache 均不是训练前置条件。

## 数据准备

默认 raw-online 训练无需先执行数据准备：dataloader 直接扫描 raw tar 并按每 10 帧步长生成 29 帧 sample window。以下命令仅用于生成可视化/排查用视频与可选 caption：

```bash
source /opt/conda/etc/profile.d/conda.sh
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

输出：

```text
/team/hyh/data/waymo_singleview_lidar_posttrain_raw1280/training/
├── videos/            # display/debug videos; not used as diffusion targets
├── rangemap_layout/   # display/debug controls; raw-online training ignores these files
├── captions/          # {"caption": "..."} JSON files
├── metadata/          # alignment/debug metadata
└── rangemap_targets/  # optional only when --write-rangemap-targets is requested
```

默认 dataloader 直接枚举 `/team/hyh/data/rds_hq_waymo/<split>/lidar_raw/*.tar` 得到 sample key，再以 `frame_start=chunk_index*10` 在线生成 target 和 layout。两者来自同一次投影和同一帧窗，训练前会检查 shape 完全一致。`--write-rangemap-targets` 仅用于需要离线缓存回退时。

默认 LiDAR display/control 与 online target contract：

```text
source lidar: /team/hyh/data/rds_hq_waymo/<split>/lidar_raw/<segment_key>.tar
frame_start: chunk_index * 10
num_frames: 29
downsampled range map: 64 x 1280
target video: 29 x 704 x 1280 x 3 uint8
layout video: 29 x 704 x 1280 x 3 uint8
online target: 3 x 29 x 704 x 1280 float32 -> Wan encode -> 16 x 8 x 88 x 160
display video normalization: range [5m, 100m] -> [-1, 1] -> uint8
valid mask: preprocessing/layout occupancy follows downsampled_range > 0
fps: 10
```

### GT Target Audit

2026-05-26 检查旧 `704x1200` tokenizer-converted 样本 `10017090168044687777_6380_000_6400_000_0` 后确认：target mp4 的 range 数值和 raw/downsampled range 基本对齐，但 target 表示法对 LiDAR generation 不理想。

- `normalize_range_map` 会把 `downsampled_range == 0` 的 invalid/empty 像素 clip 到 `min_range=5m`，再编码成 normalized `-1` / uint8 `0`。因此 target video 本身无法区分 empty 和真实近距离 5m 点。
- 该样本 `64x1200x29` 中 valid 为 `2077583 / 2227200`，invalid 为 `149617`；另有 `251037` 个 valid 点小于 5m，也会被 clip 到 5m。
- 落盘 H264 后 invalid 区域大多仍接近 0，但有压缩残留；按当前 decoder 反归一化后 invalid 区域 median 为 `5.0m`，99% quantile 为 `7.235m`。
- GT mp4 解码回米制后，在 valid mask 上相对 clipped raw 的 MAE 约 `0.3897m`，说明 GT range 对齐本身不是主因。
- 旧数据会把 `704x1200` target/control 反射 padding 到 `704x1280`，左右各 40px；新 `704x1280` contract 直接对齐 single-view latent grid，避免这类宽度 padding。

当前生成点云里的 extra points 和 range 偏远，和上述 target 表示有关：模型只靠把 empty 区域预测到非常接近 `-1` 才能避免假点，稍微高于阈值就会在 decoded point cloud 中变成 generated extra occupancy。后续仍建议保留显式 occupancy/valid 信息，或改用能区分 invalid 和 near-range 的 target 编码；宽度主线已切到 `704x1280`，不再依赖反射 padding 补齐。

Caption 可选模式：

- `--caption-mode fixed`：固定 LiDAR caption，最稳。
- `--caption-mode source`：读取 `/data/waymo/waymo_multiview_texts.json` 的原始 camera caption。
- `--caption-mode vlm`：从 `/data/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/...` 或 `--caption-model-path` 加载本地 VLM，读取 rangemap 中间帧生成 caption。

Smoke 数据：

```bash
python scripts/prepare_waymo_lidar_singleview_posttrain_dataset.py \
  --split validation \
  --output-root /tmp/waymo_lidar_layout_posttrain_smoke \
  --raw-waymo-root /team/hyh/data/rds_hq_waymo \
  --sample-list-source lidar \
  --range-map-source raw \
  --native-n-cols 1280 \
  --repeat-row 11 \
  --max-samples 1 \
  --overwrite \
  --caption-mode fixed
```


## 训练

注册实验，也是当前默认训练实验：

```text
transfer2_singleview_posttrain_waymo_lidar_wan21_online_layout_fullfinetune
```

关键配置：

- official single-view Transfer2 post-training stack
- `state_t=8`
- `num_frames=29`
- `raw_rangemap_online=true`，target 和 `rangemap_layout` 均在线从 raw tar 生成
- `online_target_key=rangemap_target`，训练时在线 encode 为 `[16,8,88,160]`
- `hint_keys=rangemap_layout`
- `min_num_conditional_frames=1`
- `max_num_conditional_frames=1`
- full-finetune 默认学习率 `1e-5`
- 初始化使用官方 single-view edge checkpoint，作为单分支 control 的最近可用预训练权重

启动：

```bash
./train_waymo_lidar_singleview_chunked.sh
```

常用覆盖：

只保留 chunked 启动脚本。raw-online 默认 job 名为 `waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8`，避免自动续接旧 layout 输入路径的实验；`--job-name` 会直接传给训练任务。如需进 tmux，使用 `--tmux`，窗口名可用 `lidar_train`。

```bash
OUTPUT_ROOT=outputs/waymo_lidar_singleview_posttrain \
NUM_GPUS=8 \
TOTAL_ITER=100000 \
CHUNK_ITER=10000 \
SAVE_ITER=1000 \
./train_waymo_lidar_singleview_chunked.sh
```

从已有 checkpoint 继续训练时，`TOTAL_ITER` 必须大于 `checkpoints/latest_checkpoint.txt` 中的 iteration；当前 raw-online 最新记录为 `iter_000053000`，因此不要再用 `TOTAL_ITER=40000` 之类低于最新 checkpoint 的值。

`state_t=8` 要求 GPU 数整除 8，推荐 `1`、`2`、`4`、`8`。只有 7 张 GPU 可见时先用 `--gpus 4`。

Dry run：

```bash
USE_TMUX=false ./train_waymo_lidar_singleview_chunked.sh \
  --raw-lidar-split validation \
  --gpus 1 \
  --total-iter 1 \
  --save-iter 1 \
  --logging-iter 1 \
  --num-workers 0 \
  --dry-run \
  --no-tmux
```

## 推理

DCP checkpoint 需要先转换成 `model_ema_bf16.pt`。建议把转换结果放到独立评估目录，避免污染训练 checkpoint：

```bash
RUN=outputs/waymo_lidar_singleview_posttrain/cosmos_transfer2_posttrain/waymo_lidar_singleview/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8
ITER=iter_000053000
EVAL_DIR=outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/$ITER

mkdir -p "$EVAL_DIR/checkpoint_pt"
python scripts/convert_distcp_to_pt.py \
  "$RUN/checkpoints/$ITER/model" \
  "$EVAL_DIR/checkpoint_pt"
```

之后使用专用入口推理：

```bash
SPLIT=training
SAMPLE=10017090168044687777_6380_000_6400_000_0

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/team/hyh/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/infer_waymo_lidar_singleview_posttrain.py \
  --checkpoint-path "$EVAL_DIR/checkpoint_pt/model_ema_bf16.pt" \
  --sample-key "$SAMPLE" \
  --raw-online-input \
  --raw-lidar-root /team/hyh/data/rds_hq_waymo \
  --raw-lidar-split "$SPLIT" \
  --output-dir "$EVAL_DIR/inference_g3_s35_text_cond1" \
  --name "${SAMPLE}_${ITER}_ema_g3_s35_text_cond1" \
  --experiment transfer2_singleview_posttrain_waymo_lidar_wan21_online_layout_fullfinetune \
  --num-conditional-frames 1 \
  --num-steps 35 \
  --guidance 3 \
  --max-frames 29 \
  --num-video-frames-per-chunk 29
```

`--num-steps 35 --guidance 3` 对齐训练 callback 的常规采样设置。online 实验会从 raw tar 重新投影 lossless `rangemap_target` 与 `rangemap_layout` 并直接注入模型；写出的 raw input MP4 只用于展示和 pipeline 尺寸载体，不作为模型 target/control。若只想隔离文本条件，可额外加 `--zero-text-embedding`。

2026-05-28 注意：raw-online 推理必须确认日志中实际出现 `num_conditional_frames: 1 is set by data_batch[NUM_CONDITIONAL_FRAMES_KEY]`。此前 inference pipeline 对首个 chunk 固定写入 `0`，会导致命令传 `--num-conditional-frames 1` 但模型实际无首帧 latent 条件；已在 `model_video_inputs` 场景修正为按传入帧数换算 latent 条件帧。旧目录 `inference_g3_s35_text` 中同名结果按 cond=0 生成，不用于当前判断；当前有效结果使用 `inference_g3_s35_text_cond1`。

输出：

```text
<name>.mp4
<name>_control_rangemap_layout.mp4
<name>_raw_target_input.mp4                  # display only
<name>_raw_layout_input.mp4                  # display only
<name>_comparison_gt_control_generated.mp4
```

## 点云可视化

Transfer2 推理保存的 `<name>.mp4` 是 decoder 后的 rangemap video。点云脚本读取同名 JSON 中的 `sample_key`，并严格复用 `docs/waymo_lidar_wan21_vae_preprocessing.md` 验证过的 raw TOP `64x1280` 投影、valid mask 和 ray/extrinsic 流程；raw GT 读取失败时直接报错，不会用 generated video 伪装成 GT。

```bash
python scripts/visualize_waymo_lidar_generation.py \
  --generated-video "$EVAL_DIR/inference_g3_s35_text_cond1/${SAMPLE}_${ITER}_ema_g3_s35_text_cond1.mp4" \
  --split "$SPLIT" \
  --raw-lidar-root /team/hyh/data/rds_hq_waymo \
  --output-dir "$EVAL_DIR/pointcloud_vis_raw" \
  --pcd-renderer plotly \
  --raw-valid-mode preprocess \
  --generated-valid-mode predicted \
  --pcd-workers 4 \
  --display-frame vehicle \
  --camera-view front_view
```

输出：

```text
point_cloud/<name>.mp4
<name>_point_cloud_summary.json
```

`--pcd-renderer auto` 会优先走 Cosmos-Drive-Dreams 的 Plotly renderer；如果参考 renderer 因 `open3d` 缺失不可用，会使用本仓库的 Plotly fallback。当前环境已安装 `plotly==6.7.0` 和 `kaleido==0.2.1`，可直接导出 mp4；想导出逐帧点云文件时加 `--save-ply`。

`--raw-valid-mode preprocess` 与训练预处理相同，按 raw projected range `> 0` 构造 GT occupancy。`--generated-valid-mode predicted` 按 decoder 输出自己的 range 阈值构造 generated occupancy，用于诊断模型自身 occupancy；`--generated-valid-mode layout` 复用 layout control 的 occupancy 作为 generated point cloud mask，用于当前 layout-conditioned 路径的低噪声点云重建；`--generated-valid-mode matched_gt` 只用于 oracle 对照。

## 评估

无条件生成很难用 paired GT 直接量化；本路径先采用 layout-conditioned evaluation，把 layout 当作可量化约束，检查生成结果是否遵守 occupancy 和 edge 结构，同时对有 GT 的样本计算 rangemap 米制误差。

```bash
python scripts/evaluate_waymo_lidar_rangemap_generation.py \
  --generated-video "$EVAL_DIR/inference_g3_s35_text_cond1/${SAMPLE}_${ITER}_ema_g3_s35_text_cond1.mp4" \
  --gt-video "$EVAL_DIR/inference_g3_s35_text_cond1/${SAMPLE}_${ITER}_ema_g3_s35_text_cond1_raw_target_input.mp4" \
  --layout-video "$EVAL_DIR/inference_g3_s35_text_cond1/${SAMPLE}_${ITER}_ema_g3_s35_text_cond1_raw_layout_input.mp4" \
  --layout-occupancy-threshold 90 \
  --layout-edge-threshold 90 \
  --output-json /tmp/<sample>_metrics.json
```

输出指标：

- `range_mae_m` / `range_rmse_m` / `range_bias_m`
- occupancy precision / recall / F1 / IoU
- edge precision / recall / F1 / IoU

数值评估入口读取展示 MP4，因此仍受 H264 量化影响；点云展示的 GT/rays 则直接来自 raw tar。Layout MP4 经 H264 编码后会有低值残留，评估默认用阈值 90 提取 occupancy/edge mask，而不是 `>0`。

### 当前 raw-online sanity 指标

2026-05-28 使用当前 raw-online checkpoint `waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000053000`、EMA bf16、`num_steps=35`、`guidance=3`、正常 fixed caption/text embedding、实际 `num_conditional_frames=1`，在 1 个 training 样本 `10017090168044687777_6380_000_6400_000_0` 上得到：

```text
range_mae_m                 1.2966
range_rmse_m                3.4973
range_bias_m               -0.5907
occupancy_iou               0.8371
occupancy_f1                0.9113
occupancy_precision/recall  0.9657 / 0.8627
edge_iou                    0.4603
edge_f1                     0.6304
edge_precision/recall       0.5629 / 0.7162
```

结果文件：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000053000/metrics_g3_s35_text_cond1/10017090168044687777_6380_000_6400_000_0_iter_000053000_ema_g3_s35_text_cond1_metrics.json
```

704x1280 decode 后的 GT / GEN / abs error 对比：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000053000/rangemap_compare_704_g3_s35_text_cond1/10017090168044687777_6380_000_6400_000_0_iter_000053000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
```

该 704x1280 对比口径下的辅助指标为 `range_mae_m=1.2970`、`range_rmse_m=3.5040`、`range_bias_m=-0.5827`、`occupancy_iou=0.8348`、`occupancy_f1=0.9100`。

点云 summary：

```text
GT valid                 2210832
generated valid          1973789
matched valid            1907370
generated extra            66419
generated missing         303462
point cloud video: outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000053000/pointcloud_vis_raw_g3_s35_text_cond1/point_cloud/10017090168044687777_6380_000_6400_000_0_iter_000053000_ema_g3_s35_text_cond1.mp4
```

判断：不建议在 `iter_000053000` 直接停止训练。理由是 raw-online 主线相对旧 tokenizer-converted 结果已经大幅改善 range 误差和 edge 指标，但当前只有单个 training 样本 sanity，occupancy recall 仍只有 `0.8627`，点云仍有 `303462` 个 missing 点，说明模型偏保守、还没有充分覆盖 layout occupancy。训练日志在 `53100-53700` 附近 loss 仍在 `0.02-0.04` 区间波动，没有看到明显发散；继续训练风险不高。

该建议已执行到 `iter_000100000`；后续是否继续训练以 100k 的 training sanity 和 validation8 结果为准。

### iter_000100000 raw-online 指标

2026-05-29 使用 `waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000`、EMA bf16、`num_steps=35`、`guidance=3`、fixed caption/text embedding、实际 `num_conditional_frames=1`，在同一 training 样本 `10017090168044687777_6380_000_6400_000_0` 上得到：

```text
range_mae_m                 1.2936
range_rmse_m                3.4953
range_bias_m               -0.5679
occupancy_iou               0.8387
occupancy_f1                0.9123
occupancy_precision/recall  0.9657 / 0.8645
edge_iou                    0.4612
edge_f1                     0.6313
edge_precision/recall       0.5640 / 0.7169
```

相对 `iter_000053000` 的同一样本，改善很小：`range_mae_m` 仅从 `1.2966` 到 `1.2936`，occupancy recall 从 `0.8627` 到 `0.8645`，点云 missing 从 `303462` 到 `299553`。

结果文件：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/metrics_g3_s35_text_cond1/10017090168044687777_6380_000_6400_000_0_iter_000100000_ema_g3_s35_text_cond1_metrics.json
```

704x1280 decode 后的 GT / GEN / abs error 对比：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/rangemap_compare_704_g3_s35_text_cond1/10017090168044687777_6380_000_6400_000_0_iter_000100000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
```

该 704x1280 对比口径下的辅助指标为 `range_mae_m=1.2942`、`range_rmse_m=3.5025`、`range_bias_m=-0.5601`、`occupancy_iou=0.8363`、`occupancy_f1=0.9109`。

点云 summary（`--generated-valid-mode predicted`，诊断模型自身 occupancy）：

```text
GT valid                 2210832
generated valid          1977921
matched valid            1911279
generated extra            66642
generated missing         299553
point cloud video: outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_vis_raw_g3_s35_text_cond1/point_cloud/10017090168044687777_6380_000_6400_000_0_iter_000100000_ema_g3_s35_text_cond1.mp4
```

点云 summary（`--generated-valid-mode layout`，layout-conditioned 点云重建）：

```text
GT valid                 2210832
generated valid          2209511
matched valid            2209220
generated extra              291
generated missing           1612
point cloud video: outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_vis_raw_g3_s35_text_cond1_layoutmask/point_cloud/10017090168044687777_6380_000_6400_000_0_iter_000100000_ema_g3_s35_text_cond1.mp4
```

同时抽了 3 个 validation segment 的首个 window 做 raw-online cond1 sanity。该集合很小，只用于发现明显问题，不能替代完整 validation：

| sample | range MAE | range RMSE | bias | occ F1 / IoU | edge F1 / IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10203656353524179475_7625_000_7645_000_0` | 4.3727 | 10.1390 | -2.6954 | 0.9311 / 0.8711 | 0.8066 / 0.6759 |
| `1024360143612057520_3580_000_3600_000_0` | 1.0334 | 3.8011 | -0.5193 | 0.9608 / 0.9245 | 0.8099 / 0.6805 |
| `10247954040621004675_2180_000_2200_000_0` | 1.1263 | 2.7844 | 0.1669 | 0.9558 / 0.9154 | 0.7424 / 0.5903 |
| mean | 2.1774 | 5.5748 | -1.0160 | 0.9492 / 0.9037 | 0.7863 / 0.6489 |

validation3 点云 totals（`predicted` generated occupancy）：

```text
GT valid                 5770664
generated valid          5850156
matched valid            5525212
generated extra           324944
generated missing         245452
extra/missing rate        5.63% / 4.25%
```

validation3 点云 totals（`layout` generated occupancy）：

```text
GT valid                 5770664
generated valid          5769098
matched valid            5767788
generated extra             1310
generated missing           2876
extra/missing rate        0.023% / 0.050%
```

validation3 结果文件：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation3_g3_s35_text_cond1_metrics/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_summary_validation3_g3_s35_text_cond1/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_summary_validation3_g3_s35_text_cond1_layoutmask/
```

2026-05-30 继续补跑 8 个 validation segment 的首个 window，仍使用 `iter_000100000` EMA bf16、`num_steps=35`、`guidance=3`、fixed caption/text embedding、实际 `num_conditional_frames=1`。推理需保留 `HF_HOME=/team/hyh/huggingface`，否则离线模式会找不到本地 Wan2.1 VAE tokenizer。

| sample | range MAE | range RMSE | bias | occ F1 / IoU | edge F1 / IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10203656353524179475_7625_000_7645_000_0` | 4.3758 | 10.1526 | -2.7149 | 0.9309 / 0.8707 | 0.8068 / 0.6762 |
| `1024360143612057520_3580_000_3600_000_0` | 1.0331 | 3.7972 | -0.5169 | 0.9612 / 0.9254 | 0.8090 / 0.6793 |
| `10247954040621004675_2180_000_2200_000_0` | 1.1286 | 2.7910 | 0.1807 | 0.9559 / 0.9154 | 0.7413 / 0.5890 |
| `10289507859301986274_4200_000_4220_000_0` | 1.7013 | 4.4741 | 0.1734 | 0.9797 / 0.9602 | 0.7546 / 0.6060 |
| `10335539493577748957_1372_870_1392_870_0` | 3.1734 | 7.2940 | -0.8891 | 0.9671 / 0.9362 | 0.7685 / 0.6240 |
| `10359308928573410754_720_000_740_000_0` | 0.8994 | 2.9106 | -0.4352 | 0.9579 / 0.9191 | 0.7579 / 0.6102 |
| `10448102132863604198_472_000_492_000_0` | 1.8919 | 5.3199 | -1.0678 | 0.8643 / 0.7611 | 0.7539 / 0.6051 |
| `10689101165701914459_2072_300_2092_300_0` | 2.9008 | 7.4150 | -1.0274 | 0.9772 / 0.9555 | 0.7763 / 0.6344 |
| mean | 2.1381 | 5.5193 | -0.7871 | 0.9493 / 0.9054 | 0.7711 / 0.6280 |

validation8 mean 的完整 binary 均值：occupancy precision/recall/F1/IoU 为 `0.9392 / 0.9599 / 0.9493 / 0.9054`，edge precision/recall/F1/IoU 为 `0.6762 / 0.8985 / 0.7711 / 0.6280`。

validation8 点云 totals（`predicted` generated occupancy）：

```text
GT valid                15949564
generated valid         16276364
matched valid           15342224
generated extra           934140
generated missing         607340
extra/missing rate        5.86% / 3.81%
```

validation8 点云 totals（`layout` generated occupancy）：

```text
GT valid                15949564
generated valid         15942951
matched valid           15938770
generated extra             4181
generated missing          10794
extra/missing rate        0.026% / 0.068%
```

704x1280 decode 后的 GT / GEN / abs error 诊断对比已补两个样本：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_rangemap_compare_704_g3_s35_text_cond1/10203656353524179475_7625_000_7645_000_0_iter_000100000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_rangemap_compare_704_g3_s35_text_cond1/10448102132863604198_472_000_492_000_0_iter_000100000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
```

704x1280 口径辅助指标：`102036...` 为 `range_mae_m=4.4080`、`range_rmse_m=10.1984`、`range_bias_m=-2.6707`；`104481...` 为 `range_mae_m=1.9079`、`range_rmse_m=5.3419`、`range_bias_m=-1.0263`。

occupancy 最差样本 `10448102132863604198_472_000_492_000_0` 的点云视频：

```text
predicted mask: outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_vis_validation8_g3_s35_text_cond1/10448102132863604198_472_000_492_000_0/point_cloud/10448102132863604198_472_000_492_000_0_iter_000100000_ema_g3_s35_text_cond1.mp4
layout mask:    outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_vis_validation8_g3_s35_text_cond1_layoutmask/10448102132863604198_472_000_492_000_0/point_cloud/10448102132863604198_472_000_492_000_0_iter_000100000_ema_g3_s35_text_cond1.mp4
```

validation8 结果文件：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_g3_s35_text_cond1/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_g3_s35_text_cond1_metrics/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_g3_s35_text_cond1_metrics/summary_iter_000100000_ema_g3_s35_text_cond1_validation8.json
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_summary_validation8_g3_s35_text_cond1/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_summary_validation8_g3_s35_text_cond1_layoutmask/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_rangemap_compare_704_g3_s35_text_cond1/
```

判断：不建议现在直接无条件继续长训。`iter_000100000` 在同一个 training sanity 上相对 53k 基本进入平台期；validation8 mean 与 validation3 接近，但样本间波动仍明显，`102036...` 的 `range_mae_m=4.3758`、`range_rmse_m=10.1526`，`104481...` 的 occupancy F1 只有 `0.8643`，predicted-mask 点云 extra/missing 达 `14.44% / 12.84%`。同时，复用 layout occupancy 后 validation8 点云 totals 的 extra/missing 从 `5.86% / 3.81%` 降到 `0.026% / 0.068%`，说明反投影几何和 raw rays 基本可用，当前主要瓶颈是 invalid/occupancy 表示与模型自身 occupancy，而不是单纯训练步数。建议先固定 validation8 或扩到 `20-50` 个 validation windows，对 53k/80k/100k 做同集合对比；如果 100k 没有稳定优于早期 checkpoint，下一轮应优先改 target/valid 表示、显式 occupancy/valid loss 或数据采样。若必须继续训练，只建议短续到 `120k`，每 `5k-10k` 用固定 validation 集 early stop。


### 下一轮训练改动：valid + edge aware latent loss

2026-05-30 开始修改训练代码，本轮不只针对 occupancy/invalid，而是把 raw rangemap 的有效测距区域和 range discontinuity/edge 区域都纳入训练权重；不改变 raw target/control 的外部数据格式：

- `SingleViewTransferDataset` 在 raw-online 路径额外输出 `rangemap_valid_mask` 和 `rangemap_edge_mask`。`valid` 来自 raw projected range `> 0`，`edge` 复用 `rangemap_layout` 的 occupancy boundary 与 range discontinuity 定义；二者 shape 均为 `[1, 29, 704, 1280]`。
- `ControlVideo2WorldModelRectifiedFlow` 在在线 encode `rangemap_target` 后，将 valid/edge mask trilinear 下采样到 latent grid `[B, 1, 8, 88, 160]`，合成 `edm_loss_weight`。
- `Text2WorldModelRectifiedFlow` 支持可选 `edm_loss_weight_key`，对 rectified-flow latent MSE 做 per-latent 加权，并按 mean weight 归一化，避免整体 loss scale 大幅漂移；context-parallel split 会同步处理 loss weight。
- 当前 LiDAR raw-online 主实验启用 `invalid=1.0`、`valid=1.25`、`edge=2.0`。最终 per-latent weight 取 valid-derived weight 与 edge-derived weight 的逐点最大值，目标是同时补强有效 range、边界/不连续结构和点云几何，而不是只优化 occupancy。

对应配置已写入：

```text
transfer2_singleview_posttrain_waymo_lidar_wan21_online_layout_fullfinetune["model"]["config"]:
  edm_loss_weight_key="edm_loss_weight"
  online_target_valid_mask_key="rangemap_valid_mask"
  online_target_edge_mask_key="rangemap_edge_mask"
  online_target_valid_loss_weight=1.25
  online_target_edge_loss_weight=2.0
  online_target_invalid_loss_weight=1.0
```

最小验证：

```text
python -m py_compile cosmos_transfer2/_src/predict2/models/text2world_model_rectified_flow.py \
  cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py \
  cosmos_transfer2/_src/transfer2/datasets/local_datasets/singleview_dataset.py \
  cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py \
  cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py

raw sample smoke:
video/layout/target = (3, 29, 704, 1280)
valid mask          = (1, 29, 704, 1280), dtype=torch.bool, count=22405922
edge mask           = (1, 29, 704, 1280), dtype=torch.bool, count=8994337

latent weight smoke:
shape=(2, 1, 8, 88, 160), min=1.0, max=2.0, mean=1.0683
```

该建议已执行并在 `iter_000134000` 用固定 validation8 复测；结论见下节。后续不要再更换验证集合，否则 53k/100k/valid-edge-weighted 结果不可直接比较。

### iter_000134000 valid + edge aware loss 验证

2026-05-30 使用 valid + edge aware latent loss 从 `iter_000100000` 续训，训练在 `134300` 附近中断；本轮评估使用最近完整 checkpoint `iter_000134000`、EMA bf16、`num_steps=35`、`guidance=3`、fixed caption/text embedding、实际 `num_conditional_frames=1`。验证集合保持上一轮固定 validation8，仍是 8 条数据、每卡一条推理。

validation8 mean 与 `iter_000100000` 对比：

| metric | iter_000100000 | iter_000134000 | delta |
| --- | ---: | ---: | ---: |
| range MAE | 2.1381 | 2.1223 | -0.0158 |
| range RMSE | 5.5193 | 5.4684 | -0.0509 |
| range bias | -0.7871 | -0.7687 | +0.0184 |
| occupancy precision / recall | 0.9392 / 0.9599 | 0.9395 / 0.9612 | +0.0003 / +0.0012 |
| occupancy F1 / IoU | 0.9493 / 0.9054 | 0.9500 / 0.9067 | +0.0007 / +0.0013 |
| edge precision / recall | 0.6762 / 0.8985 | 0.6762 / 0.8984 | +0.0000 / -0.0001 |
| edge F1 / IoU | 0.7711 / 0.6280 | 0.7710 / 0.6280 | -0.0000 / -0.0000 |

逐样本变化：

| sample | MAE 100k -> 134k | RMSE 100k -> 134k | occ F1 100k -> 134k | edge F1 100k -> 134k |
| --- | ---: | ---: | ---: | ---: |
| `10203656353524179475_7625_000_7645_000_0` | 4.3758 -> 4.2342 | 10.1526 -> 9.7633 | 0.9309 -> 0.9356 | 0.8068 -> 0.8060 |
| `1024360143612057520_3580_000_3600_000_0` | 1.0331 -> 1.0342 | 3.7972 -> 3.7975 | 0.9612 -> 0.9620 | 0.8090 -> 0.8098 |
| `10247954040621004675_2180_000_2200_000_0` | 1.1286 -> 1.1327 | 2.7910 -> 2.7908 | 0.9559 -> 0.9564 | 0.7413 -> 0.7426 |
| `10289507859301986274_4200_000_4220_000_0` | 1.7013 -> 1.7118 | 4.4741 -> 4.4804 | 0.9797 -> 0.9796 | 0.7546 -> 0.7543 |
| `10335539493577748957_1372_870_1392_870_0` | 3.1734 -> 3.1818 | 7.2940 -> 7.2949 | 0.9671 -> 0.9670 | 0.7685 -> 0.7694 |
| `10359308928573410754_720_000_740_000_0` | 0.8994 -> 0.8992 | 2.9106 -> 2.9087 | 0.9579 -> 0.9576 | 0.7579 -> 0.7581 |
| `10448102132863604198_472_000_492_000_0` | 1.8919 -> 1.8881 | 5.3199 -> 5.3048 | 0.8643 -> 0.8646 | 0.7539 -> 0.7515 |
| `10689101165701914459_2072_300_2092_300_0` | 2.9008 -> 2.8963 | 7.4150 -> 7.4063 | 0.9772 -> 0.9770 | 0.7763 -> 0.7765 |

validation8 点云 totals（`predicted` generated occupancy）：

```text
GT valid                15949564
generated valid         16289312
matched valid           15357817
generated extra           931495
generated missing         591747
extra/missing rate        5.84% / 3.71%
```

对比 `iter_000100000` 的 `5.86% / 3.81%`，extra 小幅下降 `2645` 点，missing 下降 `15593` 点；改善真实存在但幅度很小。validation8 点云 totals（`layout` generated occupancy）保持完全一致：extra/missing 仍是 `4181 / 10794 = 0.026% / 0.068%`，说明点云反投影和 layout occupancy 口径稳定。

704x1280 decode 后 GT / GEN / abs error 诊断对比已补两个样本：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_rangemap_compare_704_g3_s35_text_cond1/10203656353524179475_7625_000_7645_000_0_iter_000134000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_rangemap_compare_704_g3_s35_text_cond1/10448102132863604198_472_000_492_000_0_iter_000134000_ema_g3_s35_text_cond1_gt_gen_absdiff_704x1280.mp4
```

704x1280 口径辅助指标：`102036...` 为 `range_mae_m=4.2654`、`range_rmse_m=9.8116`、`range_bias_m=-2.5027`、occupancy F1/IoU `0.9331 / 0.8746`；`104481...` 为 `range_mae_m=1.9039`、`range_rmse_m=5.3272`、`range_bias_m=-1.0016`、occupancy F1/IoU `0.8606 / 0.7553`。

occupancy 最差样本 `10448102132863604198_472_000_492_000_0` 的点云视频：

```text
predicted mask: outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/pointcloud_vis_validation8_g3_s35_text_cond1/10448102132863604198_472_000_492_000_0/point_cloud/10448102132863604198_472_000_492_000_0_iter_000134000_ema_g3_s35_text_cond1.mp4
layout mask:    outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/pointcloud_vis_validation8_g3_s35_text_cond1_layoutmask/10448102132863604198_472_000_492_000_0/point_cloud/10448102132863604198_472_000_492_000_0_iter_000134000_ema_g3_s35_text_cond1.mp4
```

validation8 结果文件：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_g3_s35_text_cond1/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_g3_s35_text_cond1_metrics/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_g3_s35_text_cond1_metrics/summary_iter_000134000_ema_g3_s35_text_cond1_validation8.json
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/pointcloud_summary_validation8_g3_s35_text_cond1/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/pointcloud_summary_validation8_g3_s35_text_cond1_layoutmask/
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000134000/validation8_rangemap_compare_704_g3_s35_text_cond1/
```

判断：valid + edge aware latent loss 在 `100k -> 134k` 的固定 validation8 上有弱正向效果，主要体现为 range MAE/RMSE、bias、occupancy recall/F1/IoU 和 predicted-mask 点云 missing 小幅改善；edge 指标基本不变。该效果不足以证明继续长训会显著提升，且 `104481...` 的 occupancy 仍是主要短板。当前决策是不继续训到 `150k`，该改动暂不作为主线路径合入；后续主线仍以 `iter_000100000` raw-online baseline 和固定 validation8 为主要对照。valid + edge aware loss 相关提交只作为实验记录保留，下一步应转向更直接的 occupancy/invalid 表示、decoder 后处理或阈值策略。2026-05-30 已在 `docs/waymo_lidar_wan21_vae_preprocessing.md` 补充固定 validation8 的 Wan2.1 VAE roundtrip 和阈值诊断，结论是当前 frozen VAE 自身会产生 invalid bleed，应优先验证显式 valid/invalid 表示或 LiDAR VAE fine-tune。2026-05-31 的 `polyphase3_repeat + roll=640` 单帧 VAE fine-tune 已把 validation8 frame0 mean MAE/F1 改到 `0.2397m / 0.9887`，但直接用于 29 帧时 range 指标反而差于 frozen Wan2.1，因此 `step_050000` 只作为单帧表示实验参考，不能替换当前 29-frame 主线 tokenizer；若继续 VAE 方向，应新开 `num_frames=29` smoke 或保护 temporal 模块后再评估。

### iter_000100000 decode valid 后处理 sweep

2026-05-30 在不重新推理、不改训练的前提下，给 `scripts/evaluate_waymo_lidar_rangemap_generation.py` 和 `scripts/visualize_waymo_lidar_generation.py` 补齐 generated valid mask 选项：

```text
--generated-valid-mode predicted|layout|matched_gt
--generated-valid-threshold-m <meters>
```

默认仍等价于旧口径：`predicted` 且 threshold 为 `min_range + valid_min_offset_m = 5.25m`，所以历史指标不变。新增参数用于可复现实验 decode 后点云重建的 valid 策略；不要把 layout/matched_gt 模式当成无条件生成能力指标。

在 `iter_000100000` 固定 validation8 上重评估四种策略，range MAE/RMSE/Bias 均保持旧值 `2.1381m / 5.5193m / -0.7871m`，因为 range 误差仍按 GT valid 区域计算；变化只来自 generated occupancy/edge mask：

| generated valid strategy | occ F1 / IoU | edge F1 | generated valid | extra | missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `predicted @ 5.25m` | 0.9493 / 0.9054 | 0.7711 | 16276364 | 941733 | 608320 |
| `predicted @ 5.50m` | **0.9527 / 0.9114** | **0.7765** | 15843216 | 661775 | 761510 |
| `predicted @ 6.00m` | 0.9486 / 0.9044 | 0.7764 | 15502328 | 542205 | 982828 |
| `layout` | 1.0000 / 1.0000 | 0.8144 | 15942951 | 0 | 0 |

`5.50m` 是当前 predicted-mask 后处理里比较稳的折中：extra 比 `5.25m` 少约 `280k`，missing 多约 `153k`，occupancy/edge F1 同时小幅提升。`6.00m` 继续降噪但 missing 增幅过大，不建议作为默认点云重建阈值。`layout` gating 依赖 layout control occupancy，适合作为受控 point-cloud reconstruction 或可视化上界；它会隐藏模型自己的 occupancy 错误，因此不应用于主指标判断。

结果目录：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/validation8_g3_s35_text_cond1_decode_valid_sweep_metrics/
```

单样本 point-cloud summary smoke 也验证了可视化脚本的 `--generated-valid-threshold-m 5.50` 入口，样本 `10448102132863604198_472_000_492_000_0` 输出：

```text
outputs/waymo_lidar_eval/waymo_lidar_wan21_raw_online_layout_fullfinetune_i2v_t8/iter_000100000/pointcloud_summary_decode_t5p50_smoke/10448102132863604198_472_000_492_000_0_iter_000100000_ema_g3_s35_text_cond1_point_cloud_summary.json
```

后续点云可视化可先用 `predicted @ 5.50m` 作为降噪对照；论文/主报告指标仍保留 `5.25m` baseline，并额外报告 threshold/gating sweep，避免后处理选择掩盖生成模型本身的问题。

### official Waymo raw range image sanity

2026-05-30 已新增 `scripts/inspect_waymo_official_lidar.py`，直接用官方 Waymo Open Dataset API 从 `/team/hyh/data/waymo/raw` 读取 range image 并重建点云。结果记录在 `docs/waymo_lidar_wan21_vae_preprocessing.md`：官方 TOP range image 是 `[64,2650,4]`，当前 `rds_hq_waymo/*/lidar_raw/*.tar` 等价于官方 TOP return1+return2 点云；主要损失不在 RDS raw 点坐标，而在后续把 native `[64,2650]` 和双 return 结构重新投影/压缩到当前训练用 `64x1280` range map。


### 历史 tokenizer-converted sanity 指标

以下数值来自旧 `rangemap_layout_fullfinetune_i2v_t8` / tokenizer-converted checkpoint，不代表 raw-online 主线。2026-05-26 使用 `iter_000075000`、EMA bf16、`num_steps=35`、`guidance=3`、正常 fixed caption/text embedding，在 1 个 training 样本 `10017090168044687777_6380_000_6400_000_0` 上得到：

```text
range_mae_m                 14.7988
range_rmse_m                15.7232
range_bias_m                14.2980
occupancy_iou               0.9402
occupancy_f1                0.9692
occupancy_precision/recall  0.9411 / 0.9990
edge_iou                    0.3604
edge_f1                     0.5299
edge_precision/recall       0.4025 / 0.7751
```

结果文件：`outputs/waymo_lidar_eval/waymo_lidar_singleview_rangemap_layout_fullfinetune_i2v_t8/iter_000075000/metrics_train1_g3_s35_text/10017090168044687777_6380_000_6400_000_0_iter75000_ema_g3_s35_text_metrics.json`。

对应点云 summary：GT valid `2077583`，generated valid `2205174`，generated extra `129696`，generated missing `2105`。

历史对照：2026-05-25 的 `iter_000037000`、EMA bf16、`num_steps=4`、`--zero-text-embedding`、3 个 training 样本 sanity 结果保存在 `outputs/waymo_lidar_eval/waymo_lidar_singleview_rangemap_layout_fullfinetune_i2v_t8/iter_000037000/metrics_train3_fixed/summary_iter37000_ema_train3_fixed.json`。
