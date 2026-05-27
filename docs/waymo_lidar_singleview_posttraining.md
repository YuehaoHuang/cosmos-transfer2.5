# Waymo LiDAR Single-View Post-Training

更新日期：2026-05-26

本文记录新的 Waymo TOP LiDAR rangemap 生成主线。训练代码走官方 single-view post-training 框架；dataloader 每个 batch 从 raw LiDAR tar 在线投影 target 和 `rangemap_layout`，训练步再在线调用 Wan2.1 VAE `encode` 得到 diffusion target。MP4 与 `.npz` 都不进入默认监督路径。

## 当前服务器环境

H100-DriveSync 上的工作目录和环境：

```bash
cd /team/hyh/code/cosmos-transfer2.5
source /opt/conda/etc/profile.d/conda.sh
conda activate drivesync
```

当前已训练 checkpoint 使用的是 `/team/hyh/data/waymo_singleview_lidar_posttrain_from_tokenizer/training` 旧 tokenizer-converted 数据。2026-05-26 起，新的在线 encode 主线改为 raw Waymo TOP 投影的 `704x1280` contract；旧 `704x1200` 指标仅作为历史 sanity check。

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
TOTAL_ITER=40000 \
CHUNK_ITER=10000 \
SAVE_ITER=1000 \
./train_waymo_lidar_singleview_chunked.sh
```

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
RUN=outputs/waymo_lidar_singleview_posttrain/cosmos_transfer2_posttrain/waymo_lidar_singleview/waymo_lidar_singleview_rangemap_layout_fullfinetune_i2v_t8
ITER=iter_000075000
EVAL_DIR=outputs/waymo_lidar_eval/waymo_lidar_singleview_rangemap_layout_fullfinetune_i2v_t8/$ITER

python scripts/convert_distcp_to_pt.py \
  "$RUN/checkpoints/$ITER/model" \
  "$EVAL_DIR/checkpoint_pt"
```

之后使用专用入口推理：

```bash
DATASET_DIR=/team/hyh/data/waymo_singleview_lidar_posttrain_from_tokenizer/training
SAMPLE=10017090168044687777_6380_000_6400_000_0

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/team/hyh/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/infer_waymo_lidar_singleview_posttrain.py \
  --checkpoint-path "$EVAL_DIR/checkpoint_pt/model_ema_bf16.pt" \
  --dataset-dir "$DATASET_DIR" \
  --sample-key "$SAMPLE" \
  --output-dir "$EVAL_DIR/inference_train1_g3_s35_text" \
  --name "${SAMPLE}_iter75000_ema_g3_s35_text" \
  --experiment transfer2_singleview_posttrain_waymo_lidar_rangemap_layout_fullfinetune \
  --num-conditional-frames 1 \
  --num-steps 35 \
  --guidance 3 \
  --max-frames 29 \
  --num-video-frames-per-chunk 29 \
  --skip-comparison
```

`--num-steps 35 --guidance 3` 对齐训练 callback 的常规采样设置；该命令使用正常 caption/text embedding。若只想隔离 layout control 能力，可额外加 `--zero-text-embedding`；去掉 `--skip-comparison` 会额外保存 GT/control/generated 拼接视频。

输出：

```text
<name>.mp4
<name>_control_rangemap_layout.mp4
<name>_comparison_gt_control_generated.mp4  # only when not using --skip-comparison
```

## 点云可视化

Transfer2 推理保存的 `<name>.mp4` 已经是 decoder 后的 rangemap video，可以反归一化成米制 range，再按 Waymo TOP LiDAR ray/extrinsic 渲染点云。脚本会优先从同名 JSON 读取 `sample_key`、GT video 和 layout path，并尝试用原始 tokenizer tar 生成更准确的 valid mask 和 ray directions。

```bash
python scripts/visualize_waymo_lidar_generation.py \
  --generated-video "$EVAL_DIR/inference_train1_g3_s35_text/${SAMPLE}_iter75000_ema_g3_s35_text.mp4" \
  --output-dir "$EVAL_DIR/pointcloud_vis_train" \
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

`--raw-valid-mode preprocess` 会按数据预处理口径构造 GT valid mask，避免把 0-5m 的近距离点误删。`--generated-valid-mode predicted` 会按 decoded generated range 自己的阈值构造 generated mask；如需旧式逐像素 range 对齐检查，可改成 `--generated-valid-mode matched_gt`。当前已生成 Plotly 示例：`outputs/waymo_lidar_eval/waymo_lidar_singleview_rangemap_layout_fullfinetune_i2v_t8/iter_000075000/pointcloud_vis_train1_g3_s35_text_predvalid/point_cloud/10017090168044687777_6380_000_6400_000_0_iter75000_ema_g3_s35_text.mp4`。

## 评估

无条件生成很难用 paired GT 直接量化；本路径先采用 layout-conditioned evaluation，把 layout 当作可量化约束，检查生成结果是否遵守 occupancy 和 edge 结构，同时对有 GT 的样本计算 rangemap 米制误差。

```bash
python scripts/evaluate_waymo_lidar_rangemap_generation.py \
  --generated-video /path/to/generated.mp4 \
  --gt-video "$DATASET_DIR/videos/$SAMPLE.mp4" \
  --layout-video "$DATASET_DIR/rangemap_layout/$SAMPLE.mp4" \
  --layout-occupancy-threshold 90 \
  --layout-edge-threshold 90 \
  --output-json /tmp/<sample>_metrics.json
```

输出指标：

- `range_mae_m` / `range_rmse_m` / `range_bias_m`
- occupancy precision / recall / F1 / IoU
- edge precision / recall / F1 / IoU

Layout mp4 经 H264 编码后会有低值残留，评估默认用阈值 90 提取 occupancy/edge mask，而不是 `>0`。

### 当前 sanity 指标

2026-05-26 使用 `iter_000075000`、EMA bf16、`num_steps=35`、`guidance=3`、正常 fixed caption/text embedding，在 1 个 training 样本 `10017090168044687777_6380_000_6400_000_0` 上得到：

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
