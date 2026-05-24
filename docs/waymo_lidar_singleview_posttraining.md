# Waymo LiDAR Single-View Post-Training

更新日期：2026-05-10

本文记录新的 Waymo TOP LiDAR rangemap video 生成主线。训练代码走官方 single-view post-training 框架：`scripts.train`、`cosmos_transfer2/singleview_config.py`、官方 single-view dataloader/config/launcher 风格。该路径不使用 depth 作为条件，也不依赖 camera video 联动。

## 目标

- 目标视频：29 帧 Waymo TOP LiDAR rangemap，`704x1280x3`。
- 条件视频：预计算的稀疏 `rangemap_layout`，表达 occupancy 和 range discontinuity edge。
- 数据结构保持官方 local single-view 样式：

```text
datasets/your_dataset/
├── videos/
│   └── *.mp4
├── captions/
│   └── *.json
└── rangemap_layout/
    └── *.mp4
```

`depth/` 仍然是官方 depth-control 的可选目录，但本 LiDAR rangemap 生成路径不使用它。

## 数据准备

默认从 Waymo raw TOP LiDAR tar 读取 29 帧，生成 target rangemap video 和 layout control video：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-transfer2.5-merge

python scripts/prepare_waymo_lidar_singleview_posttrain_dataset.py \
  --split training \
  --output-root /data2/waymo_singleview_lidar_posttrain \
  --caption-mode fixed
```

输出：

```text
/data2/waymo_singleview_lidar_posttrain/training/
├── videos/            # TOP LiDAR rangemap target videos
├── rangemap_layout/   # sparse layout controls, not metric depth
├── captions/          # {"caption": "..."} JSON files
└── metadata/          # alignment/debug metadata
```

默认 LiDAR video contract：

```text
source raw lidar: /data2/rds_hq_waymo/<split>/lidar_raw/<segment_key>.tar
frame_start: chunk_index * 10
num_frames: 29
raw range map: 64 x 1280
target video: 29 x 704 x 1280 x 3 uint8
layout video: 29 x 704 x 1280 x 3 uint8
normalization: range [5m, 100m] -> [-1, 1] -> uint8
fps: 10
```

Caption 可选模式：

- `--caption-mode fixed`：固定 LiDAR caption，最稳。
- `--caption-mode source`：读取 `/data/waymo/waymo_multiview_texts.json` 的原始 camera caption。
- `--caption-mode vlm`：从 `/data/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/...` 或 `--caption-model-path` 加载本地 VLM，读取 rangemap 中间帧生成 caption。

Smoke 数据：

```bash
python scripts/prepare_waymo_lidar_singleview_posttrain_dataset.py \
  --split validation \
  --output-root /tmp/waymo_lidar_layout_posttrain_smoke \
  --max-samples 1 \
  --overwrite \
  --caption-mode fixed
```

## 训练

注册实验：

```text
transfer2_singleview_posttrain_waymo_lidar_rangemap_layout_fullfinetune
```

关键配置：

- official single-view Transfer2 post-training stack
- `state_t=8`
- `num_frames=29`
- `hint_keys=rangemap_layout`
- `min_num_conditional_frames=1`
- `max_num_conditional_frames=1`
- 初始化使用官方 single-view edge checkpoint，作为单分支 control 的最近可用预训练权重

启动：

```bash
./train_waymo_lidar_singleview_chunked.sh
```

常用覆盖：

```bash
DATASET_DIR=/data2/waymo_singleview_lidar_posttrain/training \
OUTPUT_ROOT=/data2/waymo_lidar_singleview_posttrain_output \
NUM_GPUS=8 \
TOTAL_ITER=5000 \
SAVE_ITER=500 \
./train_waymo_lidar_singleview_chunked.sh
```

`state_t=8` 要求 GPU 数整除 8，推荐 `1`、`2`、`4`、`8`。只有 7 张 GPU 可见时先用 `--gpus 4`。

Dry run：

```bash
USE_TMUX=false ./train_waymo_lidar_singleview_chunked.sh \
  --dataset-dir /tmp/waymo_lidar_layout_posttrain_smoke/validation \
  --gpus 1 \
  --total-iter 1 \
  --save-iter 1 \
  --logging-iter 1 \
  --num-workers 0 \
  --dry-run \
  --no-tmux
```

## 推理

DCP checkpoint 需要先转换成 `model_ema_bf16.pt`，之后使用专用入口：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/dev/shm/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/infer_waymo_lidar_singleview_posttrain.py \
  --checkpoint-path /path/to/model_ema_bf16.pt \
  --dataset-dir /data2/waymo_singleview_lidar_posttrain/validation \
  --sample-key 10203656353524179475_7625_000_7645_000_0 \
  --num-steps 4 \
  --max-frames 29 \
  --num-video-frames-per-chunk 29
```

输出：

```text
<sample>_step04.mp4
<sample>_step04_control_rangemap_layout.mp4
<sample>_step04_comparison_gt_control_generated.mp4
```

## 评估

无条件生成很难用 paired GT 直接量化；本路径先采用 layout-conditioned evaluation，把 layout 当作可量化约束，检查生成结果是否遵守 occupancy 和 edge 结构，同时对有 GT 的样本计算 rangemap 米制误差。

```bash
python scripts/evaluate_waymo_lidar_rangemap_generation.py \
  --generated-video /path/to/generated.mp4 \
  --gt-video /data2/waymo_singleview_lidar_posttrain/validation/videos/<sample>.mp4 \
  --layout-video /data2/waymo_singleview_lidar_posttrain/validation/rangemap_layout/<sample>.mp4 \
  --output-json /tmp/<sample>_metrics.json
```

输出指标：

- `range_mae_m` / `range_rmse_m` / `range_bias_m`
- occupancy precision / recall / F1 / IoU
- edge precision / recall / F1 / IoU
