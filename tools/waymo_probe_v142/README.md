# Waymo LiDAR Tokenizer Dataset Builder

This toolset converts Waymo Perception TFRecords into a segment-level dataset layout that separates sidecar metadata from tokenizer inputs:

```text
/data2/waymo_lidar_tokenizer/
  metadata/
    <segment_id>.npz
  lidar/
    <segment_id>.npz
  previews/
    <segment_id>/
      frame_00000/
        rangemap_range.png
        rangemap_intensity.png
        rangemap_valid_mask.png
        reconstructed_points.npy
        reconstructed_points.ply
        pointcloud.png
```

The design goal is:

- tokenizer training only reads `lidar/<segment_id>.npz`
- point clouds can be reconstructed later using only `metadata/<segment_id>.npz + lidar/<segment_id>.npz`

## Saved Format

### `metadata/<segment_id>.npz`

Required fields:

- `context_name`
- `segment_id`
- `split`
- `laser_name`
- `return_index`
- `timestamps_list[T]`
- `frame_indices[T]`
- `pose_list[T,4,4]`
- `frame_pose[T,4,4]`
- `lidar_extrinsic[4,4]`
- `beam_inclinations[H]`
- `beam_inclination_minmax[2]`
- `range_image_shape[2]`
- `is_raw_range_image`
- `has_per_pixel_pose`
- `range_image_top_pose[T,H,W,6]` when available
- `range_norm_type`, `range_min`, `range_max`
- `intensity_norm_type`, `intensity_min`, `intensity_max`

### `lidar/<segment_id>.npz`

Tokenizer input:

- `images[T,3,H,W]`

Raw sidecars:

- `valid_mask[T,1,H,W]`
- `range_raw[T,1,H,W]`
- `intensity_raw[T,1,H,W]`
- `elongation_raw[T,1,H,W]`
- `nlz_raw[T,1,H,W]`

The 3 tokenizer channels are fixed as:

- `ch0 = normalized_range`
- `ch1 = normalized_intensity`
- `ch2 = valid_mask`

Current normalization policy:

- `range_norm_type = linear_clip`, `range_min = 0.0`, `range_max = 75.0`
- `intensity_norm_type = linear_clip`, `intensity_min = 0.0`, `intensity_max = 1.0`

## Environment

Create the conda environment:

```bash
conda create -n waymo142-probe -c conda-forge python=3.10 -y
```

Install the parser and visualization stack:

```bash
conda run -n waymo142-probe python -m pip install tensorflow==2.11.1 'protobuf<4' open3d matplotlib pillow
conda run -n waymo142-probe python -m pip install --index-url https://pypi.org/simple waymo-open-dataset-tf-2-11-0==1.6.1
conda run -n waymo142-probe python -m pip install kaleido==0.2.1
```

Notes:

- `waymo-open-dataset-tf-2-11-0==1.6.1` is the installable official wheel that worked in this environment.
- `kaleido==0.2.1` is pinned because the Waymo wheel pulls `plotly==5.13.1`.

## Single-Segment Extraction

Extract one segment into `metadata/` and `lidar/`:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/extract_top_ri.py \
  --tfrecord_path /data/waymo/raw/training/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.tfrecord \
  --output_root /data2/waymo_lidar_tokenizer \
  --split training
```

## Preview Rendering

Render previews only from the saved files:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/render_outputs.py \
  --metadata_path /data2/waymo_lidar_tokenizer/metadata/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --lidar_path /data2/waymo_lidar_tokenizer/lidar/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --preview_root /data2/waymo_lidar_tokenizer/previews
```

## Offline Point Cloud Reconstruction

Reconstruct point clouds using only saved dataset files:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/reconstruct_pointcloud_from_saved.py \
  --metadata_path /data2/waymo_lidar_tokenizer/metadata/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --lidar_path /data2/waymo_lidar_tokenizer/lidar/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --output_root /data2/waymo_lidar_tokenizer/reconstructed_points \
  --save_ply 1
```

## Online-vs-Offline Validation

Compare saved-file reconstruction against the official Waymo online parser:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/compare_reconstruction_to_waymo.py \
  --metadata_path /data2/waymo_lidar_tokenizer/metadata/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --lidar_path /data2/waymo_lidar_tokenizer/lidar/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --tfrecord_path /data/waymo/raw/training/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.tfrecord \
  --output_json /data2/waymo_lidar_tokenizer/validation/segment-10327752107000040525_1120_000_1140_000_with_camera_labels_compare.json
```

The output JSON includes:

- per-frame point-count agreement
- per-frame `max_abs_error`
- per-frame `mean_abs_error`
- per-frame `rmse`

## Images-to-Video

Convert saved `images[T,3,H,W]` into an MP4 video:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/images_to_video.py \
  --lidar_path /data2/waymo_lidar_tokenizer/lidar/segment-10327752107000040525_1120_000_1140_000_with_camera_labels.npz \
  --output_path /data2/waymo_lidar_tokenizer/videos/segment-10327752107000040525_1120_000_1140_000_with_camera_labels_images.mp4 \
  --fps 10 \
  --scale 4
```

## Full-Dataset Processing

Process the full `training` split:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/process_dataset.py \
  --input_root /data/waymo/raw \
  --output_root /data2/waymo_lidar_tokenizer \
  --splits training \
  --render 1 \
  --skip_existing 1
```

Process multiple splits:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/process_dataset.py \
  --input_root /data/waymo/raw \
  --output_root /data2/waymo_lidar_tokenizer \
  --splits training validation testing \
  --render 1 \
  --skip_existing 1
```

Smoke test on a small sample:

```bash
conda run -n waymo142-probe python /root/workspace/cosmos-transfer2.5/tools/waymo_probe_v142/process_dataset.py \
  --input_root /data/waymo/raw \
  --output_root /data2/waymo_lidar_tokenizer \
  --splits training \
  --max_segments 1 \
  --max_frames_per_segment 2 \
  --render 1 \
  --skip_existing 0
```

## Current Defaults

- sensor: `TOP`
- return: `first return`
- no `128x3600` reprojection
- tokenizer channels: `range + intensity + valid_mask`
- `elongation`, `NLZ`, and per-pixel pose are preserved as sidecars, not fed into tokenizer forward
