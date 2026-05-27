#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Visualize decoded one-way video->LiDAR outputs in Waymo tokenizer style.

This is intentionally an offline utility: it consumes the decoded tensor saved by
``infer_waymo_video_lidar_one_way_expert.py`` and writes the same high-level
artifacts used by the Cosmos-Drive-Dreams LiDAR tokenizer docs:

  range_map_video/{sample_key}_{prediction_key}.mp4
  histogram/{sample_key}_{prediction_key}.png
  point_cloud/{sample_key}_{prediction_key}.mp4

The decoded tensors are normalized tokenizer outputs with shape
[B, C, T, repeat_row * 128, crop_width]. This script undoes the row/column repeat,
clips back to the tokenizer normalization range, unnormalizes to meters, and uses
Waymo TOP LiDAR beam elevations for range-map-to-point-cloud conversion.

By default the range-map video has three rows:
  raw/preprocessed GT, decode(gt latent), prediction.

Recommended environment:
  conda activate cosmos-transfer2.5-merge
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw


WAYMO_TOP_LIDAR_EXTRINSIC = np.array(
    [
        [-8.4777248e-01, -5.3035414e-01, -2.5136571e-03, 1.4299999e00],
        [5.3035545e-01, -8.4777534e-01, 1.8014426e-04, 0.0000000e00],
        [-2.2265569e-03, -1.1804104e-03, 9.9999684e-01, 2.1840000e00],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ],
    dtype=np.float64,
)

WAYMO_TOP_BEAM_INCLINATIONS_RAD = np.array(
    [
        0.03849,
        0.03570,
        0.03231,
        0.02941,
        0.02658,
        0.02379,
        0.02089,
        0.01803,
        0.01460,
        0.01198,
        0.00899,
        0.00617,
        0.00329,
        0.00055,
        -0.00268,
        -0.00545,
        -0.00849,
        -0.01113,
        -0.01419,
        -0.01682,
        -0.02016,
        -0.02294,
        -0.02590,
        -0.02894,
        -0.03222,
        -0.03554,
        -0.03962,
        -0.04336,
        -0.04745,
        -0.05171,
        -0.05606,
        -0.06060,
        -0.06619,
        -0.07076,
        -0.07635,
        -0.08161,
        -0.08721,
        -0.09276,
        -0.09927,
        -0.10564,
        -0.11206,
        -0.11833,
        -0.12536,
        -0.13210,
        -0.13941,
        -0.14685,
        -0.15429,
        -0.16180,
        -0.16976,
        -0.17758,
        -0.18603,
        -0.19431,
        -0.20291,
        -0.21142,
        -0.22096,
        -0.22990,
        -0.23938,
        -0.24864,
        -0.25816,
        -0.26761,
        -0.27795,
        -0.28828,
        -0.29886,
        -0.30935,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-path", required=True, help="Path to a *_decoded.pt file.")
    parser.add_argument("--output-dir", default=None, help="Visualization output directory.")
    parser.add_argument("--sample-key", default=None, help="Sample name used in output filenames.")
    parser.add_argument("--prediction-key", default="sampled", help="Prediction tensor key, or 'all'.")
    parser.add_argument("--gt-key", default="gt_tokenizer_recon")
    parser.add_argument("--gt-source", default="auto", choices=["auto", "raw", "decoded"])
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--segment-key", default=None)
    parser.add_argument("--lidar-frame-indices", default=None, help="Comma-separated raw LiDAR frame indices.")
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--pad-lidar-last", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--colormap", default="Spectral")
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=2)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--repeat-row", type=int, default=4)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--full-width", type=int, default=3600, help="Original Waymo TOP range-map width.")
    parser.add_argument("--crop-mode", default="center", choices=["center", "left", "none"])
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument("--near-buffer", type=float, default=0.1)
    parser.add_argument("--far-buffer", type=float, default=0.1)
    parser.add_argument("--no-clip-normalized", action="store_true")
    parser.add_argument("--vis-pcd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--display-frame", default="vehicle", choices=["lidar", "vehicle"])
    parser.add_argument("--camera-view", default="front_view", choices=["front_view", "top_down_view"])
    parser.add_argument("--pcd-width", type=int, default=1280)
    parser.add_argument("--pcd-height", type=int, default=720)
    parser.add_argument("--pcd-max-points", type=int, default=70000)
    parser.add_argument(
        "--pcd-renderer",
        default="auto",
        choices=["auto", "plotly", "raster"],
        help="`auto` matches the Waymo doc more closely by preferring the reference Plotly renderer.",
    )
    parser.add_argument(
        "--pcd-workers",
        type=int,
        default=1,
        help="Parallel point-cloud frame render workers. Useful for Plotly/Kaleido, which is CPU-bound.",
    )
    parser.add_argument(
        "--lidar-tokenizer-repo",
        default="/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen",
        help="Repo used for the Cosmos-Drive-Dreams Plotly point-cloud renderer.",
    )
    return parser.parse_args()


def infer_sample_key(decoded_path: Path) -> str:
    name = decoded_path.name
    suffix = "_decoded.pt"
    return name[: -len(suffix)] if name.endswith(suffix) else decoded_path.stem


def infer_segment_key(sample_key: str) -> tuple[str, int]:
    if "_" not in sample_key:
        return sample_key, 0
    segment_key, chunk = sample_key.rsplit("_", 1)
    try:
        return segment_key, int(chunk)
    except ValueError:
        return sample_key, 0


def natural_frame_key(name: str) -> tuple[Any, ...]:
    stem = name.removesuffix(".lidar_row.npz")
    parts: list[Any] = []
    for token in stem.replace("-", "_").split("_"):
        parts.append(int(token) if token.isdigit() else token)
    return tuple(parts)


def make_waymo_top_elevation_angles_128() -> np.ndarray:
    beam_deg = np.rad2deg(WAYMO_TOP_BEAM_INCLINATIONS_RAD)
    beam_asc = beam_deg[::-1]
    x_orig = np.linspace(0, 1, len(beam_asc))
    x_new = np.linspace(0, 1, 128)
    elevation_128 = np.interp(x_new, x_orig, beam_asc)
    return elevation_128[::-1].copy()


def crop_start_for_width(width: int, full_width: int, crop_mode: str) -> int:
    if crop_mode == "none" or width == full_width:
        return 0
    if crop_mode == "left":
        return 0
    if width > full_width:
        raise ValueError(f"Decoded width {width} is larger than full-width {full_width}.")
    return (full_width - width) // 2


def range_map_to_ray_directions_cropped(
    n_cols: int,
    sensor_elevation_angles: np.ndarray,
    *,
    full_width: int,
    crop_mode: str,
) -> tuple[np.ndarray, int]:
    start = crop_start_for_width(n_cols, full_width, crop_mode)
    full_azimuth = np.linspace(np.pi, -np.pi, full_width, endpoint=False)
    azimuth_angles = full_azimuth[start : start + n_cols]
    if len(azimuth_angles) != n_cols:
        raise ValueError(f"Could not take {n_cols} azimuth columns from full width {full_width}.")

    elevation_angles_rad = np.radians(sensor_elevation_angles)
    elevation_grid, azimuth_grid = np.meshgrid(elevation_angles_rad, azimuth_angles, indexing="ij")
    x = np.cos(elevation_grid) * np.cos(azimuth_grid)
    y = np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = np.sin(elevation_grid)
    return np.stack([x, y, z], axis=-1).astype(np.float32), start


def load_each_frame_from_tar_data(tar_data: tarfile.TarFile, frame_name: str, n_rows: int = 128, n_cols: int = 3600) -> np.ndarray:
    lidar_row = np.load(tar_data.extractfile(f"{frame_name}.lidar_row.npz"))["arr_0"]
    lidar_col = np.load(tar_data.extractfile(f"{frame_name}.lidar_col.npz"))["arr_0"]
    lidar_range = np.load(tar_data.extractfile(f"{frame_name}.lidar_range.npz"))["arr_0"]
    range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    range_map[lidar_row, lidar_col] = lidar_range.astype(np.float32)
    return range_map


def parse_lidar_frame_indices(args: argparse.Namespace, sample_key: str, n_frames: int) -> list[int]:
    if args.lidar_frame_indices:
        return [int(x) for x in args.lidar_frame_indices.split(",") if x.strip()]
    _, chunk_index = infer_segment_key(sample_key)
    start = chunk_index * args.lidar_chunk_stride_frames
    return list(range(start, start + n_frames))


def load_lidar_window(tar_path: Path, frame_indices: list[int], *, pad_last: bool) -> np.ndarray:
    with tarfile.open(tar_path, "r") as tar_handle:
        frame_names = sorted(
            (
                name.removesuffix(".lidar_row.npz")
                for name in tar_handle.getnames()
                if name.endswith(".lidar_row.npz")
            ),
            key=natural_frame_key,
        )
        if not frame_names:
            raise ValueError(f"No lidar_row frames found in {tar_path}")
        range_maps = []
        for frame_idx in frame_indices:
            if frame_idx >= len(frame_names):
                if not pad_last:
                    raise IndexError(
                        f"{tar_path.name}: requested frame {frame_idx}, but clip has {len(frame_names)} frames."
                    )
                frame_idx = len(frame_names) - 1
            range_maps.append(load_each_frame_from_tar_data(tar_handle, frame_names[frame_idx]))
    return np.stack(range_maps, axis=0)


def downsample_axis(data: np.ndarray, factor: int, axis: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if factor == 1:
        index_shape = data.shape[:3]
        return data, np.zeros(index_shape, dtype=np.int64)

    n_frames, height, width = data.shape[:3]
    if axis == 1:
        assert height % factor == 0, f"Height {height} must be divisible by row factor {factor}."
        if method == "every_n":
            return data[:, ::factor], np.zeros((n_frames, height // factor, width), dtype=np.int64)
        groups = data.reshape(n_frames, height // factor, factor, width)
        reduce_axis = 2
    else:
        assert width % factor == 0, f"Width {width} must be divisible by col factor {factor}."
        if method == "every_n":
            return data[:, :, ::factor], np.zeros((n_frames, height, width // factor), dtype=np.int64)
        groups = data.reshape(n_frames, height, width // factor, factor)
        reduce_axis = 3

    if method == "scatter_min":
        groups_clean = np.where(groups == 0, 1e3, groups)
        values = np.min(groups_clean, axis=reduce_axis)
        indices = np.argmin(groups_clean, axis=reduce_axis).astype(np.int64)
        values = np.where(values == 1e3, 0, values)
    elif method == "scatter_max":
        values = np.max(groups, axis=reduce_axis)
        indices = np.argmax(groups, axis=reduce_axis).astype(np.int64)
    else:
        raise ValueError(f"Unsupported downsample method: {method}")
    return values, indices


def apply_downsample_indices(data: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        if data.shape[1] == indices.shape[1]:
            return data
        factor = data.shape[1] // indices.shape[1]
        if np.all(indices == 0):
            return data[:, ::factor]
        n_frames, height, width = data.shape[:3]
        reshaped = data.reshape(n_frames, height // factor, factor, width, *data.shape[3:])
        batch_idx = np.arange(n_frames)[:, None, None]
        height_idx = np.arange(height // factor)[None, :, None]
        width_idx = np.arange(width)[None, None, :]
        return reshaped[batch_idx, height_idx, indices, width_idx]

    if data.shape[2] == indices.shape[2]:
        return data
    factor = data.shape[2] // indices.shape[2]
    if np.all(indices == 0):
        return data[:, :, ::factor]
    n_frames, height, width = data.shape[:3]
    reshaped = data.reshape(n_frames, height, width // factor, factor, *data.shape[3:])
    batch_idx = np.broadcast_to(np.arange(n_frames)[:, None, None], indices.shape)
    height_idx = np.broadcast_to(np.arange(height)[None, :, None], indices.shape)
    width_idx = np.broadcast_to(np.arange(width // factor)[None, None, :], indices.shape)
    return reshaped[batch_idx, height_idx, width_idx, indices]


def downsample_range_with_extras(
    range_maps: np.ndarray,
    extras: list[np.ndarray],
    *,
    row_factor: int,
    col_factor: int,
    method: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    current, row_indices = downsample_axis(range_maps, row_factor, axis=1, method=method)
    extras = [apply_downsample_indices(extra, row_indices, axis=1) for extra in extras]
    current, col_indices = downsample_axis(current, col_factor, axis=2, method=method)
    extras = [apply_downsample_indices(extra, col_indices, axis=2) for extra in extras]
    return current, extras


def load_raw_gt_for_decoded_crop(
    args: argparse.Namespace,
    *,
    sample_key: str,
    n_frames: int,
    decoded_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    segment_key = args.segment_key or infer_segment_key(sample_key)[0]
    tar_path = Path(args.raw_lidar_root) / args.split / "lidar" / f"{segment_key}.tar"
    frame_indices = parse_lidar_frame_indices(args, sample_key, n_frames)
    raw_range = load_lidar_window(tar_path, frame_indices, pad_last=args.pad_lidar_last)
    raw_valid = valid_range_mask(raw_range, args.min_range, args.max_range, args.near_buffer, args.far_buffer)

    elevations = make_waymo_top_elevation_angles_128()
    raw_rays, _ = range_map_to_ray_directions_cropped(
        raw_range.shape[-1],
        elevations,
        full_width=raw_range.shape[-1],
        crop_mode="none",
    )
    raw_rays = np.broadcast_to(raw_rays[None], raw_range.shape + (3,))
    down_range, [down_rays, down_valid] = downsample_range_with_extras(
        raw_range,
        [raw_rays, raw_valid],
        row_factor=args.downsample_factor_row,
        col_factor=args.downsample_factor_col,
        method=args.downsample_method,
    )
    down_range = np.clip(down_range, args.min_range, args.max_range)
    down_width = down_range.shape[-1]
    crop_start = crop_start_for_width(decoded_width, down_width, args.crop_mode)
    crop_end = crop_start + decoded_width
    meta = {
        "raw_lidar_path": str(tar_path),
        "segment_key": segment_key,
        "lidar_frame_indices": frame_indices,
        "downsampled_width": int(down_width),
        "crop_start_downsample_col": int(crop_start),
        "crop_end_downsample_col": int(crop_end),
        "crop_start_raw_col_approx": int(crop_start * args.downsample_factor_col),
        "crop_end_raw_col_approx": int(crop_end * args.downsample_factor_col),
    }
    return (
        down_range[:, :, crop_start:crop_end].astype(np.float32, copy=False),
        down_valid[:, :, crop_start:crop_end].astype(bool, copy=False),
        down_rays[:, :, crop_start:crop_end].astype(np.float32, copy=False),
        meta,
    )


def transform_points_to_vehicle_frame(points: np.ndarray) -> np.ndarray:
    rot = WAYMO_TOP_LIDAR_EXTRINSIC[:3, :3]
    trans = WAYMO_TOP_LIDAR_EXTRINSIC[:3, 3]
    return (points @ rot.T) + trans


def decoded_tensor_to_range(
    tensor: torch.Tensor,
    *,
    repeat_row: int,
    repeat_col: int,
    min_range: float,
    max_range: float,
    min_value: float,
    max_value: float,
    clip_normalized: bool,
) -> np.ndarray:
    decoded = tensor.detach().cpu().float()
    if decoded.ndim == 5:
        if decoded.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 decoded tensor, got {tuple(decoded.shape)}.")
        decoded = decoded[0]

    if decoded.ndim != 4:
        raise ValueError(f"Expected decoded tensor rank 4 or 5, got {tuple(decoded.shape)}.")

    if decoded.shape[0] in (1, 3):
        decoded = decoded.permute(1, 0, 2, 3)
    elif decoded.shape[-1] in (1, 3):
        decoded = decoded.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Could not infer channel dimension for tensor shape {tuple(decoded.shape)}.")

    row_offset = repeat_row // 2
    col_offset = repeat_col // 2
    decoded = decoded[:, :, row_offset::repeat_row, col_offset::repeat_col].mean(dim=1)
    if clip_normalized:
        decoded = decoded.clamp(min_value, max_value)

    if min_value == -1:
        range_m = (decoded + 1.0) / 2.0 * (max_range - min_range) + min_range
    else:
        range_m = decoded * (max_range - min_range) + min_range
    return range_m.numpy().astype(np.float32)


def valid_range_mask(range_m: np.ndarray, min_range: float, max_range: float, near_buffer: float, far_buffer: float) -> np.ndarray:
    return (range_m > min_range + near_buffer) & (range_m < max_range - far_buffer)


def colorcode_depth_maps(depth_maps: np.ndarray, cmap_name: str) -> np.ndarray:
    depth = torch.from_numpy(depth_maps).float()
    mask = depth <= 0
    mid = depth.shape[0] // 2
    valid_mid = depth[mid][~mask[mid]]
    if valid_mid.numel() > 0:
        near = valid_mid.quantile(0.01).log()
        far = valid_mid.quantile(0.99).log()
    else:
        valid_all = depth[~mask]
        near = valid_all.quantile(0.01).log() if valid_all.numel() > 0 else torch.tensor(0.0)
        far = valid_all.quantile(0.99).log() if valid_all.numel() > 0 else torch.tensor(1.0)
    if torch.isclose(far, near):
        far = near + 1e-3

    safe = depth.clone()
    safe[mask] = 1.0
    normalized = 1.0 - (safe.log() - near) / (far - near)
    normalized = normalized.clamp(0, 1).numpy()
    rgb = matplotlib.colormaps.get_cmap(cmap_name)(normalized)[..., :3]
    rgb[mask.numpy()] = np.array([0.82, 0.82, 0.82], dtype=np.float32)
    return (rgb * 255 + 0.5).astype(np.uint8)


def save_range_map_video(rows: list[np.ndarray], output_path: Path, *, cmap_name: str, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate(rows, axis=1)
    frames = colorcode_depth_maps(stacked, cmap_name)
    media.write_video(str(output_path), frames, fps=fps)


def save_histogram(diff: np.ndarray, metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(diff, bins=100, range=(0, 10), color="#2b7bba", alpha=0.88)
    plt.xlabel("Range Difference (m)")
    plt.ylabel("Frequency")
    plt.title(
        "Range Difference Histogram\n"
        f"RMSE: {metrics['rmse_m']:.2f} m, MAE: {metrics['mae_m']:.2f} m, Rel error: {metrics['rel_error']:.2f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    stride = int(math.ceil(points.shape[0] / max_points))
    return points[::stride]


def rasterize_points(
    points: np.ndarray,
    *,
    base_color: tuple[float, float, float],
    camera_view: str,
    width: int,
    height: int,
    max_points: int,
) -> np.ndarray:
    points = deterministic_subsample(points, max_points)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if points.size == 0:
        return canvas

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    if camera_view == "front_view":
        u = y
        v = z
        u_min, u_max = -55.0, 55.0
        v_min, v_max = -4.0, 8.0
        shade_source = np.clip(x, 0, 100)
    else:
        u = y
        v = x
        u_min, u_max = -55.0, 55.0
        v_min, v_max = -20.0, 100.0
        shade_source = np.clip(np.abs(z), 0, 8)

    keep = (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max)
    if not np.any(keep):
        return canvas
    u = u[keep]
    v = v[keep]
    shade_source = shade_source[keep]
    px = ((u - u_min) / (u_max - u_min) * (width - 1)).astype(np.int64)
    py = (height - 1 - (v - v_min) / (v_max - v_min) * (height - 1)).astype(np.int64)
    shade = 0.35 + 0.65 * (1.0 - np.clip(shade_source / 100.0, 0, 1))
    color = (np.array(base_color, dtype=np.float32)[None, :] * shade[:, None] * 255.0).astype(np.uint8)
    for channel in range(3):
        np.maximum.at(canvas[:, :, channel], (py, px), color[:, channel])
    return canvas


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((8, 8, 8 + 14 * len(label), 34), fill=(0, 0, 0))
    draw.text((14, 13), label, fill=(255, 255, 255))
    return np.asarray(pil)


def local_plotly_visualize_point_cloud(
    point_cloud: np.ndarray,
    colors: np.ndarray,
    *,
    point_size: float,
    camera_position: dict[str, Any],
    width: int,
    height: int,
    range: float = 80,
    opacity: float = 1.0,
    bgcolor: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    import io

    import plotly.graph_objects as go

    rgb = np.clip(colors, 0.0, 1.0) * 255.0
    color_strings = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]
    trace = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=color_strings, opacity=opacity),
    )
    color_str = f"rgba({bgcolor[0]},{bgcolor[1]},{bgcolor[2]})"
    axis_cfg = dict(
        range=[-range, range],
        autorange=False,
        showbackground=False,
        showticklabels=False,
        zeroline=False,
        visible=False,
        showgrid=False,
    )
    fig = go.Figure(
        data=[trace],
        layout=go.Layout(
            scene=dict(xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg, aspectmode="cube", camera=camera_position),
            paper_bgcolor=color_str,
            plot_bgcolor=color_str,
            margin=dict(l=0, r=0, b=0, t=0),
        ),
    )
    buf = io.BytesIO(fig.to_image(format="png", width=width, height=height))
    return np.asarray(Image.open(buf))[:, :, :3]


@lru_cache(maxsize=8)
def load_reference_plotly_renderer(
    lidar_tokenizer_repo: str,
    *,
    camera_view: str,
    width: int,
    height: int,
):
    camera_positions = {
        "front_view": {"eye": {"x": -0.3, "y": 0, "z": 0.2}, "center": {"x": 0.1, "y": 0, "z": 0}},
        "top_down_view": {"eye": {"x": 0, "y": -0.05, "z": 0.5}, "center": {"x": 0, "y": -0.05, "z": 0}},
    }
    local_kwargs = {
        "point_size": 0.3,
        "camera_position": camera_positions[camera_view],
        "width": width,
        "height": height,
        "range": 100,
        "bgcolor": (0.0, 0.0, 0.0),
    }

    repo_path = str(Path(lidar_tokenizer_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        from cosmos_predict1.utils.visualize.point_cloud import (  # type: ignore
            CAMERA_VIEWS as POINT_CLOUD_CAMERA_VIEWS,
            VIZ_KWARGS,
            visualize_point_cloud,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "open3d":
            raise
        return local_plotly_visualize_point_cloud, local_kwargs

    camera_key = {
        "front_view": "front_view_1",
        "top_down_view": "top_down_view_1",
    }[camera_view]
    render_kwargs = dict(VIZ_KWARGS)
    render_kwargs["camera_position"] = POINT_CLOUD_CAMERA_VIEWS[camera_key]
    render_kwargs["width"] = width
    render_kwargs["height"] = height
    return visualize_point_cloud, render_kwargs


def render_points_plotly(
    points: np.ndarray,
    *,
    base_color: tuple[float, float, float],
    width: int,
    height: int,
    max_points: int,
    visualize_point_cloud,
    render_kwargs: dict[str, Any],
) -> np.ndarray:
    points = deterministic_subsample(points, max_points).astype(np.float32, copy=False)
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    colors = np.broadcast_to(np.array([base_color], dtype=np.float32), (points.shape[0], 3))
    return visualize_point_cloud(points, colors, **render_kwargs)


def render_point_cloud_frame_worker(payload: dict[str, Any]) -> tuple[int, np.ndarray]:
    frame_idx = payload["frame_idx"]
    gt = payload["gt"]
    pred = payload["pred"]
    gt_mask = payload.get("gt_valid_mask", payload["valid_mask"])
    pred_mask = payload.get("pred_valid_mask", gt_mask)
    frame_rays = payload["ray_directions"]
    gt_points = frame_rays[gt_mask] * gt[gt_mask, None]
    pred_points = frame_rays[pred_mask] * pred[pred_mask, None]
    if payload["display_frame"] == "vehicle":
        gt_points = transform_points_to_vehicle_frame(gt_points)
        pred_points = transform_points_to_vehicle_frame(pred_points)

    actual_renderer = payload["actual_renderer"]
    if actual_renderer == "plotly":
        try:
            plotly_renderer, plotly_kwargs = load_reference_plotly_renderer(
                payload["lidar_tokenizer_repo"],
                camera_view=payload["camera_view"],
                width=payload["width"],
                height=payload["height"],
            )
            gt_img = render_points_plotly(
                gt_points,
                base_color=(1.0, 0.706, 0.0),
                width=payload["width"],
                height=payload["height"],
                max_points=payload["max_points"],
                visualize_point_cloud=plotly_renderer,
                render_kwargs=plotly_kwargs,
            )
            pred_img = render_points_plotly(
                pred_points,
                base_color=(0.0, 0.651, 0.929),
                width=payload["width"],
                height=payload["height"],
                max_points=payload["max_points"],
                visualize_point_cloud=plotly_renderer,
                render_kwargs=plotly_kwargs,
            )
        except Exception as exc:
            if payload["requested_renderer"] == "plotly":
                raise
            print(f"Plotly render failed at frame {frame_idx}, falling back to raster: {exc}", flush=True)
            actual_renderer = "raster"

    if actual_renderer == "raster":
        gt_img = rasterize_points(
            gt_points,
            base_color=(1.0, 0.706, 0.0),
            camera_view=payload["camera_view"],
            width=payload["width"],
            height=payload["height"],
            max_points=payload["max_points"],
        )
        pred_img = rasterize_points(
            pred_points,
            base_color=(0.0, 0.651, 0.929),
            camera_view=payload["camera_view"],
            width=payload["width"],
            height=payload["height"],
            max_points=payload["max_points"],
        )

    gt_img = label_image(gt_img, payload["gt_label"])
    pred_img = label_image(pred_img, payload["prediction_label"])
    separator = np.zeros((4, payload["width"], 3), dtype=np.uint8)
    return frame_idx, np.concatenate([gt_img, separator, pred_img], axis=0)


def save_point_cloud_video(
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    *,
    gt_label: str,
    prediction_label: str,
    ray_directions: np.ndarray,
    display_frame: str,
    camera_view: str,
    fps: int,
    width: int,
    height: int,
    max_points: int,
    renderer: str,
    lidar_tokenizer_repo: str,
    workers: int,
    pred_valid_mask: np.ndarray | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_renderer = "raster"
    plotly_renderer = None
    plotly_kwargs: dict[str, Any] | None = None
    if renderer in {"auto", "plotly"}:
        try:
            plotly_renderer, plotly_kwargs = load_reference_plotly_renderer(
                lidar_tokenizer_repo,
                camera_view=camera_view,
                width=width,
                height=height,
            )
            actual_renderer = "plotly"
        except Exception as exc:
            if renderer == "plotly":
                raise
            print(f"Plotly point-cloud renderer unavailable, falling back to raster: {exc}", flush=True)

    if pred_valid_mask is None:
        pred_valid_mask = valid_mask
    if pred_valid_mask.shape != valid_mask.shape:
        raise ValueError(f"pred_valid_mask shape {pred_valid_mask.shape} does not match valid_mask {valid_mask.shape}.")

    workers = max(1, int(workers))
    payloads: list[dict[str, Any]] = []
    for frame_idx in range(gt.shape[0]):
        gt_mask = valid_mask[frame_idx]
        pred_mask = pred_valid_mask[frame_idx]
        frame_rays = ray_directions[frame_idx] if ray_directions.ndim == 4 else ray_directions
        payloads.append(
            {
                "frame_idx": frame_idx,
                "gt": gt[frame_idx],
                "pred": pred[frame_idx],
                "valid_mask": gt_mask,
                "gt_valid_mask": gt_mask,
                "pred_valid_mask": pred_mask,
                "ray_directions": frame_rays,
                "display_frame": display_frame,
                "camera_view": camera_view,
                "width": width,
                "height": height,
                "max_points": max_points,
                "actual_renderer": actual_renderer,
                "requested_renderer": renderer,
                "lidar_tokenizer_repo": lidar_tokenizer_repo,
                "gt_label": gt_label,
                "prediction_label": prediction_label,
            }
        )

    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            rendered = list(executor.map(render_point_cloud_frame_worker, payloads))
    else:
        rendered = [render_point_cloud_frame_worker(payload) for payload in payloads]
    frames = [frame for _, frame in sorted(rendered, key=lambda item: item[0])]
    media.write_video(str(output_path), frames, fps=fps)
    return actual_renderer


def compute_metrics(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    diff_signed = pred[valid_mask] - gt[valid_mask]
    diff = np.abs(diff_signed)
    if diff.size == 0:
        raise ValueError("No valid pixels found for metrics.")
    metrics = {
        "rmse_m": float(np.sqrt(np.mean(diff_signed**2))),
        "mae_m": float(np.mean(diff)),
        "rel_error": float(np.mean(diff / (gt[valid_mask] + 1e-6))),
        "valid_pixels": int(diff.size),
        "valid_ratio": float(valid_mask.mean()),
        "gt_mean_m": float(gt[valid_mask].mean()),
        "pred_mean_m": float(pred[valid_mask].mean()),
    }
    return metrics, diff


def process_prediction(
    payload: dict[str, Any],
    prediction_key: str,
    *,
    args: argparse.Namespace,
    sample_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    decoded_gt_range = decoded_tensor_to_range(
        payload[args.gt_key],
        repeat_row=args.repeat_row,
        repeat_col=args.repeat_col,
        min_range=args.min_range,
        max_range=args.max_range,
        min_value=args.min_value,
        max_value=args.max_value,
        clip_normalized=not args.no_clip_normalized,
    )
    pred_range = decoded_tensor_to_range(
        payload[prediction_key],
        repeat_row=args.repeat_row,
        repeat_col=args.repeat_col,
        min_range=args.min_range,
        max_range=args.max_range,
        min_value=args.min_value,
        max_value=args.max_value,
        clip_normalized=not args.no_clip_normalized,
    )
    if args.max_frames > 0:
        decoded_gt_range = decoded_gt_range[: args.max_frames]
        pred_range = pred_range[: args.max_frames]

    gt_source = "decoded"
    raw_meta: dict[str, Any] = {}
    ray_directions = None
    try:
        if args.gt_source in {"auto", "raw"}:
            raw_gt_range, raw_valid_mask, raw_rays, raw_meta = load_raw_gt_for_decoded_crop(
                args,
                sample_key=sample_key,
                n_frames=decoded_gt_range.shape[0],
                decoded_width=decoded_gt_range.shape[-1],
            )
            gt_source = "raw"
            eval_gt_range = raw_gt_range
            valid_mask = raw_valid_mask
            ray_directions = raw_rays
        else:
            eval_gt_range = decoded_gt_range
            valid_mask = valid_range_mask(
                decoded_gt_range,
                args.min_range,
                args.max_range,
                args.near_buffer,
                args.far_buffer,
            )
    except Exception as exc:
        if args.gt_source == "raw":
            raise
        print(f"Raw LiDAR GT unavailable, falling back to decoded GT: {exc}", flush=True)
        eval_gt_range = decoded_gt_range
        valid_mask = valid_range_mask(
            decoded_gt_range,
            args.min_range,
            args.max_range,
            args.near_buffer,
            args.far_buffer,
        )

    raw_or_eval_gt_for_vis = np.where(valid_mask, eval_gt_range, 0)
    decoded_gt_for_vis = np.where(valid_mask, decoded_gt_range, 0)
    pred_for_vis = np.where(valid_mask, pred_range, 0)
    metrics, diff = compute_metrics(eval_gt_range, pred_range, valid_mask)
    decoded_gt_metrics, _ = compute_metrics(eval_gt_range, decoded_gt_range, valid_mask)

    name = f"{sample_key}_{prediction_key}"
    range_video_path = output_dir / "range_map_video" / f"{name}.mp4"
    hist_path = output_dir / "histogram" / f"{name}.png"
    save_range_map_video(
        [raw_or_eval_gt_for_vis, decoded_gt_for_vis, pred_for_vis],
        range_video_path,
        cmap_name=args.colormap,
        fps=args.fps,
    )
    save_histogram(diff, metrics, hist_path)

    if ray_directions is None:
        elevation_angles = make_waymo_top_elevation_angles_128()
        effective_full_width = max(1, args.full_width // max(1, args.downsample_factor_col))
        ray_directions_2d, crop_start = range_map_to_ray_directions_cropped(
            decoded_gt_range.shape[-1],
            elevation_angles,
            full_width=effective_full_width,
            crop_mode=args.crop_mode,
        )
        ray_directions = np.broadcast_to(ray_directions_2d[None], decoded_gt_range.shape + (3,))
        raw_meta = {
            "downsampled_width": int(effective_full_width),
            "crop_start_downsample_col": int(crop_start),
            "crop_end_downsample_col": int(crop_start + decoded_gt_range.shape[-1]),
            "crop_start_raw_col_approx": int(crop_start * args.downsample_factor_col),
            "crop_end_raw_col_approx": int((crop_start + decoded_gt_range.shape[-1]) * args.downsample_factor_col),
        }
    pcd_path = None
    pcd_renderer = None
    if args.vis_pcd:
        pcd_path = output_dir / "point_cloud" / f"{name}.mp4"
        pcd_renderer = save_point_cloud_video(
            raw_or_eval_gt_for_vis,
            pred_for_vis,
            valid_mask,
            pcd_path,
            gt_label=f"GT: {gt_source}",
            prediction_label=f"prediction: {prediction_key}",
            ray_directions=ray_directions,
            display_frame=args.display_frame,
            camera_view=args.camera_view,
            fps=args.fps,
            width=args.pcd_width,
            height=args.pcd_height,
            max_points=args.pcd_max_points,
            renderer=args.pcd_renderer,
            lidar_tokenizer_repo=args.lidar_tokenizer_repo,
            workers=args.pcd_workers,
        )

    return {
        "sample_key": sample_key,
        "prediction_key": prediction_key,
        "gt_key": args.gt_key,
        "gt_source": gt_source,
        "range_map_rows": ["raw_or_eval_gt", "decode_gt_latent", prediction_key],
        "shape_frames_rows_cols": list(decoded_gt_range.shape),
        "full_width": args.full_width,
        "downsample_factor_row": args.downsample_factor_row,
        "downsample_factor_col": args.downsample_factor_col,
        "downsample_method": args.downsample_method,
        "crop_mode": args.crop_mode,
        **raw_meta,
        "repeat_row": args.repeat_row,
        "repeat_col": args.repeat_col,
        "range_map_video": str(range_video_path),
        "histogram": str(hist_path),
        "point_cloud": str(pcd_path) if pcd_path is not None else None,
        "point_cloud_renderer": pcd_renderer,
        "point_cloud_workers": args.pcd_workers,
        "camera_view": args.camera_view,
        "display_frame": args.display_frame,
        "metrics": metrics,
        "decode_gt_vs_eval_gt_metrics": decoded_gt_metrics,
    }


def main() -> None:
    args = parse_args()
    decoded_path = Path(args.decoded_path)
    output_dir = Path(args.output_dir) if args.output_dir else decoded_path.parent / "waymo_doc_vis_4step"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_key = args.sample_key or infer_sample_key(decoded_path)

    payload = torch.load(decoded_path, map_location="cpu")
    if args.gt_key not in payload:
        raise KeyError(f"Missing gt key {args.gt_key!r}; available keys: {sorted(payload)}")

    if args.prediction_key == "all":
        prediction_keys = [key for key in payload.keys() if key != args.gt_key and torch.is_tensor(payload[key])]
    else:
        prediction_keys = [args.prediction_key]
    for key in prediction_keys:
        if key not in payload:
            raise KeyError(f"Missing prediction key {key!r}; available keys: {sorted(payload)}")

    results = [
        process_prediction(payload, key, args=args, sample_key=sample_key, output_dir=output_dir)
        for key in prediction_keys
    ]
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
