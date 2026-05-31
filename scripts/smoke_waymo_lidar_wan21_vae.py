# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test Wan2.1 VAE encode/decode on Waymo LiDAR range maps.

The default path reads converted Waymo LiDAR tokenizer tar files, preprocesses
range maps into a 3-channel Wan-compatible video tensor, then runs the frozen
Wan2.1 VAE through encode and decode. A raw lidar tar can be passed explicitly
to rebuild range maps in memory using the projection logic from the
Cosmos-Drive-Dreams conversion script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SEGMENT_KEY = "10203656353524179475_7625_000_7645_000"
DEFAULT_LIDAR_UTILS_REPO = "/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen"
DEFAULT_MPLCONFIGDIR = "/tmp/matplotlib"
DEFAULT_RAW_LIDAR_ROOT = "/data2/rds_hq_waymo"
WAN_HF_REPO_CACHE = "models--nvidia--Cosmos-Predict2.5-2B"
WAN_HF_REVISION = "f176dc95b4a70f53ce01c4b302851595e7322b00"
WAN_HF_FILENAME = "tokenizer.pth"
POLYPHASE3_LAYOUTS = {"polyphase3_repeat", "polyphase3_interp"}


# Waymo TOP LiDAR calibration, matching
# Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py.
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

WAYMO_TOP_LIDAR_EXTRINSIC = np.array(
    [
        [-8.4777248e-01, -5.3035414e-01, -2.5136571e-03, 1.4299999e00],
        [5.3035545e-01, -8.4777534e-01, 1.8014426e-04, 0.0000000e00],
        [-2.2265569e-03, -1.1804104e-03, 9.9999684e-01, 2.1840000e00],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--segment-key", default=DEFAULT_SEGMENT_KEY)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=29)
    parser.add_argument("--pad-last", action="store_true")
    parser.add_argument(
        "--preprocess-mode",
        default="tokenizer_128x3600",
        choices=["tokenizer_128x3600", "waymo_top_64x2650"],
        help="`tokenizer_128x3600` matches the LiDAR-tokenizer baseline. "
        "`waymo_top_64x2650` projects raw Waymo TOP points to the native 64-beam/2650-column range image.",
    )
    parser.add_argument("--raw-lidar-root", default=DEFAULT_RAW_LIDAR_ROOT)
    parser.add_argument("--converted-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--converted-tar", default=None)
    parser.add_argument("--raw-tar", default=None, help="Optional raw lidar tar.")
    parser.add_argument("--range-map-npz", default=None, help="Optional precomputed range map npz with `range_maps`.")
    parser.add_argument("--range-map-return", default="return1", choices=["return1", "return2"])
    parser.add_argument("--range-map-target-width", type=int, default=None)
    parser.add_argument("--range-map-eval-source-grid", action="store_true")
    parser.add_argument(
        "--range-map-layout",
        default="plain",
        choices=[
            "plain",
            "fold_sparse",
            "fold_duplicate",
            "fold_visual_copy",
            "fold_visual_interp",
            "polyphase3_repeat",
            "polyphase3_interp",
        ],
        help="Optional lossless fold from 64x2650 official range maps into a 704x1280 Wan input grid.",
    )
    parser.add_argument("--fold-target-height", type=int, default=704)
    parser.add_argument("--fold-target-width", type=int, default=1280)
    parser.add_argument("--fold-row-order", default="beam_local", choices=["beam_local", "slot_major"])
    parser.add_argument("--fold-slot-strategy", default="contiguous", choices=["contiguous", "interleave"])
    parser.add_argument("--fold-serpentine", dest="fold_serpentine", action="store_true", default=True)
    parser.add_argument("--no-fold-serpentine", dest="fold_serpentine", action="store_false")
    parser.add_argument("--fold-duplicate-reducer", default="median", choices=["mean", "median"])
    parser.add_argument("--polyphase-roll", type=int, default=0)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=2650)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--wan-spatial-align", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lidar-utils-repo", default=DEFAULT_LIDAR_UTILS_REPO)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--allow-default-wan-uri", action="store_true")
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--downsample-factor-row", type=int, default=None)
    parser.add_argument("--downsample-factor-col", type=int, default=None)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--repeat-row", type=int, default=None)
    parser.add_argument("--repeat-col", type=int, default=None)
    parser.add_argument("--input-channel-mode", default="repeat_depth", choices=["repeat_depth", "concat_inv_depth"])
    parser.add_argument("--decode-channel-mode", default=None, choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-tensors", action="store_true")
    parser.add_argument("--vis-pcd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pcd-renderer", default="plotly", choices=["auto", "plotly", "raster"])
    parser.add_argument("--pcd-display-frame", default="vehicle", choices=["lidar", "vehicle"])
    parser.add_argument("--pcd-camera-view", default="front_view", choices=["front_view", "top_down_view"])
    parser.add_argument("--pcd-width", type=int, default=1280)
    parser.add_argument("--pcd-height", type=int, default=720)
    parser.add_argument("--pcd-max-points", type=int, default=70000)
    parser.add_argument("--pcd-workers", type=int, default=1)
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.downsample_factor_row is None:
        args.downsample_factor_row = 1
    if args.downsample_factor_col is None:
        args.downsample_factor_col = 1 if args.range_map_layout != "plain" else (2 if args.preprocess_mode == "tokenizer_128x3600" else 1)
    if args.repeat_row is None:
        args.repeat_row = 1 if args.range_map_layout != "plain" else (4 if args.preprocess_mode == "tokenizer_128x3600" else 16)
    if args.repeat_col is None:
        args.repeat_col = 1
    if args.decode_channel_mode is None:
        args.decode_channel_mode = "concat_fuse" if args.input_channel_mode == "concat_inv_depth" else "mean"


def natural_key(string_: str) -> list[Any]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def make_waymo_top_elevation_angles(n_rows: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    beam_deg = np.rad2deg(WAYMO_TOP_BEAM_INCLINATIONS_RAD)
    if n_rows == len(beam_deg):
        return beam_deg.copy()
    beam_asc = beam_deg[::-1]
    x_orig = np.linspace(0, 1, len(beam_asc))
    x_new = np.linspace(0, 1, n_rows)
    elevation = np.interp(x_new, x_orig, beam_asc)
    return elevation[::-1].copy()


def nearest_elevation_rows(elevation_deg: np.ndarray, elevation_angles_deg: np.ndarray) -> np.ndarray:
    elevation_asc = elevation_angles_deg[::-1]
    right = np.searchsorted(elevation_asc, elevation_deg, side="left")
    left = np.clip(right - 1, 0, len(elevation_asc) - 1)
    right = np.clip(right, 0, len(elevation_asc) - 1)
    choose_right = np.abs(elevation_asc[right] - elevation_deg) < np.abs(elevation_asc[left] - elevation_deg)
    row_asc = np.where(choose_right, right, left)
    return (len(elevation_angles_deg) - 1 - row_asc).astype(np.int64)


def points_to_range_map(
    xyz_lidar: np.ndarray,
    n_rows: int,
    n_cols: int,
    elevation_angles_deg: np.ndarray,
    *,
    max_range: float,
) -> np.ndarray:
    x, y, z = xyz_lidar[:, 0], xyz_lidar[:, 1], xyz_lidar[:, 2]
    range_values = np.sqrt(x**2 + y**2 + z**2)
    valid_range = (range_values > 0) & (range_values < max_range)

    azimuth = -np.arctan2(y, x) + np.pi
    col_idx = ((azimuth / (2 * np.pi)) * n_cols).astype(np.int64) % n_cols

    elevation = np.arcsin(z / np.clip(range_values, 1e-6, None))
    elevation_deg = np.rad2deg(elevation)
    valid_elev = (elevation_deg >= elevation_angles_deg.min() - 0.5) & (
        elevation_deg <= elevation_angles_deg.max() + 0.5
    )
    valid = valid_range & valid_elev

    row_v = nearest_elevation_rows(elevation_deg[valid], elevation_angles_deg)
    col_v = col_idx[valid]
    range_v = range_values[valid]

    order = np.argsort(-range_v)
    range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    range_map[row_v[order], col_v[order]] = range_v[order].astype(np.float32)
    return range_map


def select_frame_names(frame_names: list[str], frame_start: int, num_frames: int, *, pad_last: bool) -> list[str]:
    if frame_start < 0:
        raise ValueError(f"frame_start must be non-negative, got {frame_start}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    selected: list[str] = []
    for frame_idx in range(frame_start, frame_start + num_frames):
        if frame_idx >= len(frame_names):
            if not pad_last:
                raise IndexError(f"Requested frame {frame_idx}, but tar has {len(frame_names)} frames")
            frame_idx = len(frame_names) - 1
        selected.append(frame_names[frame_idx])
    return selected


def load_converted_range_maps(
    tar_path: Path,
    *,
    frame_start: int,
    num_frames: int,
    pad_last: bool,
    n_rows: int = 128,
    n_cols: int = 3600,
) -> tuple[np.ndarray, list[str]]:
    with tarfile.open(tar_path, "r") as tar_handle:
        frame_names = sorted(
            name.removesuffix(".lidar_row.npz") for name in tar_handle.getnames() if name.endswith(".lidar_row.npz")
        )
        if not frame_names:
            raise ValueError(f"No converted lidar_row frames found in {tar_path}")
        selected_names = select_frame_names(frame_names, frame_start, num_frames, pad_last=pad_last)
        range_maps = []
        for frame_name in selected_names:
            lidar_row = np.load(tar_handle.extractfile(f"{frame_name}.lidar_row.npz"))["arr_0"]
            lidar_col = np.load(tar_handle.extractfile(f"{frame_name}.lidar_col.npz"))["arr_0"]
            lidar_range = np.load(tar_handle.extractfile(f"{frame_name}.lidar_range.npz"))["arr_0"]
            range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
            range_map[lidar_row, lidar_col] = lidar_range.astype(np.float32)
            range_maps.append(range_map)
    return np.stack(range_maps, axis=0), selected_names


def load_raw_range_maps(
    tar_path: Path,
    *,
    frame_start: int,
    num_frames: int,
    pad_last: bool,
    n_rows: int = 128,
    n_cols: int = 3600,
    max_projection_range: float = 105.0,
) -> tuple[np.ndarray, list[str]]:
    elevation_angles_deg = make_waymo_top_elevation_angles(n_rows)
    vehicle_to_lidar = np.linalg.inv(WAYMO_TOP_LIDAR_EXTRINSIC)
    rot = vehicle_to_lidar[:3, :3]
    trans = vehicle_to_lidar[:3, 3:4]

    with tarfile.open(tar_path, "r") as tar_handle:
        frame_names = sorted(
            (name.removesuffix(".lidar_raw.npz") for name in tar_handle.getnames() if name.endswith(".lidar_raw.npz")),
            key=natural_key,
        )
        if not frame_names:
            raise ValueError(f"No raw lidar frames found in {tar_path}")
        selected_names = select_frame_names(frame_names, frame_start, num_frames, pad_last=pad_last)
        range_maps = []
        for frame_name in selected_names:
            file_obj = tar_handle.extractfile(f"{frame_name}.lidar_raw.npz")
            if file_obj is None:
                raise FileNotFoundError(f"{frame_name}.lidar_raw.npz not found in {tar_path}")
            with file_obj:
                lidar_raw = np.load(file_obj)
                xyz_vehicle = lidar_raw["xyz"].astype(np.float32)
            xyz_lidar = (rot @ xyz_vehicle.T + trans).T.astype(np.float32)
            range_maps.append(
                points_to_range_map(
                    xyz_lidar,
                    n_rows,
                    n_cols,
                    elevation_angles_deg,
                    max_range=max_projection_range,
                )
            )
    return np.stack(range_maps, axis=0), selected_names


def rebin_range_maps_width_min(range_maps: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")
    if range_maps.shape[-1] == target_width:
        return range_maps.astype(np.float32, copy=False)
    source_width = range_maps.shape[-1]
    source_cols = np.arange(source_width, dtype=np.float64)
    target_cols = np.floor((source_cols + 0.5) * target_width / source_width).astype(np.int64)
    target_cols = np.clip(target_cols, 0, target_width - 1)
    output = np.zeros((*range_maps.shape[:-1], target_width), dtype=np.float32)
    flat_input = range_maps.reshape(-1, source_width)
    flat_output = output.reshape(-1, target_width)
    for row_idx, values in enumerate(flat_input):
        accum = np.full(target_width, np.inf, dtype=np.float32)
        valid = values > 0
        if valid.any():
            np.minimum.at(accum, target_cols[valid], values[valid].astype(np.float32, copy=False))
        flat_output[row_idx] = np.where(np.isfinite(accum), accum, 0.0)
    return output


def expand_rebinned_range_maps_width(rebinned: np.ndarray, source_width: int) -> np.ndarray:
    target_width = rebinned.shape[-1]
    source_cols = np.arange(source_width, dtype=np.float64)
    target_cols = np.floor((source_cols + 0.5) * target_width / source_width).astype(np.int64)
    target_cols = np.clip(target_cols, 0, target_width - 1)
    return rebinned[..., target_cols].astype(np.float32, copy=False)


def load_npz_range_maps(path: Path, *, range_return: str) -> tuple[np.ndarray, list[str]]:
    payload = np.load(path, allow_pickle=True)
    if "range_maps" in payload:
        range_maps = payload["range_maps"].astype(np.float32)
    else:
        key = f"range_image_{range_return}"
        range_image = payload[key].astype(np.float32)
        range_maps = range_image[..., 0]
    if range_maps.ndim == 2:
        range_maps = range_maps[None]
    if "frame_names" in payload:
        frame_names = [str(item) for item in payload["frame_names"].tolist()]
    else:
        frame_names = [f"frame_{idx:06d}" for idx in range(range_maps.shape[0])]
    return range_maps, frame_names


def fold_chunk_groups(slots_per_row: int, num_chunks: int, *, strategy: str) -> list[list[int]]:
    if strategy == "interleave":
        return [list(range(chunk_idx, slots_per_row, num_chunks)) for chunk_idx in range(num_chunks)]
    if strategy != "contiguous":
        raise ValueError(f"Unsupported fold slot strategy: {strategy}")
    base = slots_per_row // num_chunks
    extra = slots_per_row % num_chunks
    groups: list[list[int]] = []
    start = 0
    for chunk_idx in range(num_chunks):
        size = base + (1 if chunk_idx < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def fold_row_indices(source_height: int, slots_per_row: int, slot: int, *, row_order: str) -> np.ndarray:
    beam_rows = np.arange(source_height)
    if row_order == "beam_local":
        return beam_rows * slots_per_row + slot
    if row_order == "slot_major":
        return slot * source_height + beam_rows
    raise ValueError(f"Unsupported fold row order: {row_order}")


def fold_sparse_slots(slots_per_row: int, num_chunks: int) -> list[int]:
    if num_chunks == 1:
        return [slots_per_row // 2]
    return [int(round(idx * (slots_per_row - 1) / (num_chunks - 1))) for idx in range(num_chunks)]


def fold_visual_fill_range_maps_to_wan_grid(
    range_maps: np.ndarray,
    *,
    mode: str,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames, source_height, source_width = range_maps.shape
    if target_height % source_height != 0:
        raise ValueError(f"target_height {target_height} is not divisible by source_height {source_height}")
    slots_per_row = target_height // source_height
    if slots_per_row < 3:
        raise ValueError(f"Need at least 3 slots per row for visual fill, got {slots_per_row}")
    if target_width != 1280 or source_width != 2650:
        raise ValueError(f"Visual fill layouts expect source_width=2650 and target_width=1280, got {source_width}->{target_width}")

    output = np.zeros((frames, target_height, target_width), dtype=np.float32)
    for beam_idx in range(source_height):
        base = beam_idx * slots_per_row
        r0 = range_maps[:, beam_idx, 0:1280].astype(np.float32, copy=False)
        r1 = range_maps[:, beam_idx, 1280:2560].astype(np.float32, copy=False)
        r2 = np.empty((frames, target_width), dtype=np.float32)
        r2[:, :90] = range_maps[:, beam_idx, 2560:2650].astype(np.float32, copy=False)
        if mode == "fold_visual_copy":
            r2[:, 90:] = range_maps[:, beam_idx, 2559:2560].astype(np.float32, copy=False)
            output[:, base + 0, :] = r0
            output[:, base + 1, :] = r1
            output[:, base + 2, :] = r2
            output[:, base + 3 : base + slots_per_row, :] = r1[:, None, :]
        elif mode == "fold_visual_interp":
            r2[:, 90:] = range_maps[:, beam_idx, 2649:2650].astype(np.float32, copy=False)
            anchors = np.stack([r0, r1, r2], axis=1)
            positions = np.linspace(0.0, 2.0, slots_per_row, dtype=np.float32)
            lower = np.floor(positions).astype(np.int64)
            upper = np.clip(lower + 1, 0, 2)
            alpha = (positions - lower).astype(np.float32)
            smooth = anchors[:, lower, :] * (1.0 - alpha[None, :, None]) + anchors[:, upper, :] * alpha[None, :, None]
            smooth[:, 0, :] = r0
            smooth[:, 1, :] = r1
            smooth[:, 2, :] = r2
            output[:, base : base + slots_per_row, :] = smooth
        else:
            raise ValueError(f"Unsupported visual fill mode: {mode}")

    info = {
        "mode": mode,
        "source_shape": [int(frames), int(source_height), int(source_width)],
        "target_shape": [int(frames), int(target_height), int(target_width)],
        "slots_per_row": int(slots_per_row),
        "real_slots": [0, 1, 2],
        "source_valid_pixels": int((range_maps > 0).sum()),
        "packed_valid_pixels": int((output > 0).sum()),
    }
    return output, info


def fold_range_maps_to_wan_grid(
    range_maps: np.ndarray,
    *,
    mode: str,
    target_height: int,
    target_width: int,
    row_order: str,
    slot_strategy: str,
    serpentine: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode in {"fold_visual_copy", "fold_visual_interp"}:
        return fold_visual_fill_range_maps_to_wan_grid(
            range_maps,
            mode=mode,
            target_height=target_height,
            target_width=target_width,
        )
    if mode not in {"fold_sparse", "fold_duplicate"}:
        raise ValueError(f"Unsupported fold mode: {mode}")
    frames, source_height, source_width = range_maps.shape
    if target_height % source_height != 0:
        raise ValueError(f"target_height {target_height} is not divisible by source_height {source_height}")
    slots_per_row = target_height // source_height
    num_chunks = int(np.ceil(source_width / target_width))
    if num_chunks > slots_per_row:
        raise ValueError(f"Need {num_chunks} chunks but only {slots_per_row} vertical slots are available")

    output = np.zeros((frames, target_height, target_width), dtype=np.float32)
    sparse_slots = fold_sparse_slots(slots_per_row, num_chunks)
    duplicate_groups = fold_chunk_groups(slots_per_row, num_chunks, strategy=slot_strategy)
    for chunk_idx in range(num_chunks):
        start_col = chunk_idx * target_width
        end_col = min((chunk_idx + 1) * target_width, source_width)
        chunk = range_maps[:, :, start_col:end_col].astype(np.float32, copy=False)
        if serpentine and chunk_idx % 2 == 1:
            chunk = chunk[:, :, ::-1]
        slots = [sparse_slots[chunk_idx]] if mode == "fold_sparse" else duplicate_groups[chunk_idx]
        for slot in slots:
            output[:, fold_row_indices(source_height, slots_per_row, slot, row_order=row_order), : end_col - start_col] = chunk

    info = {
        "mode": mode,
        "source_shape": [int(frames), int(source_height), int(source_width)],
        "target_shape": [int(frames), int(target_height), int(target_width)],
        "slots_per_row": int(slots_per_row),
        "num_chunks": int(num_chunks),
        "sparse_slots": sparse_slots,
        "duplicate_groups": duplicate_groups,
        "serpentine": serpentine,
        "row_order": row_order,
        "slot_strategy": slot_strategy,
        "source_valid_pixels": int((range_maps > 0).sum()),
        "packed_valid_pixels": int((output > 0).sum()),
    }
    return output, info


def unfold_wan_grid_to_range_maps(
    folded: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    mode: str,
    row_order: str,
    slot_strategy: str,
    serpentine: bool,
    duplicate_reducer: str,
) -> np.ndarray:
    frames, target_height, target_width = folded.shape
    slots_per_row = target_height // source_height
    if mode in {"fold_visual_copy", "fold_visual_interp"}:
        if target_width != 1280 or source_width != 2650:
            raise ValueError(f"Visual fill layouts expect source_width=2650 and target_width=1280, got {source_width}->{target_width}")
        output = np.zeros((frames, source_height, source_width), dtype=np.float32)
        for beam_idx in range(source_height):
            base = beam_idx * slots_per_row
            output[:, beam_idx, 0:1280] = folded[:, base + 0, :]
            output[:, beam_idx, 1280:2560] = folded[:, base + 1, :]
            output[:, beam_idx, 2560:2650] = folded[:, base + 2, :90]
        return output
    num_chunks = int(np.ceil(source_width / target_width))
    sparse_slots = fold_sparse_slots(slots_per_row, num_chunks)
    duplicate_groups = fold_chunk_groups(slots_per_row, num_chunks, strategy=slot_strategy)
    output = np.zeros((frames, source_height, source_width), dtype=np.float32)
    for chunk_idx in range(num_chunks):
        start_col = chunk_idx * target_width
        end_col = min((chunk_idx + 1) * target_width, source_width)
        slots = [sparse_slots[chunk_idx]] if mode == "fold_sparse" else duplicate_groups[chunk_idx]
        rows = [fold_row_indices(source_height, slots_per_row, slot, row_order=row_order) for slot in slots]
        values = np.stack([folded[:, row_idx, : end_col - start_col] for row_idx in rows], axis=0)
        if mode == "fold_duplicate" and duplicate_reducer == "mean":
            chunk = values.mean(axis=0)
        else:
            chunk = np.median(values, axis=0)
        if serpentine and chunk_idx % 2 == 1:
            chunk = chunk[:, :, ::-1]
        output[:, :, start_col:end_col] = chunk
    return output


def log_normalize_range_maps(range_maps: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    valid = range_maps > 0
    clipped = np.clip(range_maps.astype(np.float32, copy=False), args.min_range, args.max_range)
    log_min = np.log(args.min_range)
    log_max = np.log(args.max_range)
    normalized = (np.log(clipped) - log_min) / (log_max - log_min)
    normalized = normalized * 2.0 - 1.0 if args.min_value == -1 else normalized
    return np.where(valid, normalized, args.min_value).astype(np.float32)


def log_unnormalize_range_maps(normalized: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    clipped = np.clip(normalized.astype(np.float32, copy=False), args.min_value, 1.0)
    if args.min_value == -1:
        unit = (clipped + 1.0) * 0.5
    else:
        unit = clipped
    log_min = np.log(args.min_range)
    log_max = np.log(args.max_range)
    ranges = np.exp(unit * (log_max - log_min) + log_min).astype(np.float32)
    ranges[clipped <= args.min_value + 1e-6] = 0.0
    return ranges


def encode_polyphase3_normalized(range_maps: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    frames, source_height, source_width = range_maps.shape
    if source_height != 64 or source_width != 2650:
        raise ValueError(f"polyphase3 layouts expect 64x2650 range maps, got {source_height}x{source_width}")
    rolled = np.roll(log_normalize_range_maps(range_maps, args), shift=args.polyphase_roll, axis=2)
    even = rolled[:, :, 0::2]
    odd = rolled[:, :, 1::2]
    c0 = even[:, :, :1280]
    c1 = odd[:, :, :1280]

    c0_valid = c0 > args.min_value
    c1_valid = c1 > args.min_value
    low = np.full((frames, source_height, 1280), args.min_value, dtype=np.float32)
    both = c0_valid & c1_valid
    only_c0 = c0_valid & ~c1_valid
    only_c1 = c1_valid & ~c0_valid
    low[both] = (c0[both] + c1[both]) * 0.5
    low[only_c0] = c0[only_c0]
    low[only_c1] = c1[only_c1]

    c2 = low.copy()
    c2[:, :, :45] = even[:, :, 1280:1325]
    c2[:, :, 45:90] = odd[:, :, 1280:1325]
    return np.stack([c0, c1, c2], axis=1).astype(np.float32)


def expand_polyphase3_vertical(channels: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.range_map_layout == "polyphase3_repeat":
        return np.repeat(channels, 11, axis=2).astype(np.float32)
    if args.range_map_layout == "polyphase3_interp":
        tensor = torch.from_numpy(channels).float()
        expanded = F.interpolate(tensor, size=(704, 1280), mode="bilinear", align_corners=True)
        return expanded.numpy().astype(np.float32)
    raise ValueError(f"Unsupported polyphase3 layout: {args.range_map_layout}")


def preprocess_polyphase3_range_maps(
    range_maps: np.ndarray, args: argparse.Namespace
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    channels = encode_polyphase3_normalized(range_maps, args)
    expanded = expand_polyphase3_vertical(channels, args)
    tensor = torch.from_numpy(expanded).float().permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return tensor, range_maps.astype(np.float32, copy=False), range_maps > 0


def make_polyphase3_layout_info(range_maps: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    channels = encode_polyphase3_normalized(range_maps, args)
    expanded = expand_polyphase3_vertical(channels, args)
    return {
        "mode": args.range_map_layout,
        "source_shape": [int(v) for v in range_maps.shape],
        "target_shape": [int(range_maps.shape[0]), 3, 704, 1280],
        "polyphase_roll": int(args.polyphase_roll),
        "source_valid_pixels": int((range_maps > 0).sum()),
        "packed_valid_pixels": int((expanded > args.min_value).sum()),
        "tail_layout": "C2[:, :45]=even_tail, C2[:, 45:90]=odd_tail, C2[:, 90:]=valid_aware_low_frequency_mean",
        "normalization": "log_range_to_min_value_1",
    }


def prepend_lidar_utils_repo(repo_path: str) -> None:
    resolved = str(Path(repo_path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def preprocess_range_maps(
    range_maps: np.ndarray, args: argparse.Namespace
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if args.range_map_layout in POLYPHASE3_LAYOUTS:
        return preprocess_polyphase3_range_maps(range_maps, args)

    from cosmos_predict1.utils.lidar_rangemap import RangeMapDownsampler, normalize_range_map

    downsampler = RangeMapDownsampler(
        row_factor=args.downsample_factor_row,
        col_factor=args.downsample_factor_col,
        method=args.downsample_method,
    )
    downsampled = downsampler.downsample(range_maps).astype(np.float32)
    valid_mask = downsampled > 0

    depth_normalized = normalize_range_map(
        downsampled.copy(),
        args.max_range,
        args.min_range,
        args.min_value,
        False,
    ).astype(np.float32)
    if args.input_channel_mode == "repeat_depth":
        normalized = depth_normalized[:, None, :, :].repeat(3, axis=1)
    elif args.input_channel_mode == "concat_inv_depth":
        inv_normalized = normalize_range_map(
            downsampled.copy(),
            args.max_range,
            args.min_range,
            args.min_value,
            True,
        ).astype(np.float32)
        normalized = np.concatenate(
            [inv_normalized[:, None, :, :], depth_normalized[:, None, :, :], depth_normalized[:, None, :, :]],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported input channel mode: {args.input_channel_mode}")
    normalized = normalized.repeat(args.repeat_row, axis=2)
    normalized = normalized.repeat(args.repeat_col, axis=3)
    tensor = torch.from_numpy(normalized).float().permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return tensor, downsampled, valid_mask


def pad_video_tensor_spatial(
    tensor: torch.Tensor,
    *,
    align: int,
    pad_value: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    if align <= 1:
        pad_bottom = 0
        pad_right = 0
    else:
        pad_bottom = (align - tensor.shape[-2] % align) % align
        pad_right = (align - tensor.shape[-1] % align) % align
    padding = {
        "input_height": int(tensor.shape[-2]),
        "input_width": int(tensor.shape[-1]),
        "pad_bottom": int(pad_bottom),
        "pad_right": int(pad_right),
        "wan_height": int(tensor.shape[-2] + pad_bottom),
        "wan_width": int(tensor.shape[-1] + pad_right),
    }
    if pad_bottom == 0 and pad_right == 0:
        return tensor.contiguous(), padding
    return F.pad(tensor, (0, pad_right, 0, pad_bottom), value=pad_value).contiguous(), padding


def crop_video_tensor_spatial(tensor: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    return tensor[..., :height, :width].contiguous()


def make_downsampled_ray_directions(range_maps: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from cosmos_predict1.utils.lidar_rangemap import range_map_to_ray_directions
    from visualize_decoded_lidar_waymo import downsample_range_with_extras

    elevations = make_waymo_top_elevation_angles(range_maps.shape[1])
    raw_rays = range_map_to_ray_directions(range_maps.shape[-1], elevations).astype(np.float32)
    raw_rays = np.broadcast_to(raw_rays[None], range_maps.shape + (3,))
    _, [downsampled_rays] = downsample_range_with_extras(
        range_maps,
        [raw_rays],
        row_factor=args.downsample_factor_row,
        col_factor=args.downsample_factor_col,
        method=args.downsample_method,
    )
    return downsampled_rays.astype(np.float32, copy=False)


def resolve_wan_vae_path(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path

    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home))
    data_hf = Path("/data/huggingface")
    if data_hf.exists() and data_hf not in roots:
        roots.append(data_hf)

    for root in roots:
        repo_root = root / "hub" / WAN_HF_REPO_CACHE
        candidates = [
            repo_root / "snapshots" / WAN_HF_REVISION / WAN_HF_FILENAME,
            *sorted(repo_root.glob(f"snapshots/*/{WAN_HF_FILENAME}")),
            *sorted(repo_root.glob("snapshots/*/Wan2.1_VAE.pth")),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


class WanVAEAdapter:
    def __init__(self, model: Any, dtype: torch.dtype):
        self.model = model
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.model.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latent)


def load_wan_tokenizer(args: argparse.Namespace):
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface, WanVAE

    kwargs: dict[str, Any] = {
        "name": "wan2pt1_tokenizer",
        "s3_credential_path": args.wan_s3_credential_path,
        "temporal_window": 4,
    }
    resolved_vae_path = resolve_wan_vae_path(args.wan_vae_path)
    if resolved_vae_path:
        kwargs["vae_pth"] = resolved_vae_path
        print(f"[wan] using VAE checkpoint: {resolved_vae_path}", flush=True)
    elif not args.allow_default_wan_uri:
        raise FileNotFoundError(
            "Could not find a local Wan2.1 VAE checkpoint. Pass --wan-vae-path, set HF_HOME, "
            "or pass --allow-default-wan-uri to use the configured S3/default URI."
        )
    else:
        print("[wan] local VAE checkpoint not found; using default Wan2.1 URI", flush=True)

    if args.device == "cuda" and args.dtype == "bfloat16":
        return Wan2pt1VAEInterface(chunk_duration=args.num_frames, load_mean_std=False, **kwargs)

    direct_kwargs = dict(kwargs)
    direct_kwargs.pop("name", None)
    model_dtype = getattr(torch, args.dtype)
    model = WanVAE(
        dtype=model_dtype,
        is_amp=False,
        load_mean_std=False,
        device=args.device,
        **direct_kwargs,
    )
    return WanVAEAdapter(model, dtype=model_dtype)


def unnormalize_range(
    normalized: np.ndarray,
    *,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool = False,
) -> np.ndarray:
    normalized = np.clip(normalized, min_value, 1.0)
    c_min_range = min_range if not inverse_depth else 1.0 / max_range
    c_max_range = max_range if not inverse_depth else 1.0 / min_range
    if min_value == -1:
        range_map = (normalized + 1.0) * 0.5 * (c_max_range - c_min_range) + c_min_range
    else:
        range_map = normalized * (c_max_range - c_min_range) + c_min_range
    if inverse_depth:
        range_map = 1.0 / np.clip(range_map, 1e-6, None)
    return range_map


def decode_reconstruction_range(
    video_tensor: torch.Tensor,
    *,
    args: argparse.Namespace,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    range_tensor = video_tensor[0].detach().cpu().float()
    range_tensor = range_tensor[:, :, args.repeat_row // 2 :: args.repeat_row, args.repeat_col // 2 :: args.repeat_col]
    range_tensor = range_tensor[:, :, :target_height, :target_width]

    if args.decode_channel_mode == "first":
        normalized = range_tensor[0].numpy()
        return unnormalize_range(
            normalized,
            min_range=args.min_range,
            max_range=args.max_range,
            min_value=args.min_value,
        )
    if args.decode_channel_mode == "mean":
        normalized = range_tensor.mean(dim=0).numpy()
        return unnormalize_range(
            normalized,
            min_range=args.min_range,
            max_range=args.max_range,
            min_value=args.min_value,
        )
    if args.decode_channel_mode == "median":
        normalized = range_tensor.median(dim=0).values.numpy()
        return unnormalize_range(
            normalized,
            min_range=args.min_range,
            max_range=args.max_range,
            min_value=args.min_value,
        )
    if args.decode_channel_mode == "concat_fuse":
        inv_normalized = range_tensor[0].numpy()
        depth_normalized = range_tensor[1:].mean(dim=0).numpy()
        inv_range = unnormalize_range(
            inv_normalized,
            min_range=args.min_range,
            max_range=args.max_range,
            min_value=args.min_value,
            inverse_depth=True,
        )
        depth_range = unnormalize_range(
            depth_normalized,
            min_range=args.min_range,
            max_range=args.max_range,
            min_value=args.min_value,
        )
        fused = depth_range.copy()
        use_inverse = inv_range < args.inv_depth_threshold
        fused[use_inverse] = inv_range[use_inverse]
        return fused

    raise ValueError(f"Unsupported decode channel mode: {args.decode_channel_mode}")


def decode_polyphase3_reconstruction_range(video_tensor: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    channels = video_tensor[0].detach().cpu().float().permute(1, 0, 2, 3)
    if args.range_map_layout == "polyphase3_repeat":
        frames = channels.shape[0]
        channels = channels[:, :, :704, :1280].reshape(frames, 3, 64, 11, 1280).mean(dim=3)
    elif args.range_map_layout == "polyphase3_interp":
        channels = F.interpolate(channels[:, :, :704, :1280], size=(64, 1280), mode="bilinear", align_corners=True)
    else:
        raise ValueError(f"Unsupported polyphase3 layout: {args.range_map_layout}")

    normalized = channels.numpy().astype(np.float32)
    c0 = log_unnormalize_range_maps(normalized[:, 0], args)
    c1 = log_unnormalize_range_maps(normalized[:, 1], args)
    c2 = log_unnormalize_range_maps(normalized[:, 2], args)

    frames = normalized.shape[0]
    source_height = normalized.shape[2]
    even = np.zeros((frames, source_height, 1325), dtype=np.float32)
    odd = np.zeros((frames, source_height, 1325), dtype=np.float32)
    even[:, :, :1280] = c0
    odd[:, :, :1280] = c1
    even[:, :, 1280:1325] = c2[:, :, :45]
    odd[:, :, 1280:1325] = c2[:, :, 45:90]

    rolled = np.zeros((frames, source_height, 2650), dtype=np.float32)
    rolled[:, :, 0::2] = even
    rolled[:, :, 1::2] = odd
    return np.roll(rolled, shift=-args.polyphase_roll, axis=2).astype(np.float32)


def colorize_depth_maps(range_maps: np.ndarray, valid_mask: np.ndarray, *, cmap_name: str = "turbo") -> np.ndarray:
    from matplotlib import colormaps

    frames = range_maps.astype(np.float32).copy()
    valid_values = frames[valid_mask]
    if valid_values.size == 0:
        near = np.log(1.0)
        far = np.log(2.0)
    else:
        near = np.log(np.quantile(valid_values, 0.01))
        far = np.log(np.quantile(valid_values, 0.99))
        if not np.isfinite(far - near) or abs(far - near) < 1e-6:
            far = near + 1.0

    safe = np.where(valid_mask, frames, 1.0)
    normalized = 1.0 - (np.log(np.clip(safe, 1e-6, None)) - near) / (far - near)
    normalized = np.clip(normalized, 0.0, 1.0)
    rgb = colormaps.get_cmap(cmap_name)(normalized)[..., :3]
    rgb[~valid_mask] = np.array([0.82, 0.82, 0.82], dtype=np.float32)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def colorize_error_maps(error_maps: np.ndarray, valid_mask: np.ndarray, *, cmap_name: str = "magma") -> np.ndarray:
    from matplotlib import colormaps

    valid_errors = error_maps[valid_mask]
    high = float(np.quantile(valid_errors, 0.99)) if valid_errors.size else 1.0
    if high < 1e-6:
        high = 1.0
    normalized = np.clip(error_maps / high, 0.0, 1.0)
    rgb = colormaps.get_cmap(cmap_name)(normalized)[..., :3]
    rgb[~valid_mask] = np.array([0.82, 0.82, 0.82], dtype=np.float32)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def write_video(path: Path, frames: np.ndarray, *, fps: int) -> None:
    import mediapy as media

    path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(path), frames, fps=fps)


def write_point_cloud_video(
    output_dir: Path,
    *,
    input_range: np.ndarray,
    reconstruction_range: np.ndarray,
    valid_mask: np.ndarray,
    ray_directions: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    from visualize_decoded_lidar_waymo import save_point_cloud_video

    output_path = output_dir / f"point_cloud_{args.pcd_camera_view}_{args.pcd_display_frame}.mp4"
    renderer = save_point_cloud_video(
        input_range,
        reconstruction_range,
        valid_mask,
        output_path,
        gt_label="input",
        prediction_label="wan2.1_decode",
        ray_directions=ray_directions,
        display_frame=args.pcd_display_frame,
        camera_view=args.pcd_camera_view,
        fps=args.fps,
        width=args.pcd_width,
        height=args.pcd_height,
        max_points=args.pcd_max_points,
        renderer=args.pcd_renderer,
        lidar_tokenizer_repo=args.lidar_utils_repo,
        workers=args.pcd_workers,
    )
    return output_path, renderer


def save_outputs(
    output_dir: Path,
    *,
    input_tensor: torch.Tensor,
    wan_input_tensor: torch.Tensor,
    latent: torch.Tensor,
    reconstruction: torch.Tensor,
    wan_reconstruction: torch.Tensor,
    save_dtype: torch.dtype,
    skip_tensors: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_tensors:
        return
    torch.save(input_tensor.detach().cpu().to(dtype=save_dtype), output_dir / "input_tensor.pt")
    if tuple(wan_input_tensor.shape) != tuple(input_tensor.shape):
        torch.save(wan_input_tensor.detach().cpu().to(dtype=save_dtype), output_dir / "wan_input_tensor.pt")
    torch.save(latent.detach().cpu().to(dtype=save_dtype), output_dir / "latent.pt")
    torch.save(reconstruction.detach().cpu().to(dtype=save_dtype), output_dir / "reconstruction.pt")
    if tuple(wan_reconstruction.shape) != tuple(reconstruction.shape):
        torch.save(
            wan_reconstruction.detach().cpu().to(dtype=save_dtype),
            output_dir / "wan_reconstruction.pt",
        )


def build_metrics(
    *,
    input_tensor: torch.Tensor,
    latent: torch.Tensor,
    reconstruction: torch.Tensor,
    input_range: np.ndarray,
    reconstruction_range: np.ndarray,
    valid_mask: np.ndarray,
    selected_frame_names: list[str],
    source_tar: Path,
    source_range_map_shape: tuple[int, ...],
    wan_input_tensor_shape: tuple[int, ...],
    wan_reconstruction_shape: tuple[int, ...],
    spatial_padding: dict[str, int],
    range_map_layout_info: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_float = input_tensor.detach().cpu().float()
    recon_float = reconstruction.detach().cpu().float()
    diff = recon_float - input_float

    input_range_clipped = np.clip(input_range, args.min_range, args.max_range)
    valid = valid_mask & np.isfinite(reconstruction_range)
    range_diff = reconstruction_range - input_range_clipped
    if valid.any():
        valid_range_diff = range_diff[valid]
        range_mae = float(np.mean(np.abs(valid_range_diff)))
        range_rmse = float(np.sqrt(np.mean(valid_range_diff**2)))
        range_max_abs = float(np.max(np.abs(valid_range_diff)))
    else:
        range_mae = None
        range_rmse = None
        range_max_abs = None

    generated_valid = np.isfinite(reconstruction_range) & (reconstruction_range > args.min_range + 0.25)
    matched_valid = valid_mask & generated_valid
    generated_extra = generated_valid & ~valid_mask
    generated_missing = valid_mask & ~generated_valid
    precision = float(matched_valid.sum() / max(generated_valid.sum(), 1))
    recall = float(matched_valid.sum() / max(valid_mask.sum(), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    iou = float(matched_valid.sum() / max((valid_mask | generated_valid).sum(), 1))

    return {
        "segment_key": args.segment_key,
        "split": args.split,
        "source_tar": str(source_tar),
        "selected_frame_names": selected_frame_names,
        "preprocess_mode": args.preprocess_mode,
        "source_range_map_shape": list(source_range_map_shape),
        "evaluation_range_shape": list(input_range.shape),
        "input_tensor_shape": list(input_tensor.shape),
        "wan_input_tensor_shape": list(wan_input_tensor_shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "wan_reconstruction_shape": list(wan_reconstruction_shape),
        "spatial_padding": spatial_padding,
        "normalized_mse": float(torch.mean(diff * diff).item()),
        "normalized_mae": float(torch.mean(torch.abs(diff)).item()),
        "valid_pixel_count": int(valid.sum()),
        "range_mae_m": range_mae,
        "range_rmse_m": range_rmse,
        "range_max_abs_m": range_max_abs,
        "generated_valid_threshold_m": float(args.min_range + 0.25),
        "generated_valid_pixel_count": int(generated_valid.sum()),
        "matched_valid_pixel_count": int(matched_valid.sum()),
        "generated_extra_pixel_count": int(generated_extra.sum()),
        "generated_missing_pixel_count": int(generated_missing.sum()),
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
        "occupancy_iou": iou,
        "preprocess": {
            "downsample_factor_row": args.downsample_factor_row,
            "downsample_factor_col": args.downsample_factor_col,
            "downsample_method": args.downsample_method,
            "repeat_row": args.repeat_row,
            "repeat_col": args.repeat_col,
            "input_channel_mode": args.input_channel_mode,
            "decode_channel_mode": args.decode_channel_mode,
            "inv_depth_threshold": args.inv_depth_threshold,
            "max_range": args.max_range,
            "min_range": args.min_range,
            "min_value": args.min_value,
            "native_n_rows": args.native_n_rows,
            "native_n_cols": args.native_n_cols,
            "projection_max_range": args.projection_max_range,
            "wan_spatial_align": args.wan_spatial_align,
            "range_map_target_width": args.range_map_target_width,
            "range_map_eval_source_grid": args.range_map_eval_source_grid,
            "range_map_layout": args.range_map_layout,
            "range_map_layout_info": range_map_layout_info,
            "fold_row_order": args.fold_row_order,
            "fold_slot_strategy": args.fold_slot_strategy,
            "fold_serpentine": args.fold_serpentine,
            "fold_duplicate_reducer": args.fold_duplicate_reducer,
            "polyphase_roll": args.polyphase_roll,
        },
    }


def main() -> None:
    args = parse_args()
    apply_mode_defaults(args)
    os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    prepend_lidar_utils_repo(args.lidar_utils_repo)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model_dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)

    range_map_layout_info: dict[str, Any] | None = None
    fold_source_range_maps: np.ndarray | None = None
    polyphase_source_range_maps: np.ndarray | None = None
    rebin_source_range_maps: np.ndarray | None = None

    if args.range_map_npz:
        source_tar = Path(args.range_map_npz)
        print(f"[load] precomputed range map npz: {source_tar}", flush=True)
        range_maps, selected_frame_names = load_npz_range_maps(source_tar, range_return=args.range_map_return)
        if args.range_map_target_width is not None:
            if args.range_map_layout != "plain":
                raise ValueError("--range-map-target-width is incompatible with folded range-map layouts")
            if args.range_map_eval_source_grid:
                rebin_source_range_maps = range_maps.astype(np.float32, copy=True)
            range_maps = rebin_range_maps_width_min(range_maps, args.range_map_target_width)
        args.num_frames = int(range_maps.shape[0])
        args.native_n_rows = int(range_maps.shape[1])
        args.native_n_cols = int(range_maps.shape[2])
    elif args.preprocess_mode == "waymo_top_64x2650":
        source_tar = (
            Path(args.raw_tar)
            if args.raw_tar
            else Path(args.raw_lidar_root) / args.split / "lidar_raw" / f"{args.segment_key}.tar"
        )
        print(f"[load] native Waymo TOP raw lidar tar: {source_tar}", flush=True)
        range_maps, selected_frame_names = load_raw_range_maps(
            source_tar,
            frame_start=args.frame_start,
            num_frames=args.num_frames,
            pad_last=args.pad_last,
            n_rows=args.native_n_rows,
            n_cols=args.native_n_cols,
            max_projection_range=args.projection_max_range,
        )
    elif args.raw_tar:
        source_tar = Path(args.raw_tar)
        print(f"[load] raw lidar tar: {source_tar}", flush=True)
        range_maps, selected_frame_names = load_raw_range_maps(
            source_tar,
            frame_start=args.frame_start,
            num_frames=args.num_frames,
            pad_last=args.pad_last,
            n_rows=128,
            n_cols=3600,
            max_projection_range=args.projection_max_range,
        )
    else:
        source_tar = (
            Path(args.converted_tar)
            if args.converted_tar
            else Path(args.converted_lidar_root) / args.split / "lidar" / f"{args.segment_key}.tar"
        )
        print(f"[load] converted lidar tar: {source_tar}", flush=True)
        range_maps, selected_frame_names = load_converted_range_maps(
            source_tar,
            frame_start=args.frame_start,
            num_frames=args.num_frames,
            pad_last=args.pad_last,
        )

    source_range_map_shape = tuple(range_maps.shape)
    if args.range_map_layout != "plain":
        if args.range_map_layout in POLYPHASE3_LAYOUTS:
            polyphase_source_range_maps = range_maps.astype(np.float32, copy=True)
            range_map_layout_info = make_polyphase3_layout_info(polyphase_source_range_maps, args)
            print(f"[layout] polyphase3 range maps: {tuple(source_range_map_shape)} -> (T, 3, 704, 1280)", flush=True)
        else:
            fold_source_range_maps = range_maps.astype(np.float32, copy=True)
            range_maps, range_map_layout_info = fold_range_maps_to_wan_grid(
                fold_source_range_maps,
                mode=args.range_map_layout,
                target_height=args.fold_target_height,
                target_width=args.fold_target_width,
                row_order=args.fold_row_order,
                slot_strategy=args.fold_slot_strategy,
                serpentine=args.fold_serpentine,
            )
            args.native_n_rows = int(range_maps.shape[1])
            args.native_n_cols = int(range_maps.shape[2])
            print(f"[layout] folded range maps: {tuple(source_range_map_shape)} -> {tuple(range_maps.shape)}", flush=True)
        print(f"[layout] {json.dumps(range_map_layout_info, sort_keys=True)}", flush=True)

    if args.preprocess_mode == "tokenizer_128x3600":
        output_name = args.segment_key
    else:
        output_name = f"{args.segment_key}_{args.preprocess_mode}"
        if args.repeat_row != 1:
            output_name = f"{output_name}_repeatrow{args.repeat_row}"
        if args.repeat_col != 1:
            output_name = f"{output_name}_repeatcol{args.repeat_col}"
        if args.input_channel_mode != "repeat_depth":
            output_name = f"{output_name}_{args.input_channel_mode}"
        default_decode = "concat_fuse" if args.input_channel_mode == "concat_inv_depth" else "mean"
        if args.decode_channel_mode != default_decode:
            output_name = f"{output_name}_decode{args.decode_channel_mode}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("/data2/waymo_wan_lidar_vae_smoke") / args.split / output_name
    )
    print(f"[preprocess] raw range maps: {tuple(range_maps.shape)}", flush=True)
    input_tensor, downsampled_range, valid_mask = preprocess_range_maps(range_maps, args)
    wan_input_tensor, spatial_padding = pad_video_tensor_spatial(
        input_tensor,
        align=args.wan_spatial_align,
        pad_value=args.min_value,
    )
    print(f"[preprocess] logical tensor: {tuple(input_tensor.shape)}", flush=True)
    print(f"[preprocess] Wan input tensor: {tuple(wan_input_tensor.shape)} padding={spatial_padding}", flush=True)

    tokenizer = load_wan_tokenizer(args)
    wan_input_tensor = wan_input_tensor.to(device=device, dtype=model_dtype)
    with torch.no_grad():
        latent = tokenizer.encode(wan_input_tensor).detach()
        print(f"[wan] latent: {tuple(latent.shape)}", flush=True)
        wan_reconstruction = tokenizer.decode(latent).detach()
        print(f"[wan] reconstruction: {tuple(wan_reconstruction.shape)}", flush=True)

    if tuple(wan_reconstruction.shape) != tuple(wan_input_tensor.shape):
        raise ValueError(
            f"Wan reconstruction shape {tuple(wan_reconstruction.shape)} does not match input "
            f"{tuple(wan_input_tensor.shape)}"
        )

    reconstruction = crop_video_tensor_spatial(
        wan_reconstruction,
        height=spatial_padding["input_height"],
        width=spatial_padding["input_width"],
    )
    input_tensor = input_tensor.to(device=device, dtype=model_dtype)
    save_outputs(
        output_dir,
        input_tensor=input_tensor,
        wan_input_tensor=wan_input_tensor,
        latent=latent,
        reconstruction=reconstruction,
        wan_reconstruction=wan_reconstruction,
        save_dtype=save_dtype,
        skip_tensors=args.skip_tensors,
    )

    if args.range_map_layout in POLYPHASE3_LAYOUTS:
        recon_range = decode_polyphase3_reconstruction_range(reconstruction, args)
    else:
        recon_range = decode_reconstruction_range(
            reconstruction,
            args=args,
            target_height=downsampled_range.shape[1],
            target_width=downsampled_range.shape[2],
        )
    metric_input_range = downsampled_range
    metric_recon_range = recon_range
    metric_valid_mask = valid_mask
    if polyphase_source_range_maps is not None:
        metric_input_range = polyphase_source_range_maps
        metric_recon_range = recon_range
        metric_valid_mask = metric_input_range > 0
    elif fold_source_range_maps is not None:
        metric_input_range = fold_source_range_maps
        metric_recon_range = unfold_wan_grid_to_range_maps(
            recon_range,
            source_height=fold_source_range_maps.shape[1],
            source_width=fold_source_range_maps.shape[2],
            mode=args.range_map_layout,
            row_order=args.fold_row_order,
            slot_strategy=args.fold_slot_strategy,
            serpentine=args.fold_serpentine,
            duplicate_reducer=args.fold_duplicate_reducer,
        )
        metric_valid_mask = metric_input_range > 0
    elif rebin_source_range_maps is not None:
        metric_input_range = rebin_source_range_maps
        metric_recon_range = expand_rebinned_range_maps_width(recon_range, rebin_source_range_maps.shape[2])
        metric_valid_mask = metric_input_range > 0
    error_range = np.abs(metric_recon_range - np.clip(metric_input_range, args.min_range, args.max_range))

    metrics = build_metrics(
        input_tensor=input_tensor,
        latent=latent,
        reconstruction=reconstruction,
        input_range=metric_input_range,
        reconstruction_range=metric_recon_range,
        valid_mask=metric_valid_mask,
        selected_frame_names=selected_frame_names,
        source_tar=source_tar,
        source_range_map_shape=source_range_map_shape,
        wan_input_tensor_shape=tuple(wan_input_tensor.shape),
        wan_reconstruction_shape=tuple(wan_reconstruction.shape),
        spatial_padding=spatial_padding,
        range_map_layout_info=range_map_layout_info,
        args=args,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[save] metrics: {metrics_path}", flush=True)

    if not args.skip_videos:
        input_video = colorize_depth_maps(downsampled_range, valid_mask)
        recon_video = colorize_depth_maps(recon_range, valid_mask)
        error_video = colorize_error_maps(error_range, valid_mask)
        write_video(output_dir / "input_rangemap.mp4", input_video, fps=args.fps)
        write_video(output_dir / "reconstruction_rangemap.mp4", recon_video, fps=args.fps)
        write_video(output_dir / "abs_error_rangemap.mp4", error_video, fps=args.fps)
        print(f"[save] videos: {output_dir}", flush=True)

    if args.vis_pcd:
        ray_directions = make_downsampled_ray_directions(range_maps, args)
        pcd_path, pcd_renderer = write_point_cloud_video(
            output_dir,
            input_range=np.clip(downsampled_range, args.min_range, args.max_range),
            reconstruction_range=recon_range,
            valid_mask=valid_mask,
            ray_directions=ray_directions,
            args=args,
        )
        metrics["point_cloud_video"] = str(pcd_path)
        metrics["point_cloud_renderer"] = pcd_renderer
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[save] point cloud video: {pcd_path} renderer={pcd_renderer}", flush=True)

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
