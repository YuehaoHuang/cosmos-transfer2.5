#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Inspect official Waymo TOP LiDAR range images and point-cloud reconstruction.

This script intentionally uses the official ``waymo_open_dataset`` TensorFlow
API.  On the TelaAI host it should be run from the ``waymo-kitti`` conda env:

    conda activate waymo-kitti
    python scripts/inspect_waymo_official_lidar.py \
      --split validation \
      --segment-key 10448102132863604198_472_000_492_000 \
      --num-frames 29 \
      --frame-step 1 \
      --rds-frame-suffix-step 3 \
      --compare-rds-raw \
      --save-preview
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any

# The old Waymo/TensorFlow env on the server has mismatched CUDA libraries.
# This script is CPU-only; disable GPU probing before importing TensorFlow.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-waymo-root", default="/team/hyh/data/waymo/raw")
    parser.add_argument("--rds-lidar-root", default="/team/hyh/data/rds_hq_waymo")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--segment-key", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--rds-frame-suffix-offset", type=int, default=0)
    parser.add_argument("--rds-frame-suffix-step", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare-rds-raw", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--save-ply", dest="save_ply", action="store_true", default=True)
    parser.add_argument("--no-save-ply", dest="save_ply", action="store_false")
    parser.add_argument("--ply-max-points", type=int, default=200000)
    parser.add_argument("--save-top-raw-range-npz", action="store_true")
    parser.add_argument("--save-top-range-npz", action="store_true")
    parser.add_argument("--top-range-output-width", type=int, default=1280)
    parser.add_argument(
        "--top-range-return-mode",
        default="return1",
        choices=["return1", "return1_return2_min"],
    )
    return parser.parse_args()


def tfrecord_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.raw_waymo_root)
        / args.split
        / f"segment-{args.segment_key}_with_camera_labels.tfrecord"
    )


def rds_tar_path(args: argparse.Namespace) -> Path:
    return Path(args.rds_lidar_root) / args.split / "lidar_raw" / f"{args.segment_key}.tar"


def selected_frame_indices(args: argparse.Namespace) -> list[int]:
    return [args.frame_start + idx * args.frame_step for idx in range(args.num_frames)]


def rds_frame_suffix(args: argparse.Namespace, tfrecord_frame_idx: int) -> int:
    return args.rds_frame_suffix_offset + tfrecord_frame_idx * args.rds_frame_suffix_step


def iter_selected_frames(path: Path, indices: list[int]):
    wanted = set(indices)
    max_idx = max(wanted)
    for frame_idx, data in enumerate(tf.data.TFRecordDataset(str(path))):
        if frame_idx > max_idx:
            break
        if frame_idx not in wanted:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        yield frame_idx, frame


def parse_official_range_images(frame):
    parsed = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(parsed) == 3:
        return parsed
    range_images, camera_projections, _seg_labels, range_image_top_pose = parsed
    return range_images, camera_projections, range_image_top_pose


def matrix_float_to_numpy(matrix) -> np.ndarray:
    return np.asarray(matrix.data, dtype=np.float32).reshape(matrix.shape.dims)


def laser_name(name: int) -> str:
    return dataset_pb2.LaserName.Name.Name(name)


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride]


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, points.astype(np.float32), fmt="%.5f %.5f %.5f")


def rebin_range_width_min(range_maps: list[np.ndarray], target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")
    source_height, source_width = range_maps[0].shape
    output = np.zeros((source_height, target_width), dtype=np.float32)
    source_cols = np.arange(source_width, dtype=np.float64)
    target_cols = np.floor((source_cols + 0.5) * target_width / source_width).astype(np.int64)
    target_cols = np.clip(target_cols, 0, target_width - 1)

    for row_idx in range(source_height):
        accum = np.full(target_width, np.inf, dtype=np.float32)
        for range_map in range_maps:
            values = range_map[row_idx].astype(np.float32, copy=False)
            valid = values > 0
            if valid.any():
                np.minimum.at(accum, target_cols[valid], values[valid])
        output[row_idx] = np.where(np.isfinite(accum), accum, 0.0)
    return output


def save_top_range_npz(
    path: Path,
    *,
    top_ri_r1: np.ndarray,
    top_ri_r2: np.ndarray,
    frame_idx: int,
    rds_suffix: int | None,
    frame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    range_maps = [top_ri_r1[..., 0]]
    if args.top_range_return_mode == "return1_return2_min":
        range_maps.append(top_ri_r2[..., 0])
    rebinned = rebin_range_width_min(range_maps, args.top_range_output_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_name = f"{args.segment_key}.{frame_idx:06d}"
    np.savez_compressed(
        path,
        range_maps=rebinned[None].astype(np.float32),
        frame_names=np.asarray([frame_name]),
        segment_key=args.segment_key,
        split=args.split,
        tfrecord_frame_indices=np.asarray([frame_idx], dtype=np.int64),
        rds_frame_suffixes=np.asarray([-1 if rds_suffix is None else rds_suffix], dtype=np.int64),
        timestamp_micros=np.asarray([int(frame.timestamp_micros)], dtype=np.int64),
        top_range_return_mode=args.top_range_return_mode,
        top_range_output_width=np.asarray(args.top_range_output_width, dtype=np.int64),
        official_top_return1_valid=np.asarray(int((top_ri_r1[..., 0] > 0).sum()), dtype=np.int64),
        official_top_return2_valid=np.asarray(int((top_ri_r2[..., 0] > 0).sum()), dtype=np.int64),
        rebinned_valid=np.asarray(int((rebinned > 0).sum()), dtype=np.int64),
    )
    return {
        "path": str(path),
        "return_mode": args.top_range_return_mode,
        "output_shape": list(rebinned.shape),
        "valid_pixels": int((rebinned > 0).sum()),
    }


def save_top_raw_range_npz(
    path: Path,
    *,
    top_ri_r1: np.ndarray,
    top_ri_r2: np.ndarray,
    frame_idx: int,
    rds_suffix: int | None,
    frame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_name = f"{args.segment_key}.{frame_idx:06d}"
    np.savez_compressed(
        path,
        range_image_return1=top_ri_r1.astype(np.float32),
        range_image_return2=top_ri_r2.astype(np.float32),
        frame_names=np.asarray([frame_name]),
        segment_key=args.segment_key,
        split=args.split,
        tfrecord_frame_indices=np.asarray([frame_idx], dtype=np.int64),
        rds_frame_suffixes=np.asarray([-1 if rds_suffix is None else rds_suffix], dtype=np.int64),
        timestamp_micros=np.asarray([int(frame.timestamp_micros)], dtype=np.int64),
        channel_names=np.asarray(["range", "intensity", "elongation", "nlz"]),
        official_top_return1_valid=np.asarray(int((top_ri_r1[..., 0] > 0).sum()), dtype=np.int64),
        official_top_return2_valid=np.asarray(int((top_ri_r2[..., 0] > 0).sum()), dtype=np.int64),
    )
    return {
        "path": str(path),
        "return1_shape": list(top_ri_r1.shape),
        "return2_shape": list(top_ri_r2.shape),
        "return1_valid": int((top_ri_r1[..., 0] > 0).sum()),
        "return2_valid": int((top_ri_r2[..., 0] > 0).sum()),
    }


def save_preview_png(path: Path, range_image: np.ndarray, points: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = range_image > 0
    shown = np.where(valid, range_image, np.nan)
    vmax = float(np.nanquantile(shown, 0.99)) if valid.any() else 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(shown, cmap="turbo", vmin=0.0, vmax=vmax, aspect="auto")
    axes[0].set_title("TOP range image")
    axes[0].set_xlabel("azimuth col")
    axes[0].set_ylabel("beam row")

    plot_points = deterministic_subsample(points, 70000)
    axes[1].scatter(plot_points[:, 0], plot_points[:, 1], s=0.1, c=plot_points[:, 2], cmap="viridis")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("point cloud top-down")
    axes[1].set_xlabel("x forward")
    axes[1].set_ylabel("y left")

    axes[2].scatter(plot_points[:, 0], plot_points[:, 2], s=0.1, c=plot_points[:, 1], cmap="viridis")
    axes[2].set_title("point cloud side/front")
    axes[2].set_xlabel("x forward")
    axes[2].set_ylabel("z up")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_rds_raw_xyz(tar_path: Path, segment_key: str, frame_idx: int) -> np.ndarray:
    member_name = f"{segment_key}.{frame_idx:06d}.lidar_raw.npz"
    with tarfile.open(tar_path) as tar_handle:
        file_obj = tar_handle.extractfile(member_name)
        if file_obj is None:
            raise FileNotFoundError(f"{member_name} not found in {tar_path}")
        with file_obj:
            payload = np.load(file_obj)
            return payload["xyz"].astype(np.float32)


def nearest_stats(src: np.ndarray, dst: np.ndarray) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(dst).query(src, k=1)
    return {
        "mean_m": float(distances.mean()),
        "p50_m": float(np.quantile(distances, 0.50)),
        "p90_m": float(np.quantile(distances, 0.90)),
        "p99_m": float(np.quantile(distances, 0.99)),
        "max_m": float(distances.max()),
        "lt_1mm": int((distances < 0.001).sum()),
        "lt_1cm": int((distances < 0.01).sum()),
        "lt_10cm": int((distances < 0.1).sum()),
    }


def frame_summary(
    frame_idx: int,
    rds_suffix: int | None,
    frame,
    args: argparse.Namespace,
    output_dir: Path,
    tar_path: Path | None,
) -> dict[str, Any]:
    range_images, camera_projections, range_image_top_pose = parse_official_range_images(frame)
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    top_order_idx = [idx for idx, c in enumerate(calibrations) if c.name == dataset_pb2.LaserName.TOP][0]

    points_r1, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=0
    )
    points_r2, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1
    )
    top_points_r1 = points_r1[top_order_idx].astype(np.float32)
    top_points_r2 = points_r2[top_order_idx].astype(np.float32)
    top_points_both = np.concatenate([top_points_r1, top_points_r2], axis=0)

    top_ri_r1 = matrix_float_to_numpy(range_images[dataset_pb2.LaserName.TOP][0])
    top_ri_r2 = matrix_float_to_numpy(range_images[dataset_pb2.LaserName.TOP][1])

    lasers = []
    for calib, p1, p2 in zip(calibrations, points_r1, points_r2):
        ri1 = matrix_float_to_numpy(range_images[calib.name][0])
        ri2 = matrix_float_to_numpy(range_images[calib.name][1])
        lasers.append(
            {
                "name": laser_name(calib.name),
                "range_image_shape": list(ri1.shape),
                "return1_valid": int((ri1[..., 0] > 0).sum()),
                "return2_valid": int((ri2[..., 0] > 0).sum()),
                "return1_points": int(p1.shape[0]),
                "return2_points": int(p2.shape[0]),
                "beam_inclinations_count": len(calib.beam_inclinations),
                "beam_inclination_min": float(calib.beam_inclination_min),
                "beam_inclination_max": float(calib.beam_inclination_max),
            }
        )

    result: dict[str, Any] = {
        "frame_index": frame_idx,
        "tfrecord_frame_index": frame_idx,
        "rds_frame_suffix": rds_suffix,
        "timestamp_micros": int(frame.timestamp_micros),
        "top_return1_range_shape": list(top_ri_r1.shape),
        "top_return1_valid": int((top_ri_r1[..., 0] > 0).sum()),
        "top_return2_valid": int((top_ri_r2[..., 0] > 0).sum()),
        "top_return1_points": int(top_points_r1.shape[0]),
        "top_return2_points": int(top_points_r2.shape[0]),
        "top_return1_plus_return2_points": int(top_points_both.shape[0]),
        "lasers": lasers,
    }

    if tar_path is not None:
        if rds_suffix is None:
            raise ValueError("rds_suffix is required when comparing rds_hq raw data")
        raw_xyz = load_rds_raw_xyz(tar_path, args.segment_key, rds_suffix)
        result["rds_raw_xyz_count"] = int(raw_xyz.shape[0])
        result["rds_raw_to_official_top_return1"] = nearest_stats(raw_xyz, top_points_r1)
        result["rds_raw_to_official_top_return1_plus_return2"] = nearest_stats(raw_xyz, top_points_both)
        result["official_top_return1_to_rds_raw"] = nearest_stats(top_points_r1, raw_xyz)

    if args.save_ply:
        write_ascii_ply(
            output_dir / "ply" / f"{args.segment_key}_{frame_idx:06d}_top_return1.ply",
            deterministic_subsample(top_points_r1, args.ply_max_points),
        )
        write_ascii_ply(
            output_dir / "ply" / f"{args.segment_key}_{frame_idx:06d}_top_return1_return2.ply",
            deterministic_subsample(top_points_both, args.ply_max_points),
        )

    if args.save_preview:
        save_preview_png(
            output_dir / "preview" / f"{args.segment_key}_{frame_idx:06d}_official_top.png",
            top_ri_r1[..., 0],
            top_points_both,
            f"{args.segment_key} frame {frame_idx:06d}",
        )

    if args.save_top_range_npz:
        npz_path = (
            output_dir
            / "range_npz"
            / f"{args.segment_key}_{frame_idx:06d}_top_{args.top_range_return_mode}_{args.top_range_output_width}.npz"
        )
        result["saved_top_range_npz"] = save_top_range_npz(
            npz_path,
            top_ri_r1=top_ri_r1,
            top_ri_r2=top_ri_r2,
            frame_idx=frame_idx,
            rds_suffix=rds_suffix,
            frame=frame,
            args=args,
        )

    if args.save_top_raw_range_npz:
        raw_npz_path = output_dir / "range_npz" / f"{args.segment_key}_{frame_idx:06d}_top_raw_range_image.npz"
        result["saved_top_raw_range_npz"] = save_top_raw_range_npz(
            raw_npz_path,
            top_ri_r1=top_ri_r1,
            top_ri_r2=top_ri_r2,
            frame_idx=frame_idx,
            rds_suffix=rds_suffix,
            frame=frame,
            args=args,
        )
    return result


def main() -> None:
    args = parse_args()
    input_path = tfrecord_path(args)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Waymo TFRecord: {input_path}")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/waymo_official_lidar") / args.split / args.segment_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = selected_frame_indices(args)
    tar_path = rds_tar_path(args) if args.compare_rds_raw else None
    if tar_path is not None and not tar_path.exists():
        raise FileNotFoundError(f"Missing rds_hq raw tar: {tar_path}")

    frames = []
    for frame_idx, frame in iter_selected_frames(input_path, frame_indices):
        rds_suffix = rds_frame_suffix(args, frame_idx) if args.compare_rds_raw else None
        suffix_note = f" rds_suffix={rds_suffix:06d}" if rds_suffix is not None else ""
        print(f"[official-waymo] frame {frame_idx:06d}{suffix_note}", flush=True)
        frames.append(frame_summary(frame_idx, rds_suffix, frame, args, output_dir, tar_path))

    if len(frames) != len(frame_indices):
        raise RuntimeError(f"Loaded {len(frames)} frames, expected {len(frame_indices)} from {input_path}")

    totals = {
        "top_return1_valid": int(sum(frame["top_return1_valid"] for frame in frames)),
        "top_return2_valid": int(sum(frame["top_return2_valid"] for frame in frames)),
        "top_return1_plus_return2_points": int(sum(frame["top_return1_plus_return2_points"] for frame in frames)),
    }
    if args.compare_rds_raw:
        totals["rds_raw_xyz_count"] = int(sum(frame["rds_raw_xyz_count"] for frame in frames))

    summary = {
        "raw_waymo_tfrecord": str(input_path),
        "rds_lidar_tar": str(tar_path) if tar_path is not None else None,
        "split": args.split,
        "segment_key": args.segment_key,
        "frame_indices": frame_indices,
        "tfrecord_frame_indices": frame_indices,
        "rds_frame_suffix_offset": args.rds_frame_suffix_offset,
        "rds_frame_suffix_step": args.rds_frame_suffix_step,
        "num_frames": len(frames),
        "totals": totals,
        "frames": frames,
    }
    summary_path = output_dir / "official_lidar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
