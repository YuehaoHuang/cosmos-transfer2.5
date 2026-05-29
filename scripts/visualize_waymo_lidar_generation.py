#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Render generated Waymo LiDAR rangemap videos as point-cloud videos.

The Transfer2 inference path saves decoder output as an RGB rangemap mp4. This
utility decodes that mp4 back to metric range, builds Waymo TOP LiDAR rays, and
uses the same point-cloud renderer used by the Cosmos-Drive-Dreams tokenizer
workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from evaluate_waymo_lidar_rangemap_generation import layout_masks, video_to_range
from smoke_waymo_lidar_wan21_vae import load_raw_range_maps, make_downsampled_ray_directions, prepend_lidar_utils_repo
from visualize_decoded_lidar_waymo import (
    save_point_cloud_video,
    transform_points_to_vehicle_frame,
    valid_range_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-video", required=True, help="Generated rangemap mp4 from Transfer2 inference.")
    parser.add_argument(
        "--gt-video", default=None, help="Optional GT rangemap mp4. If omitted, try metadata/dataset/raw tar."
    )
    parser.add_argument("--layout-video", default=None, help="Optional rangemap_layout mp4 used to build valid mask.")
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional inference metadata JSON. Defaults to same stem as generated mp4.",
    )
    parser.add_argument("--dataset-dir", default=None, help="Dataset dir containing videos/ and rangemap_layout/.")
    parser.add_argument("--sample-key", default=None, help="Dataset sample key. Inferred from metadata when possible.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--repeat-row", type=int, default=11)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=1280)
    parser.add_argument("--decode-channel-mode", default="mean", choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--valid-min-offset-m", type=float, default=0.25)
    parser.add_argument("--near-buffer", type=float, default=0.1)
    parser.add_argument("--far-buffer", type=float, default=0.1)
    parser.add_argument(
        "--raw-valid-mode",
        default="preprocess",
        choices=["preprocess", "metric"],
        help="preprocess uses range > 0 occupancy; metric applies min/max near/far buffers.",
    )
    parser.add_argument(
        "--generated-valid-mode",
        default="predicted",
        choices=["predicted", "matched_gt", "layout"],
        help=(
            "predicted renders generated points from decoded generated occupancy; "
            "matched_gt reuses GT occupancy; layout reuses the layout control occupancy."
        ),
    )
    parser.add_argument("--layout-occupancy-threshold", type=int, default=90)
    parser.add_argument("--layout-edge-threshold", type=int, default=90)

    parser.add_argument("--raw-lidar-root", default="/team/hyh/data/rds_hq_waymo")
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--segment-key", default=None)
    parser.add_argument("--lidar-frame-indices", default=None, help="Comma-separated raw LiDAR frame indices.")
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--pad-lidar-last", action="store_true")
    parser.add_argument("--use-raw-rays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=1280)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=1)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--full-width", type=int, default=1280)
    parser.add_argument("--crop-mode", default="none", choices=["center", "left", "none"])

    parser.add_argument("--vis-pcd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--display-frame", default="vehicle", choices=["lidar", "vehicle"])
    parser.add_argument("--camera-view", default="front_view", choices=["front_view", "top_down_view"])
    parser.add_argument("--pcd-width", type=int, default=1280)
    parser.add_argument("--pcd-height", type=int, default=720)
    parser.add_argument("--pcd-max-points", type=int, default=70000)
    parser.add_argument("--pcd-renderer", default="auto", choices=["auto", "plotly", "raster"])
    parser.add_argument("--pcd-workers", type=int, default=1)
    parser.add_argument("--lidar-tokenizer-repo", default="/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--gt-label", default="GT")
    parser.add_argument("--prediction-label", default="generated")
    parser.add_argument("--save-ply", action="store_true", help="Also write generated point clouds as ASCII PLY files.")
    parser.add_argument(
        "--ply-max-points", type=int, default=0, help="Deterministic cap per frame; 0 keeps all valid points."
    )
    return parser.parse_args()


def read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"Expected metadata dict at {path}, got {type(data).__name__}.")
    return data


def infer_metadata_path(generated_video: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidate = generated_video.with_suffix(".json")
    return candidate if candidate.exists() else None


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path | str | None]:
    generated_video = Path(args.generated_video)
    metadata_path = infer_metadata_path(generated_video, args.metadata_json)
    metadata = read_metadata(metadata_path)

    sample_key = args.sample_key or metadata.get("sample_key")
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    gt_video = Path(args.gt_video) if args.gt_video else None
    if gt_video is None and metadata.get("video_path"):
        gt_video = Path(str(metadata["video_path"]))
    if gt_video is None and dataset_dir is not None and sample_key:
        gt_video = dataset_dir / "videos" / f"{sample_key}.mp4"
    if gt_video is not None and not gt_video.exists():
        raise FileNotFoundError(f"Missing GT video: {gt_video}")

    layout_video = Path(args.layout_video) if args.layout_video else None
    if layout_video is None and metadata.get("control_path"):
        layout_video = Path(str(metadata["control_path"]))
    if layout_video is None and dataset_dir is not None and sample_key:
        layout_video = dataset_dir / "rangemap_layout" / f"{sample_key}.mp4"
    if layout_video is not None and not layout_video.exists():
        raise FileNotFoundError(f"Missing layout video: {layout_video}")

    name = args.name or generated_video.stem
    output_dir = Path(args.output_dir) if args.output_dir else generated_video.parent / "point_cloud_vis"

    return {
        "generated_video": generated_video,
        "metadata_path": metadata_path,
        "sample_key": sample_key,
        "gt_video": gt_video,
        "layout_video": layout_video,
        "output_dir": output_dir,
        "name": name,
    }


def clip_shape(range_m: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return range_m[:, : args.target_height, : args.target_width]


def decode_video(path: Path, args: argparse.Namespace) -> np.ndarray:
    return clip_shape(video_to_range(path, args), args)


def load_mainline_raw_gt_and_rays(
    args: argparse.Namespace,
    *,
    sample_key: str,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.downsample_factor_row != 1 or args.downsample_factor_col != 1:
        raise ValueError("Raw-online point-cloud visualization requires downsample factors row=1 and col=1.")
    if args.lidar_frame_indices:
        frame_indices = [int(value) for value in args.lidar_frame_indices.split(",") if value.strip()]
        if not frame_indices or frame_indices != list(range(frame_indices[0], frame_indices[0] + n_frames)):
            raise ValueError("--lidar-frame-indices must specify one contiguous generated window")
        frame_start = frame_indices[0]
    else:
        _segment_key, separator, chunk_id = sample_key.rpartition("_")
        if not separator or not chunk_id.isdigit():
            raise ValueError(f"Raw-online sample key must end with a numeric chunk index: {sample_key}")
        frame_start = int(chunk_id) * args.lidar_chunk_stride_frames
    segment_key = args.segment_key or sample_key.rpartition("_")[0]
    tar_path = Path(args.raw_lidar_root) / args.split / "lidar_raw" / f"{segment_key}.tar"
    prepend_lidar_utils_repo(args.lidar_tokenizer_repo)
    raw_range, _ = load_raw_range_maps(
        tar_path,
        frame_start=frame_start,
        num_frames=n_frames,
        pad_last=args.pad_lidar_last,
        n_rows=args.native_n_rows,
        n_cols=args.native_n_cols,
        max_projection_range=args.projection_max_range,
    )
    expected_shape = (args.target_height, args.target_width)
    if tuple(raw_range.shape[1:]) != expected_shape:
        raise ValueError(f"Raw projected GT shape {raw_range.shape[1:]} does not match expected {expected_shape}")
    if args.raw_valid_mode == "preprocess":
        valid_mask = raw_range > 0
    else:
        valid_mask = valid_range_mask(raw_range, args.min_range, args.max_range, args.near_buffer, args.far_buffer)
    ray_args = SimpleNamespace(
        downsample_factor_row=1,
        downsample_factor_col=1,
        downsample_method=args.downsample_method,
    )
    rays = make_downsampled_ray_directions(raw_range, ray_args)
    return (
        raw_range.astype(np.float32, copy=False),
        valid_mask.astype(bool, copy=False),
        rays.astype(np.float32, copy=False),
    )


def resolve_gt_mask_and_rays(
    args: argparse.Namespace,
    resolved: dict[str, Path | str | None],
    *,
    generated_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    sample_key = resolved["sample_key"]
    if args.use_raw_rays:
        if not sample_key:
            raise ValueError("--sample-key or inference metadata is required for raw-online visualization")
        raw_gt, raw_valid, raw_rays = load_mainline_raw_gt_and_rays(
            args,
            sample_key=str(sample_key),
            n_frames=generated_range.shape[0],
        )
        return raw_gt, raw_valid, raw_rays, "raw_lidar_tar"

    gt_video = resolved["gt_video"]
    if gt_video is None:
        raise ValueError(
            "--gt-video is required with --no-use-raw-rays; generated output cannot be used as its own GT."
        )
    gt_range = decode_video(Path(gt_video), args)

    layout_video = resolved["layout_video"]
    if layout_video is not None:
        valid, _edges = layout_masks(Path(layout_video), args)
        valid_mask = clip_shape(valid, args).astype(bool)
    else:
        valid_mask = valid_range_mask(gt_range, args.min_range, args.max_range, args.near_buffer, args.far_buffer)
    prepend_lidar_utils_repo(args.lidar_tokenizer_repo)
    ray_args = SimpleNamespace(
        downsample_factor_row=1, downsample_factor_col=1, downsample_method=args.downsample_method
    )
    rays = make_downsampled_ray_directions(np.zeros_like(generated_range), ray_args)
    return gt_range, valid_mask, rays, "gt_video"


def make_generated_valid_mask(
    generated_range: np.ndarray,
    gt_valid_mask: np.ndarray,
    layout_valid_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if args.generated_valid_mode == "matched_gt":
        return gt_valid_mask.copy(), "matched_gt"
    if args.generated_valid_mode == "layout":
        if layout_valid_mask is None:
            raise ValueError("--generated-valid-mode layout requires a layout video from --layout-video or metadata")
        return layout_valid_mask.copy(), "layout_occupancy_threshold"
    valid = np.isfinite(generated_range) & (generated_range > (args.min_range + args.valid_min_offset_m))
    return valid.astype(bool, copy=False), "predicted_range_threshold"


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    stride = int(math.ceil(points.shape[0] / max_points))
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


def save_generated_ply(
    output_dir: Path,
    name: str,
    generated_range: np.ndarray,
    valid_mask: np.ndarray,
    ray_directions: np.ndarray,
    args: argparse.Namespace,
) -> list[str]:
    paths: list[str] = []
    ply_dir = output_dir / "point_cloud_ply" / name
    for frame_idx in range(generated_range.shape[0]):
        rays = ray_directions[frame_idx] if ray_directions.ndim == 4 else ray_directions
        mask = valid_mask[frame_idx]
        points = rays[mask] * generated_range[frame_idx][mask, None]
        if args.display_frame == "vehicle":
            points = transform_points_to_vehicle_frame(points)
        points = deterministic_subsample(points, args.ply_max_points)
        path = ply_dir / f"{name}_{frame_idx:04d}.ply"
        write_ascii_ply(path, points)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    resolved = resolve_inputs(args)
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_range = decode_video(Path(resolved["generated_video"]), args)
    gt_range, valid_mask, ray_directions, gt_source = resolve_gt_mask_and_rays(
        args,
        resolved,
        generated_range=generated_range,
    )
    layout_valid_mask = None
    if args.generated_valid_mode == "layout":
        layout_video = resolved["layout_video"]
        if layout_video is None:
            raise ValueError("--generated-valid-mode layout requires a layout video from --layout-video or metadata")
        layout_valid, _edges = layout_masks(Path(layout_video), args)
        layout_valid_mask = clip_shape(layout_valid, args).astype(bool)

    min_frame_inputs = [generated_range.shape[0], gt_range.shape[0], valid_mask.shape[0]]
    if layout_valid_mask is not None:
        min_frame_inputs.append(layout_valid_mask.shape[0])
    min_frames = min(min_frame_inputs)
    generated_range = generated_range[:min_frames]
    gt_range = gt_range[:min_frames]
    valid_mask = valid_mask[:min_frames]
    if layout_valid_mask is not None:
        layout_valid_mask = layout_valid_mask[:min_frames]
    if ray_directions.ndim == 4:
        ray_directions = ray_directions[:min_frames]

    generated_valid_mask, generated_valid_source = make_generated_valid_mask(
        generated_range,
        valid_mask,
        layout_valid_mask,
        args,
    )
    gt_for_vis = np.where(valid_mask, gt_range, 0.0)
    generated_for_vis = np.where(generated_valid_mask, generated_range, 0.0)
    name = str(resolved["name"])

    pcd_path = None
    renderer = None
    if args.vis_pcd:
        pcd_path = output_dir / "point_cloud" / f"{name}.mp4"
        renderer = save_point_cloud_video(
            gt_for_vis,
            generated_for_vis,
            valid_mask,
            pcd_path,
            gt_label=f"{args.gt_label}: {gt_source}",
            prediction_label=args.prediction_label,
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
            pred_valid_mask=generated_valid_mask,
        )

    ply_paths = (
        save_generated_ply(output_dir, name, generated_for_vis, generated_valid_mask, ray_directions, args)
        if args.save_ply
        else []
    )
    summary = {
        "generated_video": str(resolved["generated_video"]),
        "gt_video": str(resolved["gt_video"]) if resolved["gt_video"] is not None else None,
        "layout_video": str(resolved["layout_video"]) if resolved["layout_video"] is not None else None,
        "metadata_json": str(resolved["metadata_path"]) if resolved["metadata_path"] is not None else None,
        "sample_key": resolved["sample_key"],
        "shape_frames_rows_cols": list(generated_range.shape),
        "gt_source": gt_source,
        "valid_pixel_count": int(valid_mask.sum()),
        "gt_valid_pixel_count": int(valid_mask.sum()),
        "generated_valid_pixel_count": int(generated_valid_mask.sum()),
        "matched_valid_pixel_count": int((valid_mask & generated_valid_mask).sum()),
        "generated_extra_pixel_count": int((generated_valid_mask & ~valid_mask).sum()),
        "generated_missing_pixel_count": int((valid_mask & ~generated_valid_mask).sum()),
        "generated_valid_mode": args.generated_valid_mode,
        "generated_valid_source": generated_valid_source,
        "point_cloud_video": str(pcd_path) if pcd_path is not None else None,
        "point_cloud_renderer": renderer,
        "display_frame": args.display_frame,
        "camera_view": args.camera_view,
        "raw_valid_mode": args.raw_valid_mode,
        "ply_files": ply_paths,
    }
    summary_path = output_dir / f"{name}_point_cloud_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
