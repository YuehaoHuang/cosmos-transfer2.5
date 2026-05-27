#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate generated Waymo LiDAR range-map videos against GT and layout control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smoke_waymo_lidar_wan21_vae import decode_reconstruction_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-video", required=True)
    parser.add_argument("--gt-video", required=True)
    parser.add_argument("--layout-video", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--repeat-row", type=int, default=11)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=1280)
    parser.add_argument("--edge-threshold-m", type=float, default=1.0)
    parser.add_argument("--valid-min-offset-m", type=float, default=0.25)
    parser.add_argument("--layout-occupancy-threshold", type=int, default=90)
    parser.add_argument("--layout-edge-threshold", type=int, default=90)
    parser.add_argument("--decode-channel-mode", default="mean", choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    return parser.parse_args()


def read_video_uint8(path: Path) -> np.ndarray:
    import mediapy as media

    frames = media.read_video(str(path))
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0 if frames.max() <= 1.0 else frames, 0, 255).astype(np.uint8)
    return frames


def video_to_normalized_tensor(frames: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(frames.astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor / 127.5 - 1.0


def video_to_range(path: Path, args: argparse.Namespace) -> np.ndarray:
    frames = read_video_uint8(path)
    tensor = video_to_normalized_tensor(frames)
    return decode_reconstruction_range(
        tensor,
        args=args,
        target_height=args.target_height,
        target_width=args.target_width,
    )


def edge_mask(range_maps: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    edge = np.zeros_like(valid, dtype=bool)
    safe = np.where(valid, range_maps, 0.0).astype(np.float32)
    dx = np.abs(safe[:, :, 1:] - safe[:, :, :-1])
    dy = np.abs(safe[:, 1:, :] - safe[:, :-1, :])
    both_x = valid[:, :, 1:] & valid[:, :, :-1]
    both_y = valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :, 1:] |= (dx > threshold) & both_x
    edge[:, :, :-1] |= (dx > threshold) & both_x
    edge[:, 1:, :] |= (dy > threshold) & both_y
    edge[:, :-1, :] |= (dy > threshold) & both_y
    edge[:, :, 1:] |= valid[:, :, 1:] != valid[:, :, :-1]
    edge[:, :, :-1] |= valid[:, :, 1:] != valid[:, :, :-1]
    edge[:, 1:, :] |= valid[:, 1:, :] != valid[:, :-1, :]
    edge[:, :-1, :] |= valid[:, 1:, :] != valid[:, :-1, :]
    return edge


def binary_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def layout_masks(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    frames = read_video_uint8(path)
    sampled = frames[:, args.repeat_row // 2 :: args.repeat_row, args.repeat_col // 2 :: args.repeat_col, :]
    sampled = sampled[:, : args.target_height, : args.target_width, :]
    occupancy = sampled[..., 0] > args.layout_occupancy_threshold
    edges = sampled[..., 1] > args.layout_edge_threshold
    return occupancy, edges


def main() -> None:
    args = parse_args()
    generated_range = video_to_range(Path(args.generated_video), args)
    gt_range = video_to_range(Path(args.gt_video), args)

    if args.layout_video:
        gt_valid, gt_edges = layout_masks(Path(args.layout_video), args)
    else:
        gt_valid = gt_range > (args.min_range + args.valid_min_offset_m)
        gt_edges = edge_mask(gt_range, gt_valid, args.edge_threshold_m)

    pred_valid = generated_range > (args.min_range + args.valid_min_offset_m)
    pred_edges = edge_mask(generated_range, pred_valid, args.edge_threshold_m)

    valid = gt_valid & np.isfinite(generated_range) & np.isfinite(gt_range)
    diff = generated_range - gt_range
    valid_diff = diff[valid]
    metrics = {
        "generated_video": args.generated_video,
        "gt_video": args.gt_video,
        "layout_video": args.layout_video,
        "valid_pixel_count": int(valid.sum()),
        "range_mae_m": float(np.mean(np.abs(valid_diff))) if valid_diff.size else None,
        "range_rmse_m": float(np.sqrt(np.mean(valid_diff**2))) if valid_diff.size else None,
        "range_bias_m": float(np.mean(valid_diff)) if valid_diff.size else None,
        "occupancy": binary_scores(pred_valid, gt_valid),
        "edge": binary_scores(pred_edges, gt_edges),
    }

    text = json.dumps(metrics, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
