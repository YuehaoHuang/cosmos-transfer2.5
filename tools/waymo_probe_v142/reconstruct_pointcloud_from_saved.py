#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from common import load_npz, make_absolute, reconstruct_points_from_saved, scalar_str, write_point_cloud_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct point clouds only from saved metadata/.npz and lidar/.npz.")
    parser.add_argument("--metadata_path", type=Path, required=True, help="Path to metadata/<segment_id>.npz.")
    parser.add_argument("--lidar_path", type=Path, required=True, help="Path to lidar/<segment_id>.npz.")
    parser.add_argument("--output_root", type=Path, required=True, help="Where to save reconstructed points.")
    parser.add_argument("--save_ply", type=int, default=1, choices=[0, 1], help="Save per-frame PLY files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_npz(make_absolute(args.metadata_path))
    lidar = load_npz(make_absolute(args.lidar_path))
    output_root = make_absolute(args.output_root)

    segment_id = scalar_str(metadata["segment_id"])
    segment_output_root = output_root / segment_id
    segment_output_root.mkdir(parents=True, exist_ok=True)

    points_per_frame = reconstruct_points_from_saved(metadata, lidar)
    frame_counts = []
    for frame_idx, points in enumerate(points_per_frame):
        frame_counts.append(int(points.shape[0]))
        np.save(segment_output_root / f"frame_{frame_idx:05d}_points.npy", points.astype(np.float32))
        if args.save_ply:
            write_point_cloud_ply(segment_output_root / f"frame_{frame_idx:05d}.ply", points.astype(np.float32))

    summary = {
        "segment_id": segment_id,
        "num_frames": len(points_per_frame),
        "frame_point_counts": frame_counts,
        "output_root": str(segment_output_root),
    }
    (segment_output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
