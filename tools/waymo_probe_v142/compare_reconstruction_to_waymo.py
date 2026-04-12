#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from common import import_waymo_modules, load_npz, make_absolute, reconstruct_points_from_saved, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare offline point-cloud reconstruction from saved files against official Waymo online parsing."
    )
    parser.add_argument("--metadata_path", type=Path, required=True, help="Path to metadata/<segment_id>.npz.")
    parser.add_argument("--lidar_path", type=Path, required=True, help="Path to lidar/<segment_id>.npz.")
    parser.add_argument("--tfrecord_path", type=Path, required=True, help="Original TFRecord path for online reference.")
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Where to write the comparison summary JSON.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to compare.",
    )
    return parser.parse_args()


def get_online_top_points(tfrecord_path: Path, max_frames: int | None) -> list[np.ndarray]:
    tf, dataset_pb2, frame_utils, _, _ = import_waymo_modules()

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    points_per_frame = []
    for raw_record in dataset:
        if max_frames is not None and len(points_per_frame) >= max_frames:
            break

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))
        range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(
            frame
        )
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            keep_polar_features=False,
        )
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        top_index = next(i for i, calib in enumerate(calibrations) if calib.name == dataset_pb2.LaserName.TOP)
        points_per_frame.append(points[top_index].astype(np.float32))

    return points_per_frame


def compare_frame(offline_points: np.ndarray, online_points: np.ndarray) -> dict:
    count_match = offline_points.shape[0] == online_points.shape[0]
    compared_count = min(offline_points.shape[0], online_points.shape[0])

    result = {
        "offline_count": int(offline_points.shape[0]),
        "online_count": int(online_points.shape[0]),
        "count_match": bool(count_match),
        "compared_count": int(compared_count),
    }

    if compared_count == 0:
        result.update(
            {
                "max_abs_error": None,
                "mean_abs_error": None,
                "rmse": None,
            }
        )
        return result

    offline_xyz = offline_points[:compared_count, :3]
    online_xyz = online_points[:compared_count, :3]
    diff = offline_xyz - online_xyz
    abs_diff = np.abs(diff)

    result.update(
        {
            "max_abs_error": float(abs_diff.max()),
            "mean_abs_error": float(abs_diff.mean()),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    metadata_path = make_absolute(args.metadata_path)
    lidar_path = make_absolute(args.lidar_path)
    tfrecord_path = make_absolute(args.tfrecord_path)
    output_json = make_absolute(args.output_json)

    metadata = load_npz(metadata_path)
    lidar = load_npz(lidar_path)

    offline_points_per_frame = reconstruct_points_from_saved(metadata, lidar)
    if args.max_frames is not None:
        offline_points_per_frame = offline_points_per_frame[: args.max_frames]

    online_points_per_frame = get_online_top_points(tfrecord_path, max_frames=len(offline_points_per_frame))
    compared_frames = min(len(offline_points_per_frame), len(online_points_per_frame))

    frame_results = []
    for frame_idx in range(compared_frames):
        frame_result = compare_frame(offline_points_per_frame[frame_idx], online_points_per_frame[frame_idx])
        frame_result["frame_idx"] = frame_idx
        frame_results.append(frame_result)

    summary = {
        "metadata_path": str(metadata_path),
        "lidar_path": str(lidar_path),
        "tfrecord_path": str(tfrecord_path),
        "num_offline_frames": len(offline_points_per_frame),
        "num_online_frames": len(online_points_per_frame),
        "compared_frames": compared_frames,
        "all_counts_match": all(item["count_match"] for item in frame_results),
        "max_abs_error_overall": max((item["max_abs_error"] or 0.0) for item in frame_results) if frame_results else None,
        "mean_abs_error_overall": (
            float(np.mean([item["mean_abs_error"] for item in frame_results if item["mean_abs_error"] is not None]))
            if frame_results
            else None
        ),
        "rmse_overall": (
            float(np.mean([item["rmse"] for item in frame_results if item["rmse"] is not None])) if frame_results else None
        ),
        "frames": frame_results,
    }

    save_json(output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
