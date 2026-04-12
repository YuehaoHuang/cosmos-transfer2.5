#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from common import (
    DEFAULT_INTENSITY_MAX,
    DEFAULT_INTENSITY_MIN,
    DEFAULT_RANGE_MAX,
    DEFAULT_RANGE_MIN,
    build_training_images,
    compute_top_beam_inclinations,
    get_top_calibration,
    import_waymo_modules,
    make_absolute,
    save_npz,
    segment_id_from_path,
)

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a Waymo segment into metadata/.npz and lidar/.npz files.")
    parser.add_argument("--tfrecord_path", type=Path, required=True, help="Path to a single Waymo TFRecord segment.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data2/waymo_lidar_tokenizer"),
        help="Dataset root containing metadata/, lidar/, and previews/.",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on frames to extract.")
    parser.add_argument("--laser_name", type=str, default="TOP", choices=["TOP"], help="Only TOP is supported.")
    parser.add_argument("--ri_index", type=int, default=0, choices=[0], help="Only first return is supported.")
    parser.add_argument("--split", type=str, default="training", help="Split name stored in metadata.")
    parser.add_argument("--range_min", type=float, default=DEFAULT_RANGE_MIN, help="Range normalization minimum.")
    parser.add_argument("--range_max", type=float, default=DEFAULT_RANGE_MAX, help="Range normalization maximum.")
    parser.add_argument(
        "--intensity_min", type=float, default=DEFAULT_INTENSITY_MIN, help="Intensity normalization minimum."
    )
    parser.add_argument(
        "--intensity_max", type=float, default=DEFAULT_INTENSITY_MAX, help="Intensity normalization maximum."
    )
    return parser.parse_args()


def extract_segment(
    tfrecord_path: Path,
    output_root: Path,
    max_frames: int | None,
    split: str,
    laser_name: str = "TOP",
    ri_index: int = 0,
    range_min: float = DEFAULT_RANGE_MIN,
    range_max: float = DEFAULT_RANGE_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    intensity_max: float = DEFAULT_INTENSITY_MAX,
) -> dict:
    tf, dataset_pb2, frame_utils, range_image_utils, _ = import_waymo_modules()

    tfrecord_path = make_absolute(tfrecord_path)
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord does not exist: {tfrecord_path}")

    dataset_root = make_absolute(output_root)
    metadata_root = dataset_root / "metadata"
    lidar_root = dataset_root / "lidar"
    metadata_root.mkdir(parents=True, exist_ok=True)
    lidar_root.mkdir(parents=True, exist_ok=True)

    segment_id = segment_id_from_path(tfrecord_path)
    metadata_path = metadata_root / f"{segment_id}.npz"
    lidar_path = lidar_root / f"{segment_id}.npz"

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

    frame_indices = []
    timestamps_list = []
    pose_list = []
    frame_pose_list = []
    range_raw_list = []
    intensity_raw_list = []
    elongation_raw_list = []
    nlz_raw_list = []
    valid_mask_list = []
    top_pose_list = []

    context_name = None
    lidar_extrinsic = None
    beam_inclinations = None
    beam_inclination_minmax = None
    range_image_shape = None
    has_per_pixel_pose = False

    extracted = 0
    for frame_idx, raw_record in enumerate(dataset):
        if max_frames is not None and extracted >= max_frames:
            break

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))

        range_images, _, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        top_returns = range_images.get(dataset_pb2.LaserName.TOP)
        if not top_returns or len(top_returns) <= ri_index:
            continue

        calibration = get_top_calibration(frame, dataset_pb2)
        ri = top_returns[ri_index]
        range_image = np.asarray(ri.data, dtype=np.float32).reshape(ri.shape.dims)

        if context_name is None:
            context_name = frame.context.name
            lidar_extrinsic = np.reshape(np.array(calibration.extrinsic.transform, dtype=np.float32), [4, 4])
            beam_inclinations = compute_top_beam_inclinations(calibration, range_image.shape[0], range_image_utils)
            beam_inclination_minmax = np.array(
                [calibration.beam_inclination_min, calibration.beam_inclination_max], dtype=np.float32
            )
            range_image_shape = np.array(range_image.shape[:2], dtype=np.int32)

        per_pixel_pose = None
        if range_image_top_pose is not None and len(range_image_top_pose.data) > 0:
            per_pixel_pose = np.asarray(range_image_top_pose.data, dtype=np.float32).reshape(range_image_top_pose.shape.dims)
            has_per_pixel_pose = True

        range_raw = range_image[..., 0].astype(np.float32)
        intensity_raw = range_image[..., 1].astype(np.float32)
        elongation_raw = range_image[..., 2].astype(np.float32)
        nlz_raw = range_image[..., 3].astype(np.float32)
        valid_mask = (range_raw > 0).astype(np.float32)
        frame_pose = np.reshape(np.array(frame.pose.transform, dtype=np.float32), [4, 4])

        frame_indices.append(frame_idx)
        timestamps_list.append(frame.timestamp_micros)
        pose_list.append(frame_pose)
        frame_pose_list.append(frame_pose)
        range_raw_list.append(range_raw)
        intensity_raw_list.append(intensity_raw)
        elongation_raw_list.append(elongation_raw)
        nlz_raw_list.append(nlz_raw)
        valid_mask_list.append(valid_mask)
        top_pose_list.append(None if per_pixel_pose is None else per_pixel_pose.astype(np.float32))

        extracted += 1

    if extracted == 0:
        raise RuntimeError(f"No valid TOP range images were extracted from {tfrecord_path}")

    range_raw = np.stack(range_raw_list, axis=0)[:, None, :, :]
    intensity_raw = np.stack(intensity_raw_list, axis=0)[:, None, :, :]
    elongation_raw = np.stack(elongation_raw_list, axis=0)[:, None, :, :]
    nlz_raw = np.stack(nlz_raw_list, axis=0)[:, None, :, :]
    valid_mask = np.stack(valid_mask_list, axis=0)[:, None, :, :]
    images = build_training_images(
        range_raw=range_raw,
        intensity_raw=intensity_raw,
        valid_mask=valid_mask,
        range_min=range_min,
        range_max=range_max,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
    )

    if has_per_pixel_pose:
        top_pose_shape = next(pose.shape for pose in top_pose_list if pose is not None)
        range_image_top_pose_array = np.stack(
            [pose if pose is not None else np.zeros(top_pose_shape, dtype=np.float32) for pose in top_pose_list],
            axis=0,
        ).astype(np.float32)
    else:
        range_image_top_pose_array = np.empty((0,), dtype=np.float32)

    metadata_payload = {
        "context_name": np.array(context_name),
        "segment_id": np.array(segment_id),
        "split": np.array(split),
        "laser_name": np.array(laser_name),
        "return_index": np.array(ri_index, dtype=np.int32),
        "timestamps_list": np.asarray(timestamps_list, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "pose_list": np.stack(pose_list, axis=0).astype(np.float32),
        "frame_pose": np.stack(frame_pose_list, axis=0).astype(np.float32),
        "lidar_extrinsic": lidar_extrinsic.astype(np.float32),
        "beam_inclinations": beam_inclinations.astype(np.float32),
        "beam_inclination_minmax": beam_inclination_minmax.astype(np.float32),
        "range_image_shape": range_image_shape.astype(np.int32),
        "is_raw_range_image": np.array(True),
        "has_per_pixel_pose": np.array(has_per_pixel_pose),
        "range_image_top_pose": range_image_top_pose_array,
        "range_norm_type": np.array("linear_clip"),
        "range_min": np.array(range_min, dtype=np.float32),
        "range_max": np.array(range_max, dtype=np.float32),
        "intensity_norm_type": np.array("linear_clip"),
        "intensity_min": np.array(intensity_min, dtype=np.float32),
        "intensity_max": np.array(intensity_max, dtype=np.float32),
    }

    lidar_payload = {
        "images": images.astype(np.float32),
        "valid_mask": valid_mask.astype(np.float32),
        "range_raw": range_raw.astype(np.float32),
        "intensity_raw": intensity_raw.astype(np.float32),
        "elongation_raw": elongation_raw.astype(np.float32),
        "nlz_raw": nlz_raw.astype(np.float32),
    }

    save_npz(metadata_path, metadata_payload)
    save_npz(lidar_path, lidar_payload)

    summary = {
        "wod_version": "1.4.2",
        "segment_id": segment_id,
        "split": split,
        "laser_name": laser_name,
        "return_index": ri_index,
        "tfrecord_path": str(tfrecord_path),
        "metadata_path": str(metadata_path),
        "lidar_path": str(lidar_path),
        "num_frames": extracted,
        "range_image_shape": range_image_shape.tolist(),
        "has_per_pixel_pose": has_per_pixel_pose,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = extract_segment(
        tfrecord_path=args.tfrecord_path,
        output_root=args.output_root,
        max_frames=args.max_frames,
        split=args.split,
        laser_name=args.laser_name,
        ri_index=args.ri_index,
        range_min=args.range_min,
        range_max=args.range_max,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
