#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_RANGE_MIN = 0.0
DEFAULT_RANGE_MAX = 75.0
DEFAULT_INTENSITY_MIN = 0.0
DEFAULT_INTENSITY_MAX = 1.0


def make_absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(os.path.abspath(path))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def segment_id_from_path(path: Path) -> str:
    return path.name.replace(".tfrecord", "")


def import_waymo_modules():
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

    return tf, dataset_pb2, frame_utils, range_image_utils, transform_utils


def import_open3d():
    import open3d as o3d

    return o3d


def import_plotly():
    import plotly.graph_objects as go

    return go


def scalar_or_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def scalar_str(value) -> str:
    return str(scalar_or_value(value))


def scalar_bool(value) -> bool:
    return bool(scalar_or_value(value))


def get_top_calibration(frame, dataset_pb2):
    for calibration in frame.context.laser_calibrations:
        if calibration.name == dataset_pb2.LaserName.TOP:
            return calibration
    raise RuntimeError("TOP lidar calibration not found")


def compute_top_beam_inclinations(calibration, height: int, range_image_utils) -> np.ndarray:
    if len(calibration.beam_inclinations) == 0:
        import tensorflow as tf

        beam_inclinations = range_image_utils.compute_inclination(
            tf.constant([calibration.beam_inclination_min, calibration.beam_inclination_max]),
            height=height,
        ).numpy()
    else:
        beam_inclinations = np.array(calibration.beam_inclinations, dtype=np.float32)

    # Match the orientation used by Waymo frame_utils.
    return beam_inclinations[::-1].astype(np.float32)


def linear_normalize(channel: np.ndarray, valid_mask: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    normalized = np.clip(channel.astype(np.float32), min_value, max_value)
    denom = max(max_value - min_value, 1e-6)
    normalized = (normalized - min_value) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = normalized * valid_mask.astype(np.float32)
    return normalized.astype(np.float32)


def build_training_images(
    range_raw: np.ndarray,
    intensity_raw: np.ndarray,
    valid_mask: np.ndarray,
    range_min: float = DEFAULT_RANGE_MIN,
    range_max: float = DEFAULT_RANGE_MAX,
    intensity_min: float = DEFAULT_INTENSITY_MIN,
    intensity_max: float = DEFAULT_INTENSITY_MAX,
) -> np.ndarray:
    range_norm = linear_normalize(range_raw, valid_mask, range_min, range_max)
    intensity_norm = linear_normalize(intensity_raw, valid_mask, intensity_min, intensity_max)
    return np.concatenate([range_norm, intensity_norm, valid_mask.astype(np.float32)], axis=1)


def write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    o3d = import_open3d()
    if points.size == 0:
        raise RuntimeError(f"Point cloud is empty, refusing to write {path}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.tile(np.array([[1.0, 0.706, 0.0]], dtype=np.float64), (points.shape[0], 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), point_cloud)


def reconstruct_points_from_saved(metadata: dict, lidar: dict) -> list[np.ndarray]:
    tf, _, _, range_image_utils, transform_utils = import_waymo_modules()

    range_raw = lidar["range_raw"].astype(np.float32)
    beam_inclinations = metadata["beam_inclinations"].astype(np.float32)
    lidar_extrinsic = metadata["lidar_extrinsic"].astype(np.float32)
    frame_pose = metadata["frame_pose"].astype(np.float32)
    has_per_pixel_pose = scalar_bool(metadata["has_per_pixel_pose"])
    range_image_top_pose = metadata.get("range_image_top_pose")

    points_per_frame = []
    for frame_idx in range(range_raw.shape[0]):
        frame_range = range_raw[frame_idx, 0]
        range_tensor = tf.expand_dims(tf.convert_to_tensor(frame_range, dtype=tf.float32), axis=0)
        extrinsic_tensor = tf.expand_dims(tf.convert_to_tensor(lidar_extrinsic, dtype=tf.float32), axis=0)
        beam_tensor = tf.expand_dims(tf.convert_to_tensor(beam_inclinations, dtype=tf.float32), axis=0)

        pixel_pose_local = None
        frame_pose_local = None
        if has_per_pixel_pose and range_image_top_pose is not None and range_image_top_pose.size > 0:
            pose_tensor = tf.convert_to_tensor(range_image_top_pose[frame_idx], dtype=tf.float32)
            rotation = transform_utils.get_rotation_matrix(pose_tensor[..., 0], pose_tensor[..., 1], pose_tensor[..., 2])
            translation = pose_tensor[..., 3:]
            pixel_pose_local = transform_utils.get_transform(rotation, translation)
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(tf.convert_to_tensor(frame_pose[frame_idx], dtype=tf.float32), axis=0)

        cartesian = range_image_utils.extract_point_cloud_from_range_image(
            range_tensor,
            extrinsic_tensor,
            beam_tensor,
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        cartesian = tf.squeeze(cartesian, axis=0).numpy()
        valid_mask = frame_range > 0
        points_per_frame.append(cartesian[valid_mask].astype(np.float32))

    return points_per_frame
