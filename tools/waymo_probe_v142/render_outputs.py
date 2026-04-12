#!/usr/bin/env python3
import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from common import import_plotly, load_npz, make_absolute, reconstruct_points_from_saved, scalar_str, write_point_cloud_ply


CAMERA_POSITION = {
    "eye": {"x": -0.3, "y": 0.0, "z": 0.2},
    "center": {"x": 0.1, "y": 0.0, "z": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render saved metadata/lidar segment files into previews.")
    parser.add_argument("--metadata_path", type=Path, required=True, help="Path to metadata/<segment_id>.npz.")
    parser.add_argument("--lidar_path", type=Path, required=True, help="Path to lidar/<segment_id>.npz.")
    parser.add_argument("--preview_root", type=Path, required=True, help="Preview directory for this segment.")
    parser.add_argument("--point_size", type=float, default=0.3, help="Point size for point cloud rendering.")
    parser.add_argument("--point_range", type=float, default=80.0, help="Axis range for point cloud rendering.")
    return parser.parse_args()


def normalize_valid_channel(channel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    image = np.zeros(channel.shape + (3,), dtype=np.uint8)
    if not np.any(valid_mask):
        return image

    valid_values = channel[valid_mask]
    lo = float(valid_values.min())
    hi = float(valid_values.max())
    if hi <= lo:
        scaled = np.zeros_like(channel, dtype=np.float32)
    else:
        scaled = (channel.astype(np.float32) - lo) / (hi - lo)
    gray = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    image[valid_mask] = np.stack([gray[valid_mask], gray[valid_mask], gray[valid_mask]], axis=-1)
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def render_point_cloud(points: np.ndarray, point_size: float, point_range: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    go = import_plotly()
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=np.full(points.shape[0], "#ffb400"), opacity=1.0, line=dict(width=0)),
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-point_range, point_range], visible=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[-point_range, point_range], visible=False, showgrid=False, zeroline=False),
            zaxis=dict(range=[-point_range, point_range], visible=False, showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=CAMERA_POSITION,
        ),
        paper_bgcolor="rgb(0,0,0)",
        plot_bgcolor="rgb(0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
    )
    image_bytes = fig.to_image(format="png", width=1280, height=720)
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def render_segment(metadata_path: Path, lidar_path: Path, preview_root: Path, point_size: float = 0.3, point_range: float = 80.0) -> int:
    metadata = load_npz(make_absolute(metadata_path))
    lidar = load_npz(make_absolute(lidar_path))
    preview_root = make_absolute(preview_root)
    preview_root.mkdir(parents=True, exist_ok=True)

    segment_id = scalar_str(metadata["segment_id"])
    segment_preview_root = preview_root / segment_id
    segment_preview_root.mkdir(parents=True, exist_ok=True)

    points_per_frame = reconstruct_points_from_saved(metadata, lidar)
    range_raw = lidar["range_raw"]
    intensity_raw = lidar["intensity_raw"]
    valid_mask = lidar["valid_mask"] > 0.5

    for frame_idx, points in enumerate(points_per_frame):
        frame_dir = segment_preview_root / f"frame_{frame_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        range_png = normalize_valid_channel(range_raw[frame_idx, 0], valid_mask[frame_idx, 0])
        intensity_png = normalize_valid_channel(intensity_raw[frame_idx, 0], valid_mask[frame_idx, 0])
        valid_png = np.zeros(valid_mask[frame_idx, 0].shape + (3,), dtype=np.uint8)
        valid_png[valid_mask[frame_idx, 0]] = np.array([255, 255, 255], dtype=np.uint8)

        save_png(frame_dir / "rangemap_range.png", range_png)
        save_png(frame_dir / "rangemap_intensity.png", intensity_png)
        save_png(frame_dir / "rangemap_valid_mask.png", valid_png)
        np.save(frame_dir / "reconstructed_points.npy", points.astype(np.float32))
        write_point_cloud_ply(frame_dir / "reconstructed_points.ply", points.astype(np.float32))
        save_png(frame_dir / "pointcloud.png", render_point_cloud(points, point_size=point_size, point_range=point_range))
        print(f"Rendered {frame_dir}")

    return len(points_per_frame)


def main() -> None:
    args = parse_args()
    render_segment(
        metadata_path=args.metadata_path,
        lidar_path=args.lidar_path,
        preview_root=args.preview_root,
        point_size=args.point_size,
        point_range=args.point_range,
    )


if __name__ == "__main__":
    main()
