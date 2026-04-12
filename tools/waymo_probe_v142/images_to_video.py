#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from common import load_npz, make_absolute, scalar_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert lidar images[T,3,H,W] into an mp4 video.")
    parser.add_argument("--lidar_path", type=Path, required=True, help="Path to lidar/<segment_id>.npz.")
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="MP4 output path.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale factor for easier viewing.",
    )
    return parser.parse_args()


def frame_to_uint8(frame_chw: np.ndarray, scale: int) -> np.ndarray:
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
    frame_uint8 = (np.clip(frame_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    if scale > 1:
        height, width = frame_uint8.shape[:2]
        frame_uint8 = np.asarray(
            Image.fromarray(frame_uint8).resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
        )
    return frame_uint8


def main() -> None:
    args = parse_args()
    lidar_path = make_absolute(args.lidar_path)
    output_path = make_absolute(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lidar = load_npz(lidar_path)
    images = lidar["images"].astype(np.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(f"Expected images with shape [T,3,H,W], got {images.shape}")

    frames = [frame_to_uint8(images[t], scale=args.scale) for t in range(images.shape[0])]
    imageio.mimsave(output_path, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
