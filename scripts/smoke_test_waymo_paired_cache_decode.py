# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-sample smoke test for paired Waymo video/LiDAR latent cache.

This utility follows the same dataloader path as the paired cache writer, but
only processes one sample and immediately decodes both modalities for visual
inspection:

1. save a paired-cache-compatible `.pt` payload
2. save the raw five-view video mosaic
3. decode video latent back to a previewable comparison video
4. decode LiDAR latent back to tokenizer output

For `raw_downsample`, the video decode is a luma-only approximate inverse used
only for alignment checks: the placeholder encoder averages RGB before filling
16 latent channels, so color cannot be reconstructed. For `wan2pt1`, it uses the
real VAE decode path and should preserve color.

Recommended environment:
    conda activate cosmos-transfer2.5-merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mediapy as media
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw

from cosmos_transfer2._src.predict2_multiview.datasets.multiview import collate_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_waymo_video_lidar_one_way_expert import _crop_decoded_video  # noqa: E402
from train_waymo_video_to_lidar_baseline import (  # noqa: E402
    NORMAL_LIDAR_LATENT_HW,
    OnlineLidarS3Encoder,
    OnlineVideoEncoder,
    WAYMO_CAMERAS,
    build_waymo_dataset,
    maybe_resize_lidar,
)


CANONICAL_SAMPLE_KEY = "10203656353524179475_7625_000_7645_000_0"
VIEW_LABELS = {
    "pinhole_front": "front",
    "pinhole_front_left": "front_left",
    "pinhole_front_right": "front_right",
    "pinhole_side_left": "side_left",
    "pinhole_side_right": "side_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-key", default=CANONICAL_SAMPLE_KEY)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=144)
    parser.add_argument("--preview-fps", type=int, default=10)
    parser.add_argument("--save-dtype", default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--video-tokenizer", default="raw_downsample", choices=["raw_downsample", "wan2pt1"])
    parser.add_argument("--video-tokenizer-batch-size", type=int, default=1)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--lidar-tokenizer-repo", default="/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument(
        "--lidar-tokenizer-ckpt",
        default=(
            "/data2/checkpoints/posttraining/tokenizer/"
            "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/checkpoints/iter_000035500.pt"
        ),
    )
    parser.add_argument(
        "--lidar-tokenizer-config",
        default=(
            "/data2/checkpoints/posttraining/tokenizer/"
            "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/config.yaml"
        ),
    )
    parser.add_argument("--lidar-tokenizer-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--offload-lidar-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--local-frame-start", type=int, default=0)
    parser.add_argument("--num-video-frames", type=int, default=29)
    parser.add_argument("--pad-lidar-last", action="store_true")
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=2)
    parser.add_argument("--downsample-method", default="scatter_min")
    parser.add_argument("--repeat-row", type=int, default=4)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--lidar-crop-width", type=int, default=0)
    parser.add_argument("--crop-mode", default="none", choices=["center", "left", "none"])
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--latent-frames", type=int, default=8)
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=226)
    parser.add_argument("--allow-lidar-resize-for-smoke", action="store_true")
    return parser.parse_args()


def resolve_sample_index(dataset: Any, args: argparse.Namespace) -> int:
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(dataset):
            raise IndexError(f"sample-index {args.sample_index} out of range for dataset size {len(dataset)}")
        return args.sample_index

    if not hasattr(dataset, "sample_names"):
        raise AttributeError("Dataset does not expose sample_names; cannot resolve sample-key.")

    try:
        return dataset.sample_names.index(args.sample_key)
    except ValueError as exc:
        raise KeyError(f"sample-key {args.sample_key!r} not found in split {args.split}") from exc


def _tensor_to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return video.cpu()
    video = video.detach().cpu().float().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return video


def _label_image(image: torch.Tensor, label: str) -> torch.Tensor:
    pil = Image.fromarray(image.numpy())
    draw = ImageDraw.Draw(pil)
    draw.rectangle((4, 4, 12 + 8 * len(label), 26), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return torch.from_numpy(__import__("numpy").asarray(pil))


def _resize_views(video: torch.Tensor, width: int, height: int) -> torch.Tensor:
    video = video.float()
    t, v, h, w, c = video.shape
    video = rearrange(video, "t v h w c -> (t v) c h w")
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    return rearrange(video.round().clamp(0, 255).to(torch.uint8), "(t v) c h w -> t v h w c", t=t, v=v)


def _mosaic_multiview(video: torch.Tensor) -> list[torch.Tensor]:
    frames: list[torch.Tensor] = []
    for frame in video:
        tiles = []
        for view_idx, camera in enumerate(WAYMO_CAMERAS):
            tile = _label_image(frame[view_idx], VIEW_LABELS[camera])
            tiles.append(tile)
        frames.append(torch.cat(tiles, dim=1))
    return frames


def _mosaic_multiview_grid(video: torch.Tensor) -> list[torch.Tensor]:
    frames: list[torch.Tensor] = []
    for frame in video:
        tiles = []
        for view_idx, camera in enumerate(WAYMO_CAMERAS):
            tile = _label_image(frame[view_idx], VIEW_LABELS[camera])
            tiles.append(tile)
        blank = torch.zeros_like(tiles[0])
        top = torch.cat(tiles[:3], dim=1)
        bottom = torch.cat([tiles[3], tiles[4], blank], dim=1)
        frames.append(torch.cat([top, bottom], dim=0))
    return frames


def _save_multiview_video(
    raw_video: torch.Tensor,
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    layout: str = "row",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_resized = _resize_views(raw_video, width=width, height=height)
    if layout == "grid":
        frames = [frame.numpy() for frame in _mosaic_multiview_grid(raw_resized)]
    else:
        frames = [frame.numpy() for frame in _mosaic_multiview(raw_resized)]
    media.write_video(str(output_path), frames, fps=fps)


def _save_compare_video(
    raw_video: torch.Tensor,
    decoded_video: torch.Tensor,
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    video_tokenizer: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_resized = _resize_views(raw_video, width=width, height=height)
    decoded_resized = _resize_views(decoded_video, width=width, height=height)
    raw_mosaic = _mosaic_multiview(raw_resized)
    dec_mosaic = _mosaic_multiview(decoded_resized)

    frames = []
    separator = torch.zeros((6, raw_mosaic[0].shape[1], 3), dtype=torch.uint8)
    decoded_label = (
        "decoded_video:raw_downsample_luma"
        if video_tokenizer == "raw_downsample"
        else f"decoded_video:{video_tokenizer}"
    )
    for raw_frame, dec_frame in zip(raw_mosaic, dec_mosaic):
        top = _label_image(raw_frame, "raw_video")
        bottom = _label_image(dec_frame, decoded_label)
        frames.append(torch.cat([top, separator, bottom], dim=0).numpy())
    media.write_video(str(output_path), frames, fps=fps)


def _raw_video_to_multiview(video: torch.Tensor) -> torch.Tensor:
    video = video[0].cpu()
    frames_per_view = video.shape[1] // len(WAYMO_CAMERAS)
    return rearrange(video, "c (v t) h w -> t v h w c", v=len(WAYMO_CAMERAS), t=frames_per_view)


def _decode_raw_downsample_latent(video_latent: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    latent = video_latent.float().mean(dim=1, keepdim=True)
    per_view_latent_frames = latent.shape[2] // len(WAYMO_CAMERAS)
    latent = rearrange(latent, "b c (v t) h w -> (b v) c t h w", v=len(WAYMO_CAMERAS), t=per_view_latent_frames)
    decoded = F.interpolate(
        latent,
        size=(args.num_video_frames, 720, 1280),
        mode="trilinear",
        align_corners=False,
    )
    decoded = decoded.repeat(1, 3, 1, 1, 1)
    return rearrange(decoded, "(b v) c t h w -> b c (v t) h w", b=video_latent.shape[0], v=len(WAYMO_CAMERAS))


@torch.no_grad()
def _decode_video_latent(video_encoder: OnlineVideoEncoder, video_latent: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.video_tokenizer == "raw_downsample":
        return _decode_raw_downsample_latent(video_latent, args)

    if video_encoder.tokenizer is None:
        raise RuntimeError("wan2pt1 tokenizer was requested but was not initialized.")

    per_view_latent_frames = video_latent.shape[2] // len(WAYMO_CAMERAS)
    latent = rearrange(
        video_latent.to(device=args.device, dtype=video_encoder.tokenizer.dtype),
        "b c (v t) h w -> (b v) c t h w",
        v=len(WAYMO_CAMERAS),
        t=per_view_latent_frames,
    )
    chunks = []
    mini_batch = max(1, args.video_tokenizer_batch_size)
    for start in range(0, latent.shape[0], mini_batch):
        chunks.append(video_encoder.tokenizer.decode(latent[start : start + mini_batch]).detach().cpu())
        torch.cuda.empty_cache()
    decoded = torch.cat(chunks, dim=0)
    return rearrange(decoded, "(b v) c t h w -> b c (v t) h w", b=video_latent.shape[0], v=len(WAYMO_CAMERAS))


@torch.no_grad()
def _decode_lidar_latent(
    lidar_encoder: OnlineLidarS3Encoder,
    lidar_latent: torch.Tensor,
    exact_context_latent: torch.Tensor,
    crop_region: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    dtype = getattr(torch, args.lidar_tokenizer_dtype)
    if lidar_encoder.offload_to_cpu:
        lidar_encoder.model = lidar_encoder.model.to(args.device)
    try:
        decoded = lidar_encoder.model.decode(
            lidar_latent.to(device=args.device, dtype=dtype),
            exact_context_latent=exact_context_latent.to(device=args.device, dtype=dtype),
        )
        return _crop_decoded_video(decoded.detach().cpu(), crop_region)
    finally:
        if lidar_encoder.offload_to_cpu:
            lidar_encoder.model = lidar_encoder.model.to("cpu")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


def main() -> None:
    global args
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. If one GPU is unhealthy, retry with "
            "`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7` (or another healthy subset)."
        )
    if (args.train_height, args.train_width) != NORMAL_LIDAR_LATENT_HW and not args.allow_lidar_resize_for_smoke:
        raise ValueError(
            f"Expected normal LiDAR latent size {NORMAL_LIDAR_LATENT_HW}, got {(args.train_height, args.train_width)}."
        )

    dataset = build_waymo_dataset(args)
    sample_index = resolve_sample_index(dataset, args)
    sample = dataset[sample_index]
    batch = collate_fn([sample])
    sample_key = batch["__key__"][0]
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/data2/waymo_paired_cache_smoke") / args.split / args.video_tokenizer / sample_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    video_encoder = OnlineVideoEncoder(args)
    lidar_encoder = OnlineLidarS3Encoder(args)
    save_dtype = getattr(torch, args.save_dtype)

    with torch.no_grad():
        video_latent = video_encoder.encode(batch["video"])
        lidar_latent, exact_context_latent, tokenizer_crop_region = lidar_encoder.encode_batch(
            batch["waymo_segment_key"],
            batch["waymo_lidar_frame_indices"],
            return_exact_context=True,
            return_crop_region=True,
        )
        lidar_latent = maybe_resize_lidar(
            lidar_latent,
            args.train_height,
            args.train_width,
            args.allow_lidar_resize_for_smoke,
        )
        decoded_video = _decode_video_latent(video_encoder, video_latent, args)
        decoded_lidar = _decode_lidar_latent(
            lidar_encoder,
            lidar_latent,
            exact_context_latent,
            tokenizer_crop_region[0],
            args,
        )

    cache_payload = {
        "sample_key": sample_key,
        "segment_key": batch["waymo_segment_key"][0],
        "lidar_frame_indices": batch["waymo_lidar_frame_indices"][0].cpu(),
        "video_latent": video_latent[0].detach().cpu().to(dtype=save_dtype),
        "lidar_latent": lidar_latent[0].detach().cpu().to(dtype=save_dtype),
        "exact_context_latent": exact_context_latent[0].detach().cpu().to(dtype=save_dtype),
        "tokenizer_crop_region": tokenizer_crop_region[0].detach().cpu(),
        "video_tokenizer": args.video_tokenizer,
        "lidar_tokenizer_ckpt": args.lidar_tokenizer_ckpt,
        "split": args.split,
    }
    cache_path = output_dir / f"{sample_key}.pt"
    torch.save(cache_payload, cache_path)

    raw_video = _raw_video_to_multiview(batch["video"])
    decoded_video = _raw_video_to_multiview(_tensor_to_uint8_video(decoded_video))
    five_view_video_path = output_dir / f"{sample_key}_five_view_raw.mp4"
    _save_multiview_video(
        raw_video,
        five_view_video_path,
        fps=args.preview_fps,
        width=args.preview_width,
        height=args.preview_height,
        video_tokenizer=args.video_tokenizer,
    )
    five_view_grid_video_path = output_dir / f"{sample_key}_five_view_raw_grid.mp4"
    _save_multiview_video(
        raw_video,
        five_view_grid_video_path,
        fps=args.preview_fps,
        width=args.preview_width,
        height=args.preview_height,
        layout="grid",
    )
    video_compare_path = output_dir / f"{sample_key}_video_compare.mp4"
    _save_compare_video(
        raw_video,
        decoded_video,
        video_compare_path,
        fps=args.preview_fps,
        width=args.preview_width,
        height=args.preview_height,
    )

    lidar_decoded_payload = {
        "gt_tokenizer_recon": decoded_lidar.to(torch.float16),
        "paired_cache_decode": decoded_lidar.to(torch.float16),
    }
    lidar_decoded_path = output_dir / f"{sample_key}_lidar_decoded.pt"
    torch.save(lidar_decoded_payload, lidar_decoded_path)

    metadata = {
        "sample_key": sample_key,
        "sample_index": sample_index,
        "segment_key": batch["waymo_segment_key"][0],
        "lidar_frame_indices": batch["waymo_lidar_frame_indices"][0].tolist(),
        "video_tokenizer": args.video_tokenizer,
        "video_latent_shape": list(video_latent[0].shape),
        "lidar_latent_shape": list(lidar_latent[0].shape),
        "exact_context_shape": list(exact_context_latent[0].shape),
        "tokenizer_crop_region": tokenizer_crop_region[0].tolist(),
        "cache_path": str(cache_path),
        "five_view_video_path": str(five_view_video_path),
        "five_view_grid_video_path": str(five_view_grid_video_path),
        "video_compare_path": str(video_compare_path),
        "lidar_decoded_path": str(lidar_decoded_path),
        "video_decode_kind": (
            "approx_inverse_raw_downsample_luma_no_color"
            if args.video_tokenizer == "raw_downsample"
            else "wan2pt1_decode"
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
