# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache paired Waymo video/LiDAR latents from the same dataloader window.

This is an opt-in utility for the video/LiDAR joint path. It follows the same
WaymoMultiviewDataset sampling path as the online one-way baseline, then writes
one sample per file:

    {
        "video_latent": (16, 40, 90, 160),
        "lidar_latent": (16, 8, 64, 112),
        ...
    }

The default video tokenizer is ``raw_downsample`` so cache generation can be
used immediately for throughput experiments. Switch to ``wan2pt1`` when the
real frozen video VAE path is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from cosmos_transfer2._src.predict2_multiview.datasets.multiview import collate_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_waymo_video_to_lidar_baseline import (  # noqa: E402
    NORMAL_LIDAR_LATENT_HW,
    OnlineLidarS3Encoder,
    OnlineVideoEncoder,
    build_waymo_dataset,
    maybe_resize_lidar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
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
    parser.add_argument("--lidar-crop-width", type=int, default=896)
    parser.add_argument("--crop-mode", default="center", choices=["center", "left", "none"])
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--latent-frames", type=int, default=8)
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=112)
    parser.add_argument(
        "--allow-lidar-resize-for-smoke",
        action="store_true",
        help="Allow non-64x112 LiDAR latent resizing only for local smoke tests.",
    )
    return parser.parse_args()


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _select_shard(dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}.")
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    if args.limit_samples is not None:
        indices = indices[: args.limit_samples]
    return Subset(dataset, indices)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if (args.train_height, args.train_width) != NORMAL_LIDAR_LATENT_HW and not args.allow_lidar_resize_for_smoke:
        raise ValueError(
            "Paired cache uses normal S3 LiDAR latent size by default: "
            f"{NORMAL_LIDAR_LATENT_HW}. Pass --allow-lidar-resize-for-smoke only for local smoke tests."
        )
    if args.output_dir is None:
        args.output_dir = f"/data2/waymo_paired_latents/{args.split}/{args.video_tokenizer}"
    output_dir = Path(args.output_dir)

    dataset = _select_shard(build_waymo_dataset(args), args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    video_encoder = OnlineVideoEncoder(args)
    lidar_encoder = OnlineLidarS3Encoder(args)
    save_dtype = getattr(torch, args.save_dtype)

    print(f"[paired-cache] output_dir={output_dir}")
    print(
        f"[paired-cache] split={args.split} video_tokenizer={args.video_tokenizer} "
        f"dataset_size={len(dataset)} batch_size={args.batch_size} shard={args.shard_index}/{args.num_shards}"
    )

    written = 0
    skipped = 0
    for batch in loader:
        sample_keys = batch["__key__"]
        output_paths = [output_dir / f"{sample_key}.pt" for sample_key in sample_keys]
        if all(path.exists() and not args.overwrite for path in output_paths):
            skipped += len(output_paths)
            continue

        with torch.no_grad():
            video_latent = video_encoder.encode(batch["video"]).detach().cpu().to(dtype=save_dtype)
            lidar_latent = lidar_encoder.encode_batch(batch["waymo_segment_key"], batch["waymo_lidar_frame_indices"])
            lidar_latent = maybe_resize_lidar(
                lidar_latent,
                args.train_height,
                args.train_width,
                args.allow_lidar_resize_for_smoke,
            ).detach().cpu().to(dtype=save_dtype)

        for batch_idx, output_path in enumerate(output_paths):
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            payload = {
                "sample_key": sample_keys[batch_idx],
                "segment_key": batch["waymo_segment_key"][batch_idx],
                "lidar_frame_indices": batch["waymo_lidar_frame_indices"][batch_idx].cpu(),
                "video_latent": video_latent[batch_idx],
                "lidar_latent": lidar_latent[batch_idx],
                "video_tokenizer": args.video_tokenizer,
                "lidar_tokenizer_ckpt": args.lidar_tokenizer_ckpt,
                "split": args.split,
            }
            _atomic_save(output_path, payload)
            written += 1
            if written == 1 or written % 25 == 0:
                print(
                    f"[paired-cache] written={written} skipped={skipped} "
                    f"last={output_path.name} video={tuple(payload['video_latent'].shape)} "
                    f"lidar={tuple(payload['lidar_latent'].shape)}"
                )
            if args.max_samples is not None and written >= args.max_samples:
                print(f"[paired-cache] done written={written} skipped={skipped}")
                return

    print(f"[paired-cache] done written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
