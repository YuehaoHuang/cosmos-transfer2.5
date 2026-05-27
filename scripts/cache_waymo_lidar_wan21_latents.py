# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache Waymo TOP LiDAR latents with the Wan2.1 VAE native-range-map path.

This writes only the LiDAR side of a linked paired cache. Video latents remain
the existing real Wan video latents under /data/waymo/chunk/{split}/samples and
are linked as cache_root/video.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_waymo_lidar_wan21_vae import (  # noqa: E402
    DEFAULT_LIDAR_UTILS_REPO,
    DEFAULT_MPLCONFIGDIR,
    crop_video_tensor_spatial,
    load_raw_range_maps,
    load_wan_tokenizer,
    pad_video_tensor_spatial,
    prepend_lidar_utils_repo,
    preprocess_range_maps,
)
from waymo_lidar_latent_contracts import (  # noqa: E402
    WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT,
)


WAN21_CONTRACT = WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT
WAN21_CONTRACT_VERSION = str(WAN21_CONTRACT["version"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--video-latent-dir", default=None)
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo")
    parser.add_argument("--paired-cache-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--link-video-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lidar-utils-repo", default=DEFAULT_LIDAR_UTILS_REPO)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--allow-default-wan-uri", action="store_true")
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-key", action="append", default=None, help="Exact video sample key without .pt.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pad-last", action="store_true")
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--local-frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=29)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=1280)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--wan-spatial-align", type=int, default=8)
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=1)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--repeat-row", type=int, default=11)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--input-channel-mode", default="repeat_depth", choices=["repeat_depth", "concat_inv_depth"])
    parser.add_argument("--decode-channel-mode", default="mean", choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    return parser.parse_args()


def _split_sample_key(sample_key: str) -> tuple[str, int]:
    try:
        segment_key, chunk_index = sample_key.rsplit("_", 1)
        return segment_key, int(chunk_index)
    except ValueError as exc:
        raise ValueError(f"Expected sample key '<segment_key>_<chunk_idx>', got {sample_key!r}") from exc


def _iter_sample_keys(
    video_latent_dir: Path,
    sample_keys: list[str] | None,
    max_samples: int | None,
    *,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if sample_keys:
        keys = list(sample_keys)
    else:
        keys = [path.stem for path in sorted(video_latent_dir.glob("*.pt"))]
    if num_shards > 1:
        keys = keys[shard_index::num_shards]
    if max_samples is not None:
        keys = keys[:max_samples]
    return keys


def _atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def _create_or_validate_video_link(cache_root: Path, video_latent_dir: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    link_path = cache_root / "video"
    if link_path.is_symlink():
        resolved = link_path.resolve()
        if resolved != video_latent_dir.resolve():
            raise RuntimeError(f"{link_path} points to {resolved}, expected {video_latent_dir.resolve()}")
        return
    if link_path.exists():
        if link_path.resolve() != video_latent_dir.resolve():
            raise RuntimeError(f"{link_path} exists and is not the requested video latent dir: {video_latent_dir}")
        return
    link_path.symlink_to(video_latent_dir.resolve(), target_is_directory=True)


def _validate_pinned_preprocess(args: argparse.Namespace) -> None:
    expected = {
        "native_n_rows": 64,
        "native_n_cols": 1280,
        "downsample_factor_row": 1,
        "downsample_factor_col": 1,
        "repeat_row": 11,
        "repeat_col": 1,
        "input_channel_mode": "repeat_depth",
        "decode_channel_mode": "mean",
        "wan_spatial_align": 8,
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(
            "This cache writer is pinned to the documented Wan2.1 LiDAR contract. "
            f"Expected {expected}, got {actual}."
        )


def main() -> None:
    args = parse_args()
    _validate_pinned_preprocess(args)
    os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    prepend_lidar_utils_repo(args.lidar_utils_repo)

    if args.video_latent_dir is None:
        args.video_latent_dir = f"/data/waymo/chunk/{args.split}/samples"
    if args.paired_cache_root is None:
        args.paired_cache_root = (
            f"/data2/waymo_paired_latents/{args.split}/"
            "real_video_wan21_lidar_native64x1280_repeatrow11"
        )
    if args.output_dir is None:
        args.output_dir = str(Path(args.paired_cache_root) / "lidar")

    video_latent_dir = Path(args.video_latent_dir)
    cache_root = Path(args.paired_cache_root)
    output_dir = Path(args.output_dir)
    raw_lidar_dir = Path(args.raw_lidar_root) / args.split / "lidar_raw"
    sample_keys = _iter_sample_keys(
        video_latent_dir,
        args.sample_key,
        args.max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(f"[cache-wan21] contract={WAN21_CONTRACT_VERSION}", flush=True)
    print(f"[cache-wan21] video_latent_dir={video_latent_dir}", flush=True)
    print(f"[cache-wan21] raw_lidar_dir={raw_lidar_dir}", flush=True)
    print(f"[cache-wan21] output_dir={output_dir}", flush=True)
    print(f"[cache-wan21] shard={args.shard_index}/{args.num_shards} num_sample_keys={len(sample_keys)}", flush=True)

    if args.dry_run:
        print("[cache-wan21] dry_run=true; not creating links, loading VAE, or writing latents", flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    if args.link_video_dir and output_dir == cache_root / "lidar":
        _create_or_validate_video_link(cache_root, video_latent_dir)

    tokenizer = load_wan_tokenizer(args)
    model_dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)
    expected_latent_shape = tuple(int(x) for x in WAN21_CONTRACT["latent_shape"])

    written = 0
    skipped = 0
    for sample_key in sample_keys:
        output_path = output_dir / f"{sample_key}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        segment_key, chunk_index = _split_sample_key(sample_key)
        tar_path = raw_lidar_dir / f"{segment_key}.tar"
        if not tar_path.exists():
            print(f"[cache-wan21][skip] missing tar: {tar_path}", flush=True)
            skipped += 1
            continue

        lidar_start = chunk_index * args.lidar_chunk_stride_frames + args.local_frame_start
        try:
            range_maps, selected_frame_names = load_raw_range_maps(
                tar_path,
                frame_start=lidar_start,
                num_frames=args.num_frames,
                pad_last=args.pad_last,
                n_rows=args.native_n_rows,
                n_cols=args.native_n_cols,
                max_projection_range=args.projection_max_range,
            )
            input_tensor, downsampled_range, valid_mask = preprocess_range_maps(range_maps, args)
            wan_input_tensor, spatial_padding = pad_video_tensor_spatial(
                input_tensor,
                align=args.wan_spatial_align,
                pad_value=args.min_value,
            )
            with torch.no_grad():
                latent = tokenizer.encode(wan_input_tensor.to(device=args.device, dtype=model_dtype)).detach()
            if tuple(latent.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"Wan LiDAR latent shape mismatch for {sample_key}: "
                    f"expected batch+{expected_latent_shape}, got {tuple(latent.shape)}"
                )
            reconstruction_shape = None
            if args.max_samples == 1:
                with torch.no_grad():
                    reconstruction = tokenizer.decode(latent).detach()
                reconstruction = crop_video_tensor_spatial(
                    reconstruction,
                    height=spatial_padding["input_height"],
                    width=spatial_padding["input_width"],
                )
                reconstruction_shape = list(reconstruction.shape)
        except Exception as exc:  # keep long cache jobs moving across bad clips
            print(f"[cache-wan21][skip] {sample_key}: {exc}", flush=True)
            skipped += 1
            continue

        payload = {
            "latent": latent.squeeze(0).detach().cpu().to(dtype=save_dtype),
            "sample_key": sample_key,
            "segment_key": segment_key,
            "chunk_index": chunk_index,
            "lidar_frame_start": lidar_start,
            "lidar_frame_indices": list(range(lidar_start, lidar_start + args.num_frames)),
            "lidar_frame_names": selected_frame_names,
            "source_tar": str(tar_path),
            "lidar_latent_contract": dict(WAN21_CONTRACT),
            "tokenizer_latent_kind": WAN21_CONTRACT["latent_kind"],
            "preprocessed_input_shape": list(input_tensor.shape[2:]),
            "wan_input_shape": list(wan_input_tensor.shape[2:]),
            "wan_reconstruction_shape": reconstruction_shape,
            "source_range_map_shape": list(range_maps.shape),
            "downsampled_range_shape": list(downsampled_range.shape),
            "valid_pixel_count": int(valid_mask.sum()),
            "spatial_padding": spatial_padding,
            "preprocess": {
                "preprocess_mode": WAN21_CONTRACT["preprocess_mode"],
                "native_n_rows": args.native_n_rows,
                "native_n_cols": args.native_n_cols,
                "projection_max_range": args.projection_max_range,
                "downsample_factor_row": args.downsample_factor_row,
                "downsample_factor_col": args.downsample_factor_col,
                "downsample_method": args.downsample_method,
                "repeat_row": args.repeat_row,
                "repeat_col": args.repeat_col,
                "input_channel_mode": args.input_channel_mode,
                "decode_channel_mode": args.decode_channel_mode,
                "wan_spatial_align": args.wan_spatial_align,
                "max_range": args.max_range,
                "min_range": args.min_range,
                "min_value": args.min_value,
            },
        }
        _atomic_torch_save(payload, output_path)
        written += 1
        print(
            f"[cache-wan21][write] {output_path} latent_shape={tuple(payload['latent'].shape)} "
            f"input_shape={payload['preprocessed_input_shape']} wan_input={payload['wan_input_shape']}",
            flush=True,
        )

    print(f"[cache-wan21] done written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
