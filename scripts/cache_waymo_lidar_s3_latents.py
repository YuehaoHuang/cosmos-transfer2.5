# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache Waymo LiDAR S3 latents aligned to existing video latent samples.

This is an opt-in utility for the video/LiDAR joint path. It does not touch the
existing Cosmos video generation or video post-training configs.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_LIDAR_REPO = "/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen"
DEFAULT_TOKENIZER_CKPT = (
    "/data2/checkpoints/posttraining/tokenizer/"
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/checkpoints/iter_000035500.pt"
)
DEFAULT_TOKENIZER_CONFIG = (
    "/data2/checkpoints/posttraining/tokenizer/"
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3/config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--video-latent-dir", default=None)
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lidar-tokenizer-repo", default=DEFAULT_LIDAR_REPO)
    parser.add_argument("--tokenizer-ckpt", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument("--tokenizer-config", default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-key", action="append", default=None, help="Exact video sample key without .pt.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pad-last", action="store_true")
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--local-frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=29)
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
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _resolve_repo_relative_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    config = dict(config)
    path_keys = ("frozen_image_tokenizer_ckpt",)
    for key in path_keys:
        value = config.get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            candidate = repo_root / value
            if candidate.exists():
                config[key] = str(candidate)
    return config


def _split_sample_key(sample_key: str) -> tuple[str, int]:
    try:
        segment_key, chunk_index = sample_key.rsplit("_", 1)
        return segment_key, int(chunk_index)
    except ValueError as exc:
        raise ValueError(f"Expected sample key '<segment_key>_<chunk_idx>', got {sample_key!r}") from exc


def _iter_sample_keys(video_latent_dir: Path, sample_keys: list[str] | None, max_samples: int | None) -> list[str]:
    if sample_keys:
        keys = sample_keys
    else:
        keys = [path.stem for path in sorted(video_latent_dir.glob("*.pt"))]
    if max_samples is not None:
        keys = keys[:max_samples]
    return keys


def _load_lidar_window(
    tar_path: Path,
    frame_indices: list[int],
    *,
    pad_last: bool,
) -> tuple[np.ndarray, list[str]]:
    from cosmos_predict1.utils.lidar_rangemap import load_each_frame_from_tar_data

    with tarfile.open(tar_path, "r") as tar_handle:
        frame_names = sorted(
            name.removesuffix(".lidar_row.npz")
            for name in tar_handle.getnames()
            if name.endswith(".lidar_row.npz")
        )
        if not frame_names:
            raise ValueError(f"No lidar_row frames found in {tar_path}")

        selected_names: list[str] = []
        range_maps: list[np.ndarray] = []
        for frame_idx in frame_indices:
            if frame_idx >= len(frame_names):
                if not pad_last:
                    raise IndexError(
                        f"{tar_path.name}: requested LiDAR frame {frame_idx}, "
                        f"but clip has {len(frame_names)} frames"
                    )
                frame_idx = len(frame_names) - 1
            frame_name = frame_names[frame_idx]
            selected_names.append(frame_name)
            range_maps.append(load_each_frame_from_tar_data(tar_handle, frame_name))
    return np.stack(range_maps, axis=0), selected_names


def _preprocess_range_maps(range_maps: np.ndarray, args: argparse.Namespace) -> torch.Tensor:
    from cosmos_predict1.utils.lidar_rangemap import RangeMapDownsampler, normalize_range_map

    downsampler = RangeMapDownsampler(
        row_factor=args.downsample_factor_row,
        col_factor=args.downsample_factor_col,
        method=args.downsample_method,
    )
    range_maps = downsampler.downsample(range_maps)
    range_maps = normalize_range_map(
        range_maps,
        args.max_range,
        args.min_range,
        args.min_value,
        False,
    )
    range_maps = range_maps[:, None, :, :].repeat(3, axis=1)
    range_maps = range_maps.repeat(args.repeat_row, axis=2)
    range_maps = range_maps.repeat(args.repeat_col, axis=3)

    if args.crop_mode != "none" and args.lidar_crop_width > 0:
        width = range_maps.shape[-1]
        if args.lidar_crop_width > width:
            raise ValueError(f"Cannot crop width {args.lidar_crop_width} from preprocessed width {width}")
        if args.crop_mode == "center":
            start_col = (width - args.lidar_crop_width) // 2
        elif args.crop_mode == "left":
            start_col = 0
        else:
            raise ValueError(f"Unsupported crop mode: {args.crop_mode}")
        range_maps = range_maps[:, :, :, start_col : start_col + args.lidar_crop_width]

    tensor = torch.from_numpy(range_maps).float()
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def _load_tokenizer(args: argparse.Namespace) -> torch.nn.Module:
    repo_root = Path(args.lidar_tokenizer_repo).resolve()
    sys.path.insert(0, str(repo_root))

    from cosmos_predict1.tokenizer.inference.utils import load_model

    cfg = OmegaConf.load(args.tokenizer_config)
    tokenizer_config = _normalize_config_value(OmegaConf.to_container(cfg.model.config.network, resolve=True))
    tokenizer_config = _resolve_repo_relative_paths(tokenizer_config, repo_root)
    model = load_model(args.tokenizer_ckpt, tokenizer_config=tokenizer_config, device=args.device)
    return model.to(dtype=getattr(torch, args.dtype)).eval()


def _save_latent(output_path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.video_latent_dir is None:
        args.video_latent_dir = f"/data/waymo/chunk/{args.split}/samples"
    if args.output_dir is None:
        args.output_dir = f"/data2/lidar_latents_s3/{args.split}"

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    video_latent_dir = Path(args.video_latent_dir)
    raw_lidar_dir = Path(args.raw_lidar_root) / args.split / "lidar"
    output_dir = Path(args.output_dir)
    sample_keys = _iter_sample_keys(video_latent_dir, args.sample_key, args.max_samples)

    print(f"[cache] video_latent_dir={video_latent_dir}")
    print(f"[cache] raw_lidar_dir={raw_lidar_dir}")
    print(f"[cache] output_dir={output_dir}")
    print(f"[cache] num_sample_keys={len(sample_keys)}")

    tokenizer = _load_tokenizer(args)
    save_dtype = getattr(torch, args.save_dtype)

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
            print(f"[cache][skip] missing tar: {tar_path}")
            skipped += 1
            continue

        lidar_start = chunk_index * args.lidar_chunk_stride_frames + args.local_frame_start
        frame_indices = list(range(lidar_start, lidar_start + args.num_frames))
        try:
            range_maps, selected_frame_names = _load_lidar_window(tar_path, frame_indices, pad_last=args.pad_last)
            input_tensor = _preprocess_range_maps(range_maps, args).to(device=args.device, dtype=getattr(torch, args.dtype))
            with torch.no_grad():
                encoded = tokenizer.encode(input_tensor)
                latent = encoded[0] if isinstance(encoded, tuple) else encoded
            latent = latent.detach().cpu().to(dtype=save_dtype)
        except Exception as exc:  # keep long cache jobs moving across bad clips
            print(f"[cache][skip] {sample_key}: {exc}")
            skipped += 1
            continue

        payload = {
            "latent": latent.squeeze(0),
            "sample_key": sample_key,
            "segment_key": segment_key,
            "chunk_index": chunk_index,
            "lidar_frame_indices": frame_indices,
            "lidar_frame_names": selected_frame_names,
            "source_tar": str(tar_path),
            "preprocess": {
                "downsample_factor_row": args.downsample_factor_row,
                "downsample_factor_col": args.downsample_factor_col,
                "downsample_method": args.downsample_method,
                "repeat_row": args.repeat_row,
                "repeat_col": args.repeat_col,
                "lidar_crop_width": args.lidar_crop_width,
                "crop_mode": args.crop_mode,
                "max_range": args.max_range,
                "min_range": args.min_range,
                "min_value": args.min_value,
            },
        }
        _save_latent(output_path, payload, overwrite=True)
        written += 1
        print(f"[cache][write] {output_path} latent_shape={tuple(payload['latent'].shape)}")

    print(f"[cache] done written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
