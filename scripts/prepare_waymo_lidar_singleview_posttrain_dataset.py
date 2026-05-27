# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Waymo TOP LiDAR range-map targets for single-view Transfer2 post-training.

The default output layout follows the official local single-view dataset style:

    dataset/
      videos/*.mp4       # Display-only LiDAR range-map videos
      rangemap_layout/*.mp4  # sparse LiDAR layout control videos
      rangemap_targets/*.npz # Optional lossless cache; raw-online training does not require it
      captions/*.json

Each sample is keyed by the existing Waymo chunk name, e.g.
``10203656353524179475_7625_000_7645_000_0``.  The LiDAR frame window is aligned
with the video chunk by ``frame_start = chunk_index * lidar_chunk_stride_frames``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_waymo_lidar_wan21_vae import (  # noqa: E402
    DEFAULT_MPLCONFIGDIR,
    load_converted_range_maps,
    load_raw_range_maps,
    prepend_lidar_utils_repo,
    preprocess_range_maps,
    write_video,
)


WAYMO_FRONT_CAMERA = "pinhole_front"
DEFAULT_CAPTION = "A monochrome LiDAR range-map video."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--waymo-chunk-root", default="/data/waymo/chunk")
    parser.add_argument("--raw-waymo-root", default="/data2/rds_hq_waymo")
    parser.add_argument("--converted-waymo-root", default="/team/hyh/data/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--output-root", default="/team/hyh/data/waymo_singleview_lidar_posttrain")
    parser.add_argument("--camera", default=WAYMO_FRONT_CAMERA)
    parser.add_argument("--range-map-source", default="raw", choices=["raw", "converted"])
    parser.add_argument("--sample-list-source", default="camera", choices=["camera", "lidar"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-key", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-layout", action="store_true")
    parser.add_argument("--overwrite-caption", action="store_true")
    parser.add_argument("--overwrite-target", action="store_true")
    parser.add_argument("--target-folder", default="rangemap_targets")
    parser.add_argument("--write-rangemap-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--layout-folder", default="rangemap_layout")
    parser.add_argument("--layout-edge-threshold-m", type=float, default=1.0)
    parser.add_argument("--caption-mode", default="fixed", choices=["fixed", "source", "vlm"])
    parser.add_argument("--fixed-caption", default=DEFAULT_CAPTION)
    parser.add_argument("--caption-model-path", default=None)
    parser.add_argument("--caption-device", default="cuda")
    parser.add_argument("--caption-max-new-tokens", type=int, default=80)
    parser.add_argument("--lidar-utils-repo", default="/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--num-frames", type=int, default=29)
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--pad-last", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=1280)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=1)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--repeat-row", type=int, default=11)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--input-channel-mode", default="repeat_depth", choices=["repeat_depth", "concat_inv_depth"])
    parser.add_argument("--decode-channel-mode", default="mean")
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    return parser.parse_args()


def split_sample_key(sample_key: str) -> tuple[str, int]:
    base, maybe_chunk = sample_key.rsplit("_", 1)
    if maybe_chunk.isdigit():
        return base, int(maybe_chunk)
    return sample_key, 0


def apply_source_defaults(args: argparse.Namespace) -> None:
    """Use sane defaults for converted lidar_tokenizer tars without changing the raw path."""

    if args.range_map_source != "converted":
        return
    if args.native_n_rows == 64:
        args.native_n_rows = 128
    if args.native_n_cols == 1280:
        args.native_n_cols = 3600
    if args.downsample_factor_row == 1:
        args.downsample_factor_row = 2
    if args.downsample_factor_col == 1:
        args.downsample_factor_col = 3


def lidar_frame_suffix(range_map_source: str) -> str:
    if range_map_source == "converted":
        return ".lidar_row.npz"
    return ".lidar_raw.npz"


def count_lidar_frames(tar_path: Path, *, range_map_source: str) -> int:
    suffix = lidar_frame_suffix(range_map_source)
    with tarfile.open(tar_path, "r") as tar_handle:
        return sum(1 for name in tar_handle.getnames() if name.endswith(suffix))


def list_lidar_sample_keys(lidar_dir: Path, args: argparse.Namespace) -> list[str]:
    """Enumerate chunk sample keys directly from lidar tar files."""

    if not lidar_dir.exists():
        raise FileNotFoundError(f"Missing LiDAR directory: {lidar_dir}")
    sample_keys: list[str] = []
    for tar_path in sorted(lidar_dir.glob("*.tar")):
        try:
            num_frames = count_lidar_frames(tar_path, range_map_source=args.range_map_source)
        except Exception as exc:
            print(f"[list][skip] {tar_path}: {exc}", flush=True)
            continue
        if num_frames <= 0:
            continue
        if num_frames < args.num_frames and not args.pad_last:
            continue
        if args.pad_last:
            max_chunk_index = max((num_frames - 1) // args.lidar_chunk_stride_frames, 0)
        else:
            max_chunk_index = max((num_frames - args.num_frames) // args.lidar_chunk_stride_frames, 0)
        sample_keys.extend(f"{tar_path.stem}_{chunk_index}" for chunk_index in range(max_chunk_index + 1))
    return sample_keys


def load_captions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected caption JSON dict at {path}, got {type(data).__name__}")
    return data


def caption_for_sample(captions: dict[str, Any], *, segment_key: str, camera: str) -> str:
    value = captions.get(f"{segment_key}_{camera}", DEFAULT_CAPTION)
    if isinstance(value, dict):
        return str(value.get("caption", value.get("text", DEFAULT_CAPTION)))
    if value:
        return str(value)
    return DEFAULT_CAPTION


def normalized_tensor_to_uint8_video(tensor) -> np.ndarray:
    """Convert [1, C, T, H, W] normalized [-1, 1] tensor to [T, H, W, C] uint8."""

    frames = tensor.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()
    frames = np.clip((frames + 1.0) * 127.5, 0.0, 255.0)
    return (frames + 0.5).astype(np.uint8)


def make_rangemap_layout_video(
    range_maps: np.ndarray,
    valid_mask: np.ndarray,
    *,
    repeat_row: int,
    repeat_col: int,
    edge_threshold_m: float,
) -> np.ndarray:
    """Build a sparse layout control from LiDAR occupancy and range discontinuities."""

    valid = valid_mask.astype(bool)
    edge = np.zeros_like(valid, dtype=bool)
    edge[:, :, 1:] |= valid[:, :, 1:] != valid[:, :, :-1]
    edge[:, :, :-1] |= valid[:, :, 1:] != valid[:, :, :-1]
    edge[:, 1:, :] |= valid[:, 1:, :] != valid[:, :-1, :]
    edge[:, :-1, :] |= valid[:, 1:, :] != valid[:, :-1, :]

    safe = np.where(valid, range_maps, 0.0).astype(np.float32)
    dx = np.abs(safe[:, :, 1:] - safe[:, :, :-1])
    dy = np.abs(safe[:, 1:, :] - safe[:, :-1, :])
    edge[:, :, 1:] |= (dx > edge_threshold_m) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, :-1] |= (dx > edge_threshold_m) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, 1:, :] |= (dy > edge_threshold_m) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :-1, :] |= (dy > edge_threshold_m) & valid[:, 1:, :] & valid[:, :-1, :]

    layout = np.zeros((*valid.shape, 3), dtype=np.uint8)
    layout[..., 0] = valid.astype(np.uint8) * 180
    layout[..., 1] = edge.astype(np.uint8) * 255
    layout[..., 2] = (valid & ~edge).astype(np.uint8) * 80
    layout = np.repeat(layout, repeat_row, axis=1)
    layout = np.repeat(layout, repeat_col, axis=2)
    return layout


class CaptionGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.processor = None
        self.device = args.caption_device

    def _resolve_model_path(self) -> str:
        if self.args.caption_model_path:
            return self.args.caption_model_path
        snapshots = sorted(Path("/data/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/snapshots").glob("*"))
        if snapshots:
            return str(snapshots[-1])
        raise FileNotFoundError("Pass --caption-model-path for --caption-mode vlm; no local Cosmos-Reason1-7B snapshot found.")

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        model_path = self._resolve_model_path()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = model_cls.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _fallback_caption(self) -> str:
        return self.args.fixed_caption

    def caption_from_frame(self, frame: np.ndarray) -> str:
        if self.args.caption_mode != "vlm":
            return self._fallback_caption()
        self._load()
        import torch
        from PIL import Image

        image = Image.fromarray(frame)
        prompt = (
            "Describe this monochrome LiDAR range-map frame for video generation. "
            "Mention only range-map structure, scene layout, density, and motion cues. "
            "Do not mention brands, logos, text, maps, diagrams, UI panels, or RGB camera colors."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.args.caption_max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        caption = self.processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()
        return caption or self._fallback_caption()


def write_caption(path: Path, caption: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"caption": caption}, f)


def prepare_one_sample(
    sample_key: str,
    *,
    args: argparse.Namespace,
    captions: dict[str, Any],
    lidar_dir: Path,
    output_dir: Path,
    caption_generator: CaptionGenerator,
) -> dict[str, Any]:
    segment_key, chunk_index = split_sample_key(sample_key)
    frame_start = chunk_index * args.lidar_chunk_stride_frames
    lidar_tar = lidar_dir / f"{segment_key}.tar"
    if not lidar_tar.exists():
        raise FileNotFoundError(f"Missing {args.range_map_source} LiDAR tar for {sample_key}: {lidar_tar}")

    video_out = output_dir / "videos" / f"{sample_key}.mp4"
    layout_out = output_dir / args.layout_folder / f"{sample_key}.mp4"
    target_out = output_dir / args.target_folder / f"{sample_key}.npz"
    caption_out = output_dir / "captions" / f"{sample_key}.json"
    metadata_out = output_dir / "metadata" / f"{sample_key}.json"

    needs_video = args.overwrite or not video_out.exists()
    needs_layout = args.overwrite or args.overwrite_layout or not layout_out.exists()
    needs_target = args.write_rangemap_targets and (args.overwrite or args.overwrite_target or not target_out.exists())
    needs_raw_lidar = needs_video or needs_layout or needs_target

    if needs_raw_lidar:
        if args.range_map_source == "converted":
            range_maps, frame_names = load_converted_range_maps(
                lidar_tar,
                frame_start=frame_start,
                num_frames=args.num_frames,
                pad_last=args.pad_last,
                n_rows=args.native_n_rows,
                n_cols=args.native_n_cols,
            )
        else:
            range_maps, frame_names = load_raw_range_maps(
                lidar_tar,
                frame_start=frame_start,
                num_frames=args.num_frames,
                pad_last=args.pad_last,
                n_rows=args.native_n_rows,
                n_cols=args.native_n_cols,
                max_projection_range=args.projection_max_range,
            )
        tensor, downsampled_range, valid_mask = preprocess_range_maps(range_maps, args)
        frames = normalized_tensor_to_uint8_video(tensor) if needs_video or args.caption_mode == "vlm" else None
        if needs_target:
            if args.input_channel_mode != "repeat_depth":
                raise ValueError("--write-rangemap-targets currently requires --input-channel-mode repeat_depth")
            normalized_rangemap = tensor[0, 0, :, :: args.repeat_row, :: args.repeat_col].detach().cpu().numpy()
            expected_shape = (args.num_frames, downsampled_range.shape[1], downsampled_range.shape[2])
            if normalized_rangemap.shape != expected_shape:
                raise ValueError(
                    f"Normalized rangemap base shape mismatch: expected {expected_shape}, got {normalized_rangemap.shape}"
                )
            target_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_out,
                normalized_rangemap=normalized_rangemap.astype(np.float32),
                repeat_row=np.int32(args.repeat_row),
                repeat_col=np.int32(args.repeat_col),
                input_channel_mode=np.asarray(args.input_channel_mode),
                expanded_shape=np.asarray(tensor.shape[1:], dtype=np.int64),
            )
        if needs_video:
            write_video(video_out, frames, fps=args.fps)
        if needs_layout:
            layout_frames = make_rangemap_layout_video(
                downsampled_range,
                valid_mask,
                repeat_row=args.repeat_row,
                repeat_col=args.repeat_col,
                edge_threshold_m=args.layout_edge_threshold_m,
            )
            write_video(layout_out, layout_frames, fps=args.fps)
    else:
        frame_names = []
        downsampled_range = None
        valid_mask = None
        frames = None

    if args.caption_mode == "source":
        caption = caption_for_sample(captions, segment_key=segment_key, camera=args.camera)
    elif args.caption_mode == "vlm" and (args.overwrite or args.overwrite_caption or not caption_out.exists()):
        if frames is None:
            raise RuntimeError("--caption-mode vlm requires --overwrite when videos already exist.")
        caption = caption_generator.caption_from_frame(frames[len(frames) // 2])
    else:
        caption = args.fixed_caption
    write_caption(caption_out, caption, overwrite=args.overwrite or args.overwrite_caption)

    metadata = {
        "sample_key": sample_key,
        "segment_key": segment_key,
        "chunk_index": chunk_index,
        "lidar_frame_start": frame_start,
        "lidar_num_frames": args.num_frames,
        "range_map_source": args.range_map_source,
        "lidar_tar": str(lidar_tar),
        "video_path": str(video_out),
        "rangemap_layout_path": str(layout_out),
        "rangemap_target_path": str(target_out) if args.write_rangemap_targets else None,
        "caption_path": str(caption_out),
        "caption_mode": args.caption_mode,
        "frame_names": frame_names,
        "native_range_shape": [args.num_frames, args.native_n_rows, args.native_n_cols],
    }
    if downsampled_range is not None and valid_mask is not None:
        metadata["downsampled_range_shape"] = list(downsampled_range.shape)
        if args.write_rangemap_targets:
            metadata["rangemap_target_base_shape"] = [args.num_frames, *list(downsampled_range.shape[1:])]
            metadata["rangemap_target_expanded_shape"] = [
                3,
                args.num_frames,
                downsampled_range.shape[1] * args.repeat_row,
                downsampled_range.shape[2] * args.repeat_col,
            ]
        metadata["target_video_shape"] = [
            args.num_frames,
            downsampled_range.shape[1] * args.repeat_row,
            downsampled_range.shape[2] * args.repeat_col,
            3,
        ]
        metadata["valid_pixel_ratio"] = float(valid_mask.mean())
        valid_values = downsampled_range[valid_mask]
        if valid_values.size:
            metadata["range_mean_m"] = float(valid_values.mean())
            metadata["range_std_m"] = float(valid_values.std())
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main() -> None:
    args = parse_args()
    apply_source_defaults(args)
    os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    prepend_lidar_utils_repo(args.lidar_utils_repo)

    split_root = Path(args.waymo_chunk_root) / args.split
    control_dir = split_root / "world_scenario" / args.camera
    if args.range_map_source == "converted":
        lidar_dir = Path(args.converted_waymo_root) / args.split / "lidar"
    else:
        lidar_dir = Path(args.raw_waymo_root) / args.split / "lidar_raw"
    output_dir = Path(args.output_root) / args.split
    captions = load_captions(Path(args.caption_json_path))
    caption_generator = CaptionGenerator(args)

    if args.sample_key:
        sample_keys = [args.sample_key]
    elif args.sample_list_source == "lidar":
        sample_keys = list_lidar_sample_keys(lidar_dir, args)
    else:
        control_paths = sorted(control_dir.glob("*.mp4"))
        sample_keys = [control_path.stem for control_path in control_paths]
    if args.max_samples is not None:
        sample_keys = sample_keys[: args.max_samples]
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    if args.num_shards > 1:
        start = len(sample_keys) * args.shard_index // args.num_shards
        end = len(sample_keys) * (args.shard_index + 1) // args.num_shards
        sample_keys = sample_keys[start:end]
    if not sample_keys:
        if args.sample_list_source == "lidar":
            raise RuntimeError(f"No LiDAR samples found under {lidar_dir}")
        raise RuntimeError(f"No control videos found under {control_dir}")

    written = []
    skipped = 0
    for idx, sample_key in enumerate(sample_keys, start=1):
        try:
            metadata = prepare_one_sample(
                sample_key,
                args=args,
                captions=captions,
                lidar_dir=lidar_dir,
                output_dir=output_dir,
                caption_generator=caption_generator,
            )
            written.append(metadata)
            print(f"[{idx}/{len(sample_keys)}] prepared {sample_key}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[{idx}/{len(sample_keys)}] skip {sample_key}: {exc}", flush=True)

    summary = {
        "split": args.split,
        "output_dir": str(output_dir),
        "num_prepared": len(written),
        "num_skipped": skipped,
        "num_requested": len(sample_keys),
        "camera": args.camera,
        "range_map_source": args.range_map_source,
        "sample_list_source": args.sample_list_source,
        "lidar_dir": str(lidar_dir),
        "num_frames": args.num_frames,
        "native_n_rows": args.native_n_rows,
        "native_n_cols": args.native_n_cols,
        "target_folder": args.target_folder,
        "write_rangemap_targets": args.write_rangemap_targets,
        "repeat_row": args.repeat_row,
        "repeat_col": args.repeat_col,
        "layout_folder": args.layout_folder,
        "caption_mode": args.caption_mode,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = (
        "prepare_summary.json"
        if args.num_shards == 1
        else f"prepare_summary_shard{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    with open(output_dir / summary_name, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
