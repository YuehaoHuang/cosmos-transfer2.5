#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run single-view Waymo LiDAR post-training inference with range-map layout control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import read_and_process_video

DEFAULT_DATASET_DIR = None
DEFAULT_EXPERIMENT = "transfer2_singleview_posttrain_waymo_lidar_wan21_online_layout_fullfinetune"
DEFAULT_RAW_LIDAR_ROOT = "/team/hyh/data/rds_hq_waymo"
DEFAULT_CONFIG_FILE = "cosmos_transfer2/singleview_config.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample-key", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--control-path", default=None)
    parser.add_argument("--control-key", default="rangemap_layout")
    parser.add_argument("--control-folder", default="rangemap_layout")
    parser.add_argument("--prompt-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--max-frames", type=int, default=29)
    parser.add_argument("--num-video-frames-per-chunk", type=int, default=29)
    parser.add_argument("--num-conditional-frames", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--control-weight", default="1.0")
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--zero-text-embedding", action="store_true")
    parser.add_argument("--text-embedding-tokens", type=int, default=512)
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--keep-input-resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-online-input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--online-target-key", default="rangemap_target")
    parser.add_argument("--raw-lidar-root", default=DEFAULT_RAW_LIDAR_ROOT)
    parser.add_argument("--raw-lidar-split", default="training", choices=["training", "validation"])
    parser.add_argument("--lidar-utils-repo", default="/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument(
        "--layout-edge-threshold-m", type=float, default=1.0, help="Range discontinuity threshold for raw layout."
    )
    return parser.parse_args()


def read_prompt(prompt_path: Path | None, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    if prompt_path is None:
        return "A monochrome LiDAR range-map video."

    if prompt_path.suffix == ".json":
        data = json.loads(prompt_path.read_text())
        if isinstance(data, dict):
            for key in ("caption", "prompt", "text"):
                if key in data:
                    return str(data[key])
        return str(data)
    return prompt_path.read_text().strip()


def first_sample_key(dataset_dir: Path) -> str:
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No mp4 files found under {dataset_dir / 'videos'}")
    return videos[0].stem


def use_raw_online_input(args: argparse.Namespace) -> bool:
    if args.raw_online_input is not None:
        return args.raw_online_input
    return "wan21_online" in args.experiment


def raw_window_for_sample(sample_key: str, stride: int) -> tuple[str, int]:
    segment_key, separator, chunk_id = sample_key.rpartition("_")
    if not separator or not chunk_id.isdigit():
        raise ValueError(f"Raw-online sample key must end with a numeric chunk index: {sample_key}")
    return segment_key, int(chunk_id) * stride


def build_raw_online_inputs(args: argparse.Namespace, sample_key: str) -> tuple[torch.Tensor, torch.Tensor, Path]:
    from scripts.prepare_waymo_lidar_singleview_posttrain_dataset import make_rangemap_layout_video
    from scripts.smoke_waymo_lidar_wan21_vae import load_raw_range_maps, prepend_lidar_utils_repo, preprocess_range_maps

    segment_key, frame_start = raw_window_for_sample(sample_key, args.lidar_chunk_stride_frames)
    raw_tar = Path(args.raw_lidar_root) / args.raw_lidar_split / "lidar_raw" / f"{segment_key}.tar"
    if not raw_tar.exists():
        raise FileNotFoundError(f"Missing raw LiDAR tar for {sample_key}: {raw_tar}")

    preprocess_args = argparse.Namespace(
        downsample_factor_row=1,
        downsample_factor_col=1,
        downsample_method="scatter_min",
        repeat_row=11,
        repeat_col=1,
        input_channel_mode="repeat_depth",
        max_range=100.0,
        min_range=5.0,
        min_value=-1.0,
    )
    prepend_lidar_utils_repo(args.lidar_utils_repo)
    range_maps, _ = load_raw_range_maps(
        raw_tar,
        frame_start=frame_start,
        num_frames=args.max_frames,
        pad_last=False,
        n_rows=64,
        n_cols=1280,
        max_projection_range=105.0,
    )
    target_batch, downsampled_range, valid_mask = preprocess_range_maps(range_maps, preprocess_args)
    target = target_batch.squeeze(0).contiguous()
    layout_frames = make_rangemap_layout_video(
        downsampled_range,
        valid_mask,
        repeat_row=11,
        repeat_col=1,
        edge_threshold_m=args.layout_edge_threshold_m,
    )
    layout = torch.from_numpy(layout_frames).permute(3, 0, 1, 2).contiguous()
    expected_shape = (3, 29, 704, 1280)
    if tuple(target.shape) != expected_shape or tuple(layout.shape) != expected_shape:
        raise ValueError(
            f"Raw online input shape mismatch: target={tuple(target.shape)}, layout={tuple(layout.shape)}, "
            f"expected={expected_shape}"
        )
    return target, layout, raw_tar


def save_raw_online_display_inputs(
    output_dir: Path, name: str, target: torch.Tensor, layout: torch.Tensor, *, fps: int = 10
) -> tuple[Path, Path]:
    video_stem = output_dir / f"{name}_raw_target_input"
    layout_stem = output_dir / f"{name}_raw_layout_input"
    save_img_or_video(to_save_range(target), str(video_stem), fps=fps)
    save_img_or_video(to_save_range(layout), str(layout_stem), fps=fps)
    return video_stem.with_suffix(".mp4"), layout_stem.with_suffix(".mp4")


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(".")
    sample_key = args.sample_key or args.name
    if sample_key is None and args.video_path is None:
        sample_key = first_sample_key(dataset_dir)

    video_path = Path(args.video_path) if args.video_path else dataset_dir / "videos" / f"{sample_key}.mp4"
    control_path = (
        Path(args.control_path) if args.control_path else dataset_dir / args.control_folder / f"{sample_key}.mp4"
    )
    prompt_path = Path(args.prompt_path) if args.prompt_path else dataset_dir / "captions" / f"{sample_key}.json"

    if not use_raw_online_input(args):
        if not video_path.exists():
            raise FileNotFoundError(f"Missing input/GT LiDAR video: {video_path}")
        if not control_path.exists():
            raise FileNotFoundError(f"Missing {args.control_folder} control video: {control_path}")
    if args.prompt is None and not prompt_path.exists():
        prompt_path = None

    name = args.name or f"{sample_key}_step{args.num_steps:02d}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("/data2/waymo_lidar_singleview_posttrain_inference")
        / f"iter_000005000_s{args.num_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    return {
        "sample_key": sample_key,
        "name": name,
        "video_path": video_path,
        "control_path": control_path,
        "prompt_path": prompt_path,
        "output_dir": output_dir,
    }


def to_save_range(video: torch.Tensor) -> torch.Tensor:
    """Convert CTHW or BCTHW tensor to CTHW in [0, 1]."""

    if video.dim() == 5:
        video = video[0]
    if video.dtype == torch.uint8:
        return video.float() / 255.0
    min_value = float(video.amin())
    max_value = float(video.amax())
    if min_value < -0.05:
        video = (video + 1.0) / 2.0
    elif max_value > 2.0:
        video = video / 255.0
    return video.float().clamp(0.0, 1.0)


def save_comparison(
    *,
    output_dir: Path,
    name: str,
    video_path: Path,
    gt_video_tensor: torch.Tensor | None = None,
    generated_video: torch.Tensor,
    control_video: torch.Tensor,
    resolution: str,
    max_frames: int,
    fps: int,
) -> None:
    if gt_video_tensor is None:
        gt_video, _, _, _ = read_and_process_video(str(video_path), resolution=resolution, max_frames=max_frames)
        gt_video = to_save_range(gt_video)
    else:
        gt_video = to_save_range(gt_video_tensor)
    control_video = to_save_range(control_video)
    generated_video = to_save_range(generated_video)

    min_t = min(gt_video.shape[1], control_video.shape[1], generated_video.shape[1])
    gt_video = gt_video[:, :min_t]
    control_video = control_video[:, :min_t]
    generated_video = generated_video[:, :min_t]

    comparison = torch.cat([gt_video.cpu(), control_video.cpu(), generated_video.cpu()], dim=-1)
    save_img_or_video(comparison, str(output_dir / f"{name}_comparison_gt_control_generated"), fps=fps)


def main() -> None:
    args = parse_args()
    raw_online_input = use_raw_online_input(args)
    if raw_online_input and (args.max_frames != 29 or args.num_video_frames_per_chunk != 29):
        raise ValueError(
            "Raw-online single-view inference requires exactly one 29-frame chunk to avoid GT leakage across chunks."
        )
    resolved = resolve_paths(args)

    output_dir = Path(resolved["output_dir"])
    init_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_target = None
    raw_layout = None
    raw_tar = None
    if raw_online_input:
        sample_key = resolved["sample_key"]
        if not isinstance(sample_key, str):
            raise ValueError("--sample-key is required for raw-online inference")
        raw_target, raw_layout, raw_tar = build_raw_online_inputs(args, sample_key)
        video_path, control_path = save_raw_online_display_inputs(
            output_dir, str(resolved["name"]), raw_target, raw_layout
        )
        resolved["video_path"] = video_path
        resolved["control_path"] = control_path

    prompt = read_prompt(resolved["prompt_path"], args.prompt)  # type: ignore[arg-type]
    metadata = {
        "checkpoint_path": args.checkpoint_path,
        "experiment": args.experiment,
        "config_file": args.config_file,
        "sample_key": resolved["sample_key"],
        "video_path": str(resolved["video_path"]),
        "control_path": str(resolved["control_path"]),
        "control_key": args.control_key,
        "control_folder": args.control_folder,
        "prompt_path": str(resolved["prompt_path"]) if resolved["prompt_path"] else None,
        "prompt": prompt,
        "num_steps": args.num_steps,
        "max_frames": args.max_frames,
        "num_video_frames_per_chunk": args.num_video_frames_per_chunk,
        "num_conditional_frames": args.num_conditional_frames,
        "guidance": args.guidance,
        "seed": args.seed,
        "control_weight": args.control_weight,
        "zero_text_embedding": args.zero_text_embedding,
        "raw_online_input": raw_online_input,
        "online_target_key": args.online_target_key if raw_online_input else None,
        "raw_lidar_tar": str(raw_tar) if raw_tar is not None else None,
    }
    (output_dir / f"{resolved['name']}.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / f"{resolved['name']}.txt").write_text(prompt)

    exp_override_opts = []
    if args.zero_text_embedding:
        exp_override_opts.append("model.config.text_encoder_config.compute_online=False")

    pipeline = ControlVideo2WorldInference(
        registered_exp_name=args.experiment,
        checkpoint_paths=args.checkpoint_path,
        s3_credential_path="",
        exp_override_opts=exp_override_opts,
        config_file=args.config_file,
    )

    generation_prompt: str | torch.Tensor = prompt
    if args.zero_text_embedding:
        crossattn_dim = int(pipeline.config.model.config.net.crossattn_proj_in_channels)
        generation_prompt = torch.zeros(
            1,
            args.text_embedding_tokens,
            crossattn_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

    output_video, control_video_dict, _mask_video_dict, fps, _original_hw = pipeline.generate_img2world(
        video_path=str(resolved["video_path"]),
        prompt=generation_prompt,
        negative_prompt=args.negative_prompt,
        guidance=args.guidance,
        seed=args.seed,
        resolution=args.resolution,
        control_weight=args.control_weight,
        sigma_max=args.sigma_max,
        hint_key=[args.control_key],
        input_control_video_paths={} if raw_online_input else {args.control_key: str(resolved["control_path"])},
        input_control_tensors={args.control_key: raw_layout} if raw_layout is not None else None,
        model_video_inputs={args.online_target_key: raw_target} if raw_target is not None else None,
        keep_input_resolution=args.keep_input_resolution,
        max_frames=args.max_frames,
        num_conditional_frames=args.num_conditional_frames,
        num_video_frames_per_chunk=args.num_video_frames_per_chunk,
        num_steps=args.num_steps,
    )

    generated_video = to_save_range(output_video)
    save_img_or_video(generated_video, str(output_dir / str(resolved["name"])), fps=fps)

    control_video = control_video_dict.get(args.control_key)
    if control_video is not None:
        control_video_save = to_save_range(control_video)
        save_img_or_video(
            control_video_save, str(output_dir / f"{resolved['name']}_control_{args.control_key}"), fps=fps
        )
        if not args.skip_comparison:
            save_comparison(
                output_dir=output_dir,
                name=str(resolved["name"]),
                video_path=resolved["video_path"],  # type: ignore[arg-type]
                gt_video_tensor=raw_target,
                generated_video=generated_video,
                control_video=control_video_save,
                resolution=args.resolution,
                max_frames=args.max_frames,
                fps=fps,
            )

    print(json.dumps({"output_dir": str(output_dir), "name": resolved["name"]}, indent=2), flush=True)


if __name__ == "__main__":
    init_environment()
    try:
        main()
    finally:
        cleanup_environment()
