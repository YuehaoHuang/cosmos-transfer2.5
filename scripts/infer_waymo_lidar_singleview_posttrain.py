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


DEFAULT_CHECKPOINT_PATH = (
    "/data2/waymo_lidar_singleview_posttrain_output/cosmos_transfer2_posttrain/"
    "waymo_lidar_singleview/waymo_lidar_singleview_rangemap_layout_t8/"
    "checkpoints/iter_000005000/model_ema_bf16.pt"
)
DEFAULT_DATASET_DIR = "/data2/waymo_singleview_lidar_posttrain/training"
DEFAULT_EXPERIMENT = "transfer2_singleview_posttrain_waymo_lidar_rangemap_layout"
DEFAULT_CONFIG_FILE = "cosmos_transfer2/singleview_config.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
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
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--keep-input-resolution", action=argparse.BooleanOptionalAction, default=True)
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


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    dataset_dir = Path(args.dataset_dir)
    sample_key = args.sample_key or args.name
    if sample_key is None and args.video_path is None:
        sample_key = first_sample_key(dataset_dir)

    video_path = Path(args.video_path) if args.video_path else dataset_dir / "videos" / f"{sample_key}.mp4"
    control_path = (
        Path(args.control_path) if args.control_path else dataset_dir / args.control_folder / f"{sample_key}.mp4"
    )
    prompt_path = Path(args.prompt_path) if args.prompt_path else dataset_dir / "captions" / f"{sample_key}.json"

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
    generated_video: torch.Tensor,
    control_video: torch.Tensor,
    resolution: str,
    max_frames: int,
    fps: int,
) -> None:
    gt_video, _, _, _ = read_and_process_video(str(video_path), resolution=resolution, max_frames=max_frames)
    gt_video = to_save_range(gt_video)
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
    resolved = resolve_paths(args)

    output_dir = Path(resolved["output_dir"])
    init_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    }
    (output_dir / f"{resolved['name']}.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / f"{resolved['name']}.txt").write_text(prompt)

    pipeline = ControlVideo2WorldInference(
        registered_exp_name=args.experiment,
        checkpoint_paths=args.checkpoint_path,
        s3_credential_path="",
        exp_override_opts=[],
        config_file=args.config_file,
    )

    output_video, control_video_dict, _mask_video_dict, fps, _original_hw = pipeline.generate_img2world(
        video_path=str(resolved["video_path"]),
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        guidance=args.guidance,
        seed=args.seed,
        resolution=args.resolution,
        control_weight=args.control_weight,
        sigma_max=args.sigma_max,
        hint_key=[args.control_key],
        input_control_video_paths={args.control_key: str(resolved["control_path"])},
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
        save_img_or_video(control_video_save, str(output_dir / f"{resolved['name']}_control_{args.control_key}"), fps=fps)
        if not args.skip_comparison:
            save_comparison(
                output_dir=output_dir,
                name=str(resolved["name"]),
                video_path=resolved["video_path"],  # type: ignore[arg-type]
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
