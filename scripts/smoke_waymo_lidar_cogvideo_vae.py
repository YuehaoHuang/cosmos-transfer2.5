# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test CogVideoX1.5 VAE encode/decode on Waymo LiDAR range maps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smoke_waymo_lidar_wan21_vae import (  # noqa: E402
    DEFAULT_LIDAR_UTILS_REPO,
    DEFAULT_MPLCONFIGDIR,
    DEFAULT_RAW_LIDAR_ROOT,
    DEFAULT_SEGMENT_KEY,
    colorize_depth_maps,
    colorize_error_maps,
    crop_video_tensor_spatial,
    decode_reconstruction_range,
    load_raw_range_maps,
    make_downsampled_ray_directions,
    pad_video_tensor_spatial,
    prepend_lidar_utils_repo,
    preprocess_range_maps,
    write_point_cloud_video,
    write_video,
)


DEFAULT_COGVIDEO_VAE_PATH = "/data2/CogvideoX1.5-5B-T2V/vae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--segment-key", default=DEFAULT_SEGMENT_KEY)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=29)
    parser.add_argument("--pad-last", action="store_true")
    parser.add_argument("--raw-lidar-root", default=DEFAULT_RAW_LIDAR_ROOT)
    parser.add_argument("--raw-tar", default=None)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=2650)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lidar-utils-repo", default=DEFAULT_LIDAR_UTILS_REPO)
    parser.add_argument("--cogvideo-vae-path", default=DEFAULT_COGVIDEO_VAE_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--latent-mode", default="mode", choices=["mode", "sample"])
    parser.add_argument("--enable-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-slicing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-sample-min-height", type=int, default=None)
    parser.add_argument("--tile-sample-min-width", type=int, default=None)
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=1)
    parser.add_argument("--downsample-method", default="scatter_min", choices=["scatter_min", "scatter_max", "every_n"])
    parser.add_argument("--repeat-row", type=int, default=8)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument("--input-channel-mode", default="repeat_depth", choices=["repeat_depth", "concat_inv_depth"])
    parser.add_argument("--decode-channel-mode", default=None, choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Range-map normalization minimum. CogVideo's standalone VAE demo uses image tensors in [0, 1].",
    )
    parser.add_argument("--wan-spatial-align", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-tensors", action="store_true")
    parser.add_argument("--vis-pcd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pcd-renderer", default="plotly", choices=["auto", "plotly", "raster"])
    parser.add_argument("--pcd-display-frame", default="vehicle", choices=["lidar", "vehicle"])
    parser.add_argument("--pcd-camera-view", default="front_view", choices=["front_view", "top_down_view"])
    parser.add_argument("--pcd-width", type=int, default=1280)
    parser.add_argument("--pcd-height", type=int, default=720)
    parser.add_argument("--pcd-max-points", type=int, default=70000)
    parser.add_argument("--pcd-workers", type=int, default=1)
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace) -> None:
    args.preprocess_mode = "waymo_top_64x2650"
    if args.decode_channel_mode is None:
        args.decode_channel_mode = "concat_fuse" if args.input_channel_mode == "concat_inv_depth" else "mean"


def load_cogvideo_vae(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device):
    from diffusers import AutoencoderKLCogVideoX

    vae = AutoencoderKLCogVideoX.from_pretrained(args.cogvideo_vae_path, torch_dtype=dtype).to(device)
    vae.eval()
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling(
            tile_sample_min_height=args.tile_sample_min_height,
            tile_sample_min_width=args.tile_sample_min_width,
        )
    return vae


def run_cogvideo_vae(
    vae: Any,
    tensor: torch.Tensor,
    *,
    latent_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        posterior = vae.encode(tensor).latent_dist
        if latent_mode == "mode":
            latent = posterior.mode()
        elif latent_mode == "sample":
            latent = posterior.sample()
        else:
            raise ValueError(f"Unsupported latent mode: {latent_mode}")
        reconstruction = vae.decode(latent).sample
    return latent.detach(), reconstruction.detach()


def save_outputs(
    output_dir: Path,
    *,
    input_tensor: torch.Tensor,
    cogvideo_input_tensor: torch.Tensor,
    latent: torch.Tensor,
    reconstruction: torch.Tensor,
    cogvideo_reconstruction: torch.Tensor,
    save_dtype: torch.dtype,
    skip_tensors: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_tensors:
        return
    torch.save(input_tensor.detach().cpu().to(dtype=save_dtype), output_dir / "input_tensor.pt")
    if tuple(cogvideo_input_tensor.shape) != tuple(input_tensor.shape):
        torch.save(cogvideo_input_tensor.detach().cpu().to(dtype=save_dtype), output_dir / "cogvideo_input_tensor.pt")
    torch.save(latent.detach().cpu().to(dtype=save_dtype), output_dir / "latent.pt")
    torch.save(reconstruction.detach().cpu().to(dtype=save_dtype), output_dir / "reconstruction.pt")
    if tuple(cogvideo_reconstruction.shape) != tuple(reconstruction.shape):
        torch.save(
            cogvideo_reconstruction.detach().cpu().to(dtype=save_dtype),
            output_dir / "cogvideo_reconstruction.pt",
        )


def build_metrics(
    *,
    input_tensor: torch.Tensor,
    latent: torch.Tensor,
    reconstruction: torch.Tensor,
    input_range,
    reconstruction_range,
    valid_mask,
    selected_frame_names: list[str],
    source_tar: Path,
    source_range_map_shape: tuple[int, ...],
    cogvideo_input_tensor_shape: tuple[int, ...],
    cogvideo_reconstruction_shape: tuple[int, ...],
    spatial_padding: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_float = input_tensor.detach().cpu().float()
    recon_float = reconstruction.detach().cpu().float()
    diff = recon_float - input_float

    import numpy as np

    input_range_clipped = np.clip(input_range, args.min_range, args.max_range)
    valid = valid_mask & np.isfinite(reconstruction_range)
    range_diff = reconstruction_range - input_range_clipped
    if valid.any():
        valid_range_diff = range_diff[valid]
        range_mae = float(np.mean(np.abs(valid_range_diff)))
        range_rmse = float(np.sqrt(np.mean(valid_range_diff**2)))
        range_max_abs = float(np.max(np.abs(valid_range_diff)))
    else:
        range_mae = None
        range_rmse = None
        range_max_abs = None

    return {
        "vae": "CogVideoX1.5",
        "cogvideo_vae_path": args.cogvideo_vae_path,
        "segment_key": args.segment_key,
        "split": args.split,
        "source_tar": str(source_tar),
        "selected_frame_names": selected_frame_names,
        "preprocess_mode": args.preprocess_mode,
        "source_range_map_shape": list(source_range_map_shape),
        "evaluation_range_shape": list(input_range.shape),
        "input_tensor_shape": list(input_tensor.shape),
        "cogvideo_input_tensor_shape": list(cogvideo_input_tensor_shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "cogvideo_reconstruction_shape": list(cogvideo_reconstruction_shape),
        "spatial_padding": spatial_padding,
        "normalized_mse": float(torch.mean(diff * diff).item()),
        "normalized_mae": float(torch.mean(torch.abs(diff)).item()),
        "valid_pixel_count": int(valid.sum()),
        "range_mae_m": range_mae,
        "range_rmse_m": range_rmse,
        "range_max_abs_m": range_max_abs,
        "preprocess": {
            "downsample_factor_row": args.downsample_factor_row,
            "downsample_factor_col": args.downsample_factor_col,
            "downsample_method": args.downsample_method,
            "repeat_row": args.repeat_row,
            "repeat_col": args.repeat_col,
            "input_channel_mode": args.input_channel_mode,
            "decode_channel_mode": args.decode_channel_mode,
            "inv_depth_threshold": args.inv_depth_threshold,
            "max_range": args.max_range,
            "min_range": args.min_range,
            "min_value": args.min_value,
            "native_n_rows": args.native_n_rows,
            "native_n_cols": args.native_n_cols,
            "projection_max_range": args.projection_max_range,
            "spatial_align": args.wan_spatial_align,
        },
        "cogvideo": {
            "dtype": args.dtype,
            "latent_mode": args.latent_mode,
            "enable_tiling": args.enable_tiling,
            "enable_slicing": args.enable_slicing,
            "tile_sample_min_height": args.tile_sample_min_height,
            "tile_sample_min_width": args.tile_sample_min_width,
        },
    }


def output_name_for_args(args: argparse.Namespace) -> str:
    name = f"{args.segment_key}_waymo_top_64x2650_cogvideo_repeatrow{args.repeat_row}"
    if args.min_value == -1:
        name = f"{name}_minneg1"
    if args.input_channel_mode != "repeat_depth":
        name = f"{name}_{args.input_channel_mode}"
    if args.latent_mode != "mode":
        name = f"{name}_{args.latent_mode}"
    return name


def main() -> None:
    args = parse_args()
    apply_defaults(args)
    os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    prepend_lidar_utils_repo(args.lidar_utils_repo)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model_dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)

    source_tar = (
        Path(args.raw_tar)
        if args.raw_tar
        else Path(args.raw_lidar_root) / args.split / "lidar_raw" / f"{args.segment_key}.tar"
    )
    print(f"[load] native Waymo TOP raw lidar tar: {source_tar}", flush=True)
    range_maps, selected_frame_names = load_raw_range_maps(
        source_tar,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        pad_last=args.pad_last,
        n_rows=args.native_n_rows,
        n_cols=args.native_n_cols,
        max_projection_range=args.projection_max_range,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("/data2/waymo_cogvideo_vae_smoke") / args.split / output_name_for_args(args)
    )
    print(f"[preprocess] raw range maps: {tuple(range_maps.shape)}", flush=True)
    input_tensor, downsampled_range, valid_mask = preprocess_range_maps(range_maps, args)
    cogvideo_input_tensor, spatial_padding = pad_video_tensor_spatial(
        input_tensor,
        align=args.wan_spatial_align,
        pad_value=args.min_value,
    )
    print(f"[preprocess] logical tensor: {tuple(input_tensor.shape)}", flush=True)
    print(f"[preprocess] CogVideo input tensor: {tuple(cogvideo_input_tensor.shape)} padding={spatial_padding}", flush=True)

    vae = load_cogvideo_vae(args, dtype=model_dtype, device=device)
    cogvideo_input_tensor = cogvideo_input_tensor.to(device=device, dtype=model_dtype)
    latent, cogvideo_reconstruction = run_cogvideo_vae(
        vae,
        cogvideo_input_tensor,
        latent_mode=args.latent_mode,
    )
    print(f"[cogvideo] latent: {tuple(latent.shape)}", flush=True)
    print(f"[cogvideo] reconstruction: {tuple(cogvideo_reconstruction.shape)}", flush=True)

    if cogvideo_reconstruction.shape[0:2] != cogvideo_input_tensor.shape[0:2]:
        raise ValueError(
            f"CogVideo reconstruction batch/channel shape {tuple(cogvideo_reconstruction.shape)} does not match input "
            f"{tuple(cogvideo_input_tensor.shape)}"
        )
    if cogvideo_reconstruction.shape[2] < cogvideo_input_tensor.shape[2]:
        raise ValueError(
            f"CogVideo reconstruction has fewer frames {cogvideo_reconstruction.shape[2]} than input "
            f"{cogvideo_input_tensor.shape[2]}"
        )

    reconstruction = crop_video_tensor_spatial(
        cogvideo_reconstruction[:, :, : cogvideo_input_tensor.shape[2]],
        height=spatial_padding["input_height"],
        width=spatial_padding["input_width"],
    )
    input_tensor = input_tensor.to(device=device, dtype=model_dtype)
    save_outputs(
        output_dir,
        input_tensor=input_tensor,
        cogvideo_input_tensor=cogvideo_input_tensor,
        latent=latent,
        reconstruction=reconstruction,
        cogvideo_reconstruction=cogvideo_reconstruction,
        save_dtype=save_dtype,
        skip_tensors=args.skip_tensors,
    )

    recon_range = decode_reconstruction_range(
        reconstruction,
        args=args,
        target_height=downsampled_range.shape[1],
        target_width=downsampled_range.shape[2],
    )

    import numpy as np

    error_range = np.abs(recon_range - np.clip(downsampled_range, args.min_range, args.max_range))
    metrics = build_metrics(
        input_tensor=input_tensor,
        latent=latent,
        reconstruction=reconstruction,
        input_range=downsampled_range,
        reconstruction_range=recon_range,
        valid_mask=valid_mask,
        selected_frame_names=selected_frame_names,
        source_tar=source_tar,
        source_range_map_shape=tuple(range_maps.shape),
        cogvideo_input_tensor_shape=tuple(cogvideo_input_tensor.shape),
        cogvideo_reconstruction_shape=tuple(cogvideo_reconstruction.shape),
        spatial_padding=spatial_padding,
        args=args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[save] metrics: {metrics_path}", flush=True)

    if not args.skip_videos:
        input_video = colorize_depth_maps(downsampled_range, valid_mask)
        recon_video = colorize_depth_maps(recon_range, valid_mask)
        error_video = colorize_error_maps(error_range, valid_mask)
        write_video(output_dir / "input_rangemap.mp4", input_video, fps=args.fps)
        write_video(output_dir / "reconstruction_rangemap.mp4", recon_video, fps=args.fps)
        write_video(output_dir / "abs_error_rangemap.mp4", error_video, fps=args.fps)
        print(f"[save] videos: {output_dir}", flush=True)

    if args.vis_pcd:
        ray_directions = make_downsampled_ray_directions(range_maps, args)
        pcd_path, pcd_renderer = write_point_cloud_video(
            output_dir,
            input_range=np.clip(downsampled_range, args.min_range, args.max_range),
            reconstruction_range=recon_range,
            valid_mask=valid_mask,
            ray_directions=ray_directions,
            args=args,
        )
        metrics["point_cloud_video"] = str(pcd_path)
        metrics["point_cloud_renderer"] = pcd_renderer
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[save] point cloud video: {pcd_path} renderer={pcd_renderer}", flush=True)

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
