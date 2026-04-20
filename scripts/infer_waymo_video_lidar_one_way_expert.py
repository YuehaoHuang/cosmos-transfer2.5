# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference/eval for the opt-in one-way video->LiDAR expert baseline.

This script intentionally stays separate from existing Cosmos video inference.
It teacher-forces an aligned Waymo video latent, samples a LiDAR latent with the
trained one-way expert, and writes compact per-sample outputs for quick checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from cosmos_transfer2._src.predict2_multiview.datasets.multiview import collate_fn
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_joint_policy import VideoLidarAttentionPolicy
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_one_way_expert import (
    DEFAULT_WAYMO_VIDEO_CHECKPOINT,
    DEFAULT_WAYMO_VIDEO_CONFIG,
    build_waymo_video_lidar_one_way_expert_baseline,
    configure_sdpa_backends,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_waymo_lidar_s3_latents import _load_tokenizer  # noqa: E402
from train_waymo_video_lidar_one_way_expert_baseline import (  # noqa: E402
    PairedLatentCacheDataset,
    collate_paired_latent_cache,
)
from train_waymo_video_to_lidar_baseline import (  # noqa: E402
    NORMAL_LIDAR_LATENT_HW,
    OnlineLidarS3Encoder,
    OnlineVideoEncoder,
    build_waymo_dataset,
    maybe_resize_lidar,
)


DEFAULT_CHECKPOINT = (
    "/data2/waymo_video_lidar_one_way_expert/"
    "one_way_expert_lidar14_b1_no_lidar_ckpt_main_7gpu_20260419_004808/"
    "checkpoints/step_001000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--paired-latent-cache-dir", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--frozen-video-dtype", default=None, choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--eval-t", type=float, default=0.5)
    parser.add_argument("--sdpa-backends", default="flash_only", choices=["flash_only", "flash_mem", "flash_mem_math"])
    parser.add_argument("--video-conditioning-mode", default="teacher_forced_flow", choices=["teacher_forced_flow", "clean"])
    parser.add_argument("--save-video-latent", action="store_true")
    parser.add_argument("--decode-lidar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-frames", type=int, default=6)
    parser.add_argument("--video-tokenizer", default=None, choices=["raw_downsample", "wan2pt1"])
    parser.add_argument("--video-tokenizer-batch-size", type=int, default=1)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--video-expert-checkpoint", default=None)
    parser.add_argument("--video-expert-config", default=None)
    parser.add_argument("--lidar-attention-backend", default=None, choices=["torch", "minimal_a2a"])
    parser.add_argument("--lidar-num-blocks", type=int, default=None)
    parser.add_argument("--cross-frame-rule", default=None, choices=["all", "same_step"])
    parser.add_argument("--video-kv-every-n-layers", type=int, default=None)
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=112)
    parser.add_argument("--allow-lidar-resize-for-smoke", action="store_true")
    parser.add_argument("--latent-frames", type=int, default=8)
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
    return parser.parse_args()


def _default_output_dir(args: argparse.Namespace, ckpt_step: int) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("/data2/waymo_video_lidar_one_way_expert") / "inference" / f"step{ckpt_step:06d}_{timestamp}"


def _copy_ckpt_defaults(args: argparse.Namespace, ckpt: dict[str, Any]) -> None:
    ckpt_args = ckpt.get("args", {})
    if args.video_tokenizer is None:
        args.video_tokenizer = ckpt_args.get("video_tokenizer", "raw_downsample")
    if args.frozen_video_dtype is None:
        args.frozen_video_dtype = ckpt_args.get("frozen_video_dtype", "bfloat16")
    if args.video_expert_checkpoint is None:
        args.video_expert_checkpoint = ckpt.get(
            "video_expert_checkpoint",
            ckpt_args.get("video_expert_checkpoint", DEFAULT_WAYMO_VIDEO_CHECKPOINT),
        )
    if args.video_expert_config is None:
        args.video_expert_config = ckpt.get(
            "video_expert_config",
            ckpt_args.get("video_expert_config", DEFAULT_WAYMO_VIDEO_CONFIG),
        )
    if args.lidar_attention_backend is None:
        args.lidar_attention_backend = ckpt_args.get("lidar_attention_backend", "torch")
    if args.lidar_num_blocks is None:
        args.lidar_num_blocks = ckpt_args.get("lidar_num_blocks")
    if args.cross_frame_rule is None:
        args.cross_frame_rule = ckpt_args.get("cross_frame_rule", "all")
    if args.video_kv_every_n_layers is None:
        args.video_kv_every_n_layers = ckpt_args.get("video_kv_every_n_layers", 1)
    args.train_height = ckpt_args.get("train_height", args.train_height)
    args.train_width = ckpt_args.get("train_width", args.train_width)
    args.latent_frames = ckpt_args.get("latent_frames", args.latent_frames)


def _build_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.paired_latent_cache_dir:
        dataset = PairedLatentCacheDataset(args.paired_latent_cache_dir)
    else:
        dataset = build_waymo_dataset(args)
    end = args.sample_index + args.num_samples
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} out of range for dataset size {len(dataset)}")
    return Subset(dataset, range(args.sample_index, min(end, len(dataset))))


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = (pred.float() - target.float()).detach()
    return {
        "mse": float(diff.square().mean().item()),
        "mae": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
        "pred_mean": float(pred.float().mean().item()),
        "pred_std": float(pred.float().std().item()),
        "target_mean": float(target.float().mean().item()),
        "target_std": float(target.float().std().item()),
    }


@torch.no_grad()
def _teacher_forced_denoise_eval(
    model: torch.nn.Module,
    clean_video: torch.Tensor,
    clean_lidar: torch.Tensor,
    *,
    eval_t: float,
    amp_enabled: bool,
    video_conditioning_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = clean_lidar.shape[0]
    device = clean_lidar.device
    t = torch.full((batch_size,), float(eval_t), device=device).clamp(1e-4, 1.0 - 1e-4)
    video_noise = torch.randn_like(clean_video)
    lidar_noise = torch.randn_like(clean_lidar)
    if video_conditioning_mode == "clean":
        noisy_video = clean_video
        video_t = torch.ones_like(t).clamp(1e-4, 1.0 - 1e-4)
    else:
        noisy_video = (1.0 - t.view(-1, 1, 1, 1, 1)) * video_noise + t.view(-1, 1, 1, 1, 1) * clean_video
        video_t = t
    noisy_lidar = (1.0 - t.view(-1, 1, 1, 1, 1)) * lidar_noise + t.view(-1, 1, 1, 1, 1) * clean_lidar
    target_velocity = clean_lidar - lidar_noise
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        pred_velocity = model(
            noisy_video=noisy_video,
            noisy_lidar=noisy_lidar,
            video_timesteps=video_t,
            lidar_timesteps=t,
            return_video_pred=False,
        )["lidar_pred"]
    denoised = noisy_lidar + (1.0 - t.view(-1, 1, 1, 1, 1)) * pred_velocity.float()
    metric = {
        "velocity_mse": float(F.mse_loss(pred_velocity.float(), target_velocity.float()).item()),
        "velocity_mae": float((pred_velocity.float() - target_velocity.float()).abs().mean().item()),
        "denoised_mse": float(F.mse_loss(denoised.float(), clean_lidar.float()).item()),
        "denoised_mae": float((denoised.float() - clean_lidar.float()).abs().mean().item()),
        "eval_t": float(t[0].item()),
    }
    return denoised, metric


@torch.no_grad()
def _sample_lidar_latent(
    model: torch.nn.Module,
    clean_video: torch.Tensor,
    *,
    lidar_shape: tuple[int, ...],
    sample_steps: int,
    amp_enabled: bool,
    video_conditioning_mode: str,
) -> torch.Tensor:
    batch_size = clean_video.shape[0]
    device = clean_video.device
    lidar = torch.randn(lidar_shape, device=device, dtype=torch.float32)
    video_noise = torch.randn_like(clean_video)
    dt = 1.0 / float(sample_steps)
    for step_idx in range(sample_steps):
        t_value = min(max((step_idx + 0.5) / float(sample_steps), 1e-4), 1.0 - 1e-4)
        t = torch.full((batch_size,), t_value, device=device)
        if video_conditioning_mode == "clean":
            noisy_video = clean_video
            video_t = torch.ones_like(t).clamp(1e-4, 1.0 - 1e-4)
        else:
            noisy_video = (1.0 - t.view(-1, 1, 1, 1, 1)) * video_noise + t.view(-1, 1, 1, 1, 1) * clean_video
            video_t = t
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            pred_velocity = model(
                noisy_video=noisy_video,
                noisy_lidar=lidar,
                video_timesteps=video_t,
                lidar_timesteps=t,
                return_video_pred=False,
            )["lidar_pred"]
        lidar = lidar + dt * pred_velocity.float()
        print(f"[infer] sample_step={step_idx + 1}/{sample_steps} t={t_value:.4f}")
    return lidar


def _as_uint8_grid(tensor: torch.Tensor, frames: int) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 5:
        tensor = tensor[0]
    # [C, T, H, W], visualize first channel.
    values = tensor[0]
    frame_ids = torch.linspace(0, values.shape[0] - 1, steps=min(frames, values.shape[0])).round().long()
    imgs = []
    for frame_id in frame_ids:
        img = values[int(frame_id)].clamp(-1, 1)
        img = ((img + 1.0) * 127.5).to(torch.uint8).numpy()
        imgs.append(img)
    return np.concatenate(imgs, axis=1)


def _save_preview_png(output_path: Path, decoded_gt: torch.Tensor, decoded_sample: torch.Tensor, decoded_denoised: torch.Tensor, frames: int) -> None:
    rows = [
        ("gt_tokenizer_recon", _as_uint8_grid(decoded_gt, frames)),
        ("rf_sample", _as_uint8_grid(decoded_sample, frames)),
        ("single_step_denoised", _as_uint8_grid(decoded_denoised, frames)),
    ]
    label_w = 220
    row_h, row_w = rows[0][1].shape
    canvas = Image.new("L", (label_w + row_w, row_h * len(rows)), color=0)
    draw = ImageDraw.Draw(canvas)
    for row_idx, (label, arr) in enumerate(rows):
        y = row_idx * row_h
        canvas.paste(Image.fromarray(arr, mode="L"), (label_w, y))
        draw.text((8, y + 8), label, fill=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def _decode_and_preview(
    args: argparse.Namespace,
    output_dir: Path,
    sample_key: str,
    gt_lidar: torch.Tensor,
    sampled_lidar: torch.Tensor,
    denoised_lidar: torch.Tensor,
    model: torch.nn.Module,
) -> dict[str, str]:
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer_args = argparse.Namespace(**vars(args))
    tokenizer_args.tokenizer_ckpt = args.lidar_tokenizer_ckpt
    tokenizer_args.tokenizer_config = args.lidar_tokenizer_config
    tokenizer_args.dtype = args.lidar_tokenizer_dtype
    tokenizer = _load_tokenizer(tokenizer_args)
    dtype = getattr(torch, args.lidar_tokenizer_dtype)
    gt_dec = tokenizer.decode(gt_lidar.to(device=args.device, dtype=dtype)).detach().cpu()
    sample_dec = tokenizer.decode(sampled_lidar.to(device=args.device, dtype=dtype)).detach().cpu()
    denoised_dec = tokenizer.decode(denoised_lidar.to(device=args.device, dtype=dtype)).detach().cpu()
    decoded_path = output_dir / f"{sample_key}_decoded.pt"
    torch.save(
        {
            "gt_tokenizer_recon": gt_dec.to(torch.float16),
            "sampled": sample_dec.to(torch.float16),
            "single_step_denoised": denoised_dec.to(torch.float16),
        },
        decoded_path,
    )
    preview_path = output_dir / f"{sample_key}_preview.png"
    _save_preview_png(preview_path, gt_dec, sample_dec, denoised_dec, args.preview_frames)
    return {"decoded_path": str(decoded_path), "preview_path": str(preview_path)}


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not args.allow_lidar_resize_for_smoke and (args.train_height, args.train_width) != NORMAL_LIDAR_LATENT_HW:
        raise ValueError(f"Expected normal LiDAR latent size {NORMAL_LIDAR_LATENT_HW}, got {(args.train_height, args.train_width)}")

    configure_sdpa_backends(args.sdpa_backends)
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    _copy_ckpt_defaults(args, ckpt)
    ckpt_step = int(ckpt["step"])
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args, ckpt_step)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer] checkpoint={args.checkpoint} step={ckpt_step}")
    print(f"[infer] output_dir={output_dir}")
    print(
        f"[infer] split={args.split} sample_index={args.sample_index} num_samples={args.num_samples} "
        f"sample_steps={args.sample_steps} video_conditioning_mode={args.video_conditioning_mode}"
    )

    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_paired_latent_cache if args.paired_latent_cache_dir else collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    video_encoder = None if args.paired_latent_cache_dir else OnlineVideoEncoder(args)
    lidar_encoder = None if args.paired_latent_cache_dir else OnlineLidarS3Encoder(args)

    model = build_waymo_video_lidar_one_way_expert_baseline(
        checkpoint_path=args.video_expert_checkpoint,
        config_path=args.video_expert_config,
        device=args.device,
        frozen_dtype=getattr(torch, args.frozen_video_dtype),
        lidar_max_img_h=args.train_height,
        lidar_max_img_w=args.train_width,
        lidar_max_frames=args.latent_frames,
        lidar_attention_backend=args.lidar_attention_backend,
        lidar_num_blocks=args.lidar_num_blocks,
        video_kv_every_n_layers=args.video_kv_every_n_layers,
        checkpoint_lidar_blocks=False,
        policy=VideoLidarAttentionPolicy(mode="video_to_lidar", cross_frame_rule=args.cross_frame_rule),
    )
    model.lidar_expert.load_state_dict(ckpt["lidar_expert"], strict=True)
    model.eval()
    amp_enabled = args.device.startswith("cuda") and args.precision == "bfloat16"

    all_metrics: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        sample_key = str(batch["__key__"][0])
        with torch.no_grad():
            if args.paired_latent_cache_dir:
                clean_video = batch["video_latent"].to(device=args.device, dtype=torch.float32)
                clean_lidar = batch["lidar_latent"].to(device=args.device, dtype=torch.float32)
            else:
                assert video_encoder is not None and lidar_encoder is not None
                clean_video = video_encoder.encode(batch["video"]).to(device=args.device, dtype=torch.float32)
                clean_lidar = lidar_encoder.encode_batch(
                    batch["waymo_segment_key"],
                    batch["waymo_lidar_frame_indices"],
                ).to(device=args.device, dtype=torch.float32)
                clean_lidar = maybe_resize_lidar(
                    clean_lidar,
                    args.train_height,
                    args.train_width,
                    args.allow_lidar_resize_for_smoke,
                )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        print(f"[infer] sample={sample_key} video_latent={tuple(clean_video.shape)} lidar_latent={tuple(clean_lidar.shape)}")
        start = time.perf_counter()
        denoised, denoise_metrics = _teacher_forced_denoise_eval(
            model,
            clean_video,
            clean_lidar,
            eval_t=args.eval_t,
            amp_enabled=amp_enabled,
            video_conditioning_mode=args.video_conditioning_mode,
        )
        sampled = _sample_lidar_latent(
            model,
            clean_video,
            lidar_shape=tuple(clean_lidar.shape),
            sample_steps=args.sample_steps,
            amp_enabled=amp_enabled,
            video_conditioning_mode=args.video_conditioning_mode,
        )
        elapsed = time.perf_counter() - start
        sample_metrics = _metrics(sampled, clean_lidar)
        denoised_latent_metrics = _metrics(denoised, clean_lidar)
        metrics = {
            "sample_key": sample_key,
            "checkpoint_step": ckpt_step,
            "elapsed_sec": elapsed,
            "sample_steps": args.sample_steps,
            "sampled_vs_gt": sample_metrics,
            "single_step_vs_gt": denoised_latent_metrics,
            "denoise_velocity": denoise_metrics,
        }

        payload: dict[str, Any] = {
            "sample_key": sample_key,
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt_step,
            "metrics": metrics,
            "gt_lidar_latent": clean_lidar.detach().cpu().to(torch.float16),
            "sampled_lidar_latent": sampled.detach().cpu().to(torch.float16),
            "single_step_denoised_lidar_latent": denoised.detach().cpu().to(torch.float16),
        }
        if args.save_video_latent:
            payload["video_latent"] = clean_video.detach().cpu().to(torch.float16)

        if args.decode_lidar:
            decode_paths = _decode_and_preview(
                args,
                output_dir,
                sample_key,
                clean_lidar,
                sampled,
                denoised,
                model,
            )
            metrics.update(decode_paths)
            payload["decode_paths"] = decode_paths

        output_path = output_dir / f"{sample_key}_latents.pt"
        torch.save(payload, output_path)
        metrics["latent_path"] = str(output_path)
        all_metrics.append(metrics)
        print(
            f"[infer] done sample={sample_key} elapsed={elapsed:.1f}s "
            f"sample_mse={sample_metrics['mse']:.6f} sample_mae={sample_metrics['mae']:.6f} "
            f"denoise_mse={denoise_metrics['denoised_mse']:.6f}"
        )

        # Decoding moves the model to CPU to free memory. Rebuild once if more samples remain.
        if args.decode_lidar and batch_idx + 1 < len(loader):
            model = build_waymo_video_lidar_one_way_expert_baseline(
                checkpoint_path=args.video_expert_checkpoint,
                config_path=args.video_expert_config,
                device=args.device,
                frozen_dtype=getattr(torch, args.frozen_video_dtype),
                lidar_max_img_h=args.train_height,
                lidar_max_img_w=args.train_width,
                lidar_max_frames=args.latent_frames,
                lidar_attention_backend=args.lidar_attention_backend,
                lidar_num_blocks=args.lidar_num_blocks,
                video_kv_every_n_layers=args.video_kv_every_n_layers,
                checkpoint_lidar_blocks=False,
                policy=VideoLidarAttentionPolicy(mode="video_to_lidar", cross_frame_rule=args.cross_frame_rule),
            )
            model.lidar_expert.load_state_dict(ckpt["lidar_expert"], strict=True)
            model.eval()

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[infer] wrote {metrics_path}")


if __name__ == "__main__":
    main()
