# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online Waymo video-to-LiDAR latent baseline.

This script is intentionally standalone and opt-in. It reads Waymo multiview
video frames online, extracts the exactly aligned LiDAR window online, freezes
both tokenizers, and trains only a small LiDAR latent denoiser baseline.
Existing video generation and video post-training configs are not imported or
modified by this file.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Subset

from cosmos_transfer2._src.predict2_multiview.datasets.multiview import AugmentationConfig, collate_fn
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.dataloader_local import (
    WaymoMultiviewDataset,
)
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_joint_policy import (
    VideoLidarAttentionPolicy,
    build_video_lidar_attention_mask,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_waymo_lidar_s3_latents import (
    DEFAULT_LIDAR_REPO,
    DEFAULT_TOKENIZER_CKPT,
    DEFAULT_TOKENIZER_CONFIG,
    _extract_official_ltcv_latent,
    _load_lidar_window,
    _load_tokenizer,
    _preprocess_range_maps,
)


WAYMO_CAMERAS = (
    "pinhole_front",
    "pinhole_front_left",
    "pinhole_front_right",
    "pinhole_side_left",
    "pinhole_side_right",
)
NORMAL_LIDAR_LATENT_HW = (64, 226)
WAN_HF_REPO_CACHE = "models--nvidia--Cosmos-Predict2.5-2B"
WAN_HF_REVISION = "f176dc95b4a70f53ce01c4b302851595e7322b00"
WAN_HF_FILENAME = "tokenizer.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=226)
    parser.add_argument(
        "--allow-lidar-resize-for-smoke",
        action="store_true",
        help="Allow non-64x226 LiDAR latent resizing only for local smoke tests.",
    )
    parser.add_argument("--latent-frames", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--video-tokenizer", default="raw_downsample", choices=["raw_downsample", "wan2pt1"])
    parser.add_argument("--video-tokenizer-batch-size", type=int, default=1)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--lidar-tokenizer-repo", default=DEFAULT_LIDAR_REPO)
    parser.add_argument("--lidar-tokenizer-ckpt", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument("--lidar-tokenizer-config", default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--lidar-tokenizer-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--lidar-chunk-stride-frames", type=int, default=10)
    parser.add_argument("--local-frame-start", type=int, default=0)
    parser.add_argument("--num-video-frames", type=int, default=29)
    parser.add_argument("--pad-lidar-last", action="store_true")
    parser.add_argument("--downsample-factor-row", type=int, default=1)
    parser.add_argument("--downsample-factor-col", type=int, default=2)
    parser.add_argument("--downsample-method", default="scatter_min")
    parser.add_argument("--repeat-row", type=int, default=4)
    parser.add_argument("--repeat-col", type=int, default=1)
    parser.add_argument(
        "--lidar-crop-width",
        type=int,
        default=0,
        help="Optional pre-tokenizer crop width. Default 0 keeps the full downsampled Waymo range map.",
    )
    parser.add_argument("--crop-mode", default="none", choices=["center", "left", "none"])
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    return parser.parse_args()


def build_waymo_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    dataset_dir = args.dataset_dir or f"/data/waymo/chunk/{args.split}"
    camera_view_mapping = dict(zip(WAYMO_CAMERAS, range(len(WAYMO_CAMERAS))))
    augmentation_config = AugmentationConfig(
        resolution_hw=(720, 1280),
        fps_downsample_factor=1,
        num_video_frames=args.num_video_frames,
        camera_keys=WAYMO_CAMERAS,
        camera_view_mapping=camera_view_mapping,
        camera_video_key_mapping={cam: f"video_{cam}" for cam in WAYMO_CAMERAS},
        camera_caption_key_mapping={cam: f"metas_{cam}" for cam in WAYMO_CAMERAS},
        caption_probability={"dummy": 1.0},
        single_caption_camera_name=None,
        add_view_prefix_to_caption=False,
        camera_prefix_mapping={
            "pinhole_front": "The video is captured from the front camera.",
            "pinhole_front_left": "The video is captured from the front-left camera.",
            "pinhole_front_right": "The video is captured from the front-right camera.",
            "pinhole_side_left": "The video is captured from the left side camera.",
            "pinhole_side_right": "The video is captured from the right side camera.",
        },
        camera_control_key_mapping={cam: f"world_scenario_{cam}" for cam in WAYMO_CAMERAS},
    )
    dataset = WaymoMultiviewDataset(
        dataset_dir=dataset_dir,
        caption_json_path=args.caption_json_path,
        augmentation_config=augmentation_config,
        folder_to_camera_key={cam: cam for cam in WAYMO_CAMERAS},
        control_dir_name="world_scenario",
        include_lidar_alignment_metadata=True,
        lidar_chunk_stride_frames=args.lidar_chunk_stride_frames,
    )
    if args.limit_samples is not None:
        return Subset(dataset, range(min(args.limit_samples, len(dataset))))
    return dataset


class OnlineLidarS3Encoder:
    def __init__(self, args: argparse.Namespace):
        lidar_args = argparse.Namespace(**vars(args))
        lidar_args.tokenizer_ckpt = args.lidar_tokenizer_ckpt
        lidar_args.tokenizer_config = args.lidar_tokenizer_config
        lidar_args.lidar_tokenizer_repo = args.lidar_tokenizer_repo
        lidar_args.dtype = args.lidar_tokenizer_dtype
        self.args = lidar_args
        self.raw_lidar_dir = Path(args.raw_lidar_root) / args.split / "lidar"
        self.device = torch.device(args.device)
        self.offload_to_cpu = bool(getattr(args, "offload_lidar_encoder", False)) and self.device.type == "cuda"
        self.model = _load_tokenizer(lidar_args)
        if self.offload_to_cpu:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_batch(
        self,
        segment_keys: list[str],
        frame_indices: torch.Tensor,
        *,
        return_exact_context: bool = False,
        return_crop_region: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.offload_to_cpu:
            self.model = self.model.to(self.device)
        latents = []
        exact_contexts = []
        crop_regions: list[list[int]] = []
        try:
            for batch_idx, segment_key in enumerate(segment_keys):
                tar_path = self.raw_lidar_dir / f"{segment_key}.tar"
                indices = [int(x) for x in frame_indices[batch_idx].tolist()]
                range_maps, _ = _load_lidar_window(tar_path, indices, pad_last=self.args.pad_lidar_last)
                input_tensor = _preprocess_range_maps(range_maps, self.args).to(
                    device=self.args.device,
                    dtype=getattr(torch, self.args.dtype),
                )
                latent, exact_context_latent, crop_region, _ = _extract_official_ltcv_latent(self.model, input_tensor)
                latents.append(latent.squeeze(0).detach())
                if return_exact_context:
                    if exact_context_latent is None:
                        raise RuntimeError("Tokenizer did not return exact_context_latent for LiDAR batch encode.")
                    exact_contexts.append(exact_context_latent.squeeze(0).detach())
                if return_crop_region:
                    crop_regions.append(crop_region)
        finally:
            if self.offload_to_cpu:
                self.model = self.model.to("cpu")
                torch.cuda.empty_cache()
        latent_batch = torch.stack(latents, dim=0)
        outputs: list[torch.Tensor] = [latent_batch]
        if return_exact_context:
            outputs.append(torch.stack(exact_contexts, dim=0))
        if return_crop_region:
            outputs.append(torch.tensor(crop_regions, dtype=torch.int64))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class OnlineVideoEncoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mode = args.video_tokenizer
        self.device = torch.device(args.device)
        self.tokenizer = None
        if self.mode == "wan2pt1":
            from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

            kwargs: dict[str, Any] = {
                "name": "wan2pt1_tokenizer",
                "s3_credential_path": args.wan_s3_credential_path,
                "temporal_window": 4,
            }
            resolved_vae_path = resolve_wan_vae_path(args.wan_vae_path)
            if resolved_vae_path:
                kwargs["vae_pth"] = resolved_vae_path
                print(f"[video-tokenizer] resolved wan2pt1 VAE path: {resolved_vae_path}")
            else:
                print(
                    "[video-tokenizer] wan2pt1 VAE not found under HF_HOME; "
                    "falling back to the default checkpoint URI/S3 path."
                )
            self.tokenizer = Wan2pt1VAEInterface(chunk_duration=args.num_video_frames, load_mean_std=False, **kwargs)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw_downsample":
            return self._encode_raw_downsample(video)
        return self._encode_wan2pt1(video)

    def _normalize_video(self, video: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.device if device is None else device
        return video.to(device=target_device, dtype=torch.float32) / 127.5 - 1.0

    def _encode_raw_downsample(self, video: torch.Tensor) -> torch.Tensor:
        video = self._normalize_video(video, device=torch.device("cpu"))
        bsz, channels, view_times_frames, height, width = video.shape
        num_views = len(WAYMO_CAMERAS)
        frames_per_view = view_times_frames // num_views
        video = video.reshape(bsz, channels, num_views, frames_per_view, height, width)
        latent_frame_ids = (
            torch.linspace(
                0,
                frames_per_view - 1,
                steps=self.args.latent_frames,
                device=video.device,
            )
            .round()
            .long()
        )
        video = video.index_select(dim=3, index=latent_frame_ids)
        video = rearrange(video, "b c v t h w -> (b v t) c h w")
        video = F.interpolate(video, size=(90, 160), mode="bilinear", align_corners=False)
        video = video.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        video = rearrange(video, "(b v t) c h w -> b c (v t) h w", b=bsz, v=num_views, t=self.args.latent_frames)
        return video.contiguous()

    def _encode_wan2pt1(self, video: torch.Tensor) -> torch.Tensor:
        assert self.tokenizer is not None
        video = self._normalize_video(video)
        bsz, channels, view_times_frames, height, width = video.shape
        num_views = len(WAYMO_CAMERAS)
        frames_per_view = view_times_frames // num_views
        video = rearrange(video, "b c (v t) h w -> (b v) c t h w", v=num_views, t=frames_per_view)

        chunks = []
        mini_batch = max(1, self.args.video_tokenizer_batch_size)
        for start in range(0, video.shape[0], mini_batch):
            chunks.append(self.tokenizer.encode(video[start : start + mini_batch]).detach())
        latent = torch.cat(chunks, dim=0)
        latent = rearrange(latent, "(b v) c t h w -> b c (v t) h w", b=bsz, v=num_views)
        return latent.contiguous()


class VideoToLidarConvBaseline(nn.Module):
    def __init__(self, hidden_channels: int, latent_frames: int):
        super().__init__()
        self.latent_frames = latent_frames
        self.video_encoder = nn.Sequential(
            nn.Conv3d(16, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.lidar_encoder = nn.Sequential(
            nn.Conv3d(16, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.trunk = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Conv3d(hidden_channels, 16, kernel_size=3, padding=1)

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, noisy_lidar: torch.Tensor, video_latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, _, _, target_h, target_w = noisy_lidar.shape
        num_views = video_latent.shape[2] // self.latent_frames
        video = video_latent.reshape(
            bsz,
            video_latent.shape[1],
            num_views,
            self.latent_frames,
            video_latent.shape[-2],
            video_latent.shape[-1],
        ).mean(dim=2)
        video = F.interpolate(
            video,
            size=(self.latent_frames, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        h = self.lidar_encoder(noisy_lidar) + self.video_encoder(video)
        t_emb = self.time_mlp(self.timestep_embedding(t, h.shape[1])).view(bsz, h.shape[1], 1, 1, 1)
        h = h + t_emb
        return self.out(self.trunk(h))


def maybe_resize_lidar(
    lidar: torch.Tensor,
    train_height: int,
    train_width: int,
    allow_resize_for_smoke: bool,
) -> torch.Tensor:
    if lidar.shape[-2:] == (train_height, train_width):
        return lidar
    if not allow_resize_for_smoke:
        raise ValueError(
            "LiDAR baseline must use the normal full-width S3 latent size "
            f"{NORMAL_LIDAR_LATENT_HW}. "
            f"Got latent shape {tuple(lidar.shape)} and "
            f"requested train size {(train_height, train_width)}. "
            "Pass --allow-lidar-resize-for-smoke only for explicit local smoke tests."
        )
    return F.interpolate(lidar, size=(lidar.shape[2], train_height, train_width), mode="trilinear", align_corners=False)


def resolve_wan_vae_path(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    hf_root = Path(hf_home)
    repo_root = hf_root / "hub" / WAN_HF_REPO_CACHE
    candidates = [
        repo_root / "snapshots" / WAN_HF_REVISION / WAN_HF_FILENAME,
        *sorted(repo_root.glob("snapshots/*/tokenizer.pth")),
        *sorted(hf_root.rglob("Wan2.1_VAE.pth")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def validate_one_way_policy() -> None:
    policy = VideoLidarAttentionPolicy(mode="video_to_lidar", cross_frame_rule="same_step")
    mask = build_video_lidar_attention_mask(
        video_seq_len=16,
        lidar_seq_len=16,
        policy=policy,
        video_tokens_per_step=2,
        lidar_tokens_per_step=2,
    )
    assert not mask[:16, 16:].any(), "video query rows must not attend to LiDAR keys in video_to_lidar mode"
    assert mask[16:, :16].any(), "LiDAR query rows should attend to video keys in video_to_lidar mode"


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("/data2/waymo_video_to_lidar_baseline") / f"{args.split}_{args.video_tokenizer}_{timestamp}"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.dry_run:
        args.max_steps = min(args.max_steps, 1)
        args.save_every = max(args.save_every, 1)
    if not args.allow_lidar_resize_for_smoke and (args.train_height, args.train_width) != NORMAL_LIDAR_LATENT_HW:
        raise ValueError(
            "LiDAR baseline uses the normal full-width S3 latent size by default: "
            f"{NORMAL_LIDAR_LATENT_HW}. Remove --train-height/--train-width overrides, "
            "or pass --allow-lidar-resize-for-smoke for a non-baseline smoke run."
        )

    validate_one_way_policy()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_waymo_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("No Waymo batches available for baseline training.")

    video_encoder = OnlineVideoEncoder(args)
    lidar_encoder = OnlineLidarS3Encoder(args)
    model = VideoToLidarConvBaseline(args.hidden_channels, args.latent_frames).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.device.startswith("cuda") and args.precision == "bfloat16"

    print(f"[train] output_dir={output_dir}")
    print(f"[train] video_tokenizer={args.video_tokenizer} lidar_tokenizer=S3-online")
    print(f"[train] dataset_size={len(dataset)} batch_size={args.batch_size} max_steps={args.max_steps}")

    step = 0
    running = True
    while running:
        for batch in loader:
            step += 1
            sample_keys = batch["__key__"]
            segment_keys = batch["waymo_segment_key"]
            video = batch["video"]
            lidar_frame_indices = batch["waymo_lidar_frame_indices"]

            with torch.no_grad():
                video_latent = video_encoder.encode(video)
                clean_lidar = lidar_encoder.encode_batch(segment_keys, lidar_frame_indices)
                clean_lidar = clean_lidar.to(device=args.device, dtype=torch.float32)
                clean_lidar = maybe_resize_lidar(
                    clean_lidar,
                    args.train_height,
                    args.train_width,
                    args.allow_lidar_resize_for_smoke,
                )

            noise = torch.randn_like(clean_lidar)
            t = torch.rand(clean_lidar.shape[0], device=clean_lidar.device).clamp(1e-4, 1.0 - 1e-4)
            t_view = t.view(-1, 1, 1, 1, 1)
            noisy_lidar = (1.0 - t_view) * noise + t_view * clean_lidar
            target = clean_lidar - noise

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(noisy_lidar, video_latent.to(device=args.device, dtype=torch.float32), t)
                loss = F.mse_loss(pred.float(), target.float())
            loss.backward()
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                print(
                    f"[train] step={step} loss={loss.item():.6f} "
                    f"video_latent={tuple(video_latent.shape)} lidar_latent={tuple(clean_lidar.shape)} "
                    f"sample={sample_keys[0]}"
                )

            if step % args.save_every == 0 or step >= args.max_steps:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[train] saved {ckpt_path}")

            if step >= args.max_steps:
                running = False
                break

    print("[train] done")


if __name__ == "__main__":
    main()
