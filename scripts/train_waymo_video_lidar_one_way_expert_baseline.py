# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the opt-in FastWAM-style one-way video->LiDAR expert baseline."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cosmos_transfer2._src.predict2_multiview.datasets.multiview import collate_fn
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_one_way_expert import (
    DEFAULT_WAYMO_VIDEO_CHECKPOINT,
    DEFAULT_WAYMO_VIDEO_CONFIG,
    build_waymo_video_lidar_one_way_expert_baseline,
    configure_sdpa_backends,
)
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_joint_policy import (
    VideoLidarAttentionPolicy,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_waymo_video_to_lidar_baseline import (  # noqa: E402
    NORMAL_LIDAR_LATENT_HW,
    OnlineLidarS3Encoder,
    OnlineVideoEncoder,
    build_waymo_dataset,
    maybe_resize_lidar,
    validate_one_way_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--caption-json-path", default="/data/waymo/waymo_multiview_texts.json")
    parser.add_argument("--raw-lidar-root", default="/data2/rds_hq_waymo/lidar_tokenizer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--frozen-video-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--paired-latent-cache-dir",
        default=None,
        help="Optional directory of paired video/LiDAR latent .pt files from cache_waymo_paired_video_lidar_latents.py.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional one-way expert checkpoint .pt to resume from. Loads lidar_expert and, by default, optimizer.",
    )
    parser.add_argument(
        "--resume-load-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load optimizer state from --resume-checkpoint when present.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--video-tokenizer", default="raw_downsample", choices=["raw_downsample", "wan2pt1"])
    parser.add_argument("--video-tokenizer-batch-size", type=int, default=1)
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--video-expert-checkpoint", default=DEFAULT_WAYMO_VIDEO_CHECKPOINT)
    parser.add_argument("--video-expert-config", default=DEFAULT_WAYMO_VIDEO_CONFIG)
    parser.add_argument("--lidar-attention-backend", default="torch", choices=["torch", "minimal_a2a"])
    parser.add_argument("--lidar-num-blocks", type=int, default=None)
    parser.add_argument("--cross-frame-rule", default="all", choices=["all", "same_step"])
    parser.add_argument("--sdpa-backends", default="flash_mem_math", choices=["flash_only", "flash_mem", "flash_mem_math"])
    parser.add_argument("--video-kv-every-n-layers", type=int, default=1)
    parser.add_argument(
        "--checkpoint-lidar-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activation-checkpoint LiDAR one-way blocks. Disable when memory allows to avoid recompute.",
    )
    parser.add_argument(
        "--empty-cache-after-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.empty_cache() after online encoders. Disable for speed when memory is sufficient.",
    )
    parser.add_argument("--profile-step", type=int, default=0)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--independent-video-timesteps", action="store_true")
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=112)
    parser.add_argument(
        "--allow-lidar-resize-for-smoke",
        action="store_true",
        help="Allow non-64x112 LiDAR latent resizing only for local smoke tests.",
    )
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
    parser.add_argument(
        "--offload-lidar-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the LiDAR tokenizer to GPU only during encode_batch to free memory for one-way expert training.",
    )
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
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("/data2/waymo_video_lidar_one_way_expert") / f"{args.split}_{args.video_tokenizer}_{timestamp}"


def load_resume_state(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    if args.resume_checkpoint is None:
        return 0
    checkpoint_path = Path(args.resume_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"--resume-checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.lidar_expert.load_state_dict(checkpoint["lidar_expert"], strict=True)
    if args.resume_load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    resume_step = int(checkpoint.get("step", 0))
    if args.max_steps <= resume_step:
        raise ValueError(
            f"--max-steps ({args.max_steps}) must be larger than resumed step ({resume_step}) "
            f"from {checkpoint_path}"
        )
    return resume_step


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_distributed() -> bool:
    _, _, world_size = distributed_env()
    return world_size > 1


def is_rank0() -> bool:
    rank, _, _ = distributed_env()
    return rank == 0


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    rank, local_rank, world_size = distributed_env()
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    return rank, local_rank, world_size


def enable_cuda_sdpa_flash(policy: str) -> None:
    configure_sdpa_backends(policy)
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(policy in ("flash_mem", "flash_mem_math"))
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(policy == "flash_mem_math")


def maybe_start_profiler(args: argparse.Namespace, step: int, rank: int, output_dir: Path) -> torch.profiler.profile | None:
    if args.profile_step <= 0 or step != args.profile_step or rank != 0:
        return None
    profile_dir = Path(args.profile_dir) if args.profile_dir else output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    profiler.start()
    return profiler


def stop_profiler(
    profiler: torch.profiler.profile | None,
    *,
    args: argparse.Namespace,
    step: int,
    output_dir: Path,
) -> None:
    if profiler is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    profiler.stop()
    profile_dir = Path(args.profile_dir) if args.profile_dir else output_dir / "profiles"
    trace_path = profile_dir / f"step_{step:06d}_rank0.json"
    profiler.export_chrome_trace(str(trace_path))
    print(f"[profile] saved {trace_path}")


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


NORMAL_VIDEO_LATENT_SHAPE = (16, 40, 90, 160)
NORMAL_LIDAR_LATENT_SHAPE = (16, 8, 64, 112)


def _require_shape(name: str, tensor: Tensor, expected: tuple[int, ...], path: Path) -> Tensor:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{path}: expected {name} shape {expected}, got {tuple(tensor.shape)}")
    return tensor


class PairedLatentCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str | Path, limit_samples: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.paths = sorted(self.cache_dir.glob("*.pt"))
        if limit_samples is not None:
            self.paths = self.paths[:limit_samples]
        if not self.paths:
            raise RuntimeError(f"No paired latent .pt files found in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        payload = torch.load(path, map_location="cpu")
        video_latent = _require_shape("video_latent", payload["video_latent"], NORMAL_VIDEO_LATENT_SHAPE, path)
        lidar_latent = _require_shape("lidar_latent", payload["lidar_latent"], NORMAL_LIDAR_LATENT_SHAPE, path)
        return {
            "__key__": str(payload.get("sample_key", path.stem)),
            "video_latent": video_latent,
            "lidar_latent": lidar_latent,
            "segment_key": payload.get("segment_key", ""),
            "cache_path": str(path),
        }


def collate_paired_latent_cache(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "__key__": [str(item["__key__"]) for item in items],
        "video_latent": torch.stack([item["video_latent"] for item in items], dim=0),
        "lidar_latent": torch.stack([item["lidar_latent"] for item in items], dim=0),
        "segment_key": [str(item["segment_key"]) for item in items],
        "cache_path": [str(item["cache_path"]) for item in items],
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)
    enable_cuda_sdpa_flash(args.sdpa_backends)
    torch.manual_seed(args.seed + rank)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.dry_run:
        args.max_steps = min(args.max_steps, 1)
        args.save_every = max(args.save_every, 1)
    if not args.allow_lidar_resize_for_smoke and (args.train_height, args.train_width) != NORMAL_LIDAR_LATENT_HW:
        raise ValueError(
            "LiDAR one-way expert baseline uses the normal S3 latent size by default: "
            f"{NORMAL_LIDAR_LATENT_HW}. Remove --train-height/--train-width overrides, "
            "or pass --allow-lidar-resize-for-smoke for a non-baseline smoke run."
        )

    validate_one_way_policy()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    ckpt_dir = output_dir / "checkpoints"
    if is_rank0():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    use_paired_cache = args.paired_latent_cache_dir is not None
    dataset = (
        PairedLatentCacheDataset(args.paired_latent_cache_dir, limit_samples=args.limit_samples)
        if use_paired_cache
        else build_waymo_dataset(args)
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_paired_latent_cache if use_paired_cache else collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("No Waymo batches available for one-way expert baseline training.")

    video_encoder = None if use_paired_cache else OnlineVideoEncoder(args)
    lidar_encoder = None if use_paired_cache else OnlineLidarS3Encoder(args)
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
        checkpoint_lidar_blocks=args.checkpoint_lidar_blocks,
        policy=VideoLidarAttentionPolicy(
            mode="video_to_lidar",
            cross_frame_rule=args.cross_frame_rule,
        ),
    )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    trainable_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(trainable_model.lidar_expert.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_step = load_resume_state(args=args, model=trainable_model, optimizer=optimizer)
    if args.resume_checkpoint is not None:
        barrier()
    amp_enabled = args.device.startswith("cuda") and args.precision == "bfloat16"

    if is_rank0():
        print(f"[train] output_dir={output_dir}")
        latent_source = f"paired-cache:{args.paired_latent_cache_dir}" if use_paired_cache else "online"
        print(
            f"[train] mode=one_way_expert video_tokenizer={args.video_tokenizer} "
            f"lidar_tokenizer=S3-online latent_source={latent_source}"
        )
        print(
            f"[train] cross_frame_rule={args.cross_frame_rule} "
            f"sdpa_backends={args.sdpa_backends} "
            f"lidar_num_blocks={args.lidar_num_blocks or 'full'} "
            f"video_kv_every_n_layers={args.video_kv_every_n_layers} "
            f"checkpoint_lidar_blocks={args.checkpoint_lidar_blocks} "
            f"empty_cache_after_encode={args.empty_cache_after_encode} "
            f"sdpa_flash_enabled={torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else False} "
            f"sdpa_math_enabled={torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else False}"
        )
        print(
            f"[train] frozen_video_ckpt={args.video_expert_checkpoint} "
            f"dataset_size={len(dataset)} batch_size_per_rank={args.batch_size} "
            f"world_size={world_size} global_batch={args.batch_size * world_size * args.gradient_accumulation_steps} "
            f"max_steps={args.max_steps}"
        )
        if args.resume_checkpoint is not None:
            print(
                f"[train] resumed checkpoint={args.resume_checkpoint} "
                f"resume_step={resume_step} resume_load_optimizer={args.resume_load_optimizer}"
            )

    step = resume_step
    epoch = 0
    last_log_time = time.perf_counter()
    last_log_step = step
    first_timed_log = True
    running = True
    while running:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            step += 1
            profiler = maybe_start_profiler(args, step, rank, output_dir)
            with torch.no_grad():
                sample_keys = batch["__key__"]
                if use_paired_cache:
                    clean_video = batch["video_latent"].to(device=args.device, dtype=torch.float32)
                    clean_lidar = batch["lidar_latent"].to(device=args.device, dtype=torch.float32)
                else:
                    assert video_encoder is not None and lidar_encoder is not None
                    clean_video = video_encoder.encode(batch["video"]).to(device=args.device, dtype=torch.float32)
                    clean_lidar = lidar_encoder.encode_batch(
                        batch["waymo_segment_key"],
                        batch["waymo_lidar_frame_indices"],
                    )
                    clean_lidar = clean_lidar.to(device=args.device, dtype=torch.float32)
                    clean_lidar = maybe_resize_lidar(
                        clean_lidar,
                        args.train_height,
                        args.train_width,
                        args.allow_lidar_resize_for_smoke,
                    )
                if args.empty_cache_after_encode and args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            video_noise = torch.randn_like(clean_video)
            lidar_noise = torch.randn_like(clean_lidar)
            lidar_t = torch.rand(clean_lidar.shape[0], device=clean_lidar.device).clamp(1e-4, 1.0 - 1e-4)
            if args.independent_video_timesteps:
                video_t = torch.rand(clean_video.shape[0], device=clean_video.device).clamp(1e-4, 1.0 - 1e-4)
            else:
                video_t = lidar_t

            noisy_video = (1.0 - video_t.view(-1, 1, 1, 1, 1)) * video_noise + video_t.view(-1, 1, 1, 1, 1) * clean_video
            noisy_lidar = (1.0 - lidar_t.view(-1, 1, 1, 1, 1)) * lidar_noise + lidar_t.view(-1, 1, 1, 1, 1) * clean_lidar
            lidar_target = clean_lidar - lidar_noise

            if (step - 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(
                    noisy_video=noisy_video,
                    noisy_lidar=noisy_lidar,
                    video_timesteps=video_t,
                    lidar_timesteps=lidar_t,
                    return_video_pred=False,
                )
                loss = F.mse_loss(outputs["lidar_pred"].float(), lidar_target.float())
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            should_step = step % args.gradient_accumulation_steps == 0 or step >= args.max_steps
            if should_step:
                optimizer.step()

            if step == 1 or step % args.log_every == 0:
                reduced_loss = reduce_mean(loss.detach()) * args.gradient_accumulation_steps
            if is_rank0() and (step == 1 or step % args.log_every == 0):
                now = time.perf_counter()
                sec_per_step = (now - last_log_time) / max(step - last_log_step, 1)
                warmup_label = " warmup=true" if first_timed_log else ""
                last_log_time = now
                last_log_step = step
                first_timed_log = False
                print(
                    f"[train] step={step} loss={reduced_loss.item():.6f} "
                    f"sec_per_step={sec_per_step:.2f}{warmup_label} "
                    f"video_latent={tuple(clean_video.shape)} lidar_latent={tuple(clean_lidar.shape)} "
                    f"sample={sample_keys[0]}"
                )

            if is_rank0() and (step % args.save_every == 0 or step >= args.max_steps):
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "lidar_expert": trainable_model.lidar_expert.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "resume_step": resume_step,
                        "video_expert_checkpoint": args.video_expert_checkpoint,
                        "video_expert_config": args.video_expert_config,
                        "world_size": world_size,
                    },
                    ckpt_path,
                )
                print(f"[train] saved {ckpt_path}")

            if step >= args.max_steps:
                running = False
                stop_profiler(profiler, args=args, step=step, output_dir=output_dir)
                break
            stop_profiler(profiler, args=args, step=step, output_dir=output_dir)

    barrier()
    if is_rank0():
        print("[train] done")
    cleanup_distributed()


if __name__ == "__main__":
    main()
