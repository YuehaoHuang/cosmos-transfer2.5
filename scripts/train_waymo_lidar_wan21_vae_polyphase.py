# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune Wan2.1 VAE on Waymo LiDAR polyphase range-map inputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from scripts.smoke_waymo_lidar_wan21_vae import (
    load_raw_range_maps,
    natural_key,
    pad_video_tensor_spatial,
    prepend_lidar_utils_repo,
    preprocess_range_maps,
)


DEFAULT_WAN_REPO = "/team/hyh/code/Wan2.1"
DEFAULT_RAW_LIDAR_ROOT = "/team/hyh/data/rds_hq_waymo"
DEFAULT_VAE_PATH = (
    "/team/hyh/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/"
    "snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", default=DEFAULT_WAN_REPO)
    parser.add_argument("--wan-vae-path", default=DEFAULT_VAE_PATH)
    parser.add_argument("--raw-lidar-root", default=DEFAULT_RAW_LIDAR_ROOT)
    parser.add_argument("--split", default="training")
    parser.add_argument("--lidar-utils-repo", default="/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-tars", type=int, default=0)
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=2650)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--range-map-layout", default="polyphase3_repeat", choices=["polyphase3_repeat", "polyphase3_interp"])
    parser.add_argument("--polyphase-roll", type=int, default=640)
    parser.add_argument("--loss-mse-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-scope", default="full", choices=["full", "decoder"])
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--activation-checkpoint", default="none", choices=["none", "vae"])
    parser.add_argument("--activation-offload", default="none", choices=["none", "cpu"])
    parser.add_argument("--loss-in-forward", action="store_true")
    parser.add_argument("--detach-temporal-cache-gradient", action="store_true")
    parser.add_argument("--resume-train-state", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
    return rank, world_size, local_rank, distributed


def is_main_process(rank: int) -> bool:
    return rank == 0


def raw_tar_paths(args: argparse.Namespace) -> list[Path]:
    root = Path(args.raw_lidar_root) / args.split / "lidar_raw"
    paths = sorted(root.glob("*.tar"))
    if args.limit_tars > 0:
        paths = paths[: args.limit_tars]
    if not paths:
        raise FileNotFoundError(f"No raw lidar tar files found in {root}")
    return paths


def count_raw_frames(tar_path: Path) -> int:
    with tarfile.open(tar_path, "r") as tar_handle:
        return len(
            sorted(
                (name for name in tar_handle.getnames() if name.endswith(".lidar_raw.npz")),
                key=natural_key,
            )
        )


def load_dataset_index(
    args: argparse.Namespace,
    *,
    main_process: bool,
    distributed: bool,
) -> tuple[list[Path], dict[Path, int]]:
    if main_process:
        tar_paths = raw_tar_paths(args)
        payload: list[str] = [str(path) for path in tar_paths]
    else:
        payload = []

    if distributed:
        objects: list[Any] = [payload]
        dist.broadcast_object_list(objects, src=0)
        payload = objects[0]

    tar_paths = [Path(path) for path in payload]
    return tar_paths, {}


def make_preprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        downsample_factor_row=1,
        downsample_factor_col=1,
        downsample_method="scatter_min",
        repeat_row=1,
        repeat_col=1,
        input_channel_mode="repeat_depth",
        decode_channel_mode="mean",
        max_range=args.max_range,
        min_range=args.min_range,
        min_value=args.min_value,
        range_map_layout=args.range_map_layout,
        polyphase_roll=args.polyphase_roll,
    )


def load_batch(
    tar_paths: list[Path],
    frame_counts: dict[Path, int],
    args: argparse.Namespace,
    preprocess_args: argparse.Namespace,
) -> torch.Tensor:
    samples = []
    for _ in range(args.batch_size):
        tar_path = random.choice(tar_paths)
        frame_count = frame_counts.get(tar_path)
        if frame_count is None:
            frame_count = count_raw_frames(tar_path)
            frame_counts[tar_path] = frame_count
        max_start = max(0, frame_count - args.num_frames)
        frame_start = random.randint(0, max_start)
        range_maps, _ = load_raw_range_maps(
            tar_path,
            frame_start=frame_start,
            num_frames=args.num_frames,
            pad_last=True,
            n_rows=args.native_n_rows,
            n_cols=args.native_n_cols,
            max_projection_range=args.projection_max_range,
        )
        tensor, _, _ = preprocess_range_maps(range_maps, preprocess_args)
        samples.append(tensor.squeeze(0))
    return torch.stack(samples, dim=0).contiguous()


def load_trainable_wan_vae(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(Path(args.wan_repo).resolve()))
    from wan.modules.vae import _video_vae  # type: ignore

    model = _video_vae(pretrained_path=args.wan_vae_path, z_dim=16, device=device)
    model = model.to(device=device).train().requires_grad_(True)
    if args.train_scope == "decoder":
        model.requires_grad_(False)
        model.conv2.requires_grad_(True)
        model.decoder.requires_grad_(True)
    mean = torch.tensor(
        [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
        dtype=torch.float32,
        device=device,
    )
    std = torch.tensor(
        [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
         3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
        dtype=torch.float32,
        device=device,
    )
    return model, [mean, 1.0 / std]


class WanVAERoundTrip(nn.Module):
    def __init__(self, vae: nn.Module, scale: list[torch.Tensor], train_scope: str, activation_checkpoint: str):
        super().__init__()
        self.vae = vae
        self.train_scope = train_scope
        self.activation_checkpoint = activation_checkpoint
        self.register_buffer("mean", scale[0].detach().clone(), persistent=False)
        self.register_buffer("inv_std", scale[1].detach().clone(), persistent=False)

    @property
    def scale(self) -> list[torch.Tensor]:
        return [self.mean, self.inv_std]

    def forward(
        self,
        batch: torch.Tensor,
        *,
        return_loss: bool = False,
        loss_mse_weight: float = 0.1,
        crop_height: int | None = None,
        crop_width: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.train_scope == "decoder":
            with torch.no_grad():
                latent = self.vae.encode(batch, self.scale).detach()
            if batch.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            if self.activation_checkpoint == "vae":
                latent = checkpoint(lambda x: self.vae.encode(x, self.scale), batch, use_reentrant=False)
            else:
                latent = self.vae.encode(batch, self.scale)
        if return_loss:
            target = batch
            if crop_height is not None and crop_width is not None:
                target = target[..., :crop_height, :crop_width]

            def decode_loss(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                recon = self.vae.decode(z, self.scale)
                if crop_height is not None and crop_width is not None:
                    recon = recon[..., :crop_height, :crop_width]
                l1 = F.l1_loss(recon, target)
                mse = F.mse_loss(recon, target) if loss_mse_weight != 0.0 else recon.new_zeros(())
                loss = l1 + loss_mse_weight * mse
                return loss, l1.detach(), mse.detach()

            if self.activation_checkpoint == "vae":
                loss, l1_metric, mse_metric = checkpoint(decode_loss, latent, use_reentrant=False)
            else:
                loss, l1_metric, mse_metric = decode_loss(latent)
            return loss, l1_metric, mse_metric, latent

        if self.activation_checkpoint == "vae":
            recon = checkpoint(lambda z: self.vae.decode(z, self.scale), latent, use_reentrant=False)
        else:
            recon = self.vae.decode(latent, self.scale)
        return latent, recon


def wrap_fsdp_if_needed(model: nn.Module, args: argparse.Namespace, device: torch.device, amp_dtype: torch.dtype) -> nn.Module:
    if not args.fsdp:
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    mixed_precision = None
    if amp_dtype != torch.float32:
        mixed_precision = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        use_orig_params=True,
    )


def wrap_ddp_if_needed(model: nn.Module, args: argparse.Namespace, local_rank: int, distributed: bool) -> nn.Module:
    if not distributed or args.fsdp:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )


def unwrap_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "vae."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    rank: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / f"step_{step:06d}_model.pt"
    train_path = ckpt_dir / f"step_{step:06d}_train_state.pt"
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        if rank == 0:
            vae_state = unwrap_vae_state_dict(state_dict)
            torch.save(vae_state, model_path)
            torch.save({"step": step, "model": vae_state, "args": vars(args)}, train_path)
    else:
        base_model = getattr(model, "module", model)
        vae_state = base_model.vae.state_dict() if isinstance(base_model, WanVAERoundTrip) else base_model.state_dict()
        if rank == 0:
            torch.save(vae_state, model_path)
            torch.save(
                {
                    "step": step,
                    "model": vae_state,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                train_path,
            )
    if rank == 0:
        (ckpt_dir / "latest_model.txt").write_text(str(model_path) + "\n", encoding="utf-8")
        (ckpt_dir / "latest_train_state.txt").write_text(str(train_path) + "\n", encoding="utf-8")
        print(f"[save] step={step} model={model_path}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, distributed = init_distributed(args)
    main_process = is_main_process(rank)
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/waymo_lidar_vae_finetune/polyphase3_repeat_roll640_{stamp}"
    output_dir = Path(args.output_dir)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.detach_temporal_cache_gradient:
        os.environ["WAN_VAE_DETACH_CACHE_GRAD"] = "1"
    prepend_lidar_utils_repo(args.lidar_utils_repo)

    tar_paths, frame_counts = load_dataset_index(args, main_process=main_process, distributed=distributed)
    total_windows = "lazy"
    if main_process:
        print(
            f"[data] tars={len(tar_paths)} windows~={total_windows} num_frames={args.num_frames} "
            f"layout={args.range_map_layout} roll={args.polyphase_roll} world_size={world_size} fsdp={args.fsdp}",
            flush=True,
        )
    if args.dry_run:
        return

    device = torch.device(args.device)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.amp_dtype]
    vae, scale = load_trainable_wan_vae(args, device)
    start_step = 0
    resume_state: dict[str, Any] | None = None
    if args.resume_train_state:
        resume_state = torch.load(args.resume_train_state, map_location=device)
        vae.load_state_dict(resume_state["model"])
        start_step = int(resume_state["step"])
        if main_process:
            print(f"[resume] {args.resume_train_state} step={start_step}", flush=True)

    model = WanVAERoundTrip(vae, scale, args.train_scope, args.activation_checkpoint)
    model = wrap_fsdp_if_needed(model, args, device, amp_dtype)
    model = wrap_ddp_if_needed(model, args, local_rank, distributed)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable VAE parameters for train_scope={args.train_scope}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None and "optimizer" in resume_state and not args.fsdp:
        optimizer.load_state_dict(resume_state["optimizer"])

    preprocess_args = make_preprocess_args(args)
    log_path = output_dir / "train_metrics.jsonl"
    for step in range(start_step + 1, args.max_steps + 1):
        batch = load_batch(tar_paths, frame_counts, args, preprocess_args)
        batch, padding = pad_video_tensor_spatial(batch, align=8, pad_value=args.min_value)
        batch_dtype = amp_dtype if amp_dtype != torch.float32 else torch.float32
        batch = batch.to(device=device, dtype=batch_dtype)

        optimizer.zero_grad(set_to_none=True)
        offload_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if args.activation_offload == "cpu"
            else nullcontext()
        )
        with offload_ctx:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
                if args.loss_in_forward:
                    loss, l1, mse, latent = model(
                        batch,
                        return_loss=True,
                        loss_mse_weight=args.loss_mse_weight,
                        crop_height=padding["input_height"],
                        crop_width=padding["input_width"],
                    )
                    target_shape = list(batch[..., : padding["input_height"], : padding["input_width"]].shape)
                else:
                    latent, recon = model(batch)
                    recon = recon[..., : padding["input_height"], : padding["input_width"]]
                    target = batch[..., : padding["input_height"], : padding["input_width"]]
                    target_shape = list(target.shape)
                    l1 = F.l1_loss(recon, target)
                    mse = F.mse_loss(recon, target) if args.loss_mse_weight != 0.0 else recon.new_zeros(())
                    loss = l1 + args.loss_mse_weight * mse
        if device.type == "cuda":
            torch.cuda.empty_cache()
        loss.backward()
        if args.fsdp:
            grad_norm = model.clip_grad_norm_(args.grad_clip).detach().float().item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().float().item()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == 1:
            metrics_tensor = torch.tensor(
                [float(loss.detach()), float(l1.detach()), float(mse.detach()), float(grad_norm)],
                dtype=torch.float32,
                device=device,
            )
            if distributed:
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            item: dict[str, Any] = {
                "step": step,
                "loss": float(metrics_tensor[0].detach().cpu()),
                "l1": float(metrics_tensor[1].detach().cpu()),
                "mse": float(metrics_tensor[2].detach().cpu()),
                "grad_norm": float(metrics_tensor[3].detach().cpu()),
                "latent_shape": list(latent.shape),
                "input_shape": target_shape,
            }
            if main_process:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item) + "\n")
                print("[train] " + json.dumps(item, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(output_dir, step, model, optimizer, args, rank=rank)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
