# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the opt-in FastWAM-style one-way video->LiDAR expert baseline."""

from __future__ import annotations

import argparse
import inspect
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.distributed._tensor.api import DTensor
except ImportError:  # pragma: no cover - older torch builds without DTensor
    DTensor = None

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
    OnlineLidarS3Encoder,
    OnlineVideoEncoder,
    build_waymo_dataset,
    maybe_resize_lidar,
    validate_one_way_policy,
)
from smoke_waymo_lidar_wan21_vae import (  # noqa: E402
    DEFAULT_MPLCONFIGDIR,
    load_raw_range_maps,
    load_wan_tokenizer,
    pad_video_tensor_spatial,
    prepend_lidar_utils_repo,
    preprocess_range_maps,
)
from waymo_lidar_latent_contracts import (  # noqa: E402
    KNOWN_LIDAR_LATENT_CONTRACTS,
    NORMAL_VIDEO_LATENT_SHAPE,
    OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT,
    WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT,
    contract_from_payload,
    lidar_exact_context_shape,
    lidar_latent_hw,
    lidar_latent_shape,
    normalize_lidar_latent_contract,
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--paired-latent-cache-dir",
        default=None,
        help="Optional directory of paired video/LiDAR latent .pt files from cache_waymo_paired_video_lidar_latents.py.",
    )
    parser.add_argument(
        "--precomputed-video-latent-dir",
        default=None,
        help=(
            "Optional directory of real video latent .pt files keyed by sample_key. "
            "When used with --paired-latent-cache-dir, LiDAR fields come from the paired cache "
            "and video_latent is loaded from this directory."
        ),
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
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer fused AdamW on CUDA when the local torch build supports it.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--distributed-parallelism", default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument(
        "--fsdp-shard-size",
        type=int,
        default=0,
        help="FSDP2 shard group size. Default 0 shards across the full torchrun world size.",
    )
    parser.add_argument(
        "--fsdp-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FSDP2 reshard-after-forward for LiDAR expert modules.",
    )
    parser.add_argument("--video-tokenizer", default="raw_downsample", choices=["raw_downsample", "wan2pt1"])
    parser.add_argument("--video-tokenizer-batch-size", type=int, default=1)
    parser.add_argument(
        "--lidar-tokenizer",
        default="s3_ltcv",
        choices=["s3_ltcv", "wan21"],
        help="Online LiDAR tokenizer. paired-cache mode ignores this and infers from cache payloads.",
    )
    parser.add_argument("--wan-vae-path", default=None)
    parser.add_argument("--allow-default-wan-uri", action="store_true")
    parser.add_argument("--wan-s3-credential-path", default="credentials/s3_training.secret")
    parser.add_argument("--video-expert-checkpoint", default=DEFAULT_WAYMO_VIDEO_CHECKPOINT)
    parser.add_argument("--video-expert-config", default=DEFAULT_WAYMO_VIDEO_CONFIG)
    parser.add_argument(
        "--frozen-video-wan-fp32-strategy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep frozen video rotary/modulation on Wan FP32 path. Default False uses BF16 to reduce peak memory.",
    )
    parser.add_argument(
        "--lidar-wan-fp32-strategy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep LiDAR expert rotary/modulation on Wan FP32 path. Default False uses BF16 to reduce peak memory.",
    )
    parser.add_argument(
        "--init-lidar-from-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize same-shape LiDAR expert tensors from the frozen video expert before training/resume.",
    )
    parser.add_argument("--lidar-attention-backend", default="torch", choices=["torch", "minimal_a2a"])
    parser.add_argument(
        "--lidar-num-blocks",
        type=int,
        default=28,
        help="Number of LiDAR expert blocks. Default 28 mirrors the full frozen Waymo video DiT.",
    )
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
    parser.add_argument(
        "--rf-convention",
        default="predict2",
        choices=["predict2", "legacy_forward"],
        help=(
            "RF time/target convention. predict2 matches Cosmos Predict2: "
            "x_t=sigma*noise+(1-sigma)*clean, target=noise-clean, timesteps in [0,1000]. "
            "legacy_forward preserves old one-way checkpoints: x_t=(1-t)*noise+t*clean, target=clean-noise."
        ),
    )
    parser.add_argument(
        "--rf-train-time-distribution",
        default="logitnormal",
        choices=["logitnormal", "uniform"],
        help="Training time sampler for --rf-convention predict2. Cosmos Predict2 Waymo uses logitnormal.",
    )
    parser.add_argument("--rf-shift", type=float, default=5.0, help="Predict2 RF shift before discrete timesteps.")
    parser.add_argument("--rf-num-train-timesteps", type=int, default=1000)
    parser.add_argument(
        "--cached-latent-device-dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Device dtype for cached video/LiDAR latents. auto follows --precision.",
    )
    parser.add_argument(
        "--lidar-latent-contract",
        default="auto",
        choices=["auto", *sorted(KNOWN_LIDAR_LATENT_CONTRACTS)],
        help="LiDAR latent contract. 'auto' infers from paired cache, or uses the official S3 contract online.",
    )
    parser.add_argument("--train-height", type=int, default=64)
    parser.add_argument("--train-width", type=int, default=226)
    parser.add_argument(
        "--allow-lidar-resize-for-smoke",
        action="store_true",
        help="Allow non-64x226 LiDAR latent resizing only for local smoke tests.",
    )
    parser.add_argument("--lidar-tokenizer-repo", default="/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--lidar-utils-repo", default="/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen")
    parser.add_argument("--raw-waymo-root", default="/data2/rds_hq_waymo")
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
    parser.add_argument("--native-n-rows", type=int, default=64)
    parser.add_argument("--native-n-cols", type=int, default=1280)
    parser.add_argument("--projection-max-range", type=float, default=105.0)
    parser.add_argument("--wan-spatial-align", type=int, default=8)
    parser.add_argument("--input-channel-mode", default="repeat_depth", choices=["repeat_depth", "concat_inv_depth"])
    parser.add_argument("--decode-channel-mode", default="mean", choices=["first", "mean", "median", "concat_fuse"])
    parser.add_argument("--inv-depth-threshold", type=float, default=20.0)
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
    parser.add_argument("--latent-frames", type=int, default=8)
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.paired_latent_cache_dir and (Path(args.paired_latent_cache_dir) / "video").exists():
        video_source = "linked_real_video_s3_lidar"
    elif args.precomputed_video_latent_dir:
        video_source = "precomputed_video"
    else:
        video_source = args.video_tokenizer
    return Path("/data2/waymo_video_lidar_one_way_expert") / f"{args.split}_{video_source}_{timestamp}"


def cached_latent_device_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.cached_latent_device_dtype == "auto":
        return torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    return getattr(torch, args.cached_latent_device_dtype)


def build_optimizer(args: argparse.Namespace, params) -> torch.optim.Optimizer:
    kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if args.device.startswith("cuda") and args.fused_optimizer:
        adamw_params = inspect.signature(torch.optim.AdamW).parameters
        if "fused" in adamw_params:
            kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def build_ddp(
    module: torch.nn.Module,
    *,
    local_rank: int,
) -> DDP:
    kwargs = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    ddp_params = inspect.signature(DDP).parameters
    if "gradient_as_bucket_view" in ddp_params:
        kwargs["gradient_as_bucket_view"] = True
    if "static_graph" in ddp_params:
        kwargs["static_graph"] = True
    return DDP(module, **kwargs)


def build_fsdp_mesh(args: argparse.Namespace, world_size: int):
    if world_size <= 1:
        raise ValueError("--distributed-parallelism fsdp requires torchrun with WORLD_SIZE > 1.")
    shard_size = args.fsdp_shard_size or world_size
    if shard_size < 1 or shard_size > world_size:
        raise ValueError(f"--fsdp-shard-size must be in [1, {world_size}], got {shard_size}.")
    if world_size % shard_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by --fsdp-shard-size={shard_size}.")
    if shard_size == world_size:
        return init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    replica_size = world_size // shard_size
    return init_device_mesh("cuda", (replica_size, shard_size), mesh_dim_names=("replicate", "shard"))


@torch.no_grad()
def broadcast_module_states(module: torch.nn.Module, src: int = 0) -> None:
    if not dist.is_initialized():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.detach(), src=src)


def apply_lidar_fsdp(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    world_size: int,
) -> torch.nn.Module:
    if not hasattr(model, "fully_shard_lidar_expert"):
        raise TypeError(f"{type(model).__name__} does not expose fully_shard_lidar_expert().")
    broadcast_module_states(model.lidar_expert, src=0)
    fsdp_mesh = build_fsdp_mesh(args, world_size)
    model.fully_shard_lidar_expert(
        fsdp_mesh,
        reshard_after_forward=args.fsdp_reshard_after_forward,
    )
    return model


def _materialize_tensor_for_checkpoint(tensor: torch.Tensor) -> torch.Tensor:
    if DTensor is not None and isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu()


def materialize_state_dict_for_checkpoint(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: _materialize_tensor_for_checkpoint(value)
        for key, value in module.state_dict().items()
    }


def _move_wan_tokenizer(tokenizer: Any, device: torch.device | str) -> None:
    """Move the Wan2.1 VAE interface internals between CPU and GPU."""

    device = torch.device(device)
    wan = tokenizer.model
    wan.device = str(device)
    wan.mean = wan.mean.to(device=device)
    wan.std = wan.std.to(device=device)
    wan.scale = [wan.mean, 1.0 / wan.std]
    wan.img_mean = wan.img_mean.to(device=device)
    wan.img_std = wan.img_std.to(device=device)
    wan.video_mean = wan.video_mean.to(device=device)
    wan.video_std = wan.video_std.to(device=device)
    wan.model = wan.model.to(device)


class OnlineLidarWan21Encoder:
    def __init__(self, args: argparse.Namespace):
        os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
        prepend_lidar_utils_repo(args.lidar_utils_repo)
        wan_args = argparse.Namespace(**vars(args))
        wan_args.dtype = args.lidar_tokenizer_dtype
        wan_args.num_frames = args.num_video_frames
        self.args = wan_args
        self.raw_lidar_dir = Path(args.raw_waymo_root) / args.split / "lidar_raw"
        self.device = torch.device(args.device)
        self.offload_to_cpu = bool(getattr(args, "offload_lidar_encoder", False)) and self.device.type == "cuda"
        self.tokenizer = load_wan_tokenizer(wan_args)
        if self.offload_to_cpu:
            _move_wan_tokenizer(self.tokenizer, "cpu")
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_pinned_args(args: argparse.Namespace) -> None:
        expected = WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT
        actual = {
            "native_n_rows": args.native_n_rows,
            "native_n_cols": args.native_n_cols,
            "downsample_factor_row": args.downsample_factor_row,
            "downsample_factor_col": args.downsample_factor_col,
            "repeat_row": args.repeat_row,
            "repeat_col": args.repeat_col,
            "input_channel_mode": args.input_channel_mode,
            "decode_channel_mode": args.decode_channel_mode,
            "wan_spatial_align": args.wan_spatial_align,
        }
        expected_subset = {
            "native_n_rows": expected["native_n_rows"],
            "native_n_cols": expected["native_n_cols"],
            "downsample_factor_row": expected["downsample_factor_row"],
            "downsample_factor_col": expected["downsample_factor_col"],
            "repeat_row": expected["repeat_row"],
            "repeat_col": expected["repeat_col"],
            "input_channel_mode": expected["input_channel_mode"],
            "decode_channel_mode": expected["decode_channel_mode"],
            "wan_spatial_align": expected["wan_spatial_align"],
        }
        if actual != expected_subset:
            raise ValueError(f"Wan21 LiDAR online VAE must use pinned preprocess {expected_subset}, got {actual}.")

    @torch.no_grad()
    def encode_batch(self, segment_keys: list[str], frame_indices: torch.Tensor) -> torch.Tensor:
        self._validate_pinned_args(self.args)
        enable_cuda_sdpa_flash("flash_mem_math")
        if self.offload_to_cpu:
            _move_wan_tokenizer(self.tokenizer, self.device)
        latents = []
        expected_shape = tuple(WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT["latent_shape"])
        try:
            for batch_idx, segment_key in enumerate(segment_keys):
                tar_path = self.raw_lidar_dir / f"{segment_key}.tar"
                frame_start = int(frame_indices[batch_idx][0].item())
                range_maps, _ = load_raw_range_maps(
                    tar_path,
                    frame_start=frame_start,
                    num_frames=self.args.num_video_frames,
                    pad_last=self.args.pad_lidar_last,
                    n_rows=self.args.native_n_rows,
                    n_cols=self.args.native_n_cols,
                    max_projection_range=self.args.projection_max_range,
                )
                input_tensor, _, _ = preprocess_range_maps(range_maps, self.args)
                wan_input_tensor, spatial_padding = pad_video_tensor_spatial(
                    input_tensor,
                    align=self.args.wan_spatial_align,
                    pad_value=self.args.min_value,
                )
                latent = self.tokenizer.encode(
                    wan_input_tensor.to(device=self.args.device, dtype=getattr(torch, self.args.lidar_tokenizer_dtype))
                ).detach()
                if tuple(latent.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"{tar_path}: expected Wan21 LiDAR latent batch+{expected_shape}, got {tuple(latent.shape)}"
                    )
                if spatial_padding["input_height"] != int(input_tensor.shape[-2]):
                    raise RuntimeError("Unexpected Wan21 LiDAR spatial padding height mismatch.")
                latents.append(latent.squeeze(0))
        finally:
            if self.offload_to_cpu:
                _move_wan_tokenizer(self.tokenizer, "cpu")
                torch.cuda.empty_cache()
            enable_cuda_sdpa_flash(self.args.sdpa_backends)
        return torch.stack(latents, dim=0)


def sample_rf_train_u(
    *,
    batch_size: int,
    device: torch.device | str,
    distribution: str,
) -> torch.Tensor:
    if distribution == "uniform":
        return torch.rand(batch_size, device=device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    if distribution == "logitnormal":
        return torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32)).clamp(1e-4, 1.0 - 1e-4)
    raise ValueError(f"Unsupported RF train time distribution: {distribution!r}")


def rf_discrete_timesteps(
    u: torch.Tensor,
    *,
    shift: float,
    num_train_timesteps: int,
) -> torch.Tensor:
    u = u.to(dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    shifted = shift * u / (1.0 + (shift - 1.0) * u)
    return shifted * float(num_train_timesteps)


def rf_sigmas_from_timesteps(
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int,
) -> torch.Tensor:
    return (timesteps.to(dtype=torch.float32) / float(num_train_timesteps)).clamp(0.0, 1.0)


def _expand_time_like(time: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return time.to(device=tensor.device, dtype=tensor.dtype).view(time.shape[0], *([1] * (tensor.ndim - 1)))


def make_rf_noisy_and_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    model_timesteps: torch.Tensor,
    *,
    convention: str,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if convention == "predict2":
        sigma = _expand_time_like(
            rf_sigmas_from_timesteps(model_timesteps, num_train_timesteps=num_train_timesteps),
            clean,
        )
        return sigma * noise + (1.0 - sigma) * clean, noise - clean
    if convention == "legacy_forward":
        t = _expand_time_like(model_timesteps, clean)
        return (1.0 - t) * noise + t * clean, clean - noise
    raise ValueError(f"Unsupported RF convention: {convention!r}")


def checkpoint_rf_convention(checkpoint: dict[str, object]) -> str:
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        return "legacy_forward"
    return str(checkpoint_args.get("rf_convention", "legacy_forward"))


def validate_resume_rf_config(args: argparse.Namespace, checkpoint: dict[str, object], checkpoint_path: Path) -> None:
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    ckpt_convention = checkpoint_rf_convention(checkpoint)
    if ckpt_convention != args.rf_convention:
        raise ValueError(
            f"{checkpoint_path} was trained with rf_convention={ckpt_convention!r}, "
            f"but this run requested {args.rf_convention!r}. "
            "Do not mix RF target/sign conventions. Resume with the checkpoint convention, "
            "or start a fresh run with --rf-convention predict2."
        )
    for name in ("rf_shift", "rf_num_train_timesteps"):
        if name not in checkpoint_args:
            continue
        old_value = checkpoint_args[name]
        new_value = getattr(args, name)
        if float(old_value) != float(new_value):
            raise ValueError(
                f"{checkpoint_path} was trained with {name}={old_value}, "
                f"but this run requested {name}={new_value}."
            )


def load_resume_state(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    expected_lidar_contract: str | dict[str, object],
) -> int:
    if args.resume_checkpoint is None:
        return 0
    checkpoint_path = Path(args.resume_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"--resume-checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert_checkpoint_lidar_contract(
        checkpoint,
        checkpoint_path=checkpoint_path,
        expected_contract=expected_lidar_contract,
    )
    validate_resume_rf_config(args, checkpoint, checkpoint_path)
    model.lidar_expert.load_state_dict(checkpoint["lidar_expert"], strict=True)
    if optimizer is not None and args.resume_load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif optimizer is None and args.resume_load_optimizer and "optimizer" in checkpoint and is_rank0():
        print("[train] skip optimizer resume before FSDP wrapping; FSDP optimizer-state resume is not enabled.")
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


def assert_checkpoint_lidar_contract(
    checkpoint: dict[str, object],
    *,
    checkpoint_path: Path,
    expected_contract: str | dict[str, object],
) -> None:
    contract = checkpoint.get("lidar_latent_contract")
    if contract is None:
        checkpoint_args = checkpoint.get("args", {})
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}
        legacy_height = checkpoint_args.get("train_height")
        legacy_width = checkpoint_args.get("train_width")
        legacy_crop_width = checkpoint_args.get("lidar_crop_width")
        legacy_crop_mode = checkpoint_args.get("crop_mode")
        raise ValueError(
            f"{checkpoint_path} is missing lidar_latent_contract metadata. "
            "This usually means it was trained before the 2026-04-21 switch to the official "
            f"64x226 + exact_context LiDAR target (legacy train_hw={(legacy_height, legacy_width)}, "
            f"lidar_crop_width={legacy_crop_width}, crop_mode={legacy_crop_mode}). "
            "Start a fresh run or resume from a checkpoint produced after the online-extractor fix."
        )
    normalized_contract = normalize_lidar_latent_contract(contract, source=checkpoint_path)
    normalized_expected = normalize_lidar_latent_contract(expected_contract, source="requested training run")
    if normalized_contract != normalized_expected:
        raise ValueError(
            f"{checkpoint_path} uses an incompatible LiDAR latent contract: {contract}. "
            f"Expected {normalized_expected}."
        )


def _require_shape(name: str, tensor: Tensor, expected: tuple[int, ...], path: Path) -> Tensor:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{path}: expected {name} shape {expected}, got {tuple(tensor.shape)}")
    return tensor


def _load_precomputed_video_latent(video_latent_dir: Path, sample_key: str, paired_cache_path: Path) -> Tensor:
    video_path = video_latent_dir / f"{sample_key}.pt"
    if not video_path.exists():
        raise FileNotFoundError(
            f"{paired_cache_path}: missing precomputed video latent for sample_key={sample_key!r}: {video_path}"
        )
    video_latent = torch.load(video_path, map_location="cpu")
    if not torch.is_tensor(video_latent):
        raise TypeError(f"{video_path}: expected a tensor video latent, got {type(video_latent).__name__}")
    return _require_shape("video_latent", video_latent, NORMAL_VIDEO_LATENT_SHAPE, video_path)


def _load_lidar_cache_payload(
    path: Path,
    *,
    expected_contract: str | dict[str, object] | None = None,
) -> tuple[dict[str, object], Tensor, Tensor | None, object, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    if "lidar_latent" in payload:
        lidar_latent = payload["lidar_latent"]
    elif "latent" in payload:
        lidar_latent = payload["latent"]
    else:
        raise KeyError(f"{path}: expected either 'lidar_latent' or 'latent' in LiDAR cache payload")
    if not torch.is_tensor(lidar_latent):
        raise TypeError(f"{path}: expected LiDAR latent tensor, got {type(lidar_latent).__name__}")
    lidar_contract = contract_from_payload(payload, tuple(lidar_latent.shape), source=path)
    if expected_contract is not None:
        normalized_expected = normalize_lidar_latent_contract(expected_contract, source=path)
        if lidar_contract != normalized_expected:
            raise ValueError(
                f"{path}: LiDAR latent contract mismatch: got {lidar_contract}, expected {normalized_expected}"
            )
    lidar_latent = _require_shape("lidar_latent", lidar_latent, lidar_latent_shape(lidar_contract), path)

    exact_context_latent = payload.get("exact_context_latent")
    if exact_context_latent is not None:
        expected_exact_shape = lidar_exact_context_shape(lidar_contract)
        if expected_exact_shape is not None:
            exact_context_latent = _require_shape(
                "exact_context_latent",
                exact_context_latent,
                expected_exact_shape,
                path,
            )
    tokenizer_crop_region = payload.get("tokenizer_crop_region")
    return payload, lidar_latent, exact_context_latent, tokenizer_crop_region, lidar_contract


class PairedLatentCacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        limit_samples: int | None = None,
        precomputed_video_latent_dir: str | Path | None = None,
        requested_lidar_contract: str | dict[str, object] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.requested_lidar_contract = (
            None
            if requested_lidar_contract is None
            else normalize_lidar_latent_contract(requested_lidar_contract, source=self.cache_dir)
        )
        self.precomputed_video_latent_dir = (
            None if precomputed_video_latent_dir is None else Path(precomputed_video_latent_dir)
        )
        if self.precomputed_video_latent_dir is not None and not self.precomputed_video_latent_dir.exists():
            raise FileNotFoundError(f"precomputed video latent dir does not exist: {self.precomputed_video_latent_dir}")
        self.video_dir = self.cache_dir / "video"
        self.lidar_dir = self.cache_dir / "lidar"
        self.linked_layout = self.video_dir.exists() or self.lidar_dir.exists()
        if self.linked_layout:
            if not self.video_dir.exists():
                raise FileNotFoundError(f"linked paired cache is missing video dir: {self.video_dir}")
            if not self.lidar_dir.exists():
                raise FileNotFoundError(f"linked paired cache is missing lidar dir: {self.lidar_dir}")
            if self.precomputed_video_latent_dir is not None:
                raise ValueError("--precomputed-video-latent-dir is redundant for linked paired cache layout")
            self.paths = sorted(self.lidar_dir.glob("*.pt"))
        else:
            self.paths = sorted(self.cache_dir.glob("*.pt"))
        if limit_samples is not None:
            self.paths = self.paths[:limit_samples]
        if not self.paths:
            raise RuntimeError(f"No LiDAR latent .pt files found for paired cache in {self.cache_dir}")
        _, _, _, _, inferred_contract = _load_lidar_cache_payload(
            self.paths[0],
            expected_contract=self.requested_lidar_contract,
        )
        self.lidar_contract = inferred_contract

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        payload, lidar_latent, exact_context_latent, tokenizer_crop_region, lidar_contract = _load_lidar_cache_payload(
            path,
            expected_contract=self.lidar_contract,
        )
        sample_key = str(payload.get("sample_key", path.stem))
        if self.linked_layout:
            video_latent = _load_precomputed_video_latent(self.video_dir, sample_key, path)
            video_latent_path = str(self.video_dir / f"{sample_key}.pt")
        elif self.precomputed_video_latent_dir is None:
            video_latent = _require_shape("video_latent", payload["video_latent"], NORMAL_VIDEO_LATENT_SHAPE, path)
            video_latent_path = ""
        else:
            video_latent = _load_precomputed_video_latent(self.precomputed_video_latent_dir, sample_key, path)
            video_latent_path = str(self.precomputed_video_latent_dir / f"{sample_key}.pt")
        return {
            "__key__": sample_key,
            "video_latent": video_latent,
            "lidar_latent": lidar_latent,
            "exact_context_latent": exact_context_latent,
            "tokenizer_crop_region": tokenizer_crop_region,
            "lidar_latent_contract": lidar_contract,
            "segment_key": payload.get("segment_key", ""),
            "cache_path": str(path),
            "video_latent_path": video_latent_path,
            "lidar_latent_path": str(path),
        }


def collate_paired_latent_cache(items: list[dict[str, object]]) -> dict[str, object]:
    batch = {
        "__key__": [str(item["__key__"]) for item in items],
        "video_latent": torch.stack([item["video_latent"] for item in items], dim=0),
        "lidar_latent": torch.stack([item["lidar_latent"] for item in items], dim=0),
        "lidar_latent_contract": items[0]["lidar_latent_contract"],
        "segment_key": [str(item["segment_key"]) for item in items],
        "cache_path": [str(item["cache_path"]) for item in items],
        "video_latent_path": [str(item["video_latent_path"]) for item in items],
        "lidar_latent_path": [str(item["lidar_latent_path"]) for item in items],
    }
    if all(item["exact_context_latent"] is not None for item in items):
        batch["exact_context_latent"] = torch.stack([item["exact_context_latent"] for item in items], dim=0)
    if all(item["tokenizer_crop_region"] is not None for item in items):
        batch["tokenizer_crop_region"] = torch.stack(
            [torch.as_tensor(item["tokenizer_crop_region"], dtype=torch.int64) for item in items],
            dim=0,
        )
    return batch


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

    validate_one_way_policy()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    ckpt_dir = output_dir / "checkpoints"
    if is_rank0():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    use_paired_cache = args.paired_latent_cache_dir is not None
    requested_lidar_contract = (
        None if args.lidar_latent_contract == "auto" else normalize_lidar_latent_contract(args.lidar_latent_contract)
    )
    dataset = (
        PairedLatentCacheDataset(
            args.paired_latent_cache_dir,
            limit_samples=args.limit_samples,
            precomputed_video_latent_dir=args.precomputed_video_latent_dir,
            requested_lidar_contract=requested_lidar_contract,
        )
        if use_paired_cache
        else build_waymo_dataset(args)
    )
    active_lidar_contract = (
        dataset.lidar_contract
        if use_paired_cache and isinstance(dataset, PairedLatentCacheDataset)
        else normalize_lidar_latent_contract(
            requested_lidar_contract
            or (
                WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT
                if args.lidar_tokenizer == "wan21"
                else OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT
            )
        )
    )
    args.lidar_latent_contract = str(active_lidar_contract["version"])
    contract_height, contract_width = lidar_latent_hw(active_lidar_contract)
    if (args.train_height, args.train_width) != (contract_height, contract_width):
        if use_paired_cache:
            args.train_height, args.train_width = contract_height, contract_width
        elif not args.allow_lidar_resize_for_smoke:
            raise ValueError(
                "Online LiDAR training must use the active LiDAR latent contract size "
                f"{(contract_height, contract_width)}. Got {(args.train_height, args.train_width)}."
            )
    if not use_paired_cache:
        expected_online_contract = (
            WAN21_NATIVE64X1280_REPEATROW11_LIDAR_LATENT_CONTRACT
            if args.lidar_tokenizer == "wan21"
            else OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT
        )
        if active_lidar_contract != normalize_lidar_latent_contract(expected_online_contract):
            raise ValueError(
                f"Online LiDAR tokenizer {args.lidar_tokenizer} expects contract "
                f"{expected_online_contract['version']}, got {active_lidar_contract['version']}."
            )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if world_size > 1 else None
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_paired_latent_cache if use_paired_cache else collate_fn,
        "pin_memory": args.device.startswith("cuda"),
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)
    if len(loader) == 0:
        raise RuntimeError("No Waymo batches available for one-way expert baseline training.")

    use_precomputed_video_latents = (not use_paired_cache) and args.precomputed_video_latent_dir is not None
    video_encoder = None if use_paired_cache or use_precomputed_video_latents else OnlineVideoEncoder(args)
    if use_paired_cache:
        lidar_encoder = None
    elif args.lidar_tokenizer == "wan21":
        lidar_encoder = OnlineLidarWan21Encoder(args)
    else:
        lidar_encoder = OnlineLidarS3Encoder(args)
    if world_size > 1:
        os.environ.setdefault("COSMOS_ONE_WAY_LOAD_FROZEN_VIDEO_RANK0_ONLY", "1")
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
        frozen_video_use_wan_fp32_strategy=args.frozen_video_wan_fp32_strategy,
        lidar_use_wan_fp32_strategy=args.lidar_wan_fp32_strategy,
        init_lidar_from_video=args.init_lidar_from_video,
        policy=VideoLidarAttentionPolicy(
            mode="video_to_lidar",
            cross_frame_rule=args.cross_frame_rule,
        ),
    )
    if args.distributed_parallelism == "fsdp":
        if world_size <= 1:
            raise ValueError("--distributed-parallelism fsdp requires torchrun with more than one process.")
        resume_step = load_resume_state(
            args=args,
            model=model,
            optimizer=None,
            expected_lidar_contract=active_lidar_contract,
        )
        model = apply_lidar_fsdp(model, args=args, world_size=world_size)
    else:
        if world_size > 1:
            model = build_ddp(model, local_rank=local_rank)
        resume_step = 0
    trainable_model = model.module if isinstance(model, DDP) else model
    optimizer = build_optimizer(args, trainable_model.lidar_expert.parameters())
    if args.distributed_parallelism == "ddp":
        resume_step = load_resume_state(
            args=args,
            model=trainable_model,
            optimizer=optimizer,
            expected_lidar_contract=active_lidar_contract,
        )
    if args.resume_checkpoint is not None:
        barrier()
    amp_enabled = args.device.startswith("cuda") and args.precision == "bfloat16"
    cached_device_dtype = cached_latent_device_dtype(args)

    if is_rank0():
        print(f"[train] output_dir={output_dir}")
        if use_paired_cache:
            latent_source = f"paired-cache:{args.paired_latent_cache_dir}"
            if isinstance(dataset, PairedLatentCacheDataset) and dataset.linked_layout:
                video_latent_source = f"linked:{dataset.video_dir}"
            elif args.precomputed_video_latent_dir:
                video_latent_source = f"precomputed:{args.precomputed_video_latent_dir}"
            else:
                video_latent_source = f"paired-cache:{args.paired_latent_cache_dir}"
        else:
            latent_source = "online"
            video_latent_source = (
                f"precomputed:{args.precomputed_video_latent_dir}"
                if use_precomputed_video_latents
                else args.video_tokenizer
            )
        if use_paired_cache and active_lidar_contract["tokenizer"] == "wan2pt1":
            lidar_tokenizer_label = "Wan2.1-cache"
        elif args.lidar_tokenizer == "wan21":
            lidar_tokenizer_label = "Wan2.1-online"
        else:
            lidar_tokenizer_label = "S3-online"
        print(
            f"[train] mode=one_way_expert video_tokenizer={args.video_tokenizer} "
            f"video_latent_source={video_latent_source} lidar_tokenizer={lidar_tokenizer_label} "
            f"latent_source={latent_source}"
        )
        print(
            f"[train] lidar_latent_contract={active_lidar_contract['version']} "
            f"lidar_latent_shape={tuple(active_lidar_contract['latent_shape'])} "
            f"train_hw={(args.train_height, args.train_width)}"
        )
        print(
            f"[train] cross_frame_rule={args.cross_frame_rule} "
            f"sdpa_backends={args.sdpa_backends} "
            f"lidar_num_blocks={args.lidar_num_blocks or 'full'} "
            f"video_kv_every_n_layers={args.video_kv_every_n_layers} "
            f"checkpoint_lidar_blocks={args.checkpoint_lidar_blocks} "
            f"distributed_parallelism={args.distributed_parallelism} "
            f"fsdp_shard_size={args.fsdp_shard_size or world_size if args.distributed_parallelism == 'fsdp' else 'off'} "
            f"fsdp_reshard_after_forward={args.fsdp_reshard_after_forward if args.distributed_parallelism == 'fsdp' else 'off'} "
            f"rf_convention={args.rf_convention} "
            f"rf_train_time_distribution={args.rf_train_time_distribution} "
            f"rf_shift={args.rf_shift} "
            f"frozen_video_wan_fp32_strategy={args.frozen_video_wan_fp32_strategy} "
            f"lidar_wan_fp32_strategy={args.lidar_wan_fp32_strategy} "
            f"init_lidar_from_video={args.init_lidar_from_video} "
            f"fused_optimizer={args.fused_optimizer} "
            f"empty_cache_after_encode={args.empty_cache_after_encode} "
            f"cached_latent_device_dtype={cached_device_dtype} "
            f"sdpa_flash_enabled={torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else False} "
            f"sdpa_math_enabled={torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else False}"
        )
        print(
            f"[train] frozen_video_ckpt={args.video_expert_checkpoint} "
            f"dataset_size={len(dataset)} batch_size_per_rank={args.batch_size} "
            f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'off'} "
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
                    clean_video = batch["video_latent"].to(
                        device=args.device,
                        dtype=cached_device_dtype,
                        non_blocking=True,
                    )
                    clean_lidar = batch["lidar_latent"].to(
                        device=args.device,
                        dtype=cached_device_dtype,
                        non_blocking=True,
                    )
                else:
                    assert lidar_encoder is not None
                    if use_precomputed_video_latents:
                        video_dir = Path(args.precomputed_video_latent_dir)
                        clean_video = torch.stack(
                            [
                                _load_precomputed_video_latent(video_dir, str(sample_key), Path(video_dir))
                                for sample_key in sample_keys
                            ],
                            dim=0,
                        ).to(device=args.device, dtype=cached_device_dtype, non_blocking=True)
                    else:
                        assert video_encoder is not None
                        clean_video = video_encoder.encode(batch["video"]).to(device=args.device, dtype=torch.float32)
                    clean_lidar = lidar_encoder.encode_batch(
                        batch["waymo_segment_key"],
                        batch["waymo_lidar_frame_indices"],
                    )
                    clean_lidar = clean_lidar.to(device=args.device, dtype=cached_device_dtype)
                    if args.lidar_tokenizer == "s3_ltcv":
                        clean_lidar = maybe_resize_lidar(
                            clean_lidar,
                            args.train_height,
                            args.train_width,
                            args.allow_lidar_resize_for_smoke,
                        )
                    else:
                        expected = (clean_lidar.shape[0], *lidar_latent_shape(active_lidar_contract))
                        if tuple(clean_lidar.shape) != expected:
                            raise ValueError(f"Expected online Wan21 LiDAR latent shape {expected}, got {tuple(clean_lidar.shape)}")
                if args.empty_cache_after_encode and not use_paired_cache and args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            video_noise = torch.randn_like(clean_video)
            lidar_noise = torch.randn_like(clean_lidar)
            lidar_u = sample_rf_train_u(
                batch_size=clean_lidar.shape[0],
                device=clean_lidar.device,
                distribution=args.rf_train_time_distribution,
            )
            if args.independent_video_timesteps:
                video_u = sample_rf_train_u(
                    batch_size=clean_video.shape[0],
                    device=clean_video.device,
                    distribution=args.rf_train_time_distribution,
                )
            else:
                video_u = lidar_u

            if args.rf_convention == "predict2":
                video_timesteps = rf_discrete_timesteps(
                    video_u,
                    shift=args.rf_shift,
                    num_train_timesteps=args.rf_num_train_timesteps,
                )
                lidar_timesteps = rf_discrete_timesteps(
                    lidar_u,
                    shift=args.rf_shift,
                    num_train_timesteps=args.rf_num_train_timesteps,
                )
            else:
                video_timesteps = video_u
                lidar_timesteps = lidar_u

            noisy_video, _ = make_rf_noisy_and_target(
                clean_video,
                video_noise,
                video_timesteps,
                convention=args.rf_convention,
                num_train_timesteps=args.rf_num_train_timesteps,
            )
            noisy_lidar, lidar_target = make_rf_noisy_and_target(
                clean_lidar,
                lidar_noise,
                lidar_timesteps,
                convention=args.rf_convention,
                num_train_timesteps=args.rf_num_train_timesteps,
            )

            if (step - 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(
                    noisy_video=noisy_video,
                    noisy_lidar=noisy_lidar,
                    video_timesteps=video_timesteps,
                    lidar_timesteps=lidar_timesteps,
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

            should_save = step % args.save_every == 0 or step >= args.max_steps
            if should_save:
                if args.distributed_parallelism == "fsdp" or is_rank0():
                    lidar_state = materialize_state_dict_for_checkpoint(trainable_model.lidar_expert)
                else:
                    lidar_state = None
                optimizer_state = optimizer.state_dict() if args.distributed_parallelism == "ddp" and is_rank0() else None
            if is_rank0() and should_save:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                checkpoint_payload = {
                    "step": step,
                    "lidar_expert": lidar_state,
                    "args": vars(args),
                    "lidar_latent_contract": active_lidar_contract,
                    "resume_step": resume_step,
                    "video_expert_checkpoint": args.video_expert_checkpoint,
                    "video_expert_config": args.video_expert_config,
                    "world_size": world_size,
                    "distributed_parallelism": args.distributed_parallelism,
                }
                if optimizer_state is not None:
                    checkpoint_payload["optimizer"] = optimizer_state
                else:
                    checkpoint_payload["optimizer_state_saved"] = False
                torch.save(
                    checkpoint_payload,
                    ckpt_path,
                )
                print(f"[train] saved {ckpt_path}")
            if should_save:
                del lidar_state
                del optimizer_state
                barrier()

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
