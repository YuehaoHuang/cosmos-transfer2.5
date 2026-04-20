# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in FastWAM-style one-way video->LiDAR expert baseline.

This module is intentionally standalone and is not imported by the existing
video generation or video post-training configs. It keeps the frozen Waymo
video DiT on its original path, then adds a trainable LiDAR expert whose
per-layer mixed attention reads frozen video key/value tokens while video never
reads LiDAR tokens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibility with older torch builds
    SDPBackend = None
    sdpa_kernel = None

from cosmos_transfer2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import Block, CheckpointMode, SACConfig
from cosmos_transfer2._src.predict2_multiview.networks.multiview_dit import MultiViewDiT
from cosmos_transfer2._src.transfer2_multiview.networks.video_lidar_joint_policy import VideoLidarAttentionPolicy


DEFAULT_WAYMO_VIDEO_CHECKPOINT = (
    "/data/cosmos-transfer2.5/output/20260310_104528/"
    "cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/"
    "checkpoints/iter_000033000/model_ema_bf16.pt"
)
DEFAULT_WAYMO_VIDEO_CONFIG = (
    "/data/cosmos-transfer2.5/output/20260310_104528/"
    "cosmos_transfer_v2p5/waymo_multiview/waymo_5cam_post_train/config.yaml"
)

_SDPA_BACKEND_POLICY = "flash_mem_math"


def configure_sdpa_backends(policy: str) -> None:
    valid = {"flash_only", "flash_mem", "flash_mem_math"}
    if policy not in valid:
        raise ValueError(f"Unsupported SDPA backend policy {policy!r}; expected one of {sorted(valid)}.")
    global _SDPA_BACKEND_POLICY
    _SDPA_BACKEND_POLICY = policy


def _sdpa_backends() -> Optional[list[Any]]:
    if SDPBackend is None:
        return None
    if _SDPA_BACKEND_POLICY == "flash_only":
        return [SDPBackend.FLASH_ATTENTION]
    if _SDPA_BACKEND_POLICY == "flash_mem":
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


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


def _device_from_arg(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _make_padding_mask(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)


def _make_condition_mask(
    batch_size: int,
    timesteps: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros((batch_size, 1, timesteps, height, width), device=device, dtype=dtype)


def _null_crossattn(
    batch_size: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int = 1,
) -> torch.Tensor:
    return torch.zeros((batch_size, seq_len, width), device=device, dtype=dtype)


def _flatten_step_ids_single(tokens_5d: torch.Tensor) -> torch.Tensor:
    _, timesteps, height, width, _ = tokens_5d.shape
    step_ids = torch.arange(timesteps, device=tokens_5d.device)
    return step_ids.repeat_interleave(height * width)


def _flatten_step_ids_multiview(tokens_5d: torch.Tensor, state_t: int) -> torch.Tensor:
    _, total_timesteps, height, width, _ = tokens_5d.shape
    if total_timesteps % state_t != 0:
        raise ValueError(
            f"Video token time dimension {total_timesteps} must be divisible by state_t={state_t}."
        )
    num_views = total_timesteps // state_t
    step_ids = torch.arange(state_t, device=tokens_5d.device).repeat(num_views)
    return step_ids.repeat_interleave(height * width)


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit keep-mask.

    Args:
        q: [B, Sq, H, D]
        k: [B, Sk, H, D]
        v: [B, Sk, H, D]
        keep_mask: [Sq, Sk] where True means allowed.
    """

    sq, sk = keep_mask.shape
    if q.shape[1] != sq or k.shape[1] != sk or v.shape[1] != sk:
        raise ValueError(
            "Mask and q/k/v sequence lengths must agree, got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, mask={tuple(keep_mask.shape)}"
        )

    attn_bias = torch.zeros((1, 1, sq, sk), device=q.device, dtype=q.dtype)
    attn_bias = attn_bias.masked_fill(~keep_mask.view(1, 1, sq, sk), torch.finfo(q.dtype).min)
    out = _scaled_dot_product_attention(
        rearrange(q, "b s h d -> b h s d").contiguous(),
        rearrange(k, "b s h d -> b h s d").contiguous(),
        rearrange(v, "b s h d -> b h s d").contiguous(),
        attn_mask=attn_bias,
    )
    return rearrange(out, "b h s d -> b s (h d)")


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SDPA wrapper that keeps CUDA FlashAttention enabled when eligible."""

    if q.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
        backends = _sdpa_backends()
        if attn_mask is not None and _SDPA_BACKEND_POLICY == "flash_only":
            backends = [SDPBackend.MATH]
        with sdpa_kernel(backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    out = _scaled_dot_product_attention(
        rearrange(q, "b s h d -> b h s d").contiguous(),
        rearrange(k, "b s h d -> b h s d").contiguous(),
        rearrange(v, "b s h d -> b h s d").contiguous(),
    )
    return rearrange(out, "b h s d -> b s (h d)")


def _modulate_5d(x: torch.Tensor, norm_layer: nn.Module, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return norm_layer(x) * (1 + scale) + shift


@dataclass(frozen=True)
class WaymoFrozenVideoExpertConfig:
    checkpoint_path: str = DEFAULT_WAYMO_VIDEO_CHECKPOINT
    config_path: str = DEFAULT_WAYMO_VIDEO_CONFIG


@dataclass
class FrozenVideoBackboneContext:
    tokens: torch.Tensor
    rope_emb: Optional[torch.Tensor]
    extra_pos_emb: Optional[torch.Tensor]
    t_embedding: torch.Tensor
    adaln_lora: Optional[torch.Tensor]
    crossattn_emb: torch.Tensor


def load_waymo_video_expert_kwargs(config_path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    raw = OmegaConf.to_container(cfg.model.config.net, resolve=True)
    net_cfg = _normalize_config_value(raw)
    return {
        "max_img_h": int(net_cfg["max_img_h"]),
        "max_img_w": int(net_cfg["max_img_w"]),
        "max_frames": int(net_cfg["max_frames"]),
        "in_channels": int(net_cfg["in_channels"]),
        "out_channels": int(net_cfg["out_channels"]),
        "patch_spatial": int(net_cfg["patch_spatial"]),
        "patch_temporal": int(net_cfg["patch_temporal"]),
        "concat_padding_mask": bool(net_cfg["concat_padding_mask"]),
        "model_channels": int(net_cfg["model_channels"]),
        "num_blocks": int(net_cfg["num_blocks"]),
        "num_heads": int(net_cfg["num_heads"]),
        "mlp_ratio": float(net_cfg["mlp_ratio"]),
        "atten_backend": str(net_cfg["atten_backend"]),
        "crossattn_emb_channels": int(net_cfg["crossattn_emb_channels"]),
        "use_crossattn_projection": bool(net_cfg["use_crossattn_projection"]),
        "crossattn_proj_in_channels": int(net_cfg["crossattn_proj_in_channels"]),
        "pos_emb_cls": str(net_cfg["pos_emb_cls"]),
        "pos_emb_learnable": bool(net_cfg["pos_emb_learnable"]),
        "pos_emb_interpolation": str(net_cfg["pos_emb_interpolation"]),
        "use_adaln_lora": bool(net_cfg["use_adaln_lora"]),
        "adaln_lora_dim": int(net_cfg["adaln_lora_dim"]),
        "rope_h_extrapolation_ratio": float(net_cfg["rope_h_extrapolation_ratio"]),
        "rope_w_extrapolation_ratio": float(net_cfg["rope_w_extrapolation_ratio"]),
        "rope_t_extrapolation_ratio": float(net_cfg["rope_t_extrapolation_ratio"]),
        "rope_enable_fps_modulation": bool(net_cfg["rope_enable_fps_modulation"]),
        "extra_per_block_abs_pos_emb": bool(net_cfg["extra_per_block_abs_pos_emb"]),
        "n_cameras_emb": int(net_cfg["n_cameras_emb"]),
        "view_condition_dim": int(net_cfg["view_condition_dim"]),
        "concat_view_embedding": bool(net_cfg["concat_view_embedding"]),
        "state_t": int(net_cfg["state_t"]),
        "timestep_scale": float(net_cfg["timestep_scale"]),
        "use_wan_fp32_strategy": bool(net_cfg["use_wan_fp32_strategy"]),
    }


def load_frozen_waymo_video_expert(
    *,
    checkpoint_path: str | Path = DEFAULT_WAYMO_VIDEO_CHECKPOINT,
    config_path: str | Path = DEFAULT_WAYMO_VIDEO_CONFIG,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> MultiViewDiT:
    kwargs = load_waymo_video_expert_kwargs(config_path)
    video_expert = MultiViewDiT(**kwargs)

    state = torch.load(checkpoint_path, map_location="cpu")
    stripped_state = {key.removeprefix("net."): value for key, value in state.items() if key.startswith("net.")}
    incompatible = video_expert.load_state_dict(stripped_state, strict=False)
    allowed_unexpected_prefixes = (
        "control_embedder.",
        "control_blocks.",
        "input_hint_block.",
        "accum_",
    )
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(allowed_unexpected_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected frozen video weights when loading {checkpoint_path}: {unexpected[:20]}")
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Missing frozen video weights when loading {checkpoint_path}: {incompatible.missing_keys[:20]}"
        )

    video_expert.to(device=_device_from_arg(device), dtype=dtype)
    video_expert.eval()
    for param in video_expert.parameters():
        param.requires_grad_(False)
    return video_expert


@torch.no_grad()
def extract_frozen_video_context(
    video_expert: MultiViewDiT,
    noisy_video: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    crossattn_emb: Optional[torch.Tensor] = None,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    condition_video_input_mask: Optional[torch.Tensor] = None,
    view_indices_B_T: Optional[torch.Tensor] = None,
) -> FrozenVideoBackboneContext:
    expert_dtype = video_expert.x_embedder.proj[1].weight.dtype
    noisy_video = noisy_video.to(dtype=expert_dtype)
    batch_size, _, total_t, height, width = noisy_video.shape
    device = noisy_video.device
    dtype = noisy_video.dtype

    if condition_video_input_mask is None:
        condition_video_input_mask = _make_condition_mask(batch_size, total_t, height, width, device, dtype)
    if padding_mask is None:
        padding_mask = _make_padding_mask(batch_size, height, width, device, dtype)
    elif padding_mask.ndim == 3:
        padding_mask = padding_mask.unsqueeze(1)
    if crossattn_emb is None:
        num_views = total_t // video_expert.state_t
        crossattn_width = (
            video_expert.crossattn_proj_in_channels
            if video_expert.use_crossattn_projection
            else video_expert.blocks[0].cross_attn.context_dim
        )
        crossattn_emb = _null_crossattn(batch_size, crossattn_width, device, dtype, seq_len=512 * num_views)
    else:
        crossattn_emb = crossattn_emb.to(device=device, dtype=expert_dtype)

    x = torch.cat([noisy_video, condition_video_input_mask.type_as(noisy_video)], dim=1)
    x_B_T_H_W_D, rope_emb, extra_pos_emb = video_expert.prepare_embedded_sequence(
        x,
        fps=fps,
        padding_mask=padding_mask,
        view_indices_B_T=view_indices_B_T,
    )
    if video_expert.use_crossattn_projection:
        crossattn_emb = video_expert.crossattn_proj(crossattn_emb)

    scaled_timesteps = timesteps
    if scaled_timesteps.ndim == 1:
        scaled_timesteps = scaled_timesteps.unsqueeze(1)
    scaled_timesteps = scaled_timesteps * video_expert.timestep_scale

    autocast_enabled = noisy_video.device.type == "cuda" and video_expert.use_wan_fp32_strategy
    with amp.autocast("cuda", enabled=autocast_enabled, dtype=torch.float32):
        t_embedding, adaln_lora = video_expert.t_embedder(scaled_timesteps)
        t_embedding = video_expert.t_embedding_norm(t_embedding)

    h = x_B_T_H_W_D
    return FrozenVideoBackboneContext(
        tokens=h.detach(),
        rope_emb=rope_emb,
        extra_pos_emb=None if extra_pos_emb is None else extra_pos_emb.detach(),
        t_embedding=t_embedding.detach(),
        adaln_lora=None if adaln_lora is None else adaln_lora.detach(),
        crossattn_emb=crossattn_emb.detach(),
    )


class VideoConditionedLidarExpert(MinimalV1LVGDiT):
    """LiDAR expert that reads frozen video K/V with a one-way attention policy."""

    def __init__(
        self,
        *,
        video_crossattn_proj_in_channels: int,
        policy: VideoLidarAttentionPolicy = VideoLidarAttentionPolicy(
            mode="video_to_lidar",
            cross_frame_rule="same_step",
        ),
        video_state_t: int = 8,
        video_kv_every_n_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("use_crossattn_projection", True)
        kwargs.setdefault("crossattn_proj_in_channels", video_crossattn_proj_in_channels)
        super().__init__(**kwargs)
        self.policy = policy
        self.video_state_t = video_state_t
        if video_kv_every_n_layers < 1:
            raise ValueError(f"video_kv_every_n_layers must be >= 1, got {video_kv_every_n_layers}.")
        self.video_kv_every_n_layers = video_kv_every_n_layers
        self._same_step_index_cache: dict[
            tuple[Any, ...],
            tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]],
        ] = {}

    def _build_same_step_indices(
        self,
        video_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        video_step_ids = _flatten_step_ids_multiview(video_tokens, self.video_state_t)
        lidar_step_ids = _flatten_step_ids_single(lidar_tokens)
        num_steps = int(max(video_step_ids.max().item(), lidar_step_ids.max().item())) + 1
        cache_key = (
            video_step_ids.numel(),
            lidar_step_ids.numel(),
            video_tokens.shape[1],
            video_tokens.shape[2],
            video_tokens.shape[3],
            lidar_tokens.shape[1],
            lidar_tokens.shape[2],
            lidar_tokens.shape[3],
            self.policy.mode,
            self.policy.cross_frame_rule,
            str(video_tokens.device),
        )
        if cache_key not in self._same_step_index_cache:
            video_indices = []
            lidar_indices = []
            for step in range(num_steps):
                video_indices.append(torch.nonzero(video_step_ids == step, as_tuple=False).flatten())
                lidar_indices.append(torch.nonzero(lidar_step_ids == step, as_tuple=False).flatten())
            self._same_step_index_cache[cache_key] = (
                lidar_step_ids,
                video_indices,
                lidar_indices,
            )
        return self._same_step_index_cache[cache_key]

    def _mixed_attention_same_step_indexed(
        self,
        q_l: torch.Tensor,
        k_l: torch.Tensor,
        v_l: torch.Tensor,
        k_v: torch.Tensor,
        v_v: torch.Tensor,
        video_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        use_video_kv: bool = True,
    ) -> torch.Tensor:
        _, video_indices_per_step, lidar_indices_per_step = self._build_same_step_indices(video_tokens, lidar_tokens)
        mixed = torch.empty(
            (q_l.shape[0], q_l.shape[1], q_l.shape[2] * q_l.shape[3]),
            device=q_l.device,
            dtype=q_l.dtype,
        )
        allow_video = use_video_kv and self.policy.mode in ("video_to_lidar", "bidirectional")
        allow_lidar = self.policy.allow_lidar_self
        for video_indices, lidar_indices in zip(video_indices_per_step, lidar_indices_per_step):
            if lidar_indices.numel() == 0:
                continue
            k_parts = []
            v_parts = []
            if allow_video and video_indices.numel() > 0:
                k_parts.append(k_v[:, video_indices, ...])
                v_parts.append(v_v[:, video_indices, ...])
            if allow_lidar:
                k_parts.append(k_l[:, lidar_indices, ...])
                v_parts.append(v_l[:, lidar_indices, ...])
            if not k_parts:
                raise RuntimeError("LiDAR one-way expert has no keys/values available for same-step attention.")
            mixed[:, lidar_indices, :] = _attention(
                q_l[:, lidar_indices, ...],
                torch.cat(k_parts, dim=1),
                torch.cat(v_parts, dim=1),
            )
        return mixed

    def _mixed_attention_same_step(
        self,
        q_l: torch.Tensor,
        k_l: torch.Tensor,
        v_l: torch.Tensor,
        k_v: torch.Tensor,
        v_v: torch.Tensor,
        video_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        use_video_kv: bool = True,
    ) -> torch.Tensor:
        """Same-step mixed attention with one SDPA call per layer.

        The original same-step path looped over latent frames and launched one
        small attention op per step. For Waymo, the token layout is regular, so
        we stack the 8 frame buckets into the batch dimension:
        [B, S, tokens, heads, dim] -> [B*S, tokens, heads, dim].
        This preserves strict temporal anchoring while giving SDPA a large
        enough tensor to take the FlashAttention path on CUDA/bf16.
        """

        batch_size = q_l.shape[0]
        num_heads = q_l.shape[2]
        head_dim = q_l.shape[3]
        _, video_total_t, video_h, video_w, _ = video_tokens.shape
        _, lidar_t, lidar_h, lidar_w, _ = lidar_tokens.shape

        if lidar_t != self.video_state_t or video_total_t % self.video_state_t != 0:
            return self._mixed_attention_same_step_indexed(
                q_l=q_l,
                k_l=k_l,
                v_l=v_l,
                k_v=k_v,
                v_v=v_v,
                video_tokens=video_tokens,
                lidar_tokens=lidar_tokens,
                use_video_kv=use_video_kv,
            )

        allow_video = use_video_kv and self.policy.mode in ("video_to_lidar", "bidirectional")
        allow_lidar = self.policy.allow_lidar_self
        if not allow_video and not allow_lidar:
            raise RuntimeError("LiDAR one-way expert has no keys/values available for same-step attention.")

        num_views = video_total_t // lidar_t
        video_frame_tokens = video_h * video_w
        video_tokens_per_step = num_views * video_frame_tokens
        lidar_tokens_per_step = lidar_h * lidar_w

        q_step = q_l.reshape(batch_size, lidar_t, lidar_tokens_per_step, num_heads, head_dim)
        k_parts = []
        v_parts = []
        if allow_video:
            k_v_step = (
                k_v.reshape(batch_size, num_views, lidar_t, video_frame_tokens, num_heads, head_dim)
                .transpose(1, 2)
                .reshape(batch_size, lidar_t, video_tokens_per_step, num_heads, head_dim)
            )
            v_v_step = (
                v_v.reshape(batch_size, num_views, lidar_t, video_frame_tokens, num_heads, head_dim)
                .transpose(1, 2)
                .reshape(batch_size, lidar_t, video_tokens_per_step, num_heads, head_dim)
            )
            k_parts.append(k_v_step)
            v_parts.append(v_v_step)
        if allow_lidar:
            k_parts.append(k_l.reshape(batch_size, lidar_t, lidar_tokens_per_step, num_heads, head_dim))
            v_parts.append(v_l.reshape(batch_size, lidar_t, lidar_tokens_per_step, num_heads, head_dim))

        k_step = torch.cat(k_parts, dim=2)
        v_step = torch.cat(v_parts, dim=2)
        mixed_step = _attention(
            rearrange(q_step, "b s l h d -> (b s) l h d"),
            rearrange(k_step, "b s l h d -> (b s) l h d"),
            rearrange(v_step, "b s l h d -> (b s) l h d"),
        )
        return rearrange(mixed_step, "(b s) l d -> b (s l) d", b=batch_size, s=lidar_t)

    def _prepare_inputs(
        self,
        noisy_lidar: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        crossattn_emb: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        expert_dtype = self.x_embedder.proj[1].weight.dtype
        noisy_lidar = noisy_lidar.to(dtype=expert_dtype)
        batch_size, _, total_t, height, width = noisy_lidar.shape
        device = noisy_lidar.device
        dtype = noisy_lidar.dtype

        condition_mask = _make_condition_mask(batch_size, total_t, height, width, device, dtype)
        if padding_mask is None:
            padding_mask = _make_padding_mask(batch_size, height, width, device, dtype)
        elif padding_mask.ndim == 3:
            padding_mask = padding_mask.unsqueeze(1)
        if crossattn_emb is None:
            crossattn_emb = _null_crossattn(batch_size, self.crossattn_proj_in_channels, device, dtype)
        else:
            crossattn_emb = crossattn_emb.to(device=device, dtype=expert_dtype)

        x = torch.cat([noisy_lidar, condition_mask], dim=1)
        x_tokens, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        scaled_timesteps = timesteps
        if scaled_timesteps.ndim == 1:
            scaled_timesteps = scaled_timesteps.unsqueeze(1)
        scaled_timesteps = scaled_timesteps * self.timestep_scale

        autocast_enabled = noisy_lidar.device.type == "cuda" and self.use_wan_fp32_strategy
        with amp.autocast("cuda", enabled=autocast_enabled, dtype=torch.float32):
            t_embedding, adaln_lora = self.t_embedder(scaled_timesteps)
            t_embedding = self.t_embedding_norm(t_embedding)
        return x_tokens, rope_emb, extra_pos_emb, t_embedding, adaln_lora, crossattn_emb

    def _forward_one_way_block(
        self,
        *,
        lidar_block: Block,
        video_block: Block,
        lidar_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        lidar_rope_emb: Optional[torch.Tensor],
        video_rope_emb: Optional[torch.Tensor],
        video_extra_pos_emb: Optional[torch.Tensor],
        video_t_embedding: torch.Tensor,
        video_adaln_lora: Optional[torch.Tensor],
        t_embedding: torch.Tensor,
        adaln_lora: Optional[torch.Tensor],
        crossattn_emb: torch.Tensor,
        extra_pos_emb: Optional[torch.Tensor],
        use_video_kv: bool = True,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            lidar_tokens = lidar_tokens + extra_pos_emb

        autocast_enabled = lidar_tokens.device.type == "cuda" and self.use_wan_fp32_strategy
        with amp.autocast("cuda", enabled=autocast_enabled, dtype=torch.float32):
            if lidar_block.use_adaln_lora:
                assert adaln_lora is not None
                shift_self, scale_self, gate_self = (lidar_block.adaln_modulation_self_attn(t_embedding) + adaln_lora).chunk(
                    3, dim=-1
                )
                shift_cross, scale_cross, gate_cross = (
                    lidar_block.adaln_modulation_cross_attn(t_embedding) + adaln_lora
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (lidar_block.adaln_modulation_mlp(t_embedding) + adaln_lora).chunk(
                    3, dim=-1
                )
            else:
                shift_self, scale_self, gate_self = lidar_block.adaln_modulation_self_attn(t_embedding).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = lidar_block.adaln_modulation_cross_attn(t_embedding).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = lidar_block.adaln_modulation_mlp(t_embedding).chunk(3, dim=-1)

        shift_self = rearrange(shift_self, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        scale_self = rearrange(scale_self, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        gate_self = rearrange(gate_self, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        shift_cross = rearrange(shift_cross, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        scale_cross = rearrange(scale_cross, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        gate_cross = rearrange(gate_cross, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d").type_as(lidar_tokens)
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d").type_as(lidar_tokens)

        lidar_norm = _modulate_5d(lidar_tokens, lidar_block.layer_norm_self_attn, scale_self, shift_self)
        bsz, lidar_t, lidar_h, lidar_w, _ = lidar_norm.shape
        video_t = video_tokens.shape[1]
        video_h = video_tokens.shape[2]
        video_w = video_tokens.shape[3]

        lidar_flat = rearrange(lidar_norm, "b t h w d -> b (t h w) d")
        q_l, k_l, v_l = lidar_block.self_attn.compute_qkv(lidar_flat, rope_emb=lidar_rope_emb)
        k_v = v_v = None
        keep_video = use_video_kv and self.policy.mode in ("video_to_lidar", "bidirectional")
        if keep_video:
            video_dtype = video_block.self_attn.q_proj.weight.dtype
            video_tokens = video_tokens.to(dtype=video_dtype)
            if video_extra_pos_emb is not None:
                video_tokens = video_tokens + video_extra_pos_emb.to(device=video_tokens.device, dtype=video_dtype)
            with amp.autocast("cuda", enabled=autocast_enabled, dtype=torch.float32):
                video_t_embedding = video_t_embedding.to(device=video_tokens.device, dtype=video_dtype)
                if video_adaln_lora is not None:
                    video_adaln_lora = video_adaln_lora.to(device=video_tokens.device, dtype=video_dtype)
                if video_block.use_adaln_lora:
                    assert video_adaln_lora is not None
                    video_shift_self, video_scale_self, _ = (
                        video_block.adaln_modulation_self_attn(video_t_embedding) + video_adaln_lora
                    ).chunk(3, dim=-1)
                else:
                    video_shift_self, video_scale_self, _ = video_block.adaln_modulation_self_attn(
                        video_t_embedding
                    ).chunk(3, dim=-1)
            video_shift_self = rearrange(video_shift_self, "b t d -> b t 1 1 d").to(dtype=video_dtype)
            video_scale_self = rearrange(video_scale_self, "b t d -> b t 1 1 d").to(dtype=video_dtype)
            video_norm = _modulate_5d(video_tokens, video_block.layer_norm_self_attn, video_scale_self, video_shift_self)
            video_flat = rearrange(video_norm, "b t h w d -> b (t h w) d")
            _, k_v, v_v = video_block.self_attn.compute_qkv(video_flat, rope_emb=video_rope_emb)
            k_v = k_v.to(dtype=q_l.dtype)
            v_v = v_v.to(dtype=q_l.dtype)

        if self.policy.cross_frame_rule == "same_step":
            if k_v is None or v_v is None:
                k_v = k_l.new_empty((q_l.shape[0], 0, q_l.shape[2], q_l.shape[3]))
                v_v = v_l.new_empty((q_l.shape[0], 0, q_l.shape[2], q_l.shape[3]))
            mixed = self._mixed_attention_same_step(
                q_l=q_l,
                k_l=k_l,
                v_l=v_l,
                k_v=k_v,
                v_v=v_v,
                video_tokens=video_tokens,
                lidar_tokens=lidar_tokens,
                use_video_kv=keep_video,
            )
        else:
            k_parts = []
            v_parts = []
            if keep_video:
                assert k_v is not None and v_v is not None
                k_parts.append(k_v)
                v_parts.append(v_v)
            if self.policy.allow_lidar_self:
                k_parts.append(k_l)
                v_parts.append(v_l)
            if not k_parts:
                raise RuntimeError("LiDAR one-way expert has no keys/values available for mixed attention.")
            mixed = _attention(q_l, torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1))
        mixed = lidar_block.self_attn.output_dropout(lidar_block.self_attn.output_proj(mixed))
        mixed = rearrange(mixed, "b (t h w) d -> b t h w d", t=lidar_t, h=lidar_h, w=lidar_w)
        x = lidar_tokens + gate_self * mixed

        cross_norm = _modulate_5d(x, lidar_block.layer_norm_cross_attn, scale_cross, shift_cross)
        cross_out = rearrange(
            lidar_block.cross_attn(
                rearrange(cross_norm, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=lidar_rope_emb,
            ),
            "b (t h w) d -> b t h w d",
            t=lidar_t,
            h=lidar_h,
            w=lidar_w,
        )
        x = x + gate_cross * cross_out

        mlp_norm = _modulate_5d(x, lidar_block.layer_norm_mlp, scale_mlp, shift_mlp)
        x = x + gate_mlp * lidar_block.mlp(mlp_norm)
        if video_t <= 0 or video_h <= 0 or video_w <= 0:
            raise RuntimeError("Invalid frozen video token shape for one-way expert.")
        return x

    def forward_with_video_context(
        self,
        noisy_lidar: torch.Tensor,
        *,
        video_layer_inputs: list[torch.Tensor],
        video_blocks: nn.ModuleList,
        video_rope_emb: Optional[torch.Tensor],
        video_extra_pos_emb: Optional[torch.Tensor],
        video_t_embedding: torch.Tensor,
        video_adaln_lora: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        crossattn_emb: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(video_layer_inputs) < len(self.blocks) or len(video_blocks) < len(self.blocks):
            raise ValueError(
                "Video context must provide at least as many layers as the LiDAR expert, got "
                f"video_inputs={len(video_layer_inputs)}, video_blocks={len(video_blocks)}, lidar_blocks={len(self.blocks)}"
            )

        (
            lidar_tokens,
            lidar_rope_emb,
            extra_pos_emb,
            t_embedding,
            adaln_lora,
            crossattn_emb,
        ) = self._prepare_inputs(
            noisy_lidar,
            timesteps,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
        )

        for layer_idx, lidar_block in enumerate(self.blocks):
            use_video_kv = layer_idx % self.video_kv_every_n_layers == 0
            lidar_tokens = self._forward_one_way_block(
                lidar_block=lidar_block,
                video_block=video_blocks[layer_idx],
                lidar_tokens=lidar_tokens,
                video_tokens=video_layer_inputs[layer_idx].to(device=lidar_tokens.device, dtype=lidar_tokens.dtype),
                lidar_rope_emb=lidar_rope_emb,
                video_rope_emb=video_rope_emb,
                video_extra_pos_emb=video_extra_pos_emb,
                video_t_embedding=video_t_embedding.to(device=lidar_tokens.device),
                video_adaln_lora=None if video_adaln_lora is None else video_adaln_lora.to(device=lidar_tokens.device),
                t_embedding=t_embedding,
                adaln_lora=adaln_lora,
                crossattn_emb=crossattn_emb,
                extra_pos_emb=extra_pos_emb,
                use_video_kv=use_video_kv,
            )

        pred_tokens = self.final_layer(lidar_tokens, t_embedding, adaln_lora_B_T_3D=adaln_lora)
        return self.unpatchify(pred_tokens)


class VideoLidarOneWayExpertBaseline(nn.Module):
    """Frozen video backbone + trainable LiDAR expert."""

    def __init__(
        self,
        video_expert: MultiViewDiT,
        lidar_expert: VideoConditionedLidarExpert,
        *,
        checkpoint_lidar_blocks: bool = True,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.lidar_expert = lidar_expert
        self.checkpoint_lidar_blocks = checkpoint_lidar_blocks
        for param in self.video_expert.parameters():
            param.requires_grad_(False)
        self.video_expert.eval()

    def forward(
        self,
        noisy_video: torch.Tensor,
        noisy_lidar: torch.Tensor,
        *,
        video_timesteps: torch.Tensor,
        lidar_timesteps: torch.Tensor,
        crossattn_emb: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        video_padding_mask: Optional[torch.Tensor] = None,
        lidar_padding_mask: Optional[torch.Tensor] = None,
        video_condition_video_input_mask: Optional[torch.Tensor] = None,
        return_video_pred: bool = True,
    ) -> dict[str, torch.Tensor]:
        if crossattn_emb is None:
            batch_size = noisy_video.shape[0]
            num_views = noisy_video.shape[2] // self.video_expert.state_t
            crossattn_width = (
                self.video_expert.crossattn_proj_in_channels
                if self.video_expert.use_crossattn_projection
                else self.video_expert.blocks[0].cross_attn.context_dim
            )
            crossattn_emb = _null_crossattn(
                batch_size,
                crossattn_width,
                noisy_video.device,
                self.video_expert.x_embedder.proj[1].weight.dtype,
                seq_len=512 * num_views,
            )
        with torch.no_grad():
            video_context = extract_frozen_video_context(
                self.video_expert,
                noisy_video,
                video_timesteps,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=video_padding_mask,
                condition_video_input_mask=video_condition_video_input_mask,
                view_indices_B_T=view_indices_B_T,
            )

        (
            lidar_tokens,
            lidar_rope_emb,
            lidar_extra_pos_emb,
            lidar_t_embedding,
            lidar_adaln_lora,
            lidar_crossattn_emb,
        ) = self.lidar_expert._prepare_inputs(
            noisy_lidar,
            lidar_timesteps,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=lidar_padding_mask,
        )

        video_tokens = video_context.tokens
        for layer_idx, (video_block, lidar_block) in enumerate(zip(self.video_expert.blocks, self.lidar_expert.blocks)):
            use_video_kv = layer_idx % self.lidar_expert.video_kv_every_n_layers == 0
            def _lidar_one_way_forward(
                lidar_tokens_in: torch.Tensor,
                *,
                use_video_kv_for_layer: bool = use_video_kv,
            ) -> torch.Tensor:
                return self.lidar_expert._forward_one_way_block(
                    lidar_block=lidar_block,
                    video_block=video_block,
                    lidar_tokens=lidar_tokens_in,
                    video_tokens=video_tokens,
                    lidar_rope_emb=lidar_rope_emb,
                    video_rope_emb=video_context.rope_emb,
                    video_extra_pos_emb=video_context.extra_pos_emb,
                    video_t_embedding=video_context.t_embedding,
                    video_adaln_lora=video_context.adaln_lora,
                    t_embedding=lidar_t_embedding,
                    adaln_lora=lidar_adaln_lora,
                    crossattn_emb=lidar_crossattn_emb,
                    extra_pos_emb=lidar_extra_pos_emb,
                    use_video_kv=use_video_kv_for_layer,
                )

            if self.training and self.checkpoint_lidar_blocks:
                lidar_tokens = torch.utils.checkpoint.checkpoint(
                    _lidar_one_way_forward,
                    lidar_tokens,
                    use_reentrant=False,
                )
            else:
                lidar_tokens = _lidar_one_way_forward(lidar_tokens)
            with torch.no_grad():
                video_tokens = video_block(
                    video_tokens,
                    video_context.t_embedding,
                    video_context.crossattn_emb,
                    rope_emb_L_1_1_D=video_context.rope_emb,
                    adaln_lora_B_T_3D=video_context.adaln_lora,
                    extra_per_block_pos_emb=video_context.extra_pos_emb,
                )

        lidar_pred_tokens = self.lidar_expert.final_layer(
            lidar_tokens,
            lidar_t_embedding,
            adaln_lora_B_T_3D=lidar_adaln_lora,
        )
        lidar_pred = self.lidar_expert.unpatchify(lidar_pred_tokens)
        outputs = {
            "lidar_pred": lidar_pred,
        }
        if return_video_pred:
            with torch.no_grad():
                video_pred_tokens = self.video_expert.final_layer(
                    video_tokens,
                    video_context.t_embedding,
                    adaln_lora_B_T_3D=video_context.adaln_lora,
                )
                outputs["video_pred"] = self.video_expert.unpatchify(video_pred_tokens)
        return outputs


def build_waymo_video_lidar_one_way_expert_baseline(
    *,
    checkpoint_path: str | Path = DEFAULT_WAYMO_VIDEO_CHECKPOINT,
    config_path: str | Path = DEFAULT_WAYMO_VIDEO_CONFIG,
    device: torch.device | str = "cpu",
    frozen_dtype: torch.dtype = torch.bfloat16,
    lidar_dtype: Optional[torch.dtype] = None,
    lidar_max_img_h: int = 64,
    lidar_max_img_w: int = 112,
    lidar_max_frames: int = 8,
    lidar_attention_backend: str = "torch",
    lidar_num_blocks: Optional[int] = None,
    video_kv_every_n_layers: int = 1,
    checkpoint_lidar_blocks: bool = True,
    policy: VideoLidarAttentionPolicy = VideoLidarAttentionPolicy(
        mode="video_to_lidar",
        cross_frame_rule="same_step",
    ),
) -> VideoLidarOneWayExpertBaseline:
    video_kwargs = load_waymo_video_expert_kwargs(config_path)
    video_expert = load_frozen_waymo_video_expert(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        dtype=frozen_dtype,
    )
    video_num_blocks = int(video_kwargs["num_blocks"])
    if lidar_num_blocks is None:
        lidar_num_blocks = video_num_blocks
    if lidar_num_blocks < 1 or lidar_num_blocks > video_num_blocks:
        raise ValueError(f"lidar_num_blocks must be in [1, {video_num_blocks}], got {lidar_num_blocks}.")
    lidar_sac_config = SACConfig(mode=CheckpointMode.NONE) if not checkpoint_lidar_blocks else SACConfig()
    lidar_expert = VideoConditionedLidarExpert(
        max_img_h=lidar_max_img_h,
        max_img_w=lidar_max_img_w,
        max_frames=lidar_max_frames,
        in_channels=int(video_kwargs["out_channels"]),
        out_channels=int(video_kwargs["out_channels"]),
        patch_spatial=int(video_kwargs["patch_spatial"]),
        patch_temporal=int(video_kwargs["patch_temporal"]),
        concat_padding_mask=True,
        model_channels=int(video_kwargs["model_channels"]),
        num_blocks=lidar_num_blocks,
        num_heads=int(video_kwargs["num_heads"]),
        mlp_ratio=float(video_kwargs["mlp_ratio"]),
        atten_backend=lidar_attention_backend,
        crossattn_emb_channels=int(video_kwargs["crossattn_emb_channels"]),
        crossattn_proj_in_channels=int(video_kwargs["crossattn_proj_in_channels"]),
        use_crossattn_projection=True,
        pos_emb_cls=str(video_kwargs["pos_emb_cls"]),
        pos_emb_learnable=bool(video_kwargs["pos_emb_learnable"]),
        pos_emb_interpolation=str(video_kwargs["pos_emb_interpolation"]),
        use_adaln_lora=bool(video_kwargs["use_adaln_lora"]),
        adaln_lora_dim=int(video_kwargs["adaln_lora_dim"]),
        rope_h_extrapolation_ratio=float(video_kwargs["rope_h_extrapolation_ratio"]),
        rope_w_extrapolation_ratio=float(video_kwargs["rope_w_extrapolation_ratio"]),
        rope_t_extrapolation_ratio=float(video_kwargs["rope_t_extrapolation_ratio"]),
        rope_enable_fps_modulation=bool(video_kwargs["rope_enable_fps_modulation"]),
        extra_per_block_abs_pos_emb=False,
        timestep_scale=float(video_kwargs["timestep_scale"]),
        use_wan_fp32_strategy=bool(video_kwargs["use_wan_fp32_strategy"]),
        sac_config=lidar_sac_config,
        video_crossattn_proj_in_channels=int(video_kwargs["crossattn_proj_in_channels"]),
        video_state_t=int(video_kwargs["state_t"]),
        video_kv_every_n_layers=video_kv_every_n_layers,
        policy=policy,
    )
    if lidar_dtype is None:
        lidar_dtype = frozen_dtype
    lidar_expert = lidar_expert.to(device=_device_from_arg(device), dtype=lidar_dtype)
    model = VideoLidarOneWayExpertBaseline(
        video_expert=video_expert,
        lidar_expert=lidar_expert,
        checkpoint_lidar_blocks=checkpoint_lidar_blocks,
    )
    return model.to(_device_from_arg(device))
