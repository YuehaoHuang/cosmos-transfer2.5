# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in video/LiDAR joint attention policy helpers.

This module is intentionally not imported by the existing video generation
configs. It only provides small building blocks for the future joint
video/LiDAR path, following the FastWAM-inspired rule that the default
baseline lets LiDAR attend to video while keeping video self-denoising
unchanged.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch


CrossModalMode = Literal["none", "video_to_lidar", "lidar_to_video", "bidirectional"]
CrossFrameRule = Literal["all", "same_step"]


@dataclass(frozen=True)
class VideoLidarAttentionPolicy:
    """Policy for a concatenated [video, lidar] token sequence.

    `video_to_lidar` means LiDAR query tokens may attend to video key/value
    tokens. Video query tokens still cannot attend to LiDAR tokens, which keeps
    the video denoising path isolated for the first baseline.
    """

    mode: CrossModalMode = "video_to_lidar"
    cross_frame_rule: CrossFrameRule = "all"
    allow_video_self: bool = True
    allow_lidar_self: bool = True


def concat_joint_tokens(video_tokens: torch.Tensor, lidar_tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate token streams along sequence dimension."""

    if video_tokens.ndim != 3 or lidar_tokens.ndim != 3:
        raise ValueError(
            "`video_tokens` and `lidar_tokens` must be 3D [B, S, D], "
            f"got {tuple(video_tokens.shape)} and {tuple(lidar_tokens.shape)}"
        )
    if video_tokens.shape[0] != lidar_tokens.shape[0] or video_tokens.shape[2] != lidar_tokens.shape[2]:
        raise ValueError(
            "Token stream batch/hidden dimensions must match, got "
            f"{tuple(video_tokens.shape)} and {tuple(lidar_tokens.shape)}"
        )
    return torch.cat([video_tokens, lidar_tokens], dim=1)


def split_joint_tokens(tokens: torch.Tensor, video_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a concatenated [video, lidar] token sequence."""

    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be 3D [B, S, D], got {tuple(tokens.shape)}")
    if video_seq_len < 0 or video_seq_len > tokens.shape[1]:
        raise ValueError(f"`video_seq_len`={video_seq_len} is invalid for sequence length {tokens.shape[1]}")
    return tokens[:, :video_seq_len], tokens[:, video_seq_len:]


def _validate_policy(policy: VideoLidarAttentionPolicy) -> None:
    if policy.mode not in ("none", "video_to_lidar", "lidar_to_video", "bidirectional"):
        raise ValueError(f"Unsupported cross-modal mode: {policy.mode}")
    if policy.cross_frame_rule not in ("all", "same_step"):
        raise ValueError(f"Unsupported cross-frame rule: {policy.cross_frame_rule}")


def _validate_seq_len(name: str, seq_len: int) -> None:
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError(f"`{name}` must be a positive int, got {seq_len!r}")


def _validate_custom_mask(
    name: str,
    mask: torch.Tensor,
    expected_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if mask.shape != expected_shape:
        raise ValueError(f"`{name}` must have shape {expected_shape}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def _same_step_mask(
    query_seq_len: int,
    key_seq_len: int,
    query_tokens_per_step: int,
    key_tokens_per_step: int,
    device: torch.device,
) -> torch.Tensor:
    _validate_seq_len("query_tokens_per_step", query_tokens_per_step)
    _validate_seq_len("key_tokens_per_step", key_tokens_per_step)
    if query_seq_len % query_tokens_per_step != 0:
        raise ValueError(
            f"`query_seq_len`={query_seq_len} must be divisible by "
            f"`query_tokens_per_step`={query_tokens_per_step}"
        )
    if key_seq_len % key_tokens_per_step != 0:
        raise ValueError(
            f"`key_seq_len`={key_seq_len} must be divisible by "
            f"`key_tokens_per_step`={key_tokens_per_step}"
        )
    query_steps = torch.arange(query_seq_len, device=device) // query_tokens_per_step
    key_steps = torch.arange(key_seq_len, device=device) // key_tokens_per_step
    return query_steps[:, None] == key_steps[None, :]


def _same_step_mask_from_ids(
    query_step_ids: torch.Tensor,
    key_step_ids: torch.Tensor,
    query_seq_len: int,
    key_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if query_step_ids.shape != (query_seq_len,):
        raise ValueError(f"`query_step_ids` must have shape ({query_seq_len},), got {tuple(query_step_ids.shape)}")
    if key_step_ids.shape != (key_seq_len,):
        raise ValueError(f"`key_step_ids` must have shape ({key_seq_len},), got {tuple(key_step_ids.shape)}")
    query_step_ids = query_step_ids.to(device=device)
    key_step_ids = key_step_ids.to(device=device)
    return query_step_ids[:, None] == key_step_ids[None, :]


def _cross_mask(
    query_seq_len: int,
    key_seq_len: int,
    query_tokens_per_step: Optional[int],
    key_tokens_per_step: Optional[int],
    query_step_ids: Optional[torch.Tensor],
    key_step_ids: Optional[torch.Tensor],
    policy: VideoLidarAttentionPolicy,
    device: torch.device,
) -> torch.Tensor:
    if policy.cross_frame_rule == "all":
        return torch.ones((query_seq_len, key_seq_len), dtype=torch.bool, device=device)
    if query_step_ids is not None or key_step_ids is not None:
        if query_step_ids is None or key_step_ids is None:
            raise ValueError("`same_step` cross attention requires both query_step_ids and key_step_ids")
        return _same_step_mask_from_ids(
            query_step_ids=query_step_ids,
            key_step_ids=key_step_ids,
            query_seq_len=query_seq_len,
            key_seq_len=key_seq_len,
            device=device,
        )
    if query_tokens_per_step is None or key_tokens_per_step is None:
        raise ValueError(
            "`same_step` cross attention requires either explicit step ids or both "
            "query_tokens_per_step and key_tokens_per_step"
        )
    return _same_step_mask(
        query_seq_len=query_seq_len,
        key_seq_len=key_seq_len,
        query_tokens_per_step=query_tokens_per_step,
        key_tokens_per_step=key_tokens_per_step,
        device=device,
    )


def build_video_lidar_attention_mask(
    *,
    video_seq_len: int,
    lidar_seq_len: int,
    policy: VideoLidarAttentionPolicy = VideoLidarAttentionPolicy(),
    device: torch.device | str = "cpu",
    video_tokens_per_step: Optional[int] = None,
    lidar_tokens_per_step: Optional[int] = None,
    video_step_ids: Optional[torch.Tensor] = None,
    lidar_step_ids: Optional[torch.Tensor] = None,
    video_queries_to_lidar_keys_mask: Optional[torch.Tensor] = None,
    lidar_queries_to_video_keys_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a square boolean mask for [video, lidar] mixed attention.

    Rows are query tokens and columns are key/value tokens. Custom masks can be
    used later for GALA sparse projection:

    - `video_queries_to_lidar_keys_mask`: shape [Sv, Sl]
    - `lidar_queries_to_video_keys_mask`: shape [Sl, Sv]

    For `cross_frame_rule="same_step"`, prefer explicit `*_step_ids`
    when token order is not simply [step0 tokens, step1 tokens, ...].
    """

    _validate_policy(policy)
    _validate_seq_len("video_seq_len", video_seq_len)
    _validate_seq_len("lidar_seq_len", lidar_seq_len)

    device = torch.device(device)
    total_seq_len = video_seq_len + lidar_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

    video_slice = slice(0, video_seq_len)
    lidar_slice = slice(video_seq_len, total_seq_len)

    if policy.allow_video_self:
        mask[video_slice, video_slice] = True
    if policy.allow_lidar_self:
        mask[lidar_slice, lidar_slice] = True

    if policy.mode in ("lidar_to_video", "bidirectional"):
        if video_queries_to_lidar_keys_mask is None:
            video_to_lidar_keys = _cross_mask(
                query_seq_len=video_seq_len,
                key_seq_len=lidar_seq_len,
                query_tokens_per_step=video_tokens_per_step,
                key_tokens_per_step=lidar_tokens_per_step,
                query_step_ids=video_step_ids,
                key_step_ids=lidar_step_ids,
                policy=policy,
                device=device,
            )
        else:
            video_to_lidar_keys = _validate_custom_mask(
                "video_queries_to_lidar_keys_mask",
                video_queries_to_lidar_keys_mask,
                (video_seq_len, lidar_seq_len),
                device,
            )
        mask[video_slice, lidar_slice] = video_to_lidar_keys

    if policy.mode in ("video_to_lidar", "bidirectional"):
        if lidar_queries_to_video_keys_mask is None:
            lidar_to_video_keys = _cross_mask(
                query_seq_len=lidar_seq_len,
                key_seq_len=video_seq_len,
                query_tokens_per_step=lidar_tokens_per_step,
                key_tokens_per_step=video_tokens_per_step,
                query_step_ids=lidar_step_ids,
                key_step_ids=video_step_ids,
                policy=policy,
                device=device,
            )
        else:
            lidar_to_video_keys = _validate_custom_mask(
                "lidar_queries_to_video_keys_mask",
                lidar_queries_to_video_keys_mask,
                (lidar_seq_len, video_seq_len),
                device,
            )
        mask[lidar_slice, video_slice] = lidar_to_video_keys

    return mask
