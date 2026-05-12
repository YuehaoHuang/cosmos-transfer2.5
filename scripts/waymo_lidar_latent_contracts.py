# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR latent contracts shared by Waymo video->LiDAR training utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


NORMAL_VIDEO_LATENT_SHAPE = (16, 40, 90, 160)

OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT: dict[str, Any] = {
    "version": "official_ltcv_fullwidth_v1",
    "tokenizer": "s3_ltcv",
    "latent_kind": "compressed_latent_from_encoder",
    "latent_shape": [16, 8, 64, 226],
    "exact_context_shape": [16, 1, 64, 226],
    "requires_exact_context_for_decode": True,
    "input_width": 1800,
    "tokenizer_padded_width": 1808,
    "lidar_crop_width": 0,
    "crop_mode": "none",
}

WAN21_NATIVE64X1312_REPEATROW11_LIDAR_LATENT_CONTRACT: dict[str, Any] = {
    "version": "wan21_native64x1312_repeatrow11_v1",
    "tokenizer": "wan2pt1",
    "latent_kind": "wan2pt1_vae_latent",
    "latent_shape": [16, 8, 88, 164],
    "exact_context_shape": None,
    "requires_exact_context_for_decode": False,
    "preprocess_mode": "waymo_top_64x2650",
    "native_n_rows": 64,
    "native_n_cols": 1312,
    "downsample_factor_row": 1,
    "downsample_factor_col": 1,
    "repeat_row": 11,
    "repeat_col": 1,
    "input_channel_mode": "repeat_depth",
    "decode_channel_mode": "mean",
    "wan_spatial_align": 8,
    "input_height": 704,
    "input_width": 1312,
}

KNOWN_LIDAR_LATENT_CONTRACTS: dict[str, dict[str, Any]] = {
    OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT["version"]: OFFICIAL_LTCV_LIDAR_LATENT_CONTRACT,
    WAN21_NATIVE64X1312_REPEATROW11_LIDAR_LATENT_CONTRACT[
        "version"
    ]: WAN21_NATIVE64X1312_REPEATROW11_LIDAR_LATENT_CONTRACT,
}


def _source_label(source: str | Path | None) -> str:
    return "" if source is None else f" in {source}"


def normalize_lidar_latent_contract(contract: str | dict[str, Any], *, source: str | Path | None = None) -> dict[str, Any]:
    """Return a canonical copy of a known LiDAR latent contract."""

    if isinstance(contract, str):
        version = contract
    elif isinstance(contract, dict):
        version_value = contract.get("version")
        if not isinstance(version_value, str):
            raise ValueError(f"LiDAR latent contract{_source_label(source)} is missing a string 'version'.")
        version = version_value
    else:
        raise TypeError(
            f"LiDAR latent contract{_source_label(source)} must be a version string or dict, "
            f"got {type(contract).__name__}."
        )
    if version not in KNOWN_LIDAR_LATENT_CONTRACTS:
        known = ", ".join(sorted(KNOWN_LIDAR_LATENT_CONTRACTS))
        raise ValueError(f"Unknown LiDAR latent contract{_source_label(source)}: {version!r}. Known: {known}.")
    return deepcopy(KNOWN_LIDAR_LATENT_CONTRACTS[version])


def infer_lidar_latent_contract_from_shape(shape: tuple[int, ...], *, source: str | Path | None = None) -> dict[str, Any]:
    matches = [
        contract
        for contract in KNOWN_LIDAR_LATENT_CONTRACTS.values()
        if tuple(contract["latent_shape"]) == tuple(shape)
    ]
    if len(matches) == 1:
        return deepcopy(matches[0])
    if not matches:
        known = ", ".join(
            f"{contract['version']}={tuple(contract['latent_shape'])}"
            for contract in KNOWN_LIDAR_LATENT_CONTRACTS.values()
        )
        raise ValueError(f"Cannot infer LiDAR latent contract for shape {shape}{_source_label(source)}. Known: {known}.")
    versions = ", ".join(str(contract["version"]) for contract in matches)
    raise ValueError(f"Ambiguous LiDAR latent shape {shape}{_source_label(source)}; matches: {versions}.")


def contract_from_payload(
    payload: dict[str, Any],
    latent_shape: tuple[int, ...],
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    raw_contract = payload.get("lidar_latent_contract", payload.get("latent_contract"))
    if raw_contract is None:
        return infer_lidar_latent_contract_from_shape(latent_shape, source=source)
    return normalize_lidar_latent_contract(raw_contract, source=source)


def lidar_contracts_equal(left: str | dict[str, Any], right: str | dict[str, Any]) -> bool:
    return normalize_lidar_latent_contract(left) == normalize_lidar_latent_contract(right)


def lidar_latent_shape(contract: str | dict[str, Any]) -> tuple[int, ...]:
    normalized = normalize_lidar_latent_contract(contract)
    return tuple(int(x) for x in normalized["latent_shape"])


def lidar_latent_hw(contract: str | dict[str, Any]) -> tuple[int, int]:
    shape = lidar_latent_shape(contract)
    return int(shape[-2]), int(shape[-1])


def lidar_exact_context_shape(contract: str | dict[str, Any]) -> tuple[int, ...] | None:
    normalized = normalize_lidar_latent_contract(contract)
    shape = normalized.get("exact_context_shape")
    if shape is None:
        return None
    return tuple(int(x) for x in shape)
