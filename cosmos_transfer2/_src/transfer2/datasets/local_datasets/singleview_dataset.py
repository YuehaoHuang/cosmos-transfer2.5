# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Single-view Transfer dataset for local video files with control inputs.

This dataset loader is designed for post-training Cosmos-Transfer2 models with local data.
It uses the full Transfer2 augmentor pipeline including:
- Randomized edge detection thresholds for training diversity
- Automatic resizing with aspect ratio preservation
- Reflection padding
- Text transforms for caption handling
- Control input generation or loading (edge, depth, seg, blur, rangemap_layout, etc.)

Example usage:
    dataset = SingleViewTransferDataset(
        dataset_dir="datasets/example",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,  # Enable augmentations for training
    )
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import (
    get_rangemap_layout_augmentor_for_local_datasets,
    get_video_augmentor_v2,
    get_video_augmentor_v2_with_control,
)
from cosmos_transfer2._src.transfer2.utils.input_handling import detect_aspect_ratio


# Mock URL object for augmentor compatibility
class MockUrlMeta:
    """Mock metadata object for WebDataset compatibility."""

    def __init__(self):
        self.opts = {}


class MockUrl:
    """Mock URL object that augmentors expect from WebDataset."""

    def __init__(self, url: str):
        self._url = url
        self.meta = MockUrlMeta()

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"MockUrl({self._url})"


# Mappings between control types and corresponding sub-folder names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "seg": {"folder": "seg", "format": "mp4", "data_dict_key": "segmentation"},
    "rangemap_layout": {"folder": "rangemap_layout", "format": "mp4", "data_dict_key": "rangemap_layout"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly by augmentor
    "vis": {"folder": None},  # Blur, computed on-the-fly by augmentor
}


class SingleViewTransferDataset(Dataset):
    """Dataset class for loading single-view video-to-video generation data with control inputs.

    This dataset is designed for post-training Cosmos-Transfer2 models with local video files.
    It supports various control modalities including depth, segmentation, edge,
    blur, and precomputed rangemap_layout videos.

    Dataset structure:
        dataset_dir/
        ├── videos/
        │   ├── video1.mp4
        │   └── video2.mp4
        ├── captions/
        │   ├── video1.json  ({"caption": "text description"})
        │   └── video2.json
        └── <control_type>/  (optional for depth/seg/rangemap_layout, computed on-the-fly for edge/vis)
            ├── video1.mp4  (for depth or rangemap_layout)
            └── video1.pickle  (for seg/keypoint)
        └── rangemap_targets/  (optional cached targets for online VAE encode)
            ├── video1.npz  (normalized_rangemap: float32[T, H, W])
            └── video2.npz

    Waymo LiDAR training can instead enumerate samples and build the target and rangemap_layout
    online from raw LiDAR tar files; no local MP4 manifest is required in that mode.

    Args:
        dataset_dir: Base path to the dataset directory
        num_frames: Number of frames to load per sequence
        video_size: Target size (H, W) for video frames
        resolution: Resolution key for augmentor (e.g., "720", "1080")
        hint_key: Control input type (e.g., "control_input_edge", "control_input_depth")
        is_train: Whether this is for training (affects sampling)
        caption_type: Type of caption to load (default: "t2w_qwen2p5_7b")
        rangemap_target_folder: Optional folder containing preprocessed lossless range-map targets.
        rangemap_target_key: Data-batch key used for online target encoding.
        rangemap_valid_mask_key: Data-batch key used for optional online target loss weighting.
        rangemap_edge_mask_key: Data-batch key used for optional edge-aware target loss weighting.
        rangemap_target_repeat_row: Vertical expansion applied to the stored target.
        expected_rangemap_target_shape: Optional required target video tensor shape (C, T, H, W).
        raw_rangemap_online: Build Waymo range-map target and rangemap_layout from raw LiDAR tar files at load time.
        raw_lidar_root: Waymo root containing ``<split>/lidar_raw/*.tar`` for the online path.
        fallback_caption: Caption returned when no per-sample caption exists.
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        resolution: str = "720",
        hint_key: str | None = "control_input_edge",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",  # Use Qwen2.5-7B caption type
        decord_num_threads: int | None = None,
        rangemap_target_folder: str | None = None,
        rangemap_target_key: str = "rangemap_target",
        rangemap_valid_mask_key: str = "rangemap_valid_mask",
        rangemap_edge_mask_key: str = "rangemap_edge_mask",
        rangemap_target_repeat_row: int = 11,
        rangemap_target_repeat_col: int = 1,
        expected_rangemap_target_shape: tuple[int, int, int, int] | None = None,
        raw_rangemap_online: bool = False,
        raw_lidar_root: str | None = None,
        raw_lidar_split: str = "training",
        lidar_utils_repo: str = "/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen",
        lidar_chunk_stride_frames: int = 10,
        pad_lidar_last: bool = False,
        native_n_rows: int = 64,
        native_n_cols: int = 1280,
        projection_max_range: float = 105.0,
        downsample_factor_row: int = 1,
        downsample_factor_col: int = 1,
        downsample_method: str = "scatter_min",
        input_channel_mode: str = "repeat_depth",
        max_range: float = 100.0,
        min_range: float = 5.0,
        min_value: float = -1.0,
        layout_edge_threshold_m: float = 1.0,
        online_fps: float = 10.0,
        fallback_caption: str = "a video",
        **kwargs,  # Accept extra params for config compatibility (like MultiviewTransferDataset)
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.video_size = video_size
        self.resolution = resolution
        self.is_train = is_train
        self.caption_type = caption_type
        if decord_num_threads is None:
            decord_num_threads = int(os.environ.get("DECORD_NUM_THREADS", "2"))
        if decord_num_threads < 1:
            raise ValueError(f"decord_num_threads must be positive, got: {decord_num_threads}")
        self.decord_num_threads = decord_num_threads
        self.rangemap_target_key = rangemap_target_key
        self.rangemap_valid_mask_key = rangemap_valid_mask_key
        self.rangemap_edge_mask_key = rangemap_edge_mask_key
        self.rangemap_target_repeat_row = rangemap_target_repeat_row
        self.rangemap_target_repeat_col = rangemap_target_repeat_col
        self.expected_rangemap_target_shape = (
            None if expected_rangemap_target_shape is None else tuple(expected_rangemap_target_shape)
        )
        self.rangemap_target_dir = None if rangemap_target_folder is None else Path(self.dataset_dir) / rangemap_target_folder
        if self.rangemap_target_dir is not None and not self.rangemap_target_dir.is_dir():
            raise FileNotFoundError(f"Range-map target folder does not exist: {self.rangemap_target_dir}")
        if self.rangemap_target_repeat_row < 1 or self.rangemap_target_repeat_col < 1:
            raise ValueError("Range-map target repeat factors must be positive")
        self.raw_rangemap_online = raw_rangemap_online
        self.raw_lidar_dir = None if raw_lidar_root is None else Path(raw_lidar_root) / raw_lidar_split / "lidar_raw"
        self.lidar_utils_repo = lidar_utils_repo
        self.lidar_chunk_stride_frames = lidar_chunk_stride_frames
        self.pad_lidar_last = pad_lidar_last
        self.native_n_rows = native_n_rows
        self.native_n_cols = native_n_cols
        self.projection_max_range = projection_max_range
        self.downsample_factor_row = downsample_factor_row
        self.downsample_factor_col = downsample_factor_col
        self.downsample_method = downsample_method
        self.input_channel_mode = input_channel_mode
        self.max_range = max_range
        self.min_range = min_range
        self.min_value = min_value
        self.layout_edge_threshold_m = layout_edge_threshold_m
        self.online_fps = online_fps
        self.fallback_caption = fallback_caption

        # Parse control type from hint_key. A None hint_key enables video-only
        # post-training while keeping the same local dataset structure.
        self.hint_key = hint_key
        if hint_key is None:
            self.ctrl_type = None
            self.ctrl_config = None
        else:
            self.ctrl_type = hint_key.replace("control_input_", "")
            if self.ctrl_type not in CTRL_TYPE_INFO:
                raise ValueError(
                    f"Unsupported control type: {self.ctrl_type}. Supported types: {list(CTRL_TYPE_INFO.keys())}"
                )
            self.ctrl_config = CTRL_TYPE_INFO[self.ctrl_type]

        if self.raw_rangemap_online:
            if self.ctrl_type != "rangemap_layout":
                raise ValueError("raw_rangemap_online requires hint_key='control_input_rangemap_layout'")
            if self.rangemap_target_dir is not None:
                raise ValueError("raw_rangemap_online and rangemap_target_folder are mutually exclusive")
            if self.raw_lidar_dir is None or not self.raw_lidar_dir.is_dir():
                raise FileNotFoundError(f"Raw LiDAR tar folder does not exist: {self.raw_lidar_dir}")
            if self.input_channel_mode != "repeat_depth":
                raise ValueError("raw_rangemap_online currently requires input_channel_mode='repeat_depth'")
            if self.lidar_chunk_stride_frames < 1:
                raise ValueError("lidar_chunk_stride_frames must be positive")

        # Keep synthetic MP4 names for the existing __getitem__ key handling.
        # In raw-online mode no MP4 is opened; tar files define the sample set.
        if self.raw_rangemap_online:
            self.video_paths = [f"{sample_key}.mp4" for sample_key in self._list_online_raw_sample_keys()]
        else:
            video_dir = os.path.join(self.dataset_dir, "videos")
            self.video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")])

        # Support both "captions/" and "metas/" directories
        self.caption_dir = os.path.join(self.dataset_dir, "captions")
        if not os.path.exists(self.caption_dir):
            self.caption_dir = os.path.join(self.dataset_dir, "metas")

        # Note: We no longer load T5 embeddings - captions are encoded on-the-fly
        # by the model's text encoder (Qwen2.5-VL-7B / reason1p1_7B)
        self.num_failed_loads = 0
        self.bad_video_indices = set()  # Track videos that fail to load (too short, corrupted, etc.)

        # Use proper augmentor pipeline for training quality
        # This includes randomized edge detection, reflection padding, and text transforms
        # Pass embedding_type=None since we're handling T5 embeddings ourselves
        # (if embedding_type is set, the function returns early with only video_parsing)
        if self.ctrl_type is None:
            augmentor_config = get_video_augmentor_v2(
                resolution=resolution,
                caption_type=caption_type,
                embedding_type=None,
            )
            skip_augmentors = ["video_parsing", "merge_datadict", "text_transform"]
            augmentor_config = {k: v for k, v in augmentor_config.items() if k not in skip_augmentors}
        elif self.ctrl_type == "rangemap_layout":
            augmentor_config = get_rangemap_layout_augmentor_for_local_datasets(
                resolution=resolution,
                caption_type=caption_type,
                embedding_type=None,
                control_input_type=self.ctrl_type,
                use_random=is_train,
            )
        else:
            augmentor_config = get_video_augmentor_v2_with_control(
                resolution=resolution,
                caption_type=caption_type,
                embedding_type=None,  # We handle embeddings ourselves, get full augmentor pipeline
                control_input_type=self.ctrl_type,
                use_random=is_train,  # Enable random augmentations for training
            )

            # Filter out augmentors that don't apply to local datasets
            # The augmentor pipeline includes augmentors designed for S3/WebDataset that need to be skipped:
            # - video_parsing: Decodes video bytes from S3 → we already load tensors from local MP4 files
            # - depth_parsing: Decodes depth bytes from S3 key "depth_pervideo_video_depth_anything" → we load from local depth/ folder
            # - seg_parsing: Decodes seg bytes from S3 key "segmentation_sam2_color_video_v2" → we load from local seg/ folder
            # - merge_datadict: Merges multiple WebDataset shards → not needed for single local dataset
            # - text_transform: Loads pre-computed T5 embeddings → we pass raw captions for on-the-fly encoding
            skip_augmentors = ["video_parsing", "merge_datadict", "text_transform", "depth_parsing", "seg_parsing"]
            augmentor_config = {k: v for k, v in augmentor_config.items() if k not in skip_augmentors}

        log.info(f"Filtered augmentors: {list(augmentor_config.keys())}")

        # Instantiate augmentors
        self.augmentor = {k: instantiate(v) for k, v in augmentor_config.items()}

        # Double-check text_transform is not present
        if "text_transform" in self.augmentor:
            raise RuntimeError("text_transform should have been filtered out but is still present!")

        log.info(f"Initialized SingleViewTransferDataset with {len(self.video_paths)} samples")
        log.info(f"  Dataset dir: {self.dataset_dir}")
        log.info(f"  Control type: {self.ctrl_type or 'none'}")
        log.info(f"  Resolution: {resolution}, Video size: {video_size}")
        log.info(f"  Required frames: {self.sequence_length}")
        log.info(f"  Decord threads: {self.decord_num_threads}")
        if self.rangemap_target_dir is not None:
            log.info(f"  Online VAE target: {self.rangemap_target_dir} -> {self.rangemap_target_key}")
            log.info(f"  Expected rangemap target shape: {self.expected_rangemap_target_shape}")
        if self.raw_rangemap_online:
            log.info(f"  Online raw LiDAR source: {self.raw_lidar_dir}")
            log.info(f"  Online raw LiDAR target/control key: {self.rangemap_target_key}")
            log.info(f"  Expected rangemap target shape: {self.expected_rangemap_target_shape}")

        # Quick validation: check for obviously bad videos (optional, can be slow for large datasets)
        # self._validate_videos()  # Uncomment to pre-filter bad videos at initialization

    def __str__(self) -> str:
        return f"SingleViewTransferDataset: {len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _list_online_raw_sample_keys(self) -> list[str]:
        from scripts.prepare_waymo_lidar_singleview_posttrain_dataset import list_lidar_sample_keys

        args = argparse.Namespace(
            range_map_source="raw",
            num_frames=self.sequence_length,
            pad_last=self.pad_lidar_last,
            lidar_chunk_stride_frames=self.lidar_chunk_stride_frames,
        )
        sample_keys = list_lidar_sample_keys(self.raw_lidar_dir, args)
        if not sample_keys:
            raise ValueError(f"No raw LiDAR sample windows found in {self.raw_lidar_dir}")
        return sample_keys

    def _load_rangemap_target(self, video_name: str) -> torch.Tensor:
        if self.rangemap_target_dir is None:
            raise RuntimeError("_load_rangemap_target called without rangemap_target_folder")
        target_path = self.rangemap_target_dir / f"{video_name}.npz"
        if not target_path.exists():
            raise FileNotFoundError(f"Missing range-map target: {target_path}")
        with np.load(target_path, allow_pickle=False) as payload:
            if "normalized_rangemap" not in payload:
                raise KeyError(f"Target payload has no 'normalized_rangemap' field: {target_path}")
            normalized = payload["normalized_rangemap"].astype(np.float32, copy=False)
            stored_row = int(payload["repeat_row"])
            stored_col = int(payload["repeat_col"])
        if (stored_row, stored_col) != (self.rangemap_target_repeat_row, self.rangemap_target_repeat_col):
            raise ValueError(
                f"Range-map target repeat mismatch in {target_path}: stored={(stored_row, stored_col)}, "
                f"configured={(self.rangemap_target_repeat_row, self.rangemap_target_repeat_col)}"
            )
        if normalized.ndim != 3:
            raise ValueError(f"Expected normalized target shape (T, H, W), got {normalized.shape} in {target_path}")
        if normalized.shape[0] != self.sequence_length:
            raise ValueError(
                f"Range-map target has {normalized.shape[0]} frames, expected {self.sequence_length}: {target_path}"
            )
        expanded = np.repeat(normalized, self.rangemap_target_repeat_row, axis=1)
        expanded = np.repeat(expanded, self.rangemap_target_repeat_col, axis=2)
        target = torch.from_numpy(expanded).unsqueeze(0).repeat(3, 1, 1, 1).contiguous()
        if self.expected_rangemap_target_shape is not None and tuple(target.shape) != self.expected_rangemap_target_shape:
            raise ValueError(
                f"Range-map target shape mismatch in {target_path}: expected {self.expected_rangemap_target_shape}, "
                f"got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError(f"Range-map target contains non-finite values: {target_path}")
        if target.min() < -1.0001 or target.max() > 1.0001:
            raise ValueError(f"Range-map target is outside normalized [-1, 1] bounds: {target_path}")
        return target

    def _load_online_raw_rangemap_sample(
        self, video_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, list[int]]:
        """Project a raw Waymo LiDAR window into target and control tensors."""

        from scripts.prepare_waymo_lidar_singleview_posttrain_dataset import (
            make_rangemap_layout_video,
            normalized_tensor_to_uint8_video,
        )
        from scripts.smoke_waymo_lidar_wan21_vae import (
            load_raw_range_maps,
            prepend_lidar_utils_repo,
            preprocess_range_maps,
        )

        segment_key, separator, chunk_id = video_name.rpartition("_")
        if not separator or not chunk_id.isdigit():
            raise ValueError(f"Online raw LiDAR sample key must end with an integer chunk index: {video_name}")
        frame_start = int(chunk_id) * self.lidar_chunk_stride_frames
        raw_tar = self.raw_lidar_dir / f"{segment_key}.tar"
        if not raw_tar.exists():
            raise FileNotFoundError(f"Missing raw LiDAR tar for {video_name}: {raw_tar}")

        preprocess_args = argparse.Namespace(
            downsample_factor_row=self.downsample_factor_row,
            downsample_factor_col=self.downsample_factor_col,
            downsample_method=self.downsample_method,
            repeat_row=self.rangemap_target_repeat_row,
            repeat_col=self.rangemap_target_repeat_col,
            input_channel_mode=self.input_channel_mode,
            max_range=self.max_range,
            min_range=self.min_range,
            min_value=self.min_value,
        )
        prepend_lidar_utils_repo(self.lidar_utils_repo)
        range_maps, _ = load_raw_range_maps(
            raw_tar,
            frame_start=frame_start,
            num_frames=self.sequence_length,
            pad_last=self.pad_lidar_last,
            n_rows=self.native_n_rows,
            n_cols=self.native_n_cols,
            max_projection_range=self.projection_max_range,
        )
        target_batch, downsampled_range, valid_mask = preprocess_range_maps(range_maps, preprocess_args)
        target = target_batch.squeeze(0).contiguous()
        if self.expected_rangemap_target_shape is not None and tuple(target.shape) != self.expected_rangemap_target_shape:
            raise ValueError(
                f"Online raw LiDAR target shape mismatch for {video_name}: expected "
                f"{self.expected_rangemap_target_shape}, got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all() or target.min() < -1.0001 or target.max() > 1.0001:
            raise ValueError(f"Online raw LiDAR target contains invalid normalized values: {video_name}")

        video_frames = normalized_tensor_to_uint8_video(target_batch)
        video = torch.from_numpy(video_frames).permute(3, 0, 1, 2).contiguous()
        layout_frames = make_rangemap_layout_video(
            downsampled_range,
            valid_mask,
            repeat_row=self.rangemap_target_repeat_row,
            repeat_col=self.rangemap_target_repeat_col,
            edge_threshold_m=self.layout_edge_threshold_m,
        )
        layout = torch.from_numpy(layout_frames).permute(3, 0, 1, 2).contiguous()
        if video.shape != target.shape or layout.shape != target.shape:
            raise ValueError(
                f"Online raw LiDAR target/control alignment mismatch for {video_name}: "
                f"target={tuple(target.shape)}, video={tuple(video.shape)}, layout={tuple(layout.shape)}"
            )
        valid_expanded = valid_mask[:, None, :, :]
        valid_expanded = np.repeat(valid_expanded, self.rangemap_target_repeat_row, axis=2)
        valid_expanded = np.repeat(valid_expanded, self.rangemap_target_repeat_col, axis=3)
        valid_t = torch.from_numpy(valid_expanded).permute(1, 0, 2, 3).contiguous().bool()
        edge_t = torch.from_numpy(layout_frames[..., 1] > 0).permute(0, 1, 2).unsqueeze(0).contiguous().bool()
        if valid_t.shape[1:] != target.shape[1:]:
            raise ValueError(f"Online raw LiDAR valid mask shape mismatch for {video_name}: {valid_t.shape}")
        if edge_t.shape[1:] != target.shape[1:]:
            raise ValueError(f"Online raw LiDAR edge mask shape mismatch for {video_name}: {edge_t.shape}")
        frame_ids = list(range(frame_start, frame_start + self.sequence_length))
        return video, layout, target, valid_t, edge_t, float(self.online_fps), frame_ids

    def _validate_videos(self) -> None:
        """Validate all videos and pre-mark bad ones (too short, corrupted, etc.).

        This is optional and can be slow for large datasets, but helps identify
        problematic videos upfront. Call this in __init__ if you want pre-filtering.
        """
        log.info("Validating videos for minimum frame count...")
        bad_count = 0

        for idx, video_path in enumerate(self.video_paths):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                del vr

                if total_frames < self.sequence_length:
                    self.bad_video_indices.add(idx)
                    bad_count += 1
                    log.debug(
                        f"Marking video {idx} as bad: {os.path.basename(video_path)} "
                        f"has only {total_frames} frames (need {self.sequence_length})"
                    )
            except Exception as e:
                self.bad_video_indices.add(idx)
                bad_count += 1
                log.debug(f"Marking video {idx} as bad: {os.path.basename(video_path)} - {e}")

        valid_count = len(self.video_paths) - bad_count
        log.info(
            f"Video validation complete: {valid_count} valid, {bad_count} bad "
            f"({bad_count / len(self.video_paths) * 100:.1f}% filtered)"
        )

    def _load_video(self, video_path: str, frame_ids: list[int] | None = None) -> tuple[np.ndarray, float, list[int]]:
        """Load video frames from file.

        Args:
            video_path: Path to video file
            frame_ids: Specific frame indices to load. If None, randomly samples frames.

        Returns:
            Tuple of (frames, fps, frame_ids)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=self.decord_num_threads)
        total_frames = len(vr)

        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # Sample frames if not provided
        if frame_ids is None:
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx + 1) if self.is_train else 0
            frame_ids = list(range(start_frame, start_frame + self.sequence_length))

        # Load frames
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # Reset video reader

        # Debug: Log frame loading
        log.info(
            f"Loaded video {os.path.basename(video_path)}: "
            f"total_frames={total_frames}, "
            f"requested={len(frame_ids)}, "
            f"loaded={frame_data.shape[0]}"
        )

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 24  # Default FPS

        del vr
        return frame_data, fps, frame_ids

    def _load_caption(self, video_name: str) -> str:
        """Load caption from JSON or text file.

        Args:
            video_name: Video name without extension

        Returns:
            Caption text
        """
        # Try JSON first (Transfer2 format)
        json_path = Path(self.caption_dir) / f"{video_name}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Support various JSON formats
                    if isinstance(data, dict):
                        return data.get("caption", data.get("text", data.get("prompt", "")))
                    return str(data)
            except Exception as e:
                log.warning(f"Failed to load caption from {json_path}: {e}")

        # Fall back to text file (Transfer1 format)
        txt_path = Path(self.caption_dir) / f"{video_name}.txt"
        if txt_path.exists():
            try:
                return txt_path.read_text().strip()
            except Exception as e:
                log.warning(f"Failed to load caption from {txt_path}: {e}")

        log.debug(f"No caption found for {video_name}, using fallback caption")
        return self.fallback_caption

    # Captions are now encoded on-the-fly by the model's text encoder (Qwen/reason1p1_7B)

    def _load_control_data(self, video_name: str, frame_ids: list[int]) -> dict[str, Any] | None:
        """Load control input data (depth, segmentation, etc.).

        For edge/vis, returns None (computed on-the-fly by augmentor).
        For depth, loads video frames.
        For seg/keypoint, loads pickle data.

        Args:
            video_name: Video name without extension
            frame_ids: Frame indices to load

        Returns:
            Dictionary with control data or None if computed on-the-fly
        """
        if self.ctrl_type is None or self.ctrl_config is None:
            return None

        # Edge and vis are computed on-the-fly by the augmentor
        if self.ctrl_config["folder"] is None:
            return None

        ctrl_folder = os.path.join(self.dataset_dir, self.ctrl_config["folder"])
        ctrl_format = self.ctrl_config["format"]
        ctrl_path = os.path.join(ctrl_folder, f"{video_name}.{ctrl_format}")

        if not os.path.exists(ctrl_path):
            raise FileNotFoundError(f"Control input file not found: {ctrl_path}")

        data_dict = {}

        try:
            if self.ctrl_type == "seg":
                # Load segmentation video (same format as depth)
                vr = VideoReader(ctrl_path, ctx=cpu(0), num_threads=self.decord_num_threads)
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Seg video has fewer frames than RGB video: {ctrl_path}")

                seg_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                seg_frames = seg_frames.astype(np.uint8)
                # Convert to tensor - augmentor will handle resizing to match video
                seg_t = torch.from_numpy(seg_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                seg_video = seg_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputSeg augmentor
                data_dict["segmentation"] = seg_video
                del vr

            elif self.ctrl_type == "keypoint":
                # Load keypoint pickle
                with open(ctrl_path, "rb") as f:
                    keypoint_data = pickle.load(f)
                data_dict["keypoint"] = keypoint_data

            elif self.ctrl_type == "depth":
                # Load depth video
                vr = VideoReader(ctrl_path, ctx=cpu(0), num_threads=self.decord_num_threads)
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Depth video has fewer frames than RGB video: {ctrl_path}")

                depth_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                depth_frames = depth_frames.astype(np.uint8)
                # Convert to tensor - augmentor will handle resizing to match video
                depth_t = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                depth_video = depth_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputDepth augmentor
                data_dict["depth"] = depth_video
                del vr

            elif self.ctrl_type == "rangemap_layout":
                vr = VideoReader(ctrl_path, ctx=cpu(0), num_threads=self.decord_num_threads)
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Range-map layout control video has fewer frames than target video: {ctrl_path}")

                layout_frames = vr.get_batch(frame_ids).asnumpy().astype(np.uint8)
                layout_t = torch.from_numpy(layout_frames).permute(0, 3, 1, 2)
                data_dict["rangemap_layout"] = layout_t.permute(1, 0, 2, 3)
                del vr

        except Exception as e:
            log.warning(f"Failed to load control data from {ctrl_path}: {e}")
            return None

        return data_dict

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single training sample with full augmentation.

        Returns:
            Dictionary with:
                - video: RGB video tensor [C=3, T, H, W] dtype=uint8, resized and padded by augmentor
                - control_input_<type>: Control input tensor [C=3, T, H, W] dtype=uint8, same shape as video
                  (generated with randomized parameters if is_train=True)
                - fps: Video frame rate (float)
                - aspect_ratio: Aspect ratio string (e.g., "16:9")
                - image_size: Image dimensions [H, W, H, W] tensor (after augmentation)
                - padding_mask: Padding mask [1, H, W] tensor (marks valid vs. padded regions)
                - num_frames: Number of frames (int)
                - chunk_index: Chunk index (int, typically 0)
                - __url__: Dataset directory (str)
                - __key__: Video name (str)

        Note:
            - Augmentor applies randomized edge detection thresholds during training for diversity
            - Video dimensions may differ from input video_size due to augmentor's padding/resizing
            - Model expects uint8 format and performs normalization internally:
              uint8 [0, 255] → float32 [-1, 1]
        """
        max_retries = 10  # Try up to 10 different videos
        original_index = index

        for retry in range(max_retries):
            # Skip known bad videos
            if index in self.bad_video_indices:
                index = (index + 1) % len(self.video_paths)
                continue

            try:
                video_path = self.video_paths[index]
                video_name = os.path.basename(video_path).replace(".mp4", "")

                online_rangemap_target = None
                online_rangemap_layout = None
                online_rangemap_valid_mask = None
                online_rangemap_edge_mask = None
                if self.raw_rangemap_online:
                    (
                        video,
                        online_rangemap_layout,
                        online_rangemap_target,
                        online_rangemap_valid_mask,
                        online_rangemap_edge_mask,
                        fps,
                        frame_ids,
                    ) = self._load_online_raw_rangemap_sample(video_name)
                else:
                    frames, fps, frame_ids = self._load_video(video_path)
                    frames = frames.astype(np.uint8)
                    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)
                    video = frames_t.permute(1, 0, 2, 3)
                aspect_ratio = detect_aspect_ratio((video.shape[3], video.shape[2]))  # (W, H)

                # Build data dictionary
                data = {
                    "video": video,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "num_frames": self.sequence_length,
                    "chunk_index": 0,
                    "frame_indices": frame_ids,
                    "n_orig_video_frames": len(frame_ids),
                }

                # Load caption
                caption = self._load_caption(video_name)
                data[self.caption_type] = caption

                # Create metadata structure for augmentor compatibility
                # The augmentor expects "metas" with window information
                # Map caption_type to the expected caption key in window_data
                if self.caption_type == "t2w_qwen2p5_7b":
                    caption_key_in_window = "qwen2p5_7b_caption"  # gitleaks:allow
                else:
                    caption_key_in_window = self.caption_type

                window_data = {
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1] + 1,
                    caption_key_in_window: caption,
                }
                data["metas"] = {
                    "framerate": fps,
                    "nb_frames": len(frame_ids),
                    # Create a single window spanning the entire video segment
                    # Include both windows and t2w_windows for different caption types
                    "windows": [window_data],
                    "t2w_windows": [window_data],
                    "i2w_windows_later_frames": [window_data],
                }

                # Pass raw caption for on-the-fly encoding by model's text encoder
                # (Like multiview dataset - model will encode with Qwen/reason1 encoder)
                data["ai_caption"] = caption

                # Add URL and key for logging (used by augmentors and training)
                # Use MockUrl object for augmentor compatibility (augmentors expect __url__.meta.opts)
                data["__url__"] = MockUrl(str(self.dataset_dir))
                data["__key__"] = video_name

                if online_rangemap_layout is not None:
                    data["rangemap_layout"] = online_rangemap_layout
                    ctrl_data = None
                else:
                    ctrl_data = self._load_control_data(video_name, frame_ids)
                if ctrl_data is not None:
                    data.update(ctrl_data)

                # Apply augmentation pipeline
                # This includes: resizing, padding, text transform, and control input generation
                # The augmentor will handle edge detection with randomized thresholds for training
                for aug_name, aug_fn in self.augmentor.items():
                    result = aug_fn(data)
                    # Check if augmentor returned None (e.g., filtering)
                    if result is None:
                        raise ValueError(f"Augmentor {aug_name} filtered out the sample")
                    data = result

                # Convert MockUrl back to string for DataLoader collate compatibility
                # (PyTorch's collate function can't handle custom objects)
                if isinstance(data.get("__url__"), MockUrl):
                    data["__url__"] = str(data["__url__"])

                # Add final metadata (after augmentation)
                c, t, h, w = data["video"].shape
                if "image_size" not in data:
                    data["image_size"] = torch.tensor([h, w, h, w])
                if "padding_mask" not in data:
                    data["padding_mask"] = torch.ones(1, h, w)  # All valid (no padding)

                # Validate output format after augmentation
                assert data["video"].dtype == torch.uint8, f"Video dtype is {data['video'].dtype}, expected uint8"
                assert data["video"].shape[0] == 3, f"Video should have 3 channels, got {data['video'].shape[0]}"
                assert data["video"].shape[1] == self.sequence_length, (
                    f"Video should have {self.sequence_length} frames, got {data['video'].shape[1]}"
                )

                ctrl_key = None
                if self.ctrl_type is not None:
                    # Check control input exists and has correct format
                    ctrl_key = f"control_input_{self.ctrl_type}"
                    assert ctrl_key in data, f"Control input key '{ctrl_key}' not found in data"
                    assert data[ctrl_key].dtype == torch.uint8, (
                        f"Control input dtype is {data[ctrl_key].dtype}, expected uint8"
                    )
                    assert data[ctrl_key].shape == data["video"].shape, (
                        f"Control input shape {data[ctrl_key].shape} doesn't match video shape {data['video'].shape}"
                    )

                if online_rangemap_target is not None:
                    rangemap_target = online_rangemap_target
                elif self.rangemap_target_dir is not None:
                    rangemap_target = self._load_rangemap_target(video_name)
                else:
                    rangemap_target = None
                if rangemap_target is not None:
                    if rangemap_target.shape != data["video"].shape:
                        raise ValueError(
                            f"Range-map target {rangemap_target.shape} does not align with control/video {data['video'].shape}"
                        )
                    data[self.rangemap_target_key] = rangemap_target
                if online_rangemap_valid_mask is not None:
                    if online_rangemap_valid_mask.shape[1:] != data["video"].shape[1:]:
                        raise ValueError(
                            f"Range-map valid mask {online_rangemap_valid_mask.shape} does not align with "
                            f"video {data['video'].shape}"
                        )
                    data[self.rangemap_valid_mask_key] = online_rangemap_valid_mask
                if online_rangemap_edge_mask is not None:
                    if online_rangemap_edge_mask.shape[1:] != data["video"].shape[1:]:
                        raise ValueError(
                            f"Range-map edge mask {online_rangemap_edge_mask.shape} does not align with "
                            f"video {data['video'].shape}"
                        )
                    data[self.rangemap_edge_mask_key] = online_rangemap_edge_mask

                log.debug(
                    f"Dataset sample ready: video={data['video'].shape} {data['video'].dtype}, "
                    + (
                        f"{ctrl_key}={data[ctrl_key].shape} {data[ctrl_key].dtype}, "
                        if ctrl_key is not None
                        else "control_input=None, "
                    )
                )

                return data

            except Exception as e:
                self.num_failed_loads += 1
                # Mark this video as bad so we skip it in the future
                self.bad_video_indices.add(index)

                log.warning(
                    f"Failed to load video {self.video_paths[index]} (index {index}): {e}. "
                    f"Marking as bad and trying next video. "
                    f"(attempt {retry + 1}/{max_retries}, total bad videos: {len(self.bad_video_indices)})",
                    rank0_only=False,
                )

                if retry == max_retries - 1:
                    log.error(
                        f"Failed to load data after {max_retries} attempts starting from index {original_index}. "
                        f"Total bad videos: {len(self.bad_video_indices)}/{len(self.video_paths)}"
                    )
                    raise RuntimeError(
                        f"Failed to load data after {max_retries} attempts. "
                        f"Original index: {original_index}, last tried: {video_path}"
                    )

                # Try the next video in sequence (wraps around at end)
                index = (index + 1) % len(self.video_paths)

        raise RuntimeError("Should not reach here")


if __name__ == "__main__":
    """
    Sanity check for the dataset.

    Usage:
        PYTHONPATH=. python cosmos_transfer2/_src/transfer2/datasets/local_datasets/single_view_dataset.py
    """
    import sys

    # Example dataset with edge control (computed on-the-fly)
    dataset = SingleViewTransferDataset(
        dataset_dir="datasets/hdvila",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,
    )

    log.info(f"Dataset: {dataset}")
    log.info(f"Number of videos: {len(dataset)}")

    # Test loading a few samples
    indices = [0] if len(dataset) > 0 else []
    for idx in indices:
        log.info(f"\nTesting sample {idx}:")
        try:
            data = dataset[idx]
            log.info(f"  Video shape: {data['video'].shape}")
            log.info(f"  Control input shape: {data['control_input_edge'].shape}")
            log.info(f"  Caption: {data.get('ai_caption', 'N/A')[:100]}...")
            log.info(f"  FPS: {data['fps']}")
            log.info(f"  Aspect ratio: {data['aspect_ratio']}")
            log.info("  ✅ Sample loaded successfully")
        except Exception as e:
            log.error(f"  ❌ Failed to load sample: {e}")
            sys.exit(1)

    log.info("\n✅ All tests passed!")
