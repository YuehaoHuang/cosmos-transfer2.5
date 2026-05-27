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

"""Local dataset configurations for single-view post-training."""

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset import SingleViewTransferDataset


def register_dataloader_local() -> None:
    """Register local dataloader configurations for post-training.

    Note: These are example configurations with a placeholder dataset_dir.
    Override the dataset_dir in your experiment config file to point to your actual data.
    See cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiment/exp_posttrain_singleview.py
    for examples of how to properly configure the dataset path.
    """
    cs = ConfigStore()

    # Edge control example
    dataset_edge = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",  # Override this in your experiment config
        num_frames=93,  # Match state_t=24: (24-1)*4+1=93
        video_size=(704, 1280),  # 720p, 16:9 aspect ratio
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_edge",
        node=L(get_generic_dataloader)(
            dataset=dataset_edge,
            sampler=L(get_sampler)(dataset=dataset_edge) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Depth control example
    dataset_depth = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",  # Override this in your experiment config
        num_frames=93,  # Match state_t=24: (24-1)*4+1=93
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_depth",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_depth",
        node=L(get_generic_dataloader)(
            dataset=dataset_depth,
            sampler=L(get_sampler)(dataset=dataset_depth) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Segmentation control example
    dataset_seg = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",  # Override this in your experiment config
        num_frames=93,  # Match state_t=24: (24-1)*4+1=93
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_seg",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_seg",
        node=L(get_generic_dataloader)(
            dataset=dataset_seg,
            sampler=L(get_sampler)(dataset=dataset_seg) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Vis (Blur) control example
    dataset_vis = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",  # Override this in your experiment config
        num_frames=93,  # Match state_t=24: (24-1)*4+1=93
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_vis",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_vis",
        node=L(get_generic_dataloader)(
            dataset=dataset_vis,
            sampler=L(get_sampler)(dataset=dataset_vis) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Waymo TOP LiDAR range-map generation with precomputed range-map layout control.
    dataset_rangemap_layout = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",
        num_frames=29,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_rangemap_layout",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_rangemap_layout",
        node=L(get_generic_dataloader)(
            dataset=dataset_rangemap_layout,
            sampler=L(get_sampler)(dataset=dataset_rangemap_layout) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Waymo TOP LiDAR projected from raw tar files and encoded online by Wan2.1 during training.
    dataset_rangemap_layout_wan21_online = L(SingleViewTransferDataset)(
        dataset_dir="/team/hyh/data/rds_hq_waymo/training",
        num_frames=29,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_rangemap_layout",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
        raw_rangemap_online=True,
        raw_lidar_root="/team/hyh/data/rds_hq_waymo",
        raw_lidar_split="training",
        lidar_utils_repo="/team/hyh/code/Cosmos-Drive-Dreams/cosmos-transfer-lidargen",
        lidar_chunk_stride_frames=10,
        native_n_rows=64,
        native_n_cols=1280,
        projection_max_range=105.0,
        downsample_factor_row=1,
        downsample_factor_col=1,
        downsample_method="scatter_min",
        input_channel_mode="repeat_depth",
        max_range=100.0,
        min_range=5.0,
        min_value=-1.0,
        layout_edge_threshold_m=1.0,
        fallback_caption="A monochrome LiDAR range-map video.",
        rangemap_target_key="rangemap_target",
        rangemap_target_repeat_row=11,
        rangemap_target_repeat_col=1,
        expected_rangemap_target_shape=(3, 29, 704, 1280),
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_rangemap_layout_wan21_online",
        node=L(get_generic_dataloader)(
            dataset=dataset_rangemap_layout_wan21_online,
            sampler=L(get_sampler)(dataset=dataset_rangemap_layout_wan21_online) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # Waymo TOP LiDAR range-map video-only domain post-training.
    dataset_rangemap_video = L(SingleViewTransferDataset)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",
        num_frames=29,
        video_size=(704, 1280),
        resolution="720",
        hint_key=None,
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_rangemap_video",
        node=L(get_generic_dataloader)(
            dataset=dataset_rangemap_video,
            sampler=L(get_sampler)(dataset=dataset_rangemap_video) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
