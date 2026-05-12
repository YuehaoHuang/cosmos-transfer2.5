# Waymo Joint Generation with T29 LatentCompressor and Cosmos-Transfer2.5 LoRA

## 0. 2026-04-18 Implementation Update

Project environment:

```bash
conda activate cosmos-transfer2.5-merge
```

The current implementation direction is **online-first paired loading**, not an offline latent-cache-first pipeline:

- `WaymoMultiviewDataset(include_lidar_alignment_metadata=True)` is the single clock for both modalities.
- Video frames are read online from the 5-camera Waymo mp4 chunks.
- LiDAR frames are read online from the matching `<segment_key>.tar` using `waymo_lidar_frame_indices`, then encoded by frozen OpenSora-S3.
- Cache writing is allowed only as an acceleration path keyed by full `sample_key=<segment_key>_<chunk_idx>`.
- The first runnable baseline entry is `scripts/train_waymo_video_to_lidar_baseline.py`; it trains only a small LiDAR denoiser and does not import or alter the existing video generation path.

Smoke command:

```bash
python scripts/train_waymo_video_to_lidar_baseline.py \
  --video-tokenizer raw_downsample \
  --limit-samples 1 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1 \
  --dry-run \
  --hidden-channels 8 \
  --device cuda
```

This keeps the LiDAR latent at the official LTCV contract `B x 16 x 8 x 64 x 226`. The mainline path keeps the full downsampled Waymo range map width `1800`, lets the tokenizer pad it to `1808`, stores `compressed_latent_from_encoder`, and saves `exact_context_latent` alongside it for decode. Non-`64x226` resizing is guarded behind `--allow-lidar-resize-for-smoke` and should not be used for baseline runs.

2026-04-25 Wan2.1 LiDAR latent update:
- 2026-04-26 training mainline is moving from online Wan2.1 LiDAR VAE encode to completed paired cache, because online encode made the trainer slow and under-utilized memory
- online entry remains `train_waymo_video_lidar_one_way_wan21_online.sh` for debug/repro
- cache training entry is `train_waymo_video_lidar_one_way_wan21_cache.sh`; FSDP2 opt-in entry is `train_waymo_video_lidar_one_way_wan21_cache_fsdp.sh`
- cache training keeps real video latent from `/data/waymo/chunk/training/samples` and precomputes LiDAR Wan2.1 latent from `/data2/rds_hq_waymo/<split>/lidar_raw`
- target stable config after cache completion: `LIDAR_NUM_BLOCKS=14`, `VIDEO_KV_EVERY_N_LAYERS=4`, `CHECKPOINT_LIDAR_BLOCKS=true`, `BATCH_SIZE=2`
- no-checkpoint DDP boundary: `CHECKPOINT_LIDAR_BLOCKS=false`, `BATCH_SIZE=1` reaches `step=1` after frozen-video temp cleanup but OOMs on the next frozen video MLP peak at ~`78.1GB/GPU`; it is not the stable mainline
- FSDP2 path shards only trainable LiDAR expert while keeping frozen video replicated, but currently needs a LiDAR block forward-boundary refactor because direct internal block calls trigger Tensor/DTensor mixing
- the Wan-native cache path is generated via `scripts/cache_waymo_lidar_wan21_latents.py`
- cache contract: `wan21_native64x1312_repeatrow11_v1`
- LiDAR latent shape: `16 x 8 x 88 x 164`
- video latent remains the real cached Wan video latent from `/data/waymo/chunk/training/samples`, shape `16 x 40 x 90 x 160`
- the paired cache root is `/data2/waymo_paired_latents/training/real_video_wan21_lidar_native64x1312_repeatrow11`
- the cache-training entry remains `train_waymo_video_lidar_one_way_wan21_cache.sh`
- current run: tmux `wan21_cp_b2_train_20260427_002810`, output `/data2/waymo_video_lidar_one_way_expert/real_video_wan21_lidar_lidar14_vkv4_cp_b2_7gpu_after_cache42_20260427_002810`; `step=10`, loss `3.3592 -> 1.5709`, `42.15s/step`, `40.8-41.9GB/GPU`

2026-04-21 validation note:
- the switched online extractor now matches the official `lidar_cli.py` path on the canonical Waymo sample
- decoded reconstruction metrics are `RMSE 5.8006 / MAE 1.5767 / Rel 0.0501`
- old checkpoints trained on cropped `224/225`-style LiDAR targets are not resume-compatible with this corrected `226 + exact_context` contract

2026-04-21 paired-cache visualization note:
- canonical sample: `10203656353524179475_7625_000_7645_000_0`
- paired-cache smoke payload: `/data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/10203656353524179475_7625_000_7645_000_0.pt`
- raw five-view preview: `/data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/10203656353524179475_7625_000_7645_000_0_five_view_raw.mp4`
- raw five-view grid preview: `/data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/10203656353524179475_7625_000_7645_000_0_five_view_raw_grid.mp4`
- raw-vs-video-latent preview: `/data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/10203656353524179475_7625_000_7645_000_0_video_compare.mp4`
- LiDAR point cloud visualization uses `prediction_key=paired_cache_decode`, `gt_source=raw`, `camera_view=front_view`, `display_frame=vehicle`, and `pcd_renderer=plotly`
- multi-worker Plotly/Kaleido is the default practical path for doc-matched point-cloud videos; use `--pcd-workers 12` on this 256-core host

For the real frozen-video-VAE path, switch `--video-tokenizer wan2pt1` once the Wan2.1 VAE checkpoint or S3 credential is available locally.

Doc-matched point-cloud visualization command:

```bash
conda activate cosmos-transfer2.5-merge

python scripts/visualize_decoded_lidar_waymo.py \
  --decoded-path /data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/10203656353524179475_7625_000_7645_000_0_lidar_decoded.pt \
  --output-dir /data2/waymo_paired_cache_smoke/validation/raw_downsample/10203656353524179475_7625_000_7645_000_0/lidar_vis_merge_plotly_full \
  --sample-key 10203656353524179475_7625_000_7645_000_0 \
  --prediction-key paired_cache_decode \
  --gt-key gt_tokenizer_recon \
  --gt-source auto \
  --raw-lidar-root /data2/rds_hq_waymo/lidar_tokenizer \
  --split validation \
  --segment-key 10203656353524179475_7625_000_7645_000 \
  --lidar-frame-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --vis-pcd \
  --camera-view front_view \
  --display-frame vehicle \
  --pcd-renderer plotly \
  --pcd-workers 12 \
  --lidar-tokenizer-repo /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
```

## 1. Goal

Use the current Waymo LiDAR tokenizer from `Cosmos-Drive-Dreams` as the video VAE/tokenizer for a joint generation pipeline, and adapt a Cosmos-Transfer2.5 video generation model with a DreamZero-style LoRA recipe instead of full-model retraining.

The intended setup is:
- freeze the current Waymo tokenizer
- keep the generation model backbone mostly frozen
- adapt the generator to the new latent space with LoRA / AdaLN-LoRA
- train and infer on a fixed `29 -> 8 -> 29` temporal contract

## 2. Current Tokenizer Contract

The current LiDAR tokenizer mainline is the `T29 LatentCompressor` design.

Its external contract is:
- pixel input frames: `29 = 1 + 28`
- latent frames: `8 = 1 + 7`
- pixel reconstruction frames: `29`
- temporal compression: `4x`
- spatial compression: `8x`
- latent channels: `16`

Important semantic detail:
- the first frame is preserved as an exact context frame and is not temporally compressed
- only the future `28` frames are temporally compressed into `7` latent steps

This matches the external `1 + T -> 1 + T / 4` causal contract needed by Wan-style video tokenization.

## 3. Compatibility with Cosmos-Transfer2.5

### 3.1 What already matches

At the I/O level, the tokenizer is compatible with the Wan-style frame/latent conversion rule used by Cosmos-Transfer2.5:
- `get_latent_num_frames(29) = 8`
- `get_pixel_num_frames(8) = 29`

This is the most important compatibility requirement for using the tokenizer in generation.

### 3.2 What does not match automatically

This tokenizer is **not** the original Wan2.1 VAE.

Differences:
- it is a frozen 2D tokenizer plus a latent temporal compressor, not an end-to-end Wan VAE
- the first frame is exact-bypass preserved, rather than going through the same temporal compression path as the future frames
- latent statistics are different from the original Wan tokenizer, even though shape semantics match

Implication:
- we should not expect a pretrained Cosmos-Transfer2.5 generator to work well zero-shot with this tokenizer
- but shape-level compatibility is sufficient to make LoRA-based adaptation feasible

## 4. Recommended Training Strategy

### 4.1 Recommendation

Use a DreamZero-style adaptation recipe:
- freeze the tokenizer
- freeze the base video generator weights as much as possible
- adapt with LoRA / AdaLN-LoRA
- retrain on data re-encoded by the new tokenizer

This is the lowest-risk path because it does not try to preserve exact latent statistics from Wan2.1. Instead, it teaches the generator to live in the new latent space.

### 4.2 Model state length

The generation model should be configured with:
- `state_t = 8`

Reason:
- this tokenizer produces exactly `8` latent frames for `29` pixel frames
- Cosmos-Transfer2.5 uses `tokenizer.get_pixel_num_frames(state_t)` to determine the required pixel clip length
- with `state_t = 8`, the corresponding pixel window becomes `29`

This is the cleanest alignment point for joint generation.

## 5. Minimum Integration Work

### 5.1 Add a tokenizer interface in Cosmos-Transfer2.5

Add a new tokenizer wrapper in `cosmos-transfer2.5` that exposes the same interface expected by the generation code:
- `encode(state) -> latent`
- `decode(latent) -> state`
- `latent_ch = 16`
- `spatial_compression_factor = 8`
- `temporal_compression_factor = 4`
- `get_latent_num_frames(num_pixel_frames)`
- `get_pixel_num_frames(num_latent_frames)`

The wrapper should internally call the trained `T29 LatentCompressor` full-model checkpoint.

### 5.2 Restrict v1 to fixed 29-frame windows

For the first joint-generation version, keep the tokenizer side fixed to:
- input video length `29`
- latent length `8`

Do not introduce variable-length or autoregressive tokenizer behavior in v1.

### 5.3 Re-encode training data

All generation training data should be re-encoded using the new tokenizer.

Do not mix:
- old Wan latents
- new Waymo latent-compressor latents

The generator should see a single consistent latent distribution during LoRA fine-tuning.

## 6. LoRA Fine-Tuning Recipe

### 6.1 Base approach

Start from an existing Cosmos-Transfer2.5 video generator and fine-tune with LoRA rather than full-model training.

Suggested initial target:
- attention projections: `q_proj,k_proj,v_proj,output_proj`
- MLP projections: `mlp.layer1,mlp.layer2`
- AdaLN-LoRA enabled if available in the chosen config

### 6.2 Initial hyperparameter suggestion

A practical v1 starting point:
- `state_t = 8`
- `in_channels = 16`
- `out_channels = 16`
- `lora_rank = 32`
- `lora_alpha = 32`
- `use_adaln_lora = True`
- `adaln_lora_dim = 256`
- precision `bfloat16`

If adaptation is weak, the next knobs to relax are:
- increase LoRA rank
- unfreeze input/output projection layers around the denoiser
- train a tiny latent adapter before the DiT

## 7. Data and Objective

### 7.1 Training data format

For each training example, construct:
- input/control video in pixel space
- tokenizer-encoded latent video of shape `[B, 16, 8, H/8, W/8]`
- target future latent trajectory or full target latent clip, depending on the chosen generation objective

### 7.2 Conditioning semantics

The tokenizer uses a causal `1 + 28` contract.

For generation, the simplest v1 interpretation is:
- latent frame `0` is the exact context latent
- latent frames `1..7` represent compressed future context

The generator should preserve this convention consistently in both training and inference.

### 7.3 Loss / objective on the generator side

The generator objective remains a standard diffusion / flow matching objective in latent space.

No tokenizer-side reconstruction loss is needed during generator training, because the tokenizer is frozen.

## 8. Why This Is a Reasonable DreamZero-Style Path

This plan is attractive for the same reasons DreamZero-like approaches are attractive:
- the expensive generative backbone is mostly preserved
- the tokenizer is frozen, so latent supervision is stable
- the adaptation cost is much lower than retraining a large generator from scratch
- if quality is insufficient, we can widen the trainable surface gradually instead of doing a full restart

## 9. Risks and Caveats

### 9.1 Not a drop-in replacement for pretrained Wan latent models

Even though `29 -> 8 -> 29` matches at the shape level, latent statistics are different.

So:
- zero-shot swapping is not expected to work well
- LoRA is doing real adaptation, not just cosmetic tuning

### 9.2 First-frame behavior is custom

The exact first-frame bypass is deliberate and useful, but it is not identical to the original Wan internal behavior.

This is acceptable for our own joint-generation training as long as:
- training data uses the same tokenizer
- inference uses the same tokenizer
- `state_t = 8` semantics stay fixed

### 9.3 v1 should stay fixed-window

Do not combine all of these at once in v1:
- new tokenizer
- variable-length tokenizer inference
- autoregressive latent rollout
- LoRA adaptation

Start with fixed `29` pixel frames and `8` latent frames.

## 10. Recommended Milestones

### Milestone 1

Tokenizer integration only:
- add the tokenizer wrapper to Cosmos-Transfer2.5
- verify `29` pixel frames encode to `8` latent frames
- verify decode restores `29` frames

### Milestone 2

Generator dry run:
- set `state_t = 8`
- run a tiny training/inference smoke test with the new tokenizer
- verify tensor shapes end-to-end

### Milestone 3

LoRA adaptation:
- run LoRA fine-tuning on a small subset
- compare qualitative samples against baseline controls

### Milestone 4

Full joint generation:
- scale LoRA fine-tuning to the full Waymo setup
- evaluate control fidelity, temporal consistency, and motion quality

## 11. Practical Conclusion

Yes, this is a valid path.

The correct interpretation is:
- use the current tokenizer as the new latent interface
- set the generation model to `state_t = 8`
- re-encode data with the new tokenizer
- fine-tune Cosmos-Transfer2.5 with LoRA / AdaLN-LoRA to adapt to the new latent space

This is a much safer plan than trying to force exact Wan2.1 internal equivalence before starting generation work.
