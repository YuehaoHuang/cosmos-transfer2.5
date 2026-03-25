# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib

import torch
from cosmos_transfer2._src.imaginaire.config import Config, pretty_print_overrides
from cosmos_transfer2._src.imaginaire.flags import INTERNAL, SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import misc
from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_transfer2._src.imaginaire.utils.context_managers import distributed_init
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.utils.distributed import get_rank
from cosmos_transfer2._src.imaginaire.utils.launch import log_reproducible_setup
from cosmos_transfer2._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import preprocess_databatch
from cosmos_transfer2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp
from loguru import logger as logging

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir, is_rank0

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and save GT latents from training dataloader")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "--split-roots",
        default="/data/waymo/chunk/training,/data/waymo/chunk/validation",
        help="Comma-separated Waymo split roots used to resolve output path per sample key",
    )
    parser.add_argument(
        "--samples-subdir",
        default="samples",
        help="Output subdirectory under split root, e.g. samples",
    )
    parser.add_argument(
        "--skip-train-samples",
        type=int,
        default=0,
        help="Skip the first N train samples before extraction",
    )
    parser.add_argument(
        "--skip-val-samples",
        type=int,
        default=0,
        help="Skip the first N val samples before extraction",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after N batches per loader; 0 means process full loader",
    )
    parser.add_argument(
        "--loader",
        default="both",
        choices=("train", "val", "both"),
        help="Which dataloader(s) to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Log progress every N processed batches",
    )
    parser.add_argument(
        "--log-unresolved-examples",
        type=int,
        default=5,
        help="Maximum unresolved sample keys to show per loader in final summary",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print resolved config and exit.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help='LazyConfig overrides, e.g. "job.wandb_mode=offline"',
    )
    return parser.parse_args()


class SplitRootResolver:
    def __init__(self, split_roots: list[Path], samples_subdir: str):
        self.split_roots = split_roots
        self.samples_subdir = samples_subdir
        self._cache: dict[str, Path] = {}

    def resolve(self, sample_key: str) -> Path | None:
        cached = self._cache.get(sample_key)
        if cached is not None:
            return cached

        for root in self.split_roots:
            probe = root / "videos" / "pinhole_front" / f"{sample_key}.mp4"
            if probe.exists():
                self._cache[sample_key] = root
                return root
        return None


def _list_stem_set(directory: Path, suffix: str) -> set[str]:
    if not directory.exists():
        return set()
    return {p.stem for p in directory.glob(f"*{suffix}") if p.is_file()}


def _build_split_key_index(split_root: Path, samples_subdir: str) -> tuple[set[str], set[str]]:
    """Build key index for one split root.

    Returns:
        known_video_keys: keys discovered from videos/pinhole_front/*.mp4
        missing_keys: known_video_keys that do not have samples/<key>.pt yet
    """
    video_dir = split_root / "videos" / "pinhole_front"
    samples_dir = split_root / samples_subdir
    known_video_keys = _list_stem_set(video_dir, ".mp4")
    existing_sample_keys = _list_stem_set(samples_dir, ".pt")
    missing_keys = known_video_keys - existing_sample_keys
    return known_video_keys, missing_keys


def _slice_batched_prefix(value, start_idx: int, batch_size: int):
    """Slice batched structures along the sample dimension while leaving non-batched fields unchanged."""
    if start_idx <= 0:
        return value

    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[start_idx:]
        return value

    if isinstance(value, list):
        if len(value) == batch_size:
            return value[start_idx:]
        return [_slice_batched_prefix(v, start_idx, batch_size) for v in value]

    if isinstance(value, tuple):
        if len(value) == batch_size:
            return value[start_idx:]
        return tuple(_slice_batched_prefix(v, start_idx, batch_size) for v in value)

    if isinstance(value, dict):
        return {k: _slice_batched_prefix(v, start_idx, batch_size) for k, v in value.items()}

    return value


def _apply_loader_sample_offset(loader, skip_samples: int):
    """Return a new loader starting at dataset index `skip_samples` when possible.

    Falls back to the original loader if it cannot be safely rebuilt.

    Returns:
        (new_loader, applied_skip, used_loader_offset)
    """
    if loader is None or skip_samples <= 0:
        return loader, 0, False

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader, 0, False

    try:
        dataset_len = len(dataset)
    except Exception:
        return loader, 0, False

    applied = min(int(skip_samples), int(dataset_len))
    if applied <= 0:
        return loader, 0, False

    if applied >= dataset_len:
        return [], applied, True

    subset = torch.utils.data.Subset(dataset, range(applied, dataset_len))

    sampler = getattr(loader, "sampler", None)
    new_sampler = None
    if sampler is not None:
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            new_sampler = torch.utils.data.distributed.DistributedSampler(
                subset,
                num_replicas=sampler.num_replicas,
                rank=sampler.rank,
                shuffle=sampler.shuffle,
                seed=sampler.seed,
                drop_last=sampler.drop_last,
            )
        else:
            # Keep behavior safe: unknown sampler types are left to runtime skip fallback.
            return loader, 0, False

    prefetch_factor = getattr(loader, "prefetch_factor", None)
    loader_kwargs = {
        "dataset": subset,
        "batch_size": loader.batch_size,
        "shuffle": False,
        "sampler": new_sampler,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "persistent_workers": loader.persistent_workers,
        "collate_fn": loader.collate_fn,
    }
    if loader.num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(**loader_kwargs), applied, True


def init_parallel_state(config: Config) -> None:
    """Initialize distributed + model parallel groups required by the dataloader sampler."""
    with distributed_init():
        distributed.init()
        if hasattr(config.model, "context_parallel_size"):
            if config.model_parallel.context_parallel_size > 1:
                raise ValueError(
                    "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                    "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                )
            config.model_parallel.context_parallel_size = config.model.context_parallel_size

        if USE_MEGATRON and not parallel_state.is_initialized():
            init_sig = inspect.signature(parallel_state.initialize_model_parallel).parameters
            kwargs = {
                "pipeline_model_parallel_size": config.model_parallel.pipeline_model_parallel_size,
                "tensor_model_parallel_size": config.model_parallel.tensor_model_parallel_size,
                "context_parallel_size": config.model_parallel.context_parallel_size,
            }
            if "create_gloo_process_groups" in init_sig:
                kwargs["create_gloo_process_groups"] = False
            parallel_state.initialize_model_parallel(**kwargs)
            parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
            if parallel_state.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


def _save_one_latent(latent: torch.Tensor, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp_rank{get_rank()}")
    torch.save(latent.detach().cpu(), tmp_path)
    os.replace(tmp_path, out_path)
    return True


def _build_batch_save_plan(
    data_batch: dict,
    resolver: SplitRootResolver,
    overwrite: bool,
    default_split_root: Path | None = None,
    known_video_keys: set[str] | None = None,
    pending_missing_keys: set[str] | None = None,
) -> tuple[list[Path | None] | None, int, int, list[str], int]:
    """Build per-sample output plan and precheck which samples still need saving.

    Returns:
        out_paths: list aligned with sample_keys. Each item is a target path or None if unresolved/invalid.
          None is returned if keys are missing or malformed.
        skipped_existing: Number of samples already on disk and skipped by precheck.
        unresolved: Number of samples with unresolved/invalid keys.
        unresolved_keys: Example unresolved keys.
        sample_count: Number of keys found in this batch.
    """
    if "__key__" not in data_batch:
        return None, 0, 0, [], 0

    sample_keys = data_batch["__key__"]
    if not isinstance(sample_keys, list):
        return None, 0, 0, [], 0

    out_paths: list[Path | None] = []
    skipped_existing = 0
    unresolved = 0
    unresolved_keys: list[str] = []

    for sample_key in sample_keys:
        if not isinstance(sample_key, str) or not sample_key:
            unresolved += 1
            out_paths.append(None)
            continue

        # Fast path with pre-indexed keys for this split.
        if default_split_root is not None and known_video_keys is not None and pending_missing_keys is not None:
            if sample_key not in known_video_keys:
                unresolved += 1
                unresolved_keys.append(sample_key)
                out_paths.append(None)
                continue

            out_path = default_split_root / resolver.samples_subdir / f"{sample_key}.pt"
            out_paths.append(out_path)

            if not overwrite and sample_key not in pending_missing_keys:
                skipped_existing += 1
            continue

        split_root = default_split_root
        if split_root is None:
            split_root = resolver.resolve(sample_key)
            if split_root is None:
                unresolved += 1
                unresolved_keys.append(sample_key)
                out_paths.append(None)
                continue

        out_path = split_root / resolver.samples_subdir / f"{sample_key}.pt"
        out_paths.append(out_path)

        if not overwrite and out_path.exists():
            skipped_existing += 1

    return out_paths, skipped_existing, unresolved, unresolved_keys, len(sample_keys)


def _precheck_batch_outputs(
    data_batch: dict,
    resolver: SplitRootResolver,
    overwrite: bool,
    default_split_root: Path | None = None,
    known_video_keys: set[str] | None = None,
    pending_missing_keys: set[str] | None = None,
) -> tuple[bool, list[Path | None] | None, int, int, list[str], int]:
    """Return whether latent extraction is needed for this batch before running model.encode().

    Returns:
        need_extract: True if at least one sample needs saving.
        out_paths: Per-sample target path plan aligned with sample keys.
        skipped_existing: Number of samples already on disk and skipped.
        unresolved: Number of samples with unresolved/invalid keys.
        unresolved_keys: Example unresolved keys.
        sample_count: Number of keys found in this batch.
    """
    out_paths, skipped_existing, unresolved, unresolved_keys, sample_count = _build_batch_save_plan(
        data_batch=data_batch,
        resolver=resolver,
        overwrite=overwrite,
        default_split_root=default_split_root,
        known_video_keys=known_video_keys,
        pending_missing_keys=pending_missing_keys,
    )

    # Missing/malformed keys: keep old behavior and run extraction path.
    if out_paths is None:
        return True, None, skipped_existing, unresolved, unresolved_keys, sample_count

    if overwrite:
        need_extract = any(p is not None for p in out_paths)
    else:
        if pending_missing_keys is not None and known_video_keys is not None:
            sample_keys = data_batch.get("__key__", [])
            need_extract = isinstance(sample_keys, list) and any(
                isinstance(k, str) and (k in pending_missing_keys) for k in sample_keys
            )
        else:
            need_extract = any(p is not None and not p.exists() for p in out_paths)

    return need_extract, out_paths, skipped_existing, unresolved, unresolved_keys, sample_count


@torch.no_grad()
def save_latents_from_batch(
    data_batch: dict,
    latent_state: torch.Tensor,
    resolver: SplitRootResolver,
    overwrite: bool,
    out_paths: list[Path | None] | None = None,
    default_split_root: Path | None = None,
) -> tuple[int, int, int, list[str]]:
    if "__key__" not in data_batch:
        return 0, 0, 0, []

    sample_keys = data_batch["__key__"]
    if not isinstance(sample_keys, list):
        return 0, 0, 0, []
    if latent_state.shape[0] != len(sample_keys):
        raise ValueError(f"Batch mismatch: latent batch={latent_state.shape[0]} vs keys={len(sample_keys)}")

    if out_paths is None:
        out_paths, _, _, _, _ = _build_batch_save_plan(
            data_batch=data_batch,
            resolver=resolver,
            overwrite=overwrite,
            default_split_root=default_split_root,
        )

    if out_paths is None or len(out_paths) != len(sample_keys):
        raise ValueError("Invalid out_paths plan: missing or length mismatch with sample keys")

    saved = 0
    skipped = 0
    unresolved = 0
    unresolved_keys: list[str] = []
    for idx, sample_key in enumerate(sample_keys):
        out_path = out_paths[idx]
        if out_path is None:
            unresolved += 1
            if isinstance(sample_key, str) and sample_key:
                unresolved_keys.append(sample_key)
            continue

        if _save_one_latent(latent_state[idx], out_path, overwrite=overwrite):
            saved += 1
        else:
            skipped += 1

    return saved, skipped, unresolved, unresolved_keys


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    config.validate()
    config.freeze()  # type: ignore
    log_reproducible_setup(config, args)
    init_parallel_state(config)

    load_path = config.checkpoint.load_path
    if not INTERNAL:
        from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint

        if load_path:
            load_path = download_checkpoint(load_path)
    if load_path:
        logging.info(f"Checkpoint load_path resolved to: {load_path}")

    if isinstance(load_path, str) and load_path.endswith(".pt"):
        logging.info(f"Loading model weights from consolidated checkpoint: {load_path}")
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        if load_path:
            logging.info("Model will be instantiated; checkpoint loading will be done by the checkpointer.")
        model = instantiate(config.model)

    dataloader_train = instantiate(config.dataloader_train) if args.loader in ("train", "both") else None
    dataloader_val = instantiate(config.dataloader_val) if args.loader in ("val", "both") else None

    requested_skip_by_loader: dict[str, int] = {
        "train": max(0, int(args.skip_train_samples)),
        "val": max(0, int(args.skip_val_samples)),
    }
    loader_prefixed_skip_samples: dict[str, int] = {"train": 0, "val": 0}
    loader_used_offset: dict[str, bool] = {"train": False, "val": False}
    if dataloader_train is not None and requested_skip_by_loader["train"] > 0:
        dataloader_train, applied, used_offset = _apply_loader_sample_offset(
            dataloader_train, requested_skip_by_loader["train"]
        )
        loader_prefixed_skip_samples["train"] = applied
        loader_used_offset["train"] = used_offset
        if used_offset:
            logging.info(f"[train] applied loader offset: skip_samples={applied}")
    if dataloader_val is not None and requested_skip_by_loader["val"] > 0:
        dataloader_val, applied, used_offset = _apply_loader_sample_offset(
            dataloader_val, requested_skip_by_loader["val"]
        )
        loader_prefixed_skip_samples["val"] = applied
        loader_used_offset["val"] = used_offset
        if used_offset:
            logging.info(f"[val] applied loader offset: skip_samples={applied}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    model.eval()

    split_roots = [Path(p.strip()) for p in args.split_roots.split(",") if p.strip()]
    resolver = SplitRootResolver(split_roots=split_roots, samples_subdir=args.samples_subdir)
    loader_default_roots: dict[str, Path | None] = {
        "train": split_roots[0] if len(split_roots) >= 1 else None,
        "val": split_roots[1] if len(split_roots) >= 2 else (split_roots[0] if len(split_roots) == 1 else None),
    }
    split_known_keys: dict[str, set[str] | None] = {"train": None, "val": None}
    split_pending_missing_keys: dict[str, set[str] | None] = {"train": None, "val": None}
    if not args.overwrite:
        for loader_name in ("train", "val"):
            split_root = loader_default_roots.get(loader_name)
            if split_root is None:
                continue
            known_keys, missing_keys = _build_split_key_index(split_root, args.samples_subdir)
            split_known_keys[loader_name] = known_keys
            split_pending_missing_keys[loader_name] = missing_keys
            logging.info(
                f"[{loader_name}] indexed keys: known={len(known_keys)}, missing={len(missing_keys)} under {split_root}"
            )

    def run_loader(loader_name: str, loader, skip_samples: int) -> None:
        n_batches = 0
        n_samples = 0
        total_saved = 0
        total_skipped = 0
        total_unresolved = 0
        unresolved_examples: list[str] = []
        skip_remaining = 0 if loader_used_offset.get(loader_name, False) else max(0, int(skip_samples))
        prefix_skipped_samples = int(loader_prefixed_skip_samples.get(loader_name, 0))
        prefix_skipped_batches = 0
        default_split_root = loader_default_roots.get(loader_name)
        known_video_keys = split_known_keys.get(loader_name)
        pending_missing_keys = split_pending_missing_keys.get(loader_name)

        if pending_missing_keys is not None and len(pending_missing_keys) == 0 and not args.overwrite:
            done_msg = (
                f"[{loader_name}] rank={get_rank()} done: batches=0, samples=0, saved=0, skipped=0, unresolved=0 "
                f"(all keys already extracted)"
            )
            logging.info(done_msg)
            if is_rank0():
                print(done_msg, flush=True)
            return

        for data_batch in loader:
            if skip_remaining > 0:
                sample_keys = data_batch.get("__key__", None)
                if isinstance(sample_keys, list) and sample_keys:
                    batch_size = len(sample_keys)
                    if skip_remaining >= batch_size:
                        prefix_skipped_batches += 1
                        prefix_skipped_samples += batch_size
                        skip_remaining -= batch_size
                        continue

                    # Partial-batch boundary: trim the already-skipped prefix from all batched fields.
                    data_batch = _slice_batched_prefix(data_batch, skip_remaining, batch_size)
                    prefix_skipped_samples += skip_remaining
                    prefix_skipped_batches += 1
                    skip_remaining = 0
                else:
                    logging.warning(
                        f"[{loader_name}] unable to apply skip setting for a batch without valid __key__; "
                        "continuing extraction from this batch"
                    )

            need_extract, out_paths, pre_skipped, pre_unresolved, pre_unresolved_keys, sample_count = (
                _precheck_batch_outputs(
                    data_batch=data_batch,
                    resolver=resolver,
                    overwrite=args.overwrite,
                    default_split_root=default_split_root,
                    known_video_keys=known_video_keys,
                    pending_missing_keys=pending_missing_keys,
                )
            )

            # Fast path: all outputs already exist and overwrite is disabled.
            if not need_extract:
                n_batches += 1
                n_samples += sample_count
                total_skipped += pre_skipped
                total_unresolved += pre_unresolved
                if len(unresolved_examples) < args.log_unresolved_examples:
                    remaining = args.log_unresolved_examples - len(unresolved_examples)
                    unresolved_examples.extend(pre_unresolved_keys[:remaining])

                if args.save_every > 0 and n_batches % args.save_every == 0:
                    progress_msg = (
                        f"[{loader_name}] rank={get_rank()} batches={n_batches}, "
                        f"samples={n_samples}, saved={total_saved}, skipped={total_skipped}, unresolved={total_unresolved}"
                    )
                    logging.info(progress_msg)
                    if is_rank0():
                        print(progress_msg, flush=True)

                if args.max_batches > 0 and n_batches >= args.max_batches:
                    break
                continue

            data_batch = misc.to(data_batch, device=device)
            # Match training order: move to CUDA first, then multiview view-sampling preprocess.
            data_batch = preprocess_databatch(data_batch, model.config.train_sample_views_range)

            if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
                model.inplace_compute_text_embeddings_online(data_batch)
            with torch.no_grad():
                _, latent_state, _ = model.get_data_and_condition(data_batch)
            saved, skipped, unresolved, unresolved_keys = save_latents_from_batch(
                data_batch=data_batch,
                latent_state=latent_state,
                resolver=resolver,
                overwrite=args.overwrite,
                out_paths=out_paths,
                default_split_root=default_split_root,
            )
            if pending_missing_keys is not None:
                sample_keys = data_batch.get("__key__", [])
                if isinstance(sample_keys, list):
                    for sample_key in sample_keys:
                        if isinstance(sample_key, str):
                            pending_missing_keys.discard(sample_key)

            batch_size = int(latent_state.shape[0])
            n_samples += batch_size
            total_saved += saved
            total_skipped += skipped
            total_unresolved += unresolved
            if len(unresolved_examples) < args.log_unresolved_examples:
                remaining = args.log_unresolved_examples - len(unresolved_examples)
                unresolved_examples.extend(unresolved_keys[:remaining])
            n_batches += 1

            if args.save_every > 0 and n_batches % args.save_every == 0:
                progress_msg = (
                    f"[{loader_name}] rank={get_rank()} batches={n_batches}, "
                    f"samples={n_samples}, saved={total_saved}, skipped={total_skipped}, unresolved={total_unresolved}"
                )
                logging.info(progress_msg)
                if is_rank0():
                    print(progress_msg, flush=True)

            if args.max_batches > 0 and n_batches >= args.max_batches:
                break

            if pending_missing_keys is not None and len(pending_missing_keys) == 0 and not args.overwrite:
                logging.info(f"[{loader_name}] all missing keys have been extracted; stopping early.")
                break

        done_msg = (
            f"[{loader_name}] rank={get_rank()} done: batches={n_batches}, "
            f"samples={n_samples}, saved={total_saved}, skipped={total_skipped}, unresolved={total_unresolved}, "
            f"prefix_skipped_samples={prefix_skipped_samples}, prefix_skipped_batches={prefix_skipped_batches}"
        )
        logging.info(done_msg)
        if is_rank0():
            print(done_msg, flush=True)
        if unresolved_examples:
            unresolved_msg = (
                f"[{loader_name}] rank={get_rank()} unresolved examples (up to {args.log_unresolved_examples}): "
                f"{unresolved_examples}"
            )
            logging.warning(unresolved_msg)
            if is_rank0():
                print(unresolved_msg, flush=True)

    if dataloader_train is not None:
        run_loader("train", dataloader_train, requested_skip_by_loader["train"])
    if dataloader_val is not None:
        run_loader("val", dataloader_val, requested_skip_by_loader["val"])


def main() -> None:
    init_environment()
    args = parse_args()

    try:
        config_module = get_config_module(args.config)
        config = importlib.import_module(config_module).make_config()

        overrides = list(args.opts)
        if SMOKE:
            overrides.append("trainer.max_iter=2")
            overrides.append("trainer.logging_iter=1")
            overrides.append("trainer.validation_iter=1")
        config = override(config, overrides)

        if is_rank0():
            output_dir = Path(config.job.path_local)
            init_output_dir(output_dir, profile=False)

        if args.dryrun:
            logging.info(
                "Config:\n"
                + config.pretty_print(use_color=True)
                + "\n"
                + pretty_print_overrides(args.opts, use_color=True)
            )
            os.makedirs(config.job.path_local, exist_ok=True)
            LazyConfig.save_yaml(config, f"{config.job.path_local}/extract_latents_config.yaml")
            print(f"{config.job.path_local}/extract_latents_config.yaml")
            return

        launch(config, args)
    finally:
        cleanup_environment()


if __name__ == "__main__":
    main()
