#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common import make_absolute, save_json
from extract_top_ri import extract_segment
from render_outputs import render_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process Waymo TFRecords into metadata/.npz and lidar/.npz files."
    )
    parser.add_argument("--input_root", type=Path, default=Path("/data/waymo/raw"), help="Waymo TFRecord root.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data2/waymo_lidar_tokenizer"),
        help="Dataset root containing metadata/, lidar/, and previews/.",
    )
    parser.add_argument("--splits", nargs="+", default=["training"], help="Split names under input_root.")
    parser.add_argument("--max_segments", type=int, default=None, help="Optional cap on segments per split.")
    parser.add_argument(
        "--max_frames_per_segment",
        type=int,
        default=None,
        help="Optional cap on frames per segment. Unset means all frames.",
    )
    parser.add_argument("--render", type=int, default=1, choices=[0, 1], help="Render previews after extraction.")
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1], help="Skip existing segment files.")
    parser.add_argument("--point_size", type=float, default=0.3, help="Point size for previews.")
    parser.add_argument("--point_range", type=float, default=80.0, help="Axis range for previews.")
    return parser.parse_args()


def find_tfrecords(split_root: Path, max_segments: int | None) -> list[Path]:
    tfrecords = sorted(split_root.glob("*.tfrecord"))
    if max_segments is not None:
        return tfrecords[:max_segments]
    return tfrecords


def segment_exists(output_root: Path, segment_id: str) -> bool:
    metadata_path = output_root / "metadata" / f"{segment_id}.npz"
    lidar_path = output_root / "lidar" / f"{segment_id}.npz"
    return metadata_path.exists() and lidar_path.exists()


def process_split(
    split: str,
    split_root: Path,
    output_root: Path,
    max_segments: int | None,
    max_frames_per_segment: int | None,
    render: bool,
    skip_existing: bool,
    point_size: float,
    point_range: float,
) -> dict:
    tfrecords = find_tfrecords(split_root, max_segments=max_segments)
    if not tfrecords:
        raise RuntimeError(f"No TFRecord files found under {split_root}")

    processed_segments = []
    skipped_segments = []

    for tfrecord_path in tfrecords:
        segment_id = tfrecord_path.name.replace(".tfrecord", "")
        if skip_existing and segment_exists(output_root, segment_id):
            skipped_segments.append(segment_id)
            print(f"Skipping existing {segment_id}")
            continue

        summary = extract_segment(
            tfrecord_path=tfrecord_path,
            output_root=output_root,
            max_frames=max_frames_per_segment,
            split=split,
            laser_name="TOP",
            ri_index=0,
        )
        processed_segments.append(summary)

        if render:
            render_segment(
                metadata_path=output_root / "metadata" / f"{segment_id}.npz",
                lidar_path=output_root / "lidar" / f"{segment_id}.npz",
                preview_root=output_root / "previews",
                point_size=point_size,
                point_range=point_range,
            )

    split_summary = {
        "split": split,
        "input_root": str(split_root),
        "output_root": str(output_root),
        "requested_max_segments": max_segments,
        "requested_max_frames_per_segment": max_frames_per_segment,
        "render": render,
        "skip_existing": skip_existing,
        "num_found_segments": len(tfrecords),
        "num_processed_segments": len(processed_segments),
        "num_skipped_segments": len(skipped_segments),
        "processed_segments": processed_segments,
        "skipped_segments": skipped_segments,
    }
    save_json(output_root / f"_split_summary_{split}.json", split_summary)
    return split_summary


def main() -> None:
    args = parse_args()
    input_root = make_absolute(args.input_root)
    output_root = make_absolute(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "lidar").mkdir(parents=True, exist_ok=True)
    (output_root / "previews").mkdir(parents=True, exist_ok=True)

    split_summaries = []
    for split in args.splits:
        split_summaries.append(
            process_split(
                split=split,
                split_root=input_root / split,
                output_root=output_root,
                max_segments=args.max_segments,
                max_frames_per_segment=args.max_frames_per_segment,
                render=bool(args.render),
                skip_existing=bool(args.skip_existing),
                point_size=args.point_size,
                point_range=args.point_range,
            )
        )

    dataset_summary = {
        "wod_version": "1.4.2",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "max_segments": args.max_segments,
        "max_frames_per_segment": args.max_frames_per_segment,
        "render": bool(args.render),
        "skip_existing": bool(args.skip_existing),
        "split_summaries": split_summaries,
    }
    save_json(output_root / "_dataset_summary.json", dataset_summary)
    print(json.dumps(dataset_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
