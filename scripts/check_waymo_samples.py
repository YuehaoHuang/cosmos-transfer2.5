#!/usr/bin/env python3

import argparse
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm

@dataclass
class SampleIssue:
    sample: str
    camera: str
    stream_type: str
    issue: str
    path: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Waymo multiview dataset samples before training")
    parser.add_argument("--dataset-root", default="/data/waymo/posttrain/training", help="Waymo training root")
    parser.add_argument("--control-dir", default="world_scenario", help="Control directory name under dataset root")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "pinhole_front",
            "pinhole_front_left",
            "pinhole_front_right",
            "pinhole_side_left",
            "pinhole_side_right",
        ],
        help="Camera folder names",
    )
    parser.add_argument("--expected-min-frames", type=int, default=190, help="Minimum required frame count")
    parser.add_argument(
        "--window-frames", type=int, default=29, help="Number of frames sampled per training window"
    )
    parser.add_argument(
        "--window-checks",
        type=int,
        default=8,
        help="Number of random windows checked per file (0 means only frame-count check)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for window sampling")
    parser.add_argument("--report-json", default="/tmp/waymo_bad_samples.json", help="JSON report output")
    parser.add_argument("--report-txt", default="/tmp/waymo_bad_samples.txt", help="Text report output")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    return parser.parse_args()


def list_samples(video_root: Path, reference_camera: str) -> list[str]:
    ref_dir = video_root / reference_camera
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference camera directory does not exist: {ref_dir}")
    return sorted(path.stem for path in ref_dir.glob("*.mp4"))


def sample_window_offsets(total_frames: int, window_frames: int, window_checks: int, rng: random.Random) -> list[int]:
    max_offset = total_frames - window_frames
    if max_offset < 0:
        return [0]
    offsets = {0, max_offset, max_offset // 2}
    while len(offsets) < max(0, window_checks):
        offsets.add(rng.randint(0, max_offset))
    return sorted(offsets)


def check_stream(
    stream_path: Path,
    stream_type: str,
    camera: str,
    sample_name: str,
    expected_min_frames: int,
    window_frames: int,
    window_checks: int,
    rng: random.Random,
) -> list[SampleIssue]:
    issues: list[SampleIssue] = []
    if not stream_path.exists():
        issues.append(
            SampleIssue(
                sample=sample_name,
                camera=camera,
                stream_type=stream_type,
                issue="missing_file",
                path=str(stream_path),
                detail="file does not exist",
            )
        )
        return issues

    try:
        from decord import VideoReader

        video_reader = VideoReader(str(stream_path))
        total_frames = len(video_reader)
    except Exception as exc:
        issues.append(
            SampleIssue(
                sample=sample_name,
                camera=camera,
                stream_type=stream_type,
                issue="open_or_decode_error",
                path=str(stream_path),
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return issues

    if total_frames < expected_min_frames:
        issues.append(
            SampleIssue(
                sample=sample_name,
                camera=camera,
                stream_type=stream_type,
                issue="too_short",
                path=str(stream_path),
                detail=f"frames={total_frames}, expected_min_frames={expected_min_frames}",
            )
        )
        return issues

    if window_checks <= 0:
        return issues

    try:
        offsets = sample_window_offsets(total_frames, window_frames, window_checks, rng)
        for offset in offsets:
            frame_indices = list(range(offset, offset + window_frames))
            video_reader.get_batch(frame_indices).asnumpy()
    except Exception as exc:
        issues.append(
            SampleIssue(
                sample=sample_name,
                camera=camera,
                stream_type=stream_type,
                issue="window_decode_error",
                path=str(stream_path),
                detail=f"offset={offset}, window_frames={window_frames}, error={type(exc).__name__}: {exc}",
            )
        )

    return issues


def process_sample(
    sample_name: str, 
    video_root: Path, 
    control_root: Path, 
    cameras: list[str], 
    expected_min_frames: int, 
    window_frames: int, 
    window_checks: int, 
    seed: int
) -> list[SampleIssue]:
    rng = random.Random(seed + hash(sample_name))
    issues: list[SampleIssue] = []
    
    for camera in cameras:
        video_path = video_root / camera / f"{sample_name}.mp4"
        control_path = control_root / camera / f"{sample_name}.mp4"

        issues.extend(
            check_stream(
                stream_path=video_path,
                stream_type="video",
                camera=camera,
                sample_name=sample_name,
                expected_min_frames=expected_min_frames,
                window_frames=window_frames,
                window_checks=window_checks,
                rng=rng,
            )
        )
        issues.extend(
            check_stream(
                stream_path=control_path,
                stream_type="control",
                camera=camera,
                sample_name=sample_name,
                expected_min_frames=expected_min_frames,
                window_frames=window_frames,
                window_checks=window_checks,
                rng=rng,
            )
        )
    return issues


def write_reports(issues: list[SampleIssue], json_path: Path, txt_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(issue) for issue in issues], file, ensure_ascii=False, indent=2)

    grouped: dict[str, list[SampleIssue]] = {}
    for issue in issues:
        grouped.setdefault(issue.sample, []).append(issue)

    with txt_path.open("w", encoding="utf-8") as file:
        for sample_name in sorted(grouped):
            file.write(f"sample {sample_name}\n")
            for issue in grouped[sample_name]:
                file.write(
                    f"  [{issue.camera}] {issue.stream_type} {issue.issue} -> {issue.path} | {issue.detail}\n"
                )


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    video_root = dataset_root / "videos"
    control_root = dataset_root / args.control_dir

    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")
    if not control_root.exists():
        raise FileNotFoundError(f"Control root does not exist: {control_root}")

    sample_names = list_samples(video_root, args.cameras[0])
    print(f"Total reference samples: {len(sample_names)}")
    
    max_workers = args.num_workers if args.num_workers > 0 else multiprocessing.cpu_count()
    print(f"Starting parallel check with {max_workers} processes...")

    issues: list[SampleIssue] = []
    bad_samples_set = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                sample_name,
                video_root,
                control_root,
                args.cameras,
                args.expected_min_frames,
                args.window_frames,
                args.window_checks,
                args.seed
            ): sample_name
            for sample_name in sample_names
        }

        pbar = tqdm(as_completed(futures), total=len(sample_names), desc="Checking dataset")
        for future in pbar:
            sample_name = futures[future]
            try:
                sample_issues = future.result()
                if sample_issues:
                    issues.extend(sample_issues)
                    bad_samples_set.add(sample_name)
            except Exception as exc:
                issues.append(
                    SampleIssue(
                        sample=sample_name,
                        camera="N/A",
                        stream_type="N/A",
                        issue="process_crash",
                        path="N/A",
                        detail=f"Worker process crashed: {exc}"
                    )
                )
                bad_samples_set.add(sample_name)

            pbar.set_postfix_str(f"Last checked: {sample_name} | Bad: {len(bad_samples_set)}")

    report_json = Path(args.report_json)
    report_txt = Path(args.report_txt)
    write_reports(issues, report_json, report_txt)

    bad_samples = sorted(list(bad_samples_set))
    print("\nScan finished")
    print(f"Bad samples: {len(bad_samples)}")
    print(f"Issues: {len(issues)}")
    print(f"JSON report: {report_json}")
    print(f"Text report: {report_txt}")

    if bad_samples:
        print("First bad samples:")
        for name in bad_samples[:10]:
            print(f"  - {name}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())