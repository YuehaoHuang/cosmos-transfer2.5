#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

# Map Waymo camera folders to multiview inference camera keys.
WAYMO_TO_MULTIVIEW = {
    "pinhole_front": "front_wide",
    "pinhole_front_left": "cross_left",
    "pinhole_front_right": "cross_right",
    "pinhole_side_left": "rear_left",
    "pinhole_side_right": "rear_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Waymo multiview inference spec JSON files")
    parser.add_argument("--data-root", required=True, help="Waymo inference root directory")
    parser.add_argument("--split", required=True, choices=["training", "validation"], help="Dataset split")
    parser.add_argument("--prompt-json", required=True, help="Prompt JSON path")
    parser.add_argument("--output", default="", help="Output directory for spec JSON files")
    return parser.parse_args()


def find_video_file(root: Path, sequence: str) -> Path:
    # Prefer the common *_0.mp4 suffix if present, otherwise take first match.
    preferred = root / f"{sequence}_0.mp4"
    if preferred.exists():
        return preferred

    matches = sorted(root.glob(f"{sequence}_*.mp4"))
    if matches:
        return matches[0]

    # As a final fallback, allow exact filename.
    exact = root / f"{sequence}.mp4"
    if exact.exists():
        return exact

    raise FileNotFoundError(f"No video found for sequence '{sequence}' under {root}")


def load_prompt(prompt_map: dict[str, str], sequence: str) -> str:
    front_key = f"{sequence}_pinhole_front"
    if front_key in prompt_map:
        return prompt_map[front_key]

    # Fallback: pick any prompt key for this sequence.
    prefix = f"{sequence}_"
    for key, value in prompt_map.items():
        if key.startswith(prefix):
            return value

    raise KeyError(
        f"No prompt found for sequence '{sequence}'. Expected key '{front_key}' or any key with prefix '{prefix}'."
    )


def normalize_sequence(stem: str) -> str:
    if stem.endswith("_0"):
        return stem[:-2]
    return stem


def main() -> int:
    args = parse_args()

    split_root = Path(args.data_root) / args.split
    videos_root = split_root / "videos"
    control_root = split_root / "world_scenario"

    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")
    if not control_root.exists():
        raise FileNotFoundError(f"Control root not found: {control_root}")

    prompt_map_path = Path(args.prompt_json)
    with prompt_map_path.open("r", encoding="utf-8") as f:
        prompt_map = json.load(f)
    output_dir = Path(args.output) if args.output else split_root / "specs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = videos_root / "pinhole_front"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference camera directory not found: {ref_dir}")

    stems = sorted(path.stem for path in ref_dir.glob("*.mp4"))
    sequences = sorted({normalize_sequence(stem) for stem in stems})

    written = 0
    skipped = 0
    for sequence in sequences:
        per_view_prompts: dict[str, str] = {}
        for waymo_camera, mv_camera in WAYMO_TO_MULTIVIEW.items():
            prompt_key = f"{sequence}_{waymo_camera}"
            per_view_prompts[mv_camera] = prompt_map.get(prompt_key)
        # per_view_prompts = load_prompt(prompt_map, sequence)

        spec: dict[str, object] = {
            "name": sequence,
            "prompt": per_view_prompts,
            "fps": 10,
            "enable_autoregressive": True,
            "num_chunks": 999,
            "chunk_overlap": 10,
            "save_combined_views": False,
            "save_views_in_subfolders": True,
            "save_autoregressive_chunks": True,
        }

        active_views = 0
        for waymo_camera, mv_camera in WAYMO_TO_MULTIVIEW.items():
            input_file = find_video_file(videos_root / waymo_camera, sequence)
            control_file = find_video_file(control_root / waymo_camera, sequence)
            spec[mv_camera] = {
                "input_path": str(input_file),
                "control_path": str(control_file),
                "num_conditional_frames_per_view": 1,
            }
            active_views += 1

        if active_views == 0:
            print(f"[skip] {sequence}: no active views found")
            skipped += 1
            continue

        output_path = output_dir / f"{sequence}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        written += 1

    print(f"Wrote specs: {written}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
