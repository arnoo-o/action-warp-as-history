#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT.parent / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import pandas as pd


DEFAULT_ROOT = REPO_ROOT / "data" / "nuke-000000"
DEFAULT_OUTPUT = DEFAULT_ROOT / "analysis"
ACTION_LABELS = {
    "W": "move_forward",
    "A": "move_left",
    "S": "move_backward",
    "D": "move_right",
    "J": "jump",
    "C": "crouch",
    "R": "reload_or_interact",
    "V": "use_or_inspect",
    "[": "primary_fire",
    "]": "secondary_fire",
    "-": "idle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CS2 action parquet files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--top-combos", type=int, default=20)
    return parser.parse_args()


def pct(part: float, whole: float) -> float:
    return 0.0 if whole == 0 else 100.0 * part / whole


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": sorted_values[min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))],
        "p95": sorted_values[min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))],
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(root: Path, output_dir: Path, time_bins: int, top_combos: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(root.glob("*.parquet"))
    total_videos = 0
    total_frames = 0
    total_duration_s = 0.0
    button_frame_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    fire_segment_count_by_video: dict[str, int] = {}
    fire_segments: list[dict] = []
    fire_duration_values: list[float] = []
    fire_start_values: list[float] = []
    fire_start_norm_values: list[float] = []
    fire_bin_counts = [0 for _ in range(time_bins)]
    button_presence_by_video: dict[str, set[str]] = {}

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        row = df.iloc[0]
        video_id = parquet_path.stem
        fps = float(row["fps"])
        total_time = float(row["total_time"])
        frame_data = list(row["frame_data"])
        num_frames = len(frame_data)

        total_videos += 1
        total_frames += num_frames
        total_duration_s += total_time
        button_presence_by_video[video_id] = set()

        fire_segment_start = None
        fire_segment_frames = 0

        for frame_idx, frame in enumerate(frame_data):
            actions = str(frame.get("actions", "") or "")
            combo_counter[actions] += 1
            normalized_actions = actions if actions else "-"

            if normalized_actions == "-":
                button_frame_counter["-"] += 1
                button_presence_by_video[video_id].add("-")
            else:
                for char in normalized_actions:
                    button_frame_counter[char] += 1
                    button_presence_by_video[video_id].add(char)

            has_primary_fire = "[" in normalized_actions
            if has_primary_fire:
                if fire_segment_start is None:
                    fire_segment_start = frame_idx
                    fire_segment_frames = 0
                fire_segment_frames += 1
            elif fire_segment_start is not None:
                start_s = fire_segment_start / fps
                duration_s = fire_segment_frames / fps
                start_norm = 0.0 if total_time <= 0 else min(0.999999, start_s / total_time)
                bin_idx = min(time_bins - 1, int(start_norm * time_bins))
                fire_bin_counts[bin_idx] += 1
                segment = {
                    "video_id": video_id,
                    "start_frame": fire_segment_start,
                    "end_frame": fire_segment_start + fire_segment_frames - 1,
                    "start_s": round(start_s, 6),
                    "duration_s": round(duration_s, 6),
                    "start_norm": round(start_norm, 6),
                    "video_total_time_s": round(total_time, 6),
                }
                fire_segments.append(segment)
                fire_duration_values.append(duration_s)
                fire_start_values.append(start_s)
                fire_start_norm_values.append(start_norm)
                fire_segment_count_by_video[video_id] = fire_segment_count_by_video.get(video_id, 0) + 1
                fire_segment_start = None
                fire_segment_frames = 0

        if fire_segment_start is not None:
            start_s = fire_segment_start / fps
            duration_s = fire_segment_frames / fps
            start_norm = 0.0 if total_time <= 0 else min(0.999999, start_s / total_time)
            bin_idx = min(time_bins - 1, int(start_norm * time_bins))
            fire_bin_counts[bin_idx] += 1
            segment = {
                "video_id": video_id,
                "start_frame": fire_segment_start,
                "end_frame": fire_segment_start + fire_segment_frames - 1,
                "start_s": round(start_s, 6),
                "duration_s": round(duration_s, 6),
                "start_norm": round(start_norm, 6),
                "video_total_time_s": round(total_time, 6),
            }
            fire_segments.append(segment)
            fire_duration_values.append(duration_s)
            fire_start_values.append(start_s)
            fire_start_norm_values.append(start_norm)
            fire_segment_count_by_video[video_id] = fire_segment_count_by_video.get(video_id, 0) + 1

    seconds_per_frame = 0.0 if total_frames == 0 else total_duration_s / total_frames
    button_rows = []
    for char, label in ACTION_LABELS.items():
        frames = button_frame_counter.get(char, 0)
        button_rows.append(
            {
                "action_char": char,
                "label": label,
                "frame_count": frames,
                "frame_ratio_pct": round(pct(frames, total_frames), 4),
                "active_seconds_est": round(frames * seconds_per_frame, 6),
                "video_coverage": sum(1 for present in button_presence_by_video.values() if char in present),
            }
        )

    combo_rows = [
        {"action_combo": combo, "frame_count": count, "frame_ratio_pct": round(pct(count, total_frames), 4)}
        for combo, count in combo_counter.most_common(top_combos)
    ]

    top_fire_segments = sorted(fire_segments, key=lambda item: item["duration_s"], reverse=True)[:20]
    fire_segment_rows = sorted(fire_segments, key=lambda item: (item["video_id"], item["start_frame"]))

    bin_rows = []
    for idx, count in enumerate(fire_bin_counts):
        left = idx / time_bins
        right = (idx + 1) / time_bins
        bin_rows.append(
            {
                "normalized_window": f"{left:.1f}-{right:.1f}",
                "segment_count": count,
                "segment_ratio_pct": round(pct(count, len(fire_segments)), 4),
            }
        )

    summary = {
        "dataset_root": str(root),
        "total_videos": total_videos,
        "total_frames": total_frames,
        "total_duration_s": round(total_duration_s, 6),
        "button_summary": button_rows,
        "top_action_combos": combo_rows,
        "primary_fire_summary": {
            "frame_count": button_frame_counter.get("[", 0),
            "frame_ratio_pct": round(pct(button_frame_counter.get("[", 0), total_frames), 4),
            "segment_count": len(fire_segments),
            "videos_with_fire": sum(1 for count in fire_segment_count_by_video.values() if count > 0),
            "segments_per_video_stats": summarize_numeric(list(fire_segment_count_by_video.values())),
            "duration_stats_s": summarize_numeric(fire_duration_values),
            "start_time_stats_s": summarize_numeric(fire_start_values),
            "normalized_start_stats": summarize_numeric(fire_start_norm_values),
            "normalized_timeline_distribution": bin_rows,
            "top_longest_segments": top_fire_segments,
        },
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "button_summary.csv",
        button_rows,
        ["action_char", "label", "frame_count", "frame_ratio_pct", "active_seconds_est", "video_coverage"],
    )
    write_csv(output_dir / "top_action_combos.csv", combo_rows, ["action_combo", "frame_count", "frame_ratio_pct"])
    write_csv(
        output_dir / "primary_fire_segments.csv",
        fire_segment_rows,
        ["video_id", "start_frame", "end_frame", "start_s", "duration_s", "start_norm", "video_total_time_s"],
    )
    write_csv(
        output_dir / "primary_fire_timeline_bins.csv",
        bin_rows,
        ["normalized_window", "segment_count", "segment_ratio_pct"],
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze_dataset(args.root.resolve(), args.output_dir.resolve(), args.time_bins, args.top_combos)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
