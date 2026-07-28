#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Minecraft WAH manifest to exact 16-fps event timing.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--target_fps", type=float, default=16.0)
    parser.add_argument("--max_video_frames", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) if rows else []
    for column in ("source_fps", "target_fps", "source_event_local_frame", "training_frame_stride"):
        if column not in fieldnames:
            fieldnames.append(column)

    selected = []
    categories = Counter()
    for row in rows:
        source_fps = float(row.get("source_fps") or row.get("fps") or 20.0)
        source_event = row.get("source_event_local_frame") or row.get("event_local_frame")
        if source_event in (None, ""):
            continue
        event_frame = int(round(float(source_event) * float(args.target_fps) / source_fps))
        if not 0 <= event_frame < int(args.max_video_frames):
            continue
        row["source_fps"] = f"{source_fps:g}"
        row["target_fps"] = f"{float(args.target_fps):g}"
        row["source_event_local_frame"] = str(int(float(source_event)))
        row["event_frame"] = str(event_frame)
        row["fps"] = f"{float(args.target_fps):g}"
        row["training_frame_stride"] = "1"
        selected.append(row)
        categories[str(row.get("action_type") or row.get("category") or "none")] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(
        {
            "input_rows": len(rows),
            "output_rows": len(selected),
            "target_fps": float(args.target_fps),
            "max_video_frames": int(args.max_video_frames),
            "categories": dict(categories),
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
