#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised in the H100 preparation environment.
    raise SystemExit("audit_minecraft_event_alignment.py requires OpenCV (cv2)") from exc
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Render source-frame strips around Minecraft interaction telemetry events.")
    parser.add_argument("--review_manifest", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--action_type", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--before", type=int, default=1)
    parser.add_argument("--after", type=int, default=6)
    return parser.parse_args()


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Unable to decode source frame {frame_index}")
    return frame


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.review_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("review_status") == "pending"
            and (not args.action_type or row.get("action_type") == args.action_type)
        ][: args.limit]

    reports = []
    for sample_index, row in enumerate(rows):
        video_path = args.repo_root / row["video_path"]
        source_start = int(float(row.get("source_frame_start", 0) or 0))
        telemetry_frame = int(float(row["telemetry_event_source_frame"])) - source_start
        visual_start = int(float(row["visual_start_source_frame"])) - source_start
        frame_indices = list(range(max(0, visual_start - args.before), visual_start + args.after + 1))
        cap = cv2.VideoCapture(str(video_path))
        frames = [read_frame(cap, frame_index) for frame_index in frame_indices]
        cap.release()
        height, width = frames[0].shape[:2]
        tiles = []
        scores = []
        previous = frames[0]
        for frame_index, frame in zip(frame_indices, frames):
            residual = np.mean(np.abs(frame.astype(np.float32) - previous.astype(np.float32)), axis=2)
            central = residual[int(0.12 * height) : int(0.80 * height), int(0.20 * width) : int(0.80 * width)]
            score = float(np.quantile(central, 0.95))
            scores.append(score)
            tile = cv2.resize(frame, (320, 192))
            cv2.putText(
                tile,
                f"segment {frame_index} visual offset {frame_index - visual_start:+d}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"central delta p95={score:.1f}",
                (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
            previous = frame
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[start : start + 4]) for start in range(0, len(tiles), 4)])
        filename = f"{sample_index:02d}_{row['event_id'].replace(':', '_')}.jpg"
        cv2.imwrite(str(args.output_dir / filename), sheet)
        reports.append(
            {
                "file": filename,
                "event_id": row["event_id"],
                "telemetry_source_local_frame": telemetry_frame,
                "visual_start_source_local_frame": visual_start,
                "reference_source_frame": row.get("reference_source_frame"),
                "teacher_rgb_source_frames": row.get("teacher_rgb_source_frames"),
                "frame_delta_p95": dict(zip(frame_indices, scores)),
            }
        )
    (args.output_dir / "audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
