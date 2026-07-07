#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT.parent / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import imageio.v2 as imageio
import pandas as pd


PRIMARY_FIRE_CHAR = "["
DEFAULT_ROOT = REPO_ROOT / "data" / "nuke-000000"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "nuke-000000-primary-fire-chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract >=20s CS2 chunks around left-click primary fire events.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-clip-seconds", type=float, default=20.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_click_frames(frame_data: list[dict]) -> list[int]:
    click_frames = []
    prev_pressed = False
    for idx, frame in enumerate(frame_data):
        actions = str(frame.get("actions", "") or "")
        pressed = PRIMARY_FIRE_CHAR in actions
        if pressed and not prev_pressed:
            click_frames.append(int(idx))
        prev_pressed = pressed
    return click_frames


def build_clip_windows(click_frames: list[int], total_frames: int, min_clip_frames: int) -> list[tuple[int, int]]:
    if not click_frames or total_frames <= 0:
        return []
    half = max(1, min_clip_frames // 2)
    raw_windows = []
    for click in click_frames:
        start = max(0, int(click) - half)
        end = start + int(min_clip_frames)
        if end > total_frames:
            end = total_frames
            start = max(0, end - int(min_clip_frames))
        raw_windows.append((int(start), int(end)))
    raw_windows.sort()
    merged = []
    for start, end in raw_windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(int(start), int(end)) for start, end in merged]


def write_video_clip(src_path: Path, dst_path: Path, start_frame: int, end_frame: int, fps: float) -> None:
    reader = imageio.get_reader(str(src_path))
    writer = imageio.get_writer(str(dst_path), fps=float(fps))
    try:
        for idx, frame in enumerate(reader):
            if idx < int(start_frame):
                continue
            if idx >= int(end_frame):
                break
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(root.glob("*.parquet"))
    if int(args.max_videos) > 0:
        parquet_files = parquet_files[: int(args.max_videos)]

    manifest_rows = []
    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "min_clip_seconds": float(args.min_clip_seconds),
        "videos_total": 0,
        "videos_with_clicks": 0,
        "clips_total": 0,
    }

    for parquet_path in parquet_files:
        video_path = root / f"{parquet_path.stem}.mp4"
        if not video_path.is_file():
            continue
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        row = df.iloc[0].copy()
        frame_data = list(row["frame_data"])
        fps = float(row["fps"])
        total_frames = len(frame_data)
        min_clip_frames = max(1, int(round(float(args.min_clip_seconds) * fps)))
        click_frames = extract_click_frames(frame_data)
        windows = build_clip_windows(click_frames, total_frames, min_clip_frames)

        summary["videos_total"] += 1
        if click_frames:
            summary["videos_with_clicks"] += 1

        for clip_idx, (start_frame, end_frame) in enumerate(windows):
            chunk_id = f"{parquet_path.stem}_fire_{clip_idx:03d}"
            out_mp4 = output_dir / f"{chunk_id}.mp4"
            out_parquet = output_dir / f"{chunk_id}.parquet"
            if (out_mp4.exists() or out_parquet.exists()) and not args.overwrite:
                raise FileExistsError(f"{out_mp4} or {out_parquet} already exists. Use --overwrite.")

            write_video_clip(video_path, out_mp4, start_frame, end_frame, fps)
            clipped_frame_data = frame_data[start_frame:end_frame]
            clipped_row = row.copy()
            clipped_row["video_filename"] = out_mp4.name
            clipped_row["num_frames"] = int(len(clipped_frame_data))
            clipped_row["total_time"] = float(len(clipped_frame_data) / fps)
            clipped_row["frame_data"] = clipped_frame_data
            pd.DataFrame([clipped_row]).to_parquet(out_parquet, index=False)

            clip_clicks = [frame for frame in click_frames if int(start_frame) <= frame < int(end_frame)]
            manifest_rows.append(
                {
                    "clip_id": chunk_id,
                    "video_path": out_mp4.name,
                    "parquet_path": out_parquet.name,
                    "source_video": video_path.name,
                    "start_frame": int(start_frame),
                    "end_frame_exclusive": int(end_frame),
                    "num_frames": int(len(clipped_frame_data)),
                    "fps": float(fps),
                    "duration_s": float(len(clipped_frame_data) / fps),
                    "click_count": int(len(clip_clicks)),
                    "click_frames_source": " ".join(str(x) for x in clip_clicks),
                }
            )
            summary["clips_total"] += 1

    manifest_path = output_dir / "chunk_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "video_path",
                "parquet_path",
                "source_video",
                "start_frame",
                "end_frame_exclusive",
                "num_frames",
                "fps",
                "duration_s",
                "click_count",
                "click_frames_source",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output_dir / "chunk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
