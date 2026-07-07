#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT.parent / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import imageio.v2 as imageio
import pandas as pd


PRIMARY_FIRE_CHAR = "["
ACTION_CHAR_MAP = {
    "W": ("move_forward", "forward"),
    "A": ("move_left", "strafe_left"),
    "S": ("move_backward", "backward"),
    "D": ("move_right", "strafe_right"),
    "J": ("jump", "jump"),
    "C": ("crouch", "crouch"),
    "R": ("reload", "reload_or_interact"),
    "V": ("use", "use_or_inspect"),
    "[": ("primary_fire", "primary_fire"),
    "]": ("secondary_fire", "secondary_fire"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-folder CS2 primary-fire training dataset.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--folder-glob", default="nuke-*")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "cs2-primary-fire-training")
    parser.add_argument("--min-clip-seconds", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=33)
    parser.add_argument("--summary-stride", type=int, default=12)
    parser.add_argument("--top-tokens", type=int, default=6)
    parser.add_argument("--pre-fire-frames", type=int, default=16)
    parser.add_argument("--post-fire-frames", type=int, default=24)
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--max-videos-per-folder", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_relative_text(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


def decode_action_tokens(action_text: str) -> list[str]:
    tokens = []
    for char in str(action_text or ""):
        mapped = ACTION_CHAR_MAP.get(char)
        if mapped is None:
            tokens.append(f"raw_{char}")
        else:
            tokens.append(mapped[1])
    return tokens


def action_feature_flags(action_text: str) -> dict[str, float]:
    flags = {value[0]: 0.0 for value in ACTION_CHAR_MAP.values()}
    for char in str(action_text or ""):
        mapped = ACTION_CHAR_MAP.get(char)
        if mapped is not None:
            flags[mapped[0]] = 1.0
    return flags


def speed_between(prev_frame: dict | None, cur_frame: dict, fps: float) -> float:
    if prev_frame is None:
        return 0.0
    dx = float(cur_frame.get("position_x", 0.0) or 0.0) - float(prev_frame.get("position_x", 0.0) or 0.0)
    dy = float(cur_frame.get("position_y", 0.0) or 0.0) - float(prev_frame.get("position_y", 0.0) or 0.0)
    dz = float(cur_frame.get("position_z", 0.0) or 0.0) - float(prev_frame.get("position_z", 0.0) or 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz) * float(fps)


def summarize_frame(frame: dict, prev_frame: dict | None, fps: float) -> tuple[str, dict]:
    action_text = str(frame.get("actions", "") or "")
    tokens = decode_action_tokens(action_text)
    flags = action_feature_flags(action_text)
    mouse_dx = float(frame.get("mouse_x_delta", 0.0) or 0.0)
    mouse_dy = float(frame.get("mouse_y_delta", 0.0) or 0.0)
    yaw = float(frame.get("rotation_yaw", 0.0) or 0.0)
    pitch = float(frame.get("rotation_pitch", 0.0) or 0.0)
    prev_yaw = yaw if prev_frame is None else float(prev_frame.get("rotation_yaw", yaw) or yaw)
    prev_pitch = pitch if prev_frame is None else float(prev_frame.get("rotation_pitch", pitch) or pitch)
    yaw_delta = yaw - prev_yaw
    pitch_delta = pitch - prev_pitch
    speed = speed_between(prev_frame, frame, fps)

    summary_parts = []
    if tokens:
        summary_parts.append("actions:" + "+".join(tokens[:8]))
    if abs(mouse_dx) > 0.2 or abs(mouse_dy) > 0.2:
        look_tokens = []
        if mouse_dx > 0.2:
            look_tokens.append("look_right")
        elif mouse_dx < -0.2:
            look_tokens.append("look_left")
        if mouse_dy > 0.2:
            look_tokens.append("look_down")
        elif mouse_dy < -0.2:
            look_tokens.append("look_up")
        summary_parts.append("mouse:" + "+".join(look_tokens or ["look_adjust"]))
    if speed > 80.0:
        summary_parts.append("motion:fast_move")
    elif speed > 15.0:
        summary_parts.append("motion:move")

    features = {
        "actions_raw": action_text,
        "tokens": tokens,
        "mouse_dx": mouse_dx,
        "mouse_dy": mouse_dy,
        "yaw_delta": yaw_delta,
        "pitch_delta": pitch_delta,
        "speed": speed,
        **flags,
    }
    return " ; ".join(summary_parts), features


def build_prompt(row: pd.Series) -> str:
    return (
        f"Counter-Strike 2 first-person gameplay on {row['map']}, round {int(row['round_number'])}, "
        f"competitive tactical movement, aim adjustments, weapon usage, and player-controlled camera motion."
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    chunks_dir = output_dir / "clips"
    cleaned_dir = output_dir / "cleaned"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = []
    for path in sorted(data_root.glob(args.folder_glob)):
        if not path.is_dir():
            continue
        if not path.name.startswith("nuke-"):
            continue
        if "primary-fire" in path.name.lower():
            continue
        if not any(path.glob("*.parquet")):
            continue
        source_dirs.append(path)
    if int(args.max_folders) > 0:
        source_dirs = source_dirs[: int(args.max_folders)]

    rows = []
    manifest_rows = []
    action_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    map_counter: Counter[str] = Counter()
    total_frames = 0
    clips_total = 0
    videos_total = 0

    for source_dir in source_dirs:
        parquet_files = sorted(source_dir.glob("*.parquet"))
        if int(args.max_videos_per_folder) > 0:
            parquet_files = parquet_files[: int(args.max_videos_per_folder)]
        for parquet_path in parquet_files:
            video_path = source_dir / f"{parquet_path.stem}.mp4"
            if not video_path.is_file():
                continue
            videos_total += 1
            row = pd.read_parquet(parquet_path).iloc[0].copy()
            frame_data = list(row["frame_data"])
            fps = float(row["fps"])
            total_source_frames = len(frame_data)
            min_clip_frames = max(1, int(round(float(args.min_clip_seconds) * fps)))
            click_frames = extract_click_frames(frame_data)
            windows = build_clip_windows(click_frames, total_source_frames, min_clip_frames)

            for clip_idx, (start_frame, end_frame) in enumerate(windows):
                clipped_frame_data = frame_data[start_frame:end_frame]
                if len(clipped_frame_data) < int(args.min_frames):
                    continue
                clip_id = f"{source_dir.name}_{parquet_path.stem}_fire_{clip_idx:03d}"
                out_mp4 = chunks_dir / f"{clip_id}.mp4"
                out_parquet = chunks_dir / f"{clip_id}.parquet"
                out_json = cleaned_dir / f"{clip_id}_interaction_history.json"
                if (out_mp4.exists() or out_parquet.exists() or out_json.exists()) and not args.overwrite:
                    raise FileExistsError(f"Outputs for {clip_id} already exist. Use --overwrite.")

                write_video_clip(video_path, out_mp4, start_frame, end_frame, fps)

                clipped_row = row.copy()
                clipped_row["video_filename"] = out_mp4.name
                clipped_row["num_frames"] = int(len(clipped_frame_data))
                clipped_row["total_time"] = float(len(clipped_frame_data) / fps)
                clipped_row["frame_data"] = clipped_frame_data
                pd.DataFrame([clipped_row]).to_parquet(out_parquet, index=False)

                prev_frame = None
                frame_summaries = []
                frame_features = []
                local_actions = Counter()
                local_tokens = Counter()
                for local_frame_index, frame in enumerate(clipped_frame_data):
                    frame = dict(frame)
                    summary, features = summarize_frame(frame, prev_frame, fps)
                    prev_frame = frame
                    action_text = str(features["actions_raw"])
                    local_actions[action_text] += 1
                    for token in features["tokens"]:
                        local_tokens[token] += 1
                    total_frames += 1
                    if summary and (local_frame_index % int(args.summary_stride) == 0 or action_text not in {"", "-"}):
                        frame_summaries.append(
                            {"frame": int(local_frame_index), "video_t_ms": 1000.0 * local_frame_index / fps, "summary": summary}
                        )
                    frame_features.append({"frame": int(local_frame_index), **features})

                cleaned_payload = {
                    "video_path": repo_relative_text(out_mp4),
                    "fps": fps,
                    "num_frames": int(len(clipped_frame_data)),
                    "frame_summaries": frame_summaries,
                    "frame_features": frame_features,
                    "meta": {
                        "video_filename": out_mp4.name,
                        "source_video_filename": str(row["video_filename"]),
                        "match_id": str(row["match_id"]),
                        "map": str(row["map"]),
                        "round_number": int(row["round_number"]),
                        "team": int(row["team"]),
                        "player_index": int(row["player_index"]),
                        "category": str(row["category"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "fov": float(row["fov"]),
                        "source_dir": source_dir.name,
                        "clip_start_frame": int(start_frame),
                        "clip_end_frame_exclusive": int(end_frame),
                        "click_frames_source": [int(x) for x in click_frames if int(start_frame) <= x < int(end_frame)],
                        "top_actions": local_actions.most_common(int(args.top_tokens)),
                        "top_tokens": local_tokens.most_common(int(args.top_tokens)),
                    },
                }
                out_json.write_text(json.dumps(cleaned_payload, ensure_ascii=False), encoding="utf-8")
                clip_clicks = [int(x) for x in click_frames if int(start_frame) <= x < int(end_frame)]
                time_mask = [0.0] * int(len(clipped_frame_data))
                event_windows = []
                for click_frame_source in clip_clicks:
                    local_click = int(click_frame_source) - int(start_frame)
                    window_start = max(0, local_click - int(args.pre_fire_frames))
                    window_end = min(len(clipped_frame_data), local_click + int(args.post_fire_frames) + 1)
                    event_windows.append(
                        {
                            "click_frame_local": int(local_click),
                            "window_start": int(window_start),
                            "window_end_exclusive": int(window_end),
                        }
                    )
                    for idx in range(window_start, window_end):
                        time_mask[idx] = 1.0
                out_event_json = cleaned_dir / f"{clip_id}_primary_fire_event.json"
                out_event_json.write_text(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "fps": fps,
                            "num_frames": int(len(clipped_frame_data)),
                            "click_frames_source": clip_clicks,
                            "click_frames_local": [int(x) - int(start_frame) for x in clip_clicks],
                            "event_windows": event_windows,
                            "source_frame_indices": [int(start_frame) + idx for idx in range(len(clipped_frame_data))],
                            "time_mask": time_mask,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                rows.append(
                    {
                        "id": clip_id,
                        "video_path": repo_relative_text(out_mp4),
                        "prompt": build_prompt(row),
                        "interaction_history_path": repo_relative_text(out_json),
                        "map": str(row["map"]),
                        "round_number": int(row["round_number"]),
                        "num_frames": int(len(clipped_frame_data)),
                        "fps": fps,
                        "primary_fire_event_path": repo_relative_text(out_event_json),
                        "primary_fire_loss_mask_path": "",
                        "warp_video_path": "",
                        "warp_visibility_mask_path": "",
                    }
                )
                manifest_rows.append(
                    {
                        "id": clip_id,
                        "source_dir": source_dir.name,
                        "source_video": video_path.name,
                        "video_path": repo_relative_text(out_mp4),
                        "parquet_path": repo_relative_text(out_parquet),
                        "interaction_history_path": repo_relative_text(out_json),
                        "primary_fire_event_path": repo_relative_text(out_event_json),
                        "start_frame": int(start_frame),
                        "end_frame_exclusive": int(end_frame),
                        "num_frames": int(len(clipped_frame_data)),
                        "duration_s": float(len(clipped_frame_data) / fps),
                    }
                )
                clips_total += 1
                map_counter[str(row["map"])] += 1
                action_counter.update(local_actions)
                token_counter.update(local_tokens)

    training_csv = cleaned_dir / "cs2_training.csv"
    with training_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "video_path",
                "prompt",
                "interaction_history_path",
                "map",
                "round_number",
                "num_frames",
                "fps",
                "primary_fire_event_path",
                "primary_fire_loss_mask_path",
                "warp_video_path",
                "warp_visibility_mask_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_csv = output_dir / "chunk_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_dir",
                "source_video",
                "video_path",
                "parquet_path",
                "interaction_history_path",
                "primary_fire_event_path",
                "start_frame",
                "end_frame_exclusive",
                "num_frames",
                "duration_s",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "source_dirs": [str(path) for path in source_dirs],
        "videos_total": videos_total,
        "clips_total": clips_total,
        "frames_total": total_frames,
        "maps": dict(map_counter),
        "top_actions": action_counter.most_common(20),
        "top_tokens": token_counter.most_common(20),
        "training_csv": str(training_csv),
        "manifest_csv": str(manifest_csv),
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
