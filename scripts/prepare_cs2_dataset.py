#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data"
    / "cs2_example"
    / "CS2-10k-sample"
    / "sample"
    / "data"
    / "85bf6db3-f7ee-578e-8ec4-3cab1930348c"
)

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
    parser = argparse.ArgumentParser(description="Clean CS2 sample videos and interaction parquet files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--min_frames", type=int, default=33)
    parser.add_argument("--summary_stride", type=int, default=12)
    parser.add_argument("--top_tokens", type=int, default=6)
    return parser.parse_args()


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


def repo_relative_text(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        parts = list(path.parts)
        if "data" in parts:
            return "/".join(parts[parts.index("data") :])
        return str(path).replace("\\", "/")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else root / "cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(root.glob("*.parquet"))
    cleaned_index_path = output_dir / "cs2_training.csv"
    report_path = output_dir / "cleaning_report.json"

    rows = []
    action_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    map_counter: Counter[str] = Counter()
    total_frames = 0
    kept_samples = 0

    for parquet_path in parquet_files:
        stem = parquet_path.stem
        video_path = root / f"{stem}.mp4"
        if not video_path.is_file():
            continue
        row = pd.read_parquet(parquet_path).iloc[0]
        frame_data = list(row["frame_data"])
        if len(frame_data) < int(args.min_frames):
            continue

        fps = float(row["fps"])
        prev_frame = None
        frame_summaries = []
        frame_features = []
        local_actions = Counter()
        local_tokens = Counter()
        for frame_index, frame in enumerate(frame_data):
            frame = dict(frame)
            summary, features = summarize_frame(frame, prev_frame, fps)
            prev_frame = frame
            action_text = str(features["actions_raw"])
            local_actions[action_text] += 1
            for token in features["tokens"]:
                local_tokens[token] += 1
            total_frames += 1
            if summary and (frame_index % int(args.summary_stride) == 0 or action_text not in {"", "-"}):
                frame_summaries.append({"frame": int(frame_index), "video_t_ms": 1000.0 * frame_index / fps, "summary": summary})
            frame_features.append({"frame": int(frame_index), **features})

        cleaned_payload = {
            "video_path": repo_relative_text(video_path),
            "fps": fps,
            "num_frames": int(row["num_frames"]),
            "frame_summaries": frame_summaries,
            "frame_features": frame_features,
            "meta": {
                "video_filename": str(row["video_filename"]),
                "match_id": str(row["match_id"]),
                "map": str(row["map"]),
                "round_number": int(row["round_number"]),
                "team": int(row["team"]),
                "player_index": int(row["player_index"]),
                "category": str(row["category"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "fov": float(row["fov"]),
                "top_actions": local_actions.most_common(int(args.top_tokens)),
                "top_tokens": local_tokens.most_common(int(args.top_tokens)),
            },
        }
        cleaned_json_path = output_dir / f"{stem}_interaction_history.json"
        cleaned_json_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False), encoding="utf-8")

        rows.append(
            {
                "id": stem,
                "video_path": repo_relative_text(video_path),
                "prompt": build_prompt(row),
                "interaction_history_path": repo_relative_text(cleaned_json_path),
                "map": str(row["map"]),
                "round_number": int(row["round_number"]),
                "num_frames": int(row["num_frames"]),
                "fps": fps,
            }
        )
        kept_samples += 1
        map_counter[str(row["map"])] += 1
        action_counter.update(local_actions)
        token_counter.update(local_tokens)

    with cleaned_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "video_path", "prompt", "interaction_history_path", "map", "round_number", "num_frames", "fps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input_root": str(root),
        "output_dir": str(output_dir),
        "samples_total": len(parquet_files),
        "samples_kept": kept_samples,
        "frames_total": total_frames,
        "maps": dict(map_counter),
        "top_actions": action_counter.most_common(20),
        "top_tokens": token_counter.most_common(20),
        "training_csv": str(cleaned_index_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
