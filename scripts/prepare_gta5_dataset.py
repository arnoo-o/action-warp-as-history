#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "gta5"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "cleaned"
DEFAULT_VIDEO_PATH = DEFAULT_DATA_DIR / "clip.mp4"
DEFAULT_FRAME_EVENTS_PATH = DEFAULT_DATA_DIR / "frame_events.json"
DEFAULT_METADATA_PATH = DEFAULT_DATA_DIR / "metadata.json"

RELEVANT_EVENT_TYPES = {"key", "click", "drag", "scroll", "mouse_move", "modifier_change"}
NOISE_EVENT_TYPES = {
    "active_app",
    "screen_config",
    "appearance_change",
    "battery_status",
    "wifi_change",
    "network_change",
    "power_source_change",
    "input_source_change",
    "memory_pressure",
    "window_moved",
    "window_resized",
    "app_hidden",
    "idle",
    "resume",
}
KEY_NAME_MAP = {
    "w": "accelerate",
    "a": "steer_left",
    "s": "brake_reverse",
    "d": "steer_right",
    " ": "handbrake",
    "f": "enter_exit",
    "r": "cinematic",
    "e": "interact",
    "q": "cover",
    "shift": "sprint",
    "ctrl": "duck",
    "c": "look_back",
    "m": "menu",
    "tab": "weapon_wheel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean GTA5 gameplay events and export Warp-as-History training inputs.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--video_path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--frame_events_path", type=Path, default=DEFAULT_FRAME_EVENTS_PATH)
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt",
        default=(
            "GTA V third-person gameplay with driving, combat, menu navigation, and open-world interactions, "
            "dynamic player-controlled camera motion."
        ),
    )
    parser.add_argument("--max_summary_items", type=int, default=6)
    return parser.parse_args()


def normalize_key_name(event: dict) -> str:
    text = str(event.get("characters") or "").strip().lower()
    if text:
        return KEY_NAME_MAP.get(text, text)
    code = event.get("keyCode")
    return f"keycode_{code}" if code is not None else "unknown_key"


def quantize_delta(value: float, threshold: float = 8.0) -> str:
    if value >= threshold:
        return "positive"
    if value <= -threshold:
        return "negative"
    return "neutral"


def summarize_frame_events(events: list[dict], previous_mouse: tuple[float, float] | None):
    kept_events: list[dict] = []
    counts: Counter[str] = Counter()
    keys_down: list[str] = []
    keys_up: list[str] = []
    click_tokens: list[str] = []
    drag_buttons: list[str] = []
    scroll_tokens: list[str] = []
    mouse_dx = 0.0
    mouse_dy = 0.0
    mouse_moves = 0
    last_mouse = previous_mouse

    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in RELEVANT_EVENT_TYPES:
            continue
        trimmed = {"type": event_type}
        counts[event_type] += 1

        if event_type == "key":
            key_name = normalize_key_name(event)
            if bool(event.get("isDown", False)):
                keys_down.append(key_name)
                trimmed["action"] = "down"
            else:
                keys_up.append(key_name)
                trimmed["action"] = "up"
            trimmed["key"] = key_name
        elif event_type == "click":
            button = str(event.get("button") or "unknown")
            phase = "down" if bool(event.get("isDown", False)) else "up"
            click_tokens.append(f"{button}_{phase}")
            trimmed["button"] = button
            trimmed["phase"] = phase
        elif event_type == "drag":
            button = str(event.get("button") or "unknown")
            drag_buttons.append(button)
            trimmed["button"] = button
        elif event_type == "scroll":
            delta_y = float(event.get("deltaY", 0.0) or 0.0)
            delta_x = float(event.get("deltaX", 0.0) or 0.0)
            if abs(delta_y) >= abs(delta_x):
                scroll_tokens.append("scroll_down" if delta_y < 0 else "scroll_up")
            else:
                scroll_tokens.append("scroll_right" if delta_x > 0 else "scroll_left")
        elif event_type == "modifier_change":
            trimmed["modifiers"] = int(event.get("modifiers", 0) or 0)
        elif event_type == "mouse_move":
            x = float(event.get("x", 0.0) or 0.0)
            y = float(event.get("y", 0.0) or 0.0)
            if last_mouse is not None:
                mouse_dx += x - last_mouse[0]
                mouse_dy += y - last_mouse[1]
            last_mouse = (x, y)
            mouse_moves += 1

        kept_events.append(trimmed)

    summary_parts: list[str] = []
    if keys_down:
        summary_parts.append("keys_down:" + "+".join(sorted(dict.fromkeys(keys_down))))
    if keys_up:
        summary_parts.append("keys_up:" + "+".join(sorted(dict.fromkeys(keys_up))))
    if click_tokens:
        summary_parts.append("clicks:" + "+".join(sorted(dict.fromkeys(click_tokens))))
    if drag_buttons:
        summary_parts.append("drag:" + "+".join(sorted(dict.fromkeys(drag_buttons))))
    if scroll_tokens:
        summary_parts.append("scroll:" + "+".join(sorted(dict.fromkeys(scroll_tokens))))
    if mouse_moves:
        horizontal = quantize_delta(mouse_dx)
        vertical = quantize_delta(-mouse_dy)
        move_tokens = []
        if horizontal == "positive":
            move_tokens.append("look_right")
        elif horizontal == "negative":
            move_tokens.append("look_left")
        if vertical == "positive":
            move_tokens.append("look_up")
        elif vertical == "negative":
            move_tokens.append("look_down")
        if not move_tokens:
            move_tokens.append("look_adjust")
        summary_parts.append("mouse:" + "+".join(move_tokens))

    summary = " ; ".join(summary_parts)
    return kept_events, counts, summary, last_mouse


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    video_path = args.video_path.expanduser().resolve()
    frame_events_path = args.frame_events_path.expanduser().resolve()
    metadata_path = args.metadata_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cleaned_frame_path = output_dir / "frame_events_cleaned.jsonl"
    summary_path = output_dir / "frame_action_summaries.json"
    csv_path = output_dir / "gta5_training.csv"
    report_path = output_dir / "cleaning_report.json"

    frame_summaries: list[dict] = []
    type_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    total_frames = 0
    nonempty_frames = 0
    previous_mouse = None

    with frame_events_path.open("r", encoding="utf-8") as src, cleaned_frame_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            total_frames += 1
            events = list(row.get("events", []))
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type not in RELEVANT_EVENT_TYPES:
                    dropped_counts[event_type] += 1

            kept_events, counts, summary, previous_mouse = summarize_frame_events(events, previous_mouse)
            type_counts.update(counts)
            if kept_events:
                nonempty_frames += 1

            cleaned_row = {
                "frame": int(row["frame"]),
                "video_t_ms": float(row["video_t_ms"]),
                "events": kept_events,
                "summary": summary,
            }
            dst.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")

            if summary:
                frame_summaries.append(
                    {
                        "frame": int(row["frame"]),
                        "video_t_ms": float(row["video_t_ms"]),
                        "summary": summary,
                    }
                )

    summary_payload = {
        "video_path": str(video_path),
        "fps": 30.0,
        "frame_summaries": frame_summaries,
        "meta": {
            "title": metadata.get("title"),
            "game": metadata.get("game"),
            "workflow_id": metadata.get("workflow_id"),
            "prompt": str(args.prompt),
            "relevant_event_types": sorted(RELEVANT_EVENT_TYPES),
            "noise_event_types": sorted(NOISE_EVENT_TYPES),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "video_path", "prompt", "interaction_history_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "gta5_session_000",
                "video_path": str(video_path),
                "prompt": str(args.prompt),
                "interaction_history_path": str(summary_path),
            }
        )

    top_summaries = [item["summary"] for item in frame_summaries[: max(int(args.max_summary_items), 0)]]
    report = {
        "input": {
            "data_dir": str(data_dir),
            "video_path": str(video_path),
            "frame_events_path": str(frame_events_path),
            "metadata_path": str(metadata_path),
        },
        "output": {
            "cleaned_frame_events_path": str(cleaned_frame_path),
            "frame_action_summary_path": str(summary_path),
            "training_csv_path": str(csv_path),
        },
        "frames": {
            "total": int(total_frames),
            "with_relevant_events": int(nonempty_frames),
            "with_relevant_events_ratio": round(nonempty_frames / max(total_frames, 1), 6),
        },
        "events": {
            "kept_type_counts": dict(type_counts),
            "dropped_type_counts": dict(dropped_counts),
        },
        "metadata": {
            "title": metadata.get("title"),
            "game": metadata.get("game"),
            "duration_ms": metadata.get("total_duration_ms"),
            "declared_event_count": metadata.get("event_count"),
        },
        "examples": top_summaries,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
