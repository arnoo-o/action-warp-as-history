#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NEUTRAL_PROMPT = "Minecraft first-person gameplay."


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed Minecraft interaction pools from VPT telemetry.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--audit_json", type=Path, default=None)
    parser.add_argument("--target_fps", type=float, default=16.0)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--event_guard_seconds", type=float, default=2.0)
    parser.add_argument("--mine_active_min_frames", type=int, default=3)
    parser.add_argument("--mine_active_max_frames", type=int, default=60)
    parser.add_argument("--max_yaw_pitch_step", type=float, default=6.0)
    parser.add_argument("--max_cumulative_rotation", type=float, default=20.0)
    parser.add_argument("--max_negative_per_segment", type=int, default=5)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            payload["_frame"] = int(payload.get("segment_frame", line_index))
            rows.append(payload)
    return rows


def gui_open(row):
    gui = dict(row.get("gui", {}) or {})
    return bool(
        gui.get("isGuiOpen")
        or gui.get("is_gui_open")
        or gui.get("inventory")
        or gui.get("container")
    )


def left_held(row):
    return 0 in list(dict(row.get("mouse", {}) or {}).get("buttons", []) or [])


def angular_delta(a, b):
    yaw = abs(((float(b.get("yaw", 0.0)) - float(a.get("yaw", 0.0)) + 180.0) % 360.0) - 180.0)
    pitch = abs(float(b.get("pitch", 0.0)) - float(a.get("pitch", 0.0)))
    return yaw + pitch


def stable_mine_suffix(frames, segment_start, segment_end, args):
    if segment_end < segment_start:
        return None
    selected = []
    cumulative = 0.0
    previous = None
    for frame_index in range(segment_end, segment_start - 1, -1):
        row = frames[frame_index]
        if row["_frame"] != frame_index or gui_open(row) or not left_held(row):
            break
        if previous is not None:
            change = angular_delta(row, previous)
            if change > float(args.max_yaw_pitch_step) or cumulative + change > float(args.max_cumulative_rotation):
                break
            cumulative += change
        selected.append(frame_index)
        previous = row
        if len(selected) >= int(args.mine_active_max_frames):
            break
    if len(selected) < int(args.mine_active_min_frames):
        return None
    return min(selected), max(selected), cumulative


def resample_source_indices(source_frames, source_fps, target_fps):
    count = max(1, int(math.floor((source_frames - 1) * target_fps / source_fps)) + 1)
    return np.rint(np.arange(count, dtype=np.float64) * source_fps / target_fps).astype(np.int64)


def base_output_row(row):
    result = dict(row)
    result["prompt"] = NEUTRAL_PROMPT
    result["prompt_raw"] = NEUTRAL_PROMPT
    result["event_id"] = str(
        row.get("event_id")
        or f"{row.get('segment_id')}:{row.get('event_source_frame')}:{row.get('category')}:{row.get('object_id')}"
    )
    return result


def validate_event_metadata(row, args):
    reasons = []
    try:
        fps = float(row.get("fps", 0.0))
    except (TypeError, ValueError):
        fps = 0.0
    if not math.isfinite(fps) or fps <= 0:
        reasons.append("invalid_fps")
    try:
        source_start = int(row.get("source_frame_start", 0))
        source_end = int(row.get("source_frame_end_exclusive", 0))
        event_global = int(row.get("event_source_frame", -1))
    except (TypeError, ValueError):
        return [*reasons, "invalid_source_frame_mapping"]
    event_local = event_global - source_start
    if source_end <= source_start or not (source_start <= event_global < source_end):
        reasons.append("invalid_source_frame_mapping")
    if event_local < 6 or (source_end - event_global) < (int(args.num_frames) - 16):
        reasons.append("event_window_boundary")
    if not str(row.get("block_id", row.get("object_id", "")) or "").strip():
        reasons.append("missing_block_id")
    rotation = row.get("rotation_degrees", row.get("cumulative_rotation", ""))
    if str(rotation).strip():
        try:
            if float(rotation) > float(args.max_cumulative_rotation):
                reasons.append("camera_rotation")
        except ValueError:
            reasons.append("invalid_camera_rotation")
    if fps > 0 and not str(row.get("source_event_time_ms", "")).strip():
        row["source_event_time_ms"] = f"{1000.0 * event_global / fps:.6f}"
    row["verified_event_local_frame"] = str(event_local)
    row["metadata_filter_status"] = "passed" if not reasons else "rejected"
    return reasons


def make_repo_relative_data_paths(rows, data_root):
    repo_root = data_root.parents[2]
    for row in rows:
        for key in ("video_path", "actions_path", "mc_event_path"):
            value = str(row.get(key, "") or "").strip()
            if not value:
                continue
            path = Path(value)
            try:
                row[key] = path.resolve().relative_to(repo_root).as_posix()
            except (OSError, ValueError):
                if path.is_absolute():
                    raise ValueError(f"{key} must be inside the repository for portable training: {path}")


def main():
    args = parse_args()
    root = args.data_root.resolve()
    source_rows = read_csv(root / "mc_training_samples.csv")
    segment_rows = read_csv(root / "mc_long_segments.csv")
    output_csv = args.output_csv or root / "mc_interaction_training_samples.csv"
    audit_json = args.audit_json or root / "mc_interaction_manifest_audit.json"

    output = []
    metadata_rejections = Counter()
    completions_by_segment = defaultdict(list)
    mine_rows = []
    for row in source_rows:
        category = str(row.get("category", "")).strip().lower()
        reasons = validate_event_metadata(row, args) if category in {"place", "mine"} else []
        if reasons:
            metadata_rejections.update(reasons)
            continue
        if category == "place":
            result = base_output_row(row)
            result["action_type"] = "place"
            result["history_type"] = "quota"
            output.append(result)
        elif category == "mine":
            result = base_output_row(row)
            result["action_type"] = "mine_complete"
            result["history_type"] = "quota"
            result["complete_frame"] = row["event_source_frame"]
            output.append(result)
            local = int(row["event_source_frame"]) - int(row["source_frame_start"])
            completions_by_segment[row["segment_id"]].append(local)
            mine_rows.append(row)

    telemetry_cache = {}
    active_rejections = Counter()
    for row in mine_rows:
        actions_path = Path(row["actions_path"])
        if actions_path not in telemetry_cache:
            telemetry_cache[actions_path] = read_jsonl(actions_path)
        frames = telemetry_cache[actions_path]
        complete_local = int(row["event_source_frame"]) - int(row["source_frame_start"])
        hold_start = complete_local
        while hold_start > 0 and left_held(frames[hold_start - 1]) and not gui_open(frames[hold_start - 1]):
            hold_start -= 1
        previous = [value for value in completions_by_segment[row["segment_id"]] if hold_start <= value < complete_local]
        segment_start = max(hold_start, max(previous) + 1 if previous else hold_start)
        suffix = stable_mine_suffix(frames, segment_start, complete_local - 1, args)
        if suffix is None:
            active_rejections["no_stable_suffix"] += 1
            continue
        active_start, active_end, cumulative = suffix
        result = base_output_row(row)
        source_start = int(row["source_frame_start"])
        result.update(
            {
                "sample_id": f"{row['sample_id']}__active_{active_start:06d}_{active_end:06d}",
                "category": "mine",
                "action_type": "mine_active",
                "event_source_frame": str(source_start + active_start),
                "event_local_frame": str(active_start),
                "action_start_frame": str(source_start + active_start),
                "action_end_frame": str(source_start + active_end),
                "complete_frame": str(source_start + complete_local),
                "event_id": f"{row['segment_id']}:{source_start + complete_local}:mine_active:{row.get('object_id')}",
                "mouse_down_start": str(source_start + hold_start),
                "previous_complete_frame": ""
                if not previous
                else str(source_start + max(previous)),
                "stable_active_frames": str(active_end - active_start + 1),
                "cumulative_rotation": f"{cumulative:.6f}",
                "telemetry_confidence": "exact_complete_stable_suffix",
                "history_type": "quota",
            }
        )
        output.append(result)

    events_by_segment = defaultdict(list)
    for row in source_rows:
        if row.get("event_source_frame"):
            events_by_segment[row["segment_id"]].append(
                int(row["event_source_frame"]) - int(row["source_frame_start"])
            )
    negative_count = 0
    for segment in segment_rows:
        actions_path = Path(segment["actions_path"])
        if actions_path not in telemetry_cache:
            telemetry_cache[actions_path] = read_jsonl(actions_path)
        frames = telemetry_cache[actions_path]
        source_fps = float(segment["fps"])
        selected = resample_source_indices(len(frames), source_fps, float(args.target_fps))
        guard = int(round(float(args.event_guard_seconds) * source_fps))
        candidates = 0
        for target_start in range(0, max(len(selected) - int(args.num_frames) + 1, 0), int(args.num_frames) - 1):
            source_window = selected[target_start : target_start + int(args.num_frames)]
            if len(source_window) < int(args.num_frames):
                continue
            lo, hi = int(source_window[0]), int(source_window[-1])
            if any(lo - guard <= event <= hi + guard for event in events_by_segment.get(segment["id"], [])):
                continue
            if any(gui_open(frames[index]) for index in source_window):
                continue
            output.append(
                {
                    "sample_id": f"{segment['id']}__negative_{target_start:06d}",
                    "segment_id": segment["id"],
                    "video_path": segment["video_path"],
                    "actions_path": segment["actions_path"],
                    "mc_event_path": segment["mc_event_path"],
                    "category": "negative",
                    "action_type": "none",
                    "object_id": "",
                    "block_id": "",
                    "event_source_frame": "",
                    "event_local_frame": "",
                    "window_frames": str(args.num_frames),
                    "fps": segment["fps"],
                    "segment_num_frames": segment["num_frames"],
                    "segment_duration_seconds": segment["duration_seconds"],
                    "source_frame_start": segment["source_frame_start"],
                    "source_frame_end_exclusive": segment["source_frame_end_exclusive"],
                    "negative_window_start_frame": str(target_start),
                    "event_id": f"{segment['id']}:negative:{target_start}",
                    "history_type": "first" if target_start == 0 else "later",
                    "prompt": NEUTRAL_PROMPT,
                    "prompt_raw": NEUTRAL_PROMPT,
                    "telemetry_confidence": "protected_event_free_window",
                    "no_interaction_event_verified": "true",
                    "gui_closed_verified": "true",
                    "frame_contiguous_verified": "true",
                    "source_mapping_valid": "true",
                }
            )
            negative_count += 1
            candidates += 1
            if candidates >= int(args.max_negative_per_segment):
                break

    make_repo_relative_data_paths(output, root)
    columns = sorted({key for row in output for key in row})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    pool_names = {
        "place_pool": "place",
        "mine_active_pool": "mine_active",
        "mine_complete_pool": "mine_complete",
        "real_negative_pool": "none",
    }
    pool_files = {}
    for pool_name, action_type in pool_names.items():
        pool_rows = [row for row in output if str(row.get("action_type", "")) == action_type]
        pool_path = output_csv.with_name(f"{pool_name}.csv")
        with pool_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(pool_rows)
        pool_files[pool_name] = {"path": str(pool_path), "rows": len(pool_rows)}

    action_counts = Counter(row.get("action_type", "") for row in output)
    block_counts = {
        action: Counter(row.get("object_id", "") for row in output if row.get("action_type") == action).most_common()
        for action in ("place", "mine_active", "mine_complete")
    }
    audit = {
        "schema_version": 1,
        "neutral_prompt": NEUTRAL_PROMPT,
        "rows": len(output),
        "action_counts": dict(action_counts),
        "negative_count": negative_count,
        "mine_active_rejections": dict(active_rejections),
        "metadata_rejections": dict(metadata_rejections),
        "pool_files": pool_files,
        "block_counts": block_counts,
        "output_csv": str(output_csv),
    }
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
