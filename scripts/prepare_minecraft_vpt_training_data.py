#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "vpt_9x_100"
DEFAULT_OUTPUT = DEFAULT_INPUT / "wah_mc_training"

MOVEMENT_KEYS = {
    "key.keyboard.w",
    "key.keyboard.a",
    "key.keyboard.s",
    "key.keyboard.d",
    "key.keyboard.space",
    "key.keyboard.left.shift",
    "key.keyboard.left.control",
}

# Keep this precision-oriented. Ambiguous use_item stats are excluded instead of
# turning food, tools, buckets, or GUI interactions into false block placements.
NON_PLACEABLE_EXACT = {
    "air",
    "apple",
    "baked_potato",
    "beef",
    "beetroot",
    "beetroot_soup",
    "bow",
    "bread",
    "carrot",
    "chicken",
    "cod",
    "cooked_beef",
    "cooked_chicken",
    "cooked_cod",
    "cooked_mutton",
    "cooked_porkchop",
    "cooked_rabbit",
    "cooked_salmon",
    "crossbow",
    "egg",
    "ender_pearl",
    "fishing_rod",
    "flint_and_steel",
    "golden_apple",
    "golden_carrot",
    "honey_bottle",
    "map",
    "melon_slice",
    "milk_bucket",
    "mushroom_stew",
    "mutton",
    "poisonous_potato",
    "porkchop",
    "potato",
    "pufferfish",
    "pumpkin_pie",
    "rabbit",
    "rabbit_stew",
    "rotten_flesh",
    "salmon",
    "shears",
    "shield",
    "snowball",
    "spider_eye",
    "suspicious_stew",
    "sweet_berries",
    "trident",
    "tropical_fish",
    "water_bucket",
    "writable_book",
}
NON_PLACEABLE_SUFFIXES = (
    "_axe",
    "_boots",
    "_chestplate",
    "_helmet",
    "_hoe",
    "_leggings",
    "_pickaxe",
    "_shovel",
    "_sword",
)
PLACEABLE_EXACT = {
    "anvil",
    "barrel",
    "beacon",
    "bedrock",
    "bell",
    "blast_furnace",
    "bookshelf",
    "brewing_stand",
    "campfire",
    "cartography_table",
    "cauldron",
    "chest",
    "clay",
    "cobweb",
    "composter",
    "crafting_table",
    "dirt",
    "dispenser",
    "dropper",
    "enchanting_table",
    "end_rod",
    "farmland",
    "fletching_table",
    "furnace",
    "glass",
    "glowstone",
    "grass_block",
    "gravel",
    "grindstone",
    "hay_block",
    "hopper",
    "ice",
    "jukebox",
    "ladder",
    "lantern",
    "lectern",
    "lever",
    "loom",
    "magma_block",
    "note_block",
    "observer",
    "piston",
    "podzol",
    "redstone_torch",
    "redstone_wire",
    "sand",
    "scaffolding",
    "slime_block",
    "smithing_table",
    "smoker",
    "snow",
    "soul_campfire",
    "soul_lantern",
    "soul_sand",
    "soul_soil",
    "spawner",
    "stone",
    "stonecutter",
    "target",
    "tnt",
    "torch",
    "trapped_chest",
    "tripwire_hook",
    "water",
}
PLACEABLE_SUFFIXES = (
    "_banner",
    "_bed",
    "_button",
    "_carpet",
    "_concrete",
    "_concrete_powder",
    "_coral",
    "_coral_block",
    "_coral_fan",
    "_door",
    "_fence",
    "_fence_gate",
    "_glass",
    "_glass_pane",
    "_leaves",
    "_log",
    "_planks",
    "_pressure_plate",
    "_rail",
    "_sapling",
    "_shulker_box",
    "_sign",
    "_slab",
    "_stairs",
    "_stem",
    "_trapdoor",
    "_wall",
    "_wool",
)
PLACEABLE_CONTAINS = (
    "bricks",
    "mushroom_block",
    "ore",
    "purpur",
    "quartz_block",
    "terracotta",
)


@dataclass
class Event:
    category: str
    frame: int
    object_id: str = ""
    delta: int = 1
    confidence: str = "exact_stat_delta"
    segment_id: str = ""


@dataclass
class Segment:
    segment_id: str
    source_id: str
    video_path: Path
    actions_path: Path
    fps: float
    start: int
    end: int
    gui_intervals: list[tuple[int, int]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return self.end - self.start

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select GUI-safe Minecraft VPT long segments and balanced WAH training centers."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-segment-seconds", type=float, default=30.0)
    parser.add_argument("--gui-guard-frames", type=int, default=20)
    parser.add_argument("--window-frames", type=int, default=33)
    parser.add_argument("--place-ratio", type=float, default=0.5)
    parser.add_argument("--mine-ratio", type=float, default=0.3)
    parser.add_argument("--movement-ratio", type=float, default=0.2)
    parser.add_argument("--min-event-spacing-frames", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 keeps the largest feasible balanced set.")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--materialize", action="store_true", help="Render selected long MP4/JSONL segment files.")
    parser.add_argument(
        "--materialize-existing",
        action="store_true",
        help="Resume materialization directly from an existing mc_long_segments.csv.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify materialized MP4/JSONL pairs and write verification_report.json.",
    )
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument(
        "--validate-videos",
        action="store_true",
        help="Use ffmpeg to exclude unreadable source videos before balancing.",
    )
    return parser.parse_args()


def is_placeable_item(item_id: str) -> bool:
    item_id = item_id.removeprefix("minecraft:")
    if item_id in NON_PLACEABLE_EXACT or item_id.endswith(NON_PLACEABLE_SUFFIXES):
        return False
    return (
        item_id in PLACEABLE_EXACT
        or item_id.endswith(PLACEABLE_SUFFIXES)
        or any(token in item_id for token in PLACEABLE_CONTAINS)
    )


def stat_deltas(previous: dict, current: dict, prefix: str) -> list[tuple[str, int]]:
    found = []
    for key, value in current.items():
        if not key.startswith(prefix):
            continue
        old = int(previous.get(key, 0) or 0)
        delta = int(value or 0) - old
        if delta > 0:
            found.append((key[len(prefix) :], delta))
    return found


def infer_fps(millis: list[int]) -> float:
    diffs = [b - a for a, b in zip(millis, millis[1:]) if 20 <= b - a <= 200]
    if not diffs:
        return 20.0
    raw = 1000.0 / statistics.median(diffs)
    candidates = (20.0, 25.0, 30.0, 50.0, 60.0)
    nearest = min(candidates, key=lambda value: abs(value - raw))
    return nearest if abs(nearest - raw) / nearest < 0.15 else raw


def true_intervals(values: list[bool]) -> list[tuple[int, int]]:
    intervals = []
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index))
            start = None
    return intervals


def dilate_gui(gui: list[bool], guard: int) -> list[bool]:
    unsafe = [False] * len(gui)
    for start, end in true_intervals(gui):
        left = max(0, start - guard)
        right = min(len(gui), end + guard)
        unsafe[left:right] = [True] * (right - left)
    return unsafe


def safe_runs(unsafe: list[bool], min_frames: int) -> list[tuple[int, int]]:
    return [(start, end) for start, end in true_intervals([not item for item in unsafe]) if end - start >= min_frames]


def select_spaced(events: list[Event], spacing: int, rng: random.Random) -> list[Event]:
    by_segment: dict[str, list[Event]] = {}
    for event in events:
        by_segment.setdefault(event.segment_id, []).append(event)
    selected = []
    for segment_events in by_segment.values():
        segment_events.sort(key=lambda item: item.frame)
        groups: list[list[Event]] = []
        for event in segment_events:
            if not groups or event.frame - groups[-1][-1].frame >= spacing:
                groups.append([event])
            else:
                groups[-1].append(event)
        selected.extend(rng.choice(group) for group in groups)
    return selected


def inspect_pair(video_path: Path, actions_path: Path, args: argparse.Namespace) -> tuple[list[Segment], dict]:
    gui: list[bool] = []
    invalid: list[bool] = []
    millis: list[int] = []
    events: list[Event] = []
    movement_scores: list[float] = []
    previous_stats: dict = {}
    gui_type_counts: Counter[str] = Counter()
    use_item_counts: Counter[str] = Counter()
    rejected_use_item_counts: Counter[str] = Counter()
    invalid_json_lines = 0

    with actions_path.open("r", encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Preserve one action row per video frame. Treat corrupt rows as
                # unsafe boundaries instead of dropping them and shifting alignment.
                invalid_json_lines += 1
                gui.append(False)
                invalid.append(True)
                millis.append(millis[-1] + 50 if millis else 0)
                movement_scores.append(0.0)
                continue
            is_gui = bool(row.get("isGuiOpen", False))
            gui.append(is_gui)
            invalid.append(False)
            millis.append(int(row.get("milli", 0) or 0))
            if is_gui:
                gui_type_counts["inventory" if row.get("isGuiInventory") else "other_gui"] += 1

            stats = row.get("stats") or {}
            if not is_gui:
                for object_id, delta in stat_deltas(
                    previous_stats, stats, "minecraft.mine_block:minecraft."
                ):
                    events.append(Event("mine", frame, object_id, delta))
                for object_id, delta in stat_deltas(
                    previous_stats, stats, "minecraft.use_item:minecraft."
                ):
                    use_item_counts[object_id] += delta
                    if is_placeable_item(object_id):
                        events.append(Event("place", frame, object_id, delta))
                    else:
                        rejected_use_item_counts[object_id] += delta
            previous_stats = stats

            keyboard = row.get("keyboard") or {}
            keys = set(keyboard.get("keys") or [])
            mouse = row.get("mouse") or {}
            dx = float(mouse.get("dx", 0.0) or 0.0)
            dy = float(mouse.get("dy", 0.0) or 0.0)
            movement_scores.append(
                float(len(keys & MOVEMENT_KEYS)) + min(4.0, (abs(dx) + abs(dy)) / 12.0)
                if not is_gui
                else 0.0
            )

    fps = infer_fps(millis)
    unsafe = dilate_gui(
        [gui_value or invalid_value for gui_value, invalid_value in zip(gui, invalid)],
        args.gui_guard_frames,
    )
    minimum = int(math.ceil(args.min_segment_seconds * fps))
    runs = safe_runs(unsafe, minimum)
    source_id = actions_path.stem
    half_window = args.window_frames // 2
    segments = []

    for run_index, (start, end) in enumerate(runs):
        segment_id = f"{source_id}__s{run_index:03d}_{start:06d}_{end:06d}"
        segment = Segment(segment_id, source_id, video_path, actions_path, fps, start, end)
        for event in events:
            if start + half_window <= event.frame < end - half_window:
                event.segment_id = segment_id
                segment.events.append(event)

        occupied = [False] * (end - start)
        for event in segment.events:
            left = max(start, event.frame - args.gui_guard_frames)
            right = min(end, event.frame + args.gui_guard_frames + 1)
            occupied[left - start : right - start] = [True] * (right - left)
        stride = max(args.window_frames, int(round(fps)))
        for frame in range(start + half_window, end - half_window, stride):
            local = frame - start
            if movement_scores[frame] > 0 and not occupied[local]:
                segment.events.append(
                    Event("movement", frame, "", 1, "keyboard_or_camera_motion", segment_id)
                )
        segments.append(segment)

    audit = {
        "source_id": source_id,
        "frames": len(gui),
        "fps": fps,
        "duration_seconds": len(gui) / fps,
        "raw_gui_intervals": len(true_intervals(gui)),
        "raw_gui_frames": sum(gui),
        "invalid_json_lines": invalid_json_lines,
        "unsafe_frames_with_guard": sum(unsafe),
        "safe_long_segments": len(segments),
        "safe_long_segment_seconds": sum(item.duration for item in segments),
        "use_item_stat_counts": dict(use_item_counts),
        "rejected_non_placeable_use_item_counts": dict(rejected_use_item_counts),
        "gui_type_frame_counts": dict(gui_type_counts),
    }
    return segments, audit


def balanced_selection(segments: list[Segment], args: argparse.Namespace) -> tuple[list[Event], dict]:
    rng = random.Random(args.seed)
    ratios = {
        "place": args.place_ratio,
        "mine": args.mine_ratio,
        "movement": args.movement_ratio,
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("Sampling ratios must sum to a positive value.")
    ratios = {key: value / ratio_sum for key, value in ratios.items()}
    pools = {}
    for category in ratios:
        events = [event for segment in segments for event in segment.events if event.category == category]
        pools[category] = select_spaced(events, args.min_event_spacing_frames, rng)
        rng.shuffle(pools[category])

    feasible_total = min(
        int(len(pools[category]) / ratio)
        for category, ratio in ratios.items()
        if ratio > 0
    )
    if args.max_samples > 0:
        feasible_total = min(feasible_total, args.max_samples)
    counts = {category: int(round(feasible_total * ratio)) for category, ratio in ratios.items()}
    difference = feasible_total - sum(counts.values())
    if difference:
        priority = sorted(ratios, key=ratios.get, reverse=True)
        counts[priority[0]] += difference

    selected = []
    for category, count in counts.items():
        selected.extend(pools[category][:count])
    rng.shuffle(selected)
    return selected, {
        "available_candidates": {key: len(value) for key, value in pools.items()},
        "selected_counts": Counter(event.category for event in selected),
        "requested_ratios": ratios,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_ffmpeg(explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as exc:
        raise RuntimeError(
            "--materialize requires ffmpeg. Pass --ffmpeg PATH or install imageio-ffmpeg."
        ) from exc


def video_is_readable(video_path: Path, ffmpeg: Path) -> bool:
    result = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-t",
            "0.05",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def materialize_segment(segment: Segment, output_dir: Path, ffmpeg: Path) -> tuple[Path, Path]:
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    video_out = segment_dir / f"{segment.segment_id}.mp4"
    actions_out = segment_dir / f"{segment.segment_id}.jsonl"
    if not video_out.exists():
        partial_video = video_out.with_suffix(".partial.mp4")
        if partial_video.exists():
            partial_video.unlink()
        start_seconds = segment.start / segment.fps
        duration_seconds = segment.frames / segment.fps
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(segment.video_path),
            "-t",
            f"{duration_seconds:.6f}",
            "-an",
            "-vf",
            f"fps={segment.fps:g}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(partial_video),
        ]
        subprocess.run(command, check=True)
        partial_video.replace(video_out)
    if not actions_out.exists():
        partial_actions = actions_out.with_suffix(".partial.jsonl")
        if partial_actions.exists():
            partial_actions.unlink()
        with segment.actions_path.open("r", encoding="utf-8") as source, partial_actions.open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for frame, line in enumerate(source):
                if frame >= segment.end:
                    break
                if frame >= segment.start:
                    row = json.loads(line)
                    row["source_frame"] = frame
                    row["segment_frame"] = frame - segment.start
                    target.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        partial_actions.replace(actions_out)
    return video_out, actions_out


def materialize_existing(output_dir: Path, ffmpeg: Path) -> None:
    segment_csv = output_dir / "mc_long_segments.csv"
    sample_csv = output_dir / "mc_training_samples.csv"
    if not segment_csv.exists() or not sample_csv.exists():
        raise FileNotFoundError("--materialize-existing requires existing segment and sample CSV files.")
    with segment_csv.open("r", encoding="utf-8", newline="") as handle:
        segment_rows = list(csv.DictReader(handle))
    path_mapping = {}
    for index, row in enumerate(segment_rows, 1):
        event_payload = json.loads(Path(row["mc_event_path"]).read_text(encoding="utf-8"))
        segment = Segment(
            segment_id=row["id"],
            source_id=row["id"].split("__s", 1)[0],
            video_path=Path(event_payload["source_video_path"]),
            actions_path=Path(event_payload["source_actions_path"]),
            fps=float(row["fps"]),
            start=int(row["source_frame_start"]),
            end=int(row["source_frame_end_exclusive"]),
        )
        print(
            json.dumps(
                {
                    "event": "materializing_segment",
                    "index": index,
                    "total": len(segment_rows),
                    "segment_id": segment.segment_id,
                    "duration_seconds": round(segment.duration, 3),
                }
            ),
            flush=True,
        )
        video_path, actions_path = materialize_segment(segment, output_dir, ffmpeg)
        row["video_path"] = str(video_path)
        row["actions_path"] = str(actions_path)
        row["materialized"] = True
        path_mapping[segment.segment_id] = (str(video_path), str(actions_path))

    write_csv(segment_csv, list(segment_rows[0]), segment_rows)
    with sample_csv.open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    for row in sample_rows:
        row["video_path"], row["actions_path"] = path_mapping[row["segment_id"]]
    write_csv(sample_csv, list(sample_rows[0]), sample_rows)
    report_path = output_dir / "preprocessing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["materialized"] = True
    report["materialized_segments"] = len(segment_rows)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "materialization_complete", "segments": len(segment_rows)}), flush=True)


def verify_existing(output_dir: Path, ffmpeg: Path) -> None:
    with (output_dir / "mc_long_segments.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for index, row in enumerate(rows, 1):
        expected = int(row["num_frames"])
        video_path = Path(row["video_path"])
        actions_path = Path(row["actions_path"])
        result = subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-f",
                "null",
                "-",
                "-progress",
                "pipe:1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        frame_values = [
            int(line.split("=", 1)[1])
            for line in result.stdout.splitlines()
            if line.startswith("frame=")
        ]
        video_frames = frame_values[-1] if frame_values else -1
        action_frames = 0
        gui_frames = 0
        with actions_path.open("r", encoding="utf-8") as action_handle:
            for line in action_handle:
                action_frames += 1
                if bool(json.loads(line).get("isGuiOpen", False)):
                    gui_frames += 1
        if (
            result.returncode != 0
            or video_frames != expected
            or action_frames != expected
            or gui_frames
        ):
            failures.append(
                {
                    "id": row["id"],
                    "expected_frames": expected,
                    "video_frames": video_frames,
                    "action_frames": action_frames,
                    "gui_frames": gui_frames,
                    "ffmpeg_returncode": result.returncode,
                }
            )
        if index % 25 == 0 or index == len(rows):
            print(json.dumps({"event": "verification_progress", "checked": index, "total": len(rows)}), flush=True)
    report = {
        "segments": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "all_video_action_frames_aligned": not failures,
        "all_segments_gui_free": not any(item["gui_frames"] for item in failures),
        "failures": failures,
    }
    (output_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps({"event": "verification_complete", **report}), flush=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.verify_existing:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        verify_existing(output_dir, ffmpeg)
        return
    if args.materialize_existing:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        materialize_existing(output_dir, ffmpeg)
        return
    data_dir = input_dir / "data"
    pairs = []
    rejected_video_paths = []
    validation_ffmpeg = find_ffmpeg(args.ffmpeg) if args.validate_videos else None
    for actions_path in sorted(data_dir.rglob("*.jsonl")):
        video_path = actions_path.with_suffix(".mp4")
        if video_path.exists():
            if validation_ffmpeg is not None and not video_is_readable(video_path, validation_ffmpeg):
                rejected_video_paths.append(str(video_path))
                continue
            pairs.append((video_path, actions_path))
    if not pairs:
        raise FileNotFoundError(f"No .mp4/.jsonl pairs found under {data_dir}")

    all_segments = []
    audits = []
    for index, (video_path, actions_path) in enumerate(pairs, 1):
        segments, audit = inspect_pair(video_path, actions_path, args)
        all_segments.extend(segments)
        audits.append(audit)
        print(
            json.dumps(
                {
                    "event": "source_scanned",
                    "index": index,
                    "total": len(pairs),
                    "source": actions_path.stem,
                    "safe_segments": len(segments),
                }
            ),
            flush=True,
        )

    selected, selection_audit = balanced_selection(all_segments, args)
    selected_segment_ids = {event.segment_id for event in selected}
    selected_segments = [segment for segment in all_segments if segment.segment_id in selected_segment_ids]
    selected_by_segment: dict[str, list[Event]] = {}
    for event in selected:
        selected_by_segment.setdefault(event.segment_id, []).append(event)

    ffmpeg = find_ffmpeg(args.ffmpeg) if args.materialize else None
    segment_rows = []
    event_rows = []
    category_counts = Counter()
    object_counts = {"place": Counter(), "mine": Counter()}

    for segment_index, segment in enumerate(selected_segments, 1):
        video_path = segment.video_path
        actions_path = segment.actions_path
        if ffmpeg is not None:
            print(
                json.dumps(
                    {
                        "event": "materializing_segment",
                        "index": segment_index,
                        "total": len(selected_segments),
                        "segment_id": segment.segment_id,
                        "duration_seconds": round(segment.duration, 3),
                    }
                ),
                flush=True,
            )
            video_path, actions_path = materialize_segment(segment, output_dir, ffmpeg)
        segment_events = sorted(selected_by_segment[segment.segment_id], key=lambda event: event.frame)
        event_path = output_dir / "events" / f"{segment.segment_id}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_payload = {
            "schema_version": 1,
            "segment_id": segment.segment_id,
            "fps": segment.fps,
            "num_frames": segment.frames,
            "source_video_path": str(segment.video_path),
            "source_actions_path": str(segment.actions_path),
            "source_frame_start": segment.start,
            "source_frame_end_exclusive": segment.end,
            "gui_guard_frames": args.gui_guard_frames,
            "contains_gui_frames": False,
            "selected_events": [
                {
                    "category": event.category,
                    "source_frame": event.frame,
                    "local_frame": event.frame - segment.start,
                    "object_id": event.object_id or None,
                    "delta": event.delta,
                    "confidence": event.confidence,
                }
                for event in segment_events
            ],
        }
        event_path.write_text(json.dumps(event_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        segment_rows.append(
            {
                "id": segment.segment_id,
                "video_path": str(video_path),
                "actions_path": str(actions_path),
                "mc_event_path": str(event_path),
                "prompt": "Minecraft first-person survival gameplay with player-controlled camera and movement.",
                "fps": f"{segment.fps:.6f}",
                "num_frames": segment.frames,
                "duration_seconds": f"{segment.duration:.3f}",
                "source_frame_start": segment.start,
                "source_frame_end_exclusive": segment.end,
                "materialized": bool(ffmpeg),
            }
        )
        for event in segment_events:
            category_counts[event.category] += 1
            if event.category in object_counts:
                object_counts[event.category][event.object_id] += 1
            event_rows.append(
                {
                    "sample_id": f"{segment.segment_id}__{event.category}_{event.frame:06d}",
                    "segment_id": segment.segment_id,
                    "video_path": str(video_path),
                    "actions_path": str(actions_path),
                    "mc_event_path": str(event_path),
                    "category": event.category,
                    "object_id": event.object_id,
                    "event_source_frame": event.frame,
                    "event_local_frame": event.frame - segment.start,
                    "window_frames": args.window_frames,
                    "fps": f"{segment.fps:.6f}",
                    "segment_num_frames": segment.frames,
                    "segment_duration_seconds": f"{segment.duration:.3f}",
                    "source_frame_start": segment.start,
                    "source_frame_end_exclusive": segment.end,
                }
            )

    write_csv(
        output_dir / "mc_long_segments.csv",
        list(segment_rows[0]) if segment_rows else [],
        segment_rows,
    )
    write_csv(
        output_dir / "mc_training_samples.csv",
        list(event_rows[0]) if event_rows else [],
        event_rows,
    )
    total = sum(category_counts.values())
    report = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_pairs": len(pairs),
        "rejected_unreadable_videos": rejected_video_paths,
        "all_gui_safe_long_segments": len(all_segments),
        "selected_long_segments": len(selected_segments),
        "selected_long_segment_hours": sum(segment.duration for segment in selected_segments) / 3600.0,
        "minimum_segment_seconds": args.min_segment_seconds,
        "minimum_selected_segment_seconds": min(
            (segment.duration for segment in selected_segments), default=0.0
        ),
        "gui_guard_frames": args.gui_guard_frames,
        "training_samples": total,
        "category_counts": dict(category_counts),
        "category_ratios": {
            key: (value / total if total else 0.0) for key, value in category_counts.items()
        },
        "top_place_objects": object_counts["place"].most_common(50),
        "top_mined_objects": object_counts["mine"].most_common(50),
        "selection": {
            **selection_audit,
            "selected_counts": dict(selection_audit["selected_counts"]),
        },
        "materialized": bool(ffmpeg),
        "source_audit_path": str(output_dir / "source_audit.json"),
        "validation": {
            "all_segments_at_least_minimum_duration": all(
                segment.duration >= args.min_segment_seconds for segment in selected_segments
            ),
            "all_event_windows_inside_segments": all(
                segment.start + args.window_frames // 2 <= event.frame
                and event.frame < segment.end - args.window_frames // 2
                for segment in selected_segments
                for event in selected_by_segment[segment.segment_id]
            ),
            "all_place_and_mine_events_have_object_id": all(
                bool(event.object_id)
                for event in selected
                if event.category in {"place", "mine"}
            ),
            "maximum_ratio_absolute_error": max(
                (
                    abs(category_counts[key] / total - args_ratio)
                    for key, args_ratio in {
                        "place": args.place_ratio,
                        "mine": args.mine_ratio,
                        "movement": args.movement_ratio,
                    }.items()
                ),
                default=0.0,
            )
            if total
            else 0.0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **report}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
