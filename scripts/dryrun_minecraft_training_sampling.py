#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLING_PATH = REPO_ROOT / "warp_as_history" / "minecraft_sampling.py"
SAMPLING_SPEC = importlib.util.spec_from_file_location("wah_minecraft_sampling", SAMPLING_PATH)
SAMPLING = importlib.util.module_from_spec(SAMPLING_SPEC)
SAMPLING_SPEC.loader.exec_module(SAMPLING)
StepCategorySampler = SAMPLING.StepCategorySampler
build_interaction_event_window = SAMPLING.build_interaction_event_window


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run Minecraft step sampling without loading models.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--place-step-ratio", type=float, default=0.5)
    parser.add_argument("--mine-step-ratio", type=float, default=0.3)
    parser.add_argument("--other-step-ratio", type=float, default=0.2)
    parser.add_argument("--interaction-event-local-min", type=int, default=6)
    parser.add_argument("--interaction-event-local-max", type=int, default=16)
    parser.add_argument("--online-first-chunk-prob", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def canonical_category(row):
    category = str(row.get("training_category", row.get("category", "movement"))).strip().lower()
    return category if category in {"place", "mine"} else "other"


def resample_indices(source_fps, target_fps, source_frames):
    count = max(1, int(np.floor((int(source_frames) - 1) * float(target_fps) / float(source_fps))) + 1)
    return np.rint(np.arange(count, dtype=np.float64) * float(source_fps) / float(target_fps)).astype(np.int64)


def prepare_case(row, sampled_category, args, rng):
    source_fps = float(row.get("fps", 0.0) or 0.0)
    source_frames = int(row.get("segment_num_frames", 0) or 0)
    if source_fps <= 0.0 or source_frames <= 0:
        raise ValueError("manifest row lacks fps or segment_num_frames")
    source_indices = resample_indices(source_fps, args.target_fps, source_frames)
    target_frames = len(source_indices)
    if sampled_category in {"place", "mine"}:
        source_start = int(row.get("source_frame_start", 0) or 0)
        source_event = int(row["event_source_frame"])
        segment_event = source_event - source_start
        event_frame = int(np.argmin(np.abs(source_indices - segment_event)))
        target_indices, event_local = build_interaction_event_window(
            event_frame,
            num_source_frames=target_frames,
            window_size=args.num_frames,
            rng=rng,
            local_min=args.interaction_event_local_min,
            local_max=args.interaction_event_local_max,
            require_later=True,
        )
        return {
            "sampled_category": sampled_category,
            "training_category": sampled_category,
            "event_valid": 1,
            "event_local_frame": event_local,
            "first_chunk": False,
            "interaction_payload": {
                "event_frame": event_local,
                "event_valid": 1.0,
                "action_type": "place" if sampled_category == "place" else "mine_complete",
                "block_id": row.get("object_id"),
            },
            "source_event_frame": source_event,
            "source_event_time_ms": 1000.0 * segment_event / source_fps,
            "resampled_event_frame": event_frame,
            "target_indices": target_indices,
        }
    first_chunk = rng.random() < float(args.online_first_chunk_prob)
    if first_chunk:
        target_indices = list(range(args.num_frames))
    else:
        max_start = target_frames - args.num_frames
        if max_start <= 0:
            raise ValueError("movement row has no later window")
        start = rng.randint(1, max_start)
        target_indices = list(range(start, start + args.num_frames))
    return {
        "sampled_category": "other",
        "training_category": "movement",
        "event_valid": 0,
        "event_local_frame": None,
        "first_chunk": first_chunk,
        "interaction_payload": None,
        "target_indices": target_indices,
    }


def main():
    args = parse_args()
    with args.manifest.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pools = {"place": [], "mine": [], "other": []}
    for index, row in enumerate(rows):
        pools[canonical_category(row)].append(index)
    sampler = StepCategorySampler(
        pools,
        {
            "place": args.place_step_ratio,
            "mine": args.mine_step_ratio,
            "other": args.other_step_ratio,
        },
        args.cases,
        args.seed,
    )
    counters = Counter()
    audited = []
    for step in range(args.cases):
        sampled_category, row_index = sampler.sample(step)
        case = None
        for retry in range(args.max_retries):
            candidate = sampler.sample_category(sampled_category, step + retry)
            rng = random.Random(args.seed * 1_000_003 + step * 97 + retry)
            try:
                case = prepare_case(rows[candidate], sampled_category, args, rng)
                row_index = candidate
                counters["invalid_event_retries"] += retry
                break
            except ValueError:
                continue
        if case is None:
            raise RuntimeError(f"Unable to prepare {sampled_category} after {args.max_retries} retries.")
        counters[f"sampled_{sampled_category}_steps"] += 1
        counters[f"valid_{sampled_category}_steps"] += int(case["event_valid"])
        counters["first_chunk_steps"] += int(case["first_chunk"])
        counters["later_chunk_steps"] += int(not case["first_chunk"])
        if sampled_category in {"place", "mine"}:
            if case["event_valid"] != 1 or case["interaction_payload"] is None:
                raise AssertionError("positive interaction lost event payload")
            if case["first_chunk"]:
                raise AssertionError("positive interaction used first chunk")
            if not (
                args.interaction_event_local_min
                <= int(case["event_local_frame"])
                <= args.interaction_event_local_max
            ):
                raise AssertionError("positive event escaped local range")
        elif case["interaction_payload"] is not None:
            raise AssertionError("movement entered interaction Router")
        if len(audited) < 100:
            audited.append({"step": step, "row_index": row_index, **case})

    report = {
        "cases": args.cases,
        "manifest": str(args.manifest),
        "pools": {name: len(values) for name, values in pools.items()},
        "sampler": sampler.report(args.cases),
        "counters": dict(counters),
        "event_audit": audited,
        "assertions": {
            "positive_event_valid": True,
            "positive_no_first_chunk": True,
            "movement_router_bypassed": True,
            "event_local_range": [
                args.interaction_event_local_min,
                args.interaction_event_local_max,
            ],
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
