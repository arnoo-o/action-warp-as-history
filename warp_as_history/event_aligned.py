from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path


def event_aligned_indices(event_frame: int, total_frames: int, window_frames: int = 33):
    """Return the fixed reference/render/target indices for one known event."""
    event_frame = int(event_frame)
    total_frames = int(total_frames)
    window_frames = int(window_frames)
    if event_frame < 1:
        raise ValueError("event-aligned generation requires a reference frame at e-1")
    target_indices = list(range(event_frame, event_frame + window_frames))
    if not target_indices or target_indices[-1] >= total_frames:
        raise ValueError(
            f"event-aligned target [{event_frame}, {event_frame + window_frames}) "
            f"exceeds {total_frames} frames"
        )
    reference_frame_index = event_frame - 1
    return {
        "reference_frame_index": reference_frame_index,
        "target_indices": target_indices,
        "render_pose_indices": [reference_frame_index, *target_indices],
        "event_local_frame": 0,
    }


def group_scripted_events(events: Iterable[dict]):
    """Sort a known action timeline and merge payloads that share one frame."""
    grouped: dict[int, list[dict]] = {}
    for event in events:
        frame = int(event["event_frame"])
        grouped.setdefault(frame, []).append(dict(event))
    return [
        {"event_frame": frame, "events": grouped[frame]}
        for frame in sorted(grouped)
    ]


def scripted_event_segments(events: Iterable[dict], total_frames: int, window_frames: int = 33):
    """Plan fixed-size forwards while committing only up to the next event boundary."""
    grouped = group_scripted_events(events)
    segments = []
    for index, item in enumerate(grouped):
        event_frame = int(item["event_frame"])
        aligned = event_aligned_indices(event_frame, total_frames, window_frames)
        next_event = int(grouped[index + 1]["event_frame"]) if index + 1 < len(grouped) else total_frames
        commit_end = min(event_frame + int(window_frames), next_event, int(total_frames))
        segments.append(
            {
                **aligned,
                "events": list(item["events"]),
                "commit_start_frame": event_frame,
                "commit_end_frame_exclusive": commit_end,
                "committed_frame_count": max(commit_end - event_frame, 0),
            }
        )
    return segments


def validate_generated_cleanup_path(candidate, allowed_paths):
    """Resolve a generated-data cleanup target and enforce an exact whitelist."""
    raw = str(candidate or "").strip()
    if not raw:
        raise ValueError("cleanup target is empty")
    resolved = Path(raw).expanduser().resolve(strict=False)
    if resolved == resolved.parent:
        raise ValueError(f"refusing to clean a filesystem root: {resolved}")
    allowed = {Path(path).expanduser().resolve(strict=False) for path in allowed_paths}
    if resolved not in allowed:
        raise ValueError(f"cleanup target is not allowlisted: {resolved}")
    return resolved


def drop_reference_render_frame(sequence, time_axis: int = 0):
    """Drop the first frame from a 34-frame reference-inclusive render."""
    if int(sequence.shape[time_axis]) < 2:
        raise ValueError("reference-inclusive render must contain at least two frames")
    slices = [slice(None)] * int(sequence.ndim)
    slices[int(time_axis)] = slice(1, None)
    return sequence[tuple(slices)]


def run_scripted_event_timeline(
    initial_reference,
    events: Iterable[dict],
    total_frames: int,
    generate_segment: Callable,
    *,
    window_frames: int = 33,
    oracle_reference_frames=None,
):
    """Run route-three offline inference without leaking future target frames.

    ``generate_segment`` receives ``reference``, ``segment`` and ``window_frames``.
    Standard inference always advances the reference from the last committed generated
    frame.  Oracle references are accepted only through the explicit diagnostic input.
    """
    committed = []
    reference = initial_reference
    oracle_mode = oracle_reference_frames is not None
    for segment in scripted_event_segments(events, total_frames, window_frames):
        if oracle_mode:
            reference = oracle_reference_frames[int(segment["reference_frame_index"])]
        prediction = list(generate_segment(reference, segment, int(window_frames)))
        if len(prediction) != int(window_frames):
            raise ValueError(f"scripted event forward returned {len(prediction)} frames")
        keep = int(segment["committed_frame_count"])
        committed.extend(prediction[:keep])
        if keep:
            reference = committed[-1]
    return {
        "frames": committed,
        "segments": scripted_event_segments(events, total_frames, window_frames),
        "oracle_reference": bool(oracle_mode),
    }
