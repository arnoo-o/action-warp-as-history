#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


KEY_W = "key.keyboard.w"
KEY_A = "key.keyboard.a"
KEY_S = "key.keyboard.s"
KEY_D = "key.keyboard.d"
KEY_JUMP = "key.keyboard.space"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a deterministic camera controller from Minecraft VPT telemetry."
    )
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=132)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--event-frame", type=int, default=100)
    parser.add_argument("--minimum-translation", type=float, default=3.0)
    parser.add_argument("--block-id", default="oak_planks")
    parser.add_argument("--include-jump", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stationary", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def wrapped_degrees(delta: float) -> float:
    return (float(delta) + 180.0) % 360.0 - 180.0


def resolve_data_path(repo_root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def robust_median(values: list[float], fallback: float) -> float:
    if not values:
        return float(fallback)
    values_array = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(values_array, [0.05, 0.95])
    trimmed = values_array[(values_array >= low) & (values_array <= high)]
    return float(np.median(trimmed if trimmed.size else values_array))


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def conditioning_frame_for_output_frame(output_frame: int, chunk_frames: int = 33) -> int:
    conditioning_frame = int(output_frame)
    while conditioning_frame - conditioning_frame // int(chunk_frames) < int(output_frame):
        conditioning_frame += 1
    return conditioning_frame


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["rows"]]


def learn_controller(action_paths: list[Path], fps: float) -> dict:
    pure_motion = {"forward": [], "backward": [], "left": [], "right": []}
    yaw_ratios: list[float] = []
    pitch_ratios: list[float] = []
    mouse_dx_values: list[float] = []
    mouse_dy_values: list[float] = []
    jump_profiles: list[np.ndarray] = []
    telemetry_frames = 0
    jump_events = 0

    for path in action_paths:
        frames = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("isGuiOpen", False):
                    frames.append(row)
        telemetry_frames += len(frames)
        if len(frames) < 2:
            continue

        for index in range(1, len(frames)):
            previous = frames[index - 1]
            current = frames[index]
            required = ("xpos", "ypos", "zpos", "yaw", "pitch")
            if not all(key in previous and key in current for key in required):
                continue
            dx = float(current["xpos"]) - float(previous["xpos"])
            dz = float(current["zpos"]) - float(previous["zpos"])
            yaw_radians = math.radians(
                0.5 * (float(previous["yaw"]) + float(current["yaw"]))
            )
            forward = np.asarray([-math.sin(yaw_radians), math.cos(yaw_radians)])
            # Minecraft's positive world axes make camera-right the negative
            # perpendicular of the forward vector used above.
            right = np.asarray([-math.cos(yaw_radians), -math.sin(yaw_radians)])
            horizontal = np.asarray([dx, dz])
            local_forward = float(horizontal @ forward)
            local_right = float(horizontal @ right)

            keys = set((current.get("keyboard") or {}).get("keys") or [])
            wasd = keys & {KEY_W, KEY_A, KEY_S, KEY_D}
            if wasd == {KEY_W} and local_forward > 0:
                pure_motion["forward"].append(local_forward)
            elif wasd == {KEY_S} and local_forward < 0:
                pure_motion["backward"].append(-local_forward)
            elif wasd == {KEY_A} and local_right < 0:
                pure_motion["left"].append(-local_right)
            elif wasd == {KEY_D} and local_right > 0:
                pure_motion["right"].append(local_right)

            mouse = current.get("mouse") or {}
            mouse_dx = float(mouse.get("dx", 0.0) or 0.0)
            mouse_dy = float(mouse.get("dy", 0.0) or 0.0)
            yaw_delta = wrapped_degrees(float(current["yaw"]) - float(previous["yaw"]))
            pitch_delta = float(current["pitch"]) - float(previous["pitch"])
            if abs(mouse_dx) >= 1.0 and abs(yaw_delta) <= 30.0:
                yaw_ratios.append(math.radians(yaw_delta) / mouse_dx)
                mouse_dx_values.append(abs(mouse_dx))
            if abs(mouse_dy) >= 1.0 and abs(pitch_delta) <= 30.0:
                pitch_ratios.append(math.radians(pitch_delta) / mouse_dy)
                mouse_dy_values.append(abs(mouse_dy))

        profile_frames = max(8, int(round(1.25 * fps)))
        previous_jump = False
        for index, frame in enumerate(frames):
            keys = set((frame.get("keyboard") or {}).get("keys") or [])
            pressed = KEY_JUMP in keys
            if pressed and not previous_jump and index + profile_frames <= len(frames):
                baseline = float(frames[index]["ypos"])
                profile = np.asarray(
                    [float(frames[index + offset]["ypos"]) - baseline for offset in range(profile_frames)],
                    dtype=np.float32,
                )
                peak = float(profile.max())
                if 0.35 <= peak <= 2.0 and abs(float(profile[-1])) <= 0.5:
                    jump_profiles.append(profile)
                    jump_events += 1
            previous_jump = pressed

    speed_defaults = {"forward": 0.20, "backward": 0.16, "left": 0.18, "right": 0.18}
    speeds = {
        name: robust_median(values, speed_defaults[name])
        for name, values in pure_motion.items()
    }
    if jump_profiles:
        jump_profile = np.median(np.stack(jump_profiles, axis=0), axis=0).astype(np.float32)
        peak_index = int(jump_profile.argmax())
        landing_index = next(
            (
                index
                for index in range(peak_index + 1, len(jump_profile))
                if abs(float(jump_profile[index])) <= 0.1
            ),
            len(jump_profile) - 1,
        )
        jump_profile = jump_profile[: landing_index + 1]
    else:
        duration = max(8, int(round(1.0 * fps)))
        jump_profile = np.asarray(
            [1.25 * math.sin(math.pi * index / max(duration - 1, 1)) for index in range(duration)],
            dtype=np.float32,
        )
    jump_profile -= jump_profile[0]

    return {
        "fps": float(fps),
        "telemetry_files": len(action_paths),
        "telemetry_frames": telemetry_frames,
        "pure_motion_samples": {name: len(values) for name, values in pure_motion.items()},
        "speed_per_frame": speeds,
        "yaw_radians_per_mouse_dx": robust_median(yaw_ratios, math.radians(0.15)),
        "pitch_radians_per_mouse_dy": robust_median(pitch_ratios, math.radians(0.15)),
        "typical_mouse_dx": robust_median(mouse_dx_values, 4.0),
        "typical_mouse_dy": robust_median(mouse_dy_values, 3.0),
        "jump_events": jump_events,
        "jump_profile": [float(value) for value in jump_profile],
    }


def build_trajectory(
    controller: dict,
    num_frames: int,
    minimum_translation: float,
    event_frame: int,
    include_jump: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], int(num_frames), axis=0)
    pose = np.eye(4, dtype=np.float32)
    speeds = controller["speed_per_frame"]
    schedule: list[dict] = [{"action": "idle", "mouse_dx": 0.0, "mouse_dy": 0.0} for _ in range(num_frames)]

    cursor = 1
    direction_specs = (
        ("forward", 2, 1.0),
        ("backward", 2, -1.0),
        ("left", 0, -1.0),
        ("right", 0, 1.0),
    )
    for action, axis, sign in direction_specs:
        speed = max(float(speeds[action]), float(minimum_translation) / 20.0, 1e-4)
        duration = max(2, int(math.ceil(float(minimum_translation) / speed)))
        for frame in range(cursor, min(cursor + duration, num_frames)):
            schedule[frame]["action"] = action
            schedule[frame]["translation_axis"] = axis
            schedule[frame]["translation_delta"] = sign * speed
        cursor += duration

    mouse_dx = float(controller["typical_mouse_dx"])
    mouse_dy = float(controller["typical_mouse_dy"])
    yaw_start = min(max(cursor, 64), max(int(event_frame) - 28, 1))
    for frame in range(yaw_start, min(yaw_start + 8, num_frames)):
        schedule[frame]["mouse_dx"] = mouse_dx
    for frame in range(yaw_start + 8, min(yaw_start + 16, num_frames)):
        schedule[frame]["mouse_dx"] = -mouse_dx
    for frame in range(yaw_start + 16, min(yaw_start + 22, num_frames)):
        schedule[frame]["mouse_dy"] = -mouse_dy
    for frame in range(yaw_start + 22, min(yaw_start + 28, num_frames)):
        schedule[frame]["mouse_dy"] = mouse_dy

    if include_jump:
        jump_profile = np.asarray(controller["jump_profile"], dtype=np.float32)
        jump_start = min(max(yaw_start, 72), max(num_frames - len(jump_profile) - 1, 1))
        previous_height = 0.0
        for offset, height in enumerate(jump_profile):
            frame = jump_start + offset
            if frame >= num_frames:
                break
            schedule[frame]["jump_delta"] = float(height - previous_height)
            schedule[frame]["jump_height"] = float(height)
            schedule[frame]["action"] = (
                f"{schedule[frame]['action']}+jump"
                if schedule[frame]["action"] != "idle"
                else "jump"
            )
            previous_height = float(height)

    debug = []
    for frame in range(num_frames):
        command = schedule[frame]
        translation = np.zeros(3, dtype=np.float32)
        if "translation_axis" in command:
            translation[int(command["translation_axis"])] = float(command["translation_delta"])
        translation[1] += float(command.get("jump_delta", 0.0))
        yaw_delta = float(command["mouse_dx"]) * float(controller["yaw_radians_per_mouse_dx"])
        pitch_delta = float(command["mouse_dy"]) * float(controller["pitch_radians_per_mouse_dy"])
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = rotation_y(yaw_delta) @ rotation_x(pitch_delta)
        delta[:3, 3] = translation
        if frame > 0:
            pose = (pose @ delta).astype(np.float32)
        poses[frame] = pose
        debug.append(
            {
                "frame": frame,
                "action": command["action"],
                "mouse_dx": command["mouse_dx"],
                "mouse_dy": command["mouse_dy"],
                "position": [float(value) for value in pose[:3, 3]],
            }
        )
    return poses, debug


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.train_config.expanduser().resolve())
    action_paths = sorted(
        {
            resolve_data_path(repo_root, row["actions_path"])
            for row in rows
            if str(row.get("actions_path", "")).strip()
        }
    )
    controller = learn_controller(action_paths, float(args.fps))
    if bool(args.stationary):
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], int(args.num_frames), axis=0)
        debug = [
            {
                "frame": frame,
                "action": "idle",
                "mouse_dx": 0.0,
                "mouse_dy": 0.0,
                "position": [0.0, 0.0, 0.0],
            }
            for frame in range(int(args.num_frames))
        ]
    else:
        poses, debug = build_trajectory(
            controller,
            num_frames=int(args.num_frames),
            minimum_translation=float(args.minimum_translation),
            event_frame=int(args.event_frame),
            include_jump=bool(args.include_jump),
        )
    np.savez(
        output_dir / "camera_poses.npz",
        camera_poses=poses,
        fps=np.asarray(float(args.fps), dtype=np.float32),
    )
    (output_dir / "learned_controller.json").write_text(
        json.dumps(controller, indent=2),
        encoding="utf-8",
    )
    (output_dir / "debug_trajectory.json").write_text(
        json.dumps(debug, indent=2),
        encoding="utf-8",
    )
    conditioning_event_frame = conditioning_frame_for_output_frame(int(args.event_frame))
    event = {
        "event_frame": conditioning_event_frame,
        "requested_output_frame": int(args.event_frame),
        "action_type": "place",
        "object_id": str(args.block_id),
        "block_id": str(args.block_id),
        "event_valid": 1.0,
    }
    (output_dir / "interaction_event.json").write_text(
        json.dumps(event, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "infer.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first_frame_path", "prompt", "camera_poses_path", "interaction_event_path"])
        writer.writerow(
            [
                "first_frame.png",
                "Minecraft gameplay with player movement and camera rotation.",
                "camera_poses.npz",
                "interaction_event.json",
            ]
        )
    summary = {
        "output_dir": str(output_dir),
        "num_frames": int(args.num_frames),
        "fps": float(args.fps),
        "chunks": int(math.ceil(int(args.num_frames) / 33.0)),
        "requested_output_event_frame": int(args.event_frame),
        "conditioning_event_frame": conditioning_event_frame,
        "event_time_seconds": float(args.event_frame) / float(args.fps),
        "include_jump": bool(args.include_jump),
        "stationary": bool(args.stationary),
        "controller": controller,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
