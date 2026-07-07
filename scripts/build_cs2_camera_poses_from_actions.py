#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WAH camera_poses.npz from CS2 parquet/action logs.")
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--jump-height", type=float, default=0.18)
    parser.add_argument("--jump-duration-frames", type=int, default=12)
    parser.add_argument("--jump-smooth-mode", choices=["sine", "triangle"], default="sine")
    parser.add_argument("--crouch-height", type=float, default=0.26)
    parser.add_argument("--crouch-smooth-frames", type=int, default=6)
    parser.add_argument("--move-speed", type=float, default=0.055)
    parser.add_argument("--mouse-yaw-scale", type=float, default=0.0022)
    parser.add_argument("--mouse-pitch-scale", type=float, default=0.0016)
    parser.add_argument("--base-camera-height", type=float, default=1.62)
    return parser.parse_args()


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    yaw_m = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pitch_m = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    return yaw_m @ pitch_m


def jump_curve(frame_offset: int, duration: int, height: float, mode: str) -> float:
    if frame_offset < 0 or frame_offset >= duration:
        return 0.0
    phase = frame_offset / max(duration - 1, 1)
    if mode == "triangle":
        return float(height) * (1.0 - abs(2.0 * phase - 1.0))
    return float(height) * math.sin(math.pi * phase)


def smooth_step(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    args = parse_args()
    parquet_path = args.parquet_path.expanduser().resolve()
    out_dir = (args.output_dir or parquet_path.parent / f"{parquet_path.stem}_camera_poses").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    row = pd.read_parquet(parquet_path).iloc[0]
    frames = [dict(frame) for frame in row["frame_data"]]
    fps = float(args.fps) if float(args.fps) > 0 else float(row.get("fps", 16.0))
    real_pose_available = all(
        key in frames[0] for key in ("position_x", "position_y", "position_z", "rotation_yaw", "rotation_pitch")
    )

    camera_poses = np.zeros((len(frames), 4, 4), dtype=np.float32)
    debug_rows = []
    position = np.zeros(3, dtype=np.float32)
    yaw = 0.0
    pitch = 0.0
    crouch_state = 0.0
    active_jumps: list[int] = []

    if not real_pose_available:
        print("warning: real pose fields not found or incomplete, synthesizing trajectory from actions/mouse deltas.")

    for idx, frame in enumerate(frames):
        actions = str(frame.get("actions", "") or "")
        if "J" in actions and (idx == 0 or "J" not in str(frames[idx - 1].get("actions", "") or "")):
            active_jumps.append(idx)

        if real_pose_available:
            position = np.asarray(
                [
                    float(frame.get("position_x", 0.0) or 0.0),
                    float(frame.get("position_y", 0.0) or 0.0),
                    float(frame.get("position_z", 0.0) or 0.0),
                ],
                dtype=np.float32,
            )
            yaw = float(frame.get("rotation_yaw", 0.0) or 0.0)
            pitch = float(frame.get("rotation_pitch", 0.0) or 0.0)
        else:
            yaw += float(frame.get("mouse_x_delta", 0.0) or 0.0) * float(args.mouse_yaw_scale)
            pitch += float(frame.get("mouse_y_delta", 0.0) or 0.0) * float(args.mouse_pitch_scale)
            pitch = max(-1.2, min(1.2, pitch))
            forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
            right = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
            move = np.zeros(3, dtype=np.float32)
            if "W" in actions:
                move += forward
            if "S" in actions:
                move -= forward
            if "D" in actions:
                move += right
            if "A" in actions:
                move -= right
            if float(np.linalg.norm(move)) > 1e-6:
                move = move / float(np.linalg.norm(move))
            position = position + move * float(args.move_speed)

        jump_offset = sum(
            jump_curve(idx - start, int(args.jump_duration_frames), float(args.jump_height), str(args.jump_smooth_mode))
            for start in active_jumps
        )
        active_jumps = [start for start in active_jumps if idx - start < int(args.jump_duration_frames)]

        target_crouch = 1.0 if "C" in actions else 0.0
        crouch_alpha = 1.0 / max(float(args.crouch_smooth_frames), 1.0)
        crouch_state += (target_crouch - crouch_state) * crouch_alpha
        crouch_offset = -float(args.crouch_height) * smooth_step(crouch_state)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_matrix(yaw, pitch)
        pose[:3, 3] = position.astype(np.float32)
        pose[2, 3] = float(pose[2, 3]) + float(args.base_camera_height) + float(jump_offset) + float(crouch_offset)
        camera_poses[idx] = pose
        debug_rows.append(
            {
                "frame": int(idx),
                "position_x": float(pose[0, 3]),
                "position_y": float(pose[1, 3]),
                "position_z": float(pose[2, 3]),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "jump_offset": float(jump_offset),
                "crouch_offset": float(crouch_offset),
                "actions": actions,
            }
        )

    np.savez(out_dir / "camera_poses.npz", camera_poses=camera_poses, fps=np.asarray(fps, dtype=np.float32))
    (out_dir / "debug_camera_motion.json").write_text(
        json.dumps(
            {
                "parquet_path": str(parquet_path),
                "fps": fps,
                "real_pose_available": bool(real_pose_available),
                "frames": debug_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "debug_trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "position_x", "position_y", "position_z", "yaw", "pitch", "jump_offset", "crouch_offset", "actions"],
        )
        writer.writeheader()
        writer.writerows(debug_rows)
    try:
        import matplotlib.pyplot as plt

        xs = [row["position_x"] for row in debug_rows]
        ys = [row["position_y"] for row in debug_rows]
        zs = [row["position_z"] for row in debug_rows]
        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, label="xy trajectory")
        plt.scatter(xs[:1], ys[:1], c="green", label="start", s=30)
        plt.scatter(xs[-1:], ys[-1:], c="red", label="end", s=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "debug_trajectory.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.plot(zs, label="camera height")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "debug_height.png", dpi=180)
        plt.close()
    except Exception as exc:
        print(f"warning: failed to render trajectory plots: {exc}")

    print(
        json.dumps(
            {
                "event": "camera_poses_built",
                "parquet_path": str(parquet_path),
                "output_dir": str(out_dir),
                "fps": fps,
                "num_frames": len(frames),
                "real_pose_available": bool(real_pose_available),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
