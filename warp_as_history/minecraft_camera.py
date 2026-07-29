"""Shared Minecraft camera-pose conversion used by training and inference tools."""
from __future__ import annotations

import math

import numpy as np


POSE_CONVENTION = "opencv_c2w_relative"


def wrapped_degrees(delta: float) -> float:
    return (float(delta) + 180.0) % 360.0 - 180.0


def rotation_x(angle_radians: float) -> np.ndarray:
    c, s = math.cos(float(angle_radians)), math.sin(float(angle_radians))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def rotation_y(angle_radians: float) -> np.ndarray:
    c, s = math.cos(float(angle_radians)), math.sin(float(angle_radians))
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def relative_opencv_c2w(poses) -> np.ndarray:
    """Normalize absolute OpenCV c2w poses to the first pose."""
    array = np.asarray(poses, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera poses [T,4,4], got {array.shape}.")
    first_inverse = np.linalg.inv(array[0]).astype(np.float32)
    relative = first_inverse[None] @ array
    relative[:, 3] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return relative.astype(np.float32)


def vpt_rows_to_relative_opencv_c2w(
    pose_rows,
    source_index,
    target_indices,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Convert Minecraft x/y/z/yaw/pitch telemetry to relative OpenCV c2w."""
    source = pose_rows[int(source_index)]
    source_yaw = np.deg2rad(float(source["yaw"]))
    right = np.asarray([-np.cos(source_yaw), 0.0, -np.sin(source_yaw)], dtype=np.float32)
    forward = np.asarray([-np.sin(source_yaw), 0.0, np.cos(source_yaw)], dtype=np.float32)
    source_position = np.asarray(
        [float(source["xpos"]), float(source["ypos"]), float(source["zpos"])],
        dtype=np.float32,
    )
    poses = []
    for target_index in target_indices:
        target = pose_rows[int(target_index)]
        target_position = np.asarray(
            [float(target["xpos"]), float(target["ypos"]), float(target["zpos"])],
            dtype=np.float32,
        )
        delta = target_position - source_position
        yaw = np.deg2rad(wrapped_degrees(float(target["yaw"]) - float(source["yaw"])))
        pitch = np.deg2rad(float(target["pitch"]) - float(source["pitch"]))
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_y(yaw) @ rotation_x(pitch)
        pose[:3, 3] = (
            np.asarray([np.dot(delta, right), delta[1], np.dot(delta, forward)], dtype=np.float32)
            * float(translation_scale)
        )
        poses.append(pose)
    return np.stack(poses, axis=0)


def integrate_local_camera_deltas(commands) -> np.ndarray:
    """Integrate local translation/yaw/pitch deltas into relative OpenCV c2w poses."""
    commands = list(commands)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(commands), axis=0)
    pose = np.eye(4, dtype=np.float32)
    for frame, command in enumerate(commands):
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = rotation_y(float(command.get("yaw_delta", 0.0))) @ rotation_x(
            float(command.get("pitch_delta", 0.0))
        )
        delta[:3, 3] = np.asarray(command.get("translation", (0.0, 0.0, 0.0)), dtype=np.float32)
        if frame > 0:
            pose = (pose @ delta).astype(np.float32)
        poses[frame] = pose
    return relative_opencv_c2w(poses)


def effective_translation_scale(
    base_scale: float,
    median_scene_depth: float,
    *,
    multiply_by_depth: bool,
) -> float:
    scale = float(base_scale)
    if bool(multiply_by_depth):
        scale *= float(median_scene_depth)
    return scale


def pose_motion_statistics(raw_poses, rendered_poses=None) -> dict:
    raw = relative_opencv_c2w(raw_poses)
    rendered = raw if rendered_poses is None else relative_opencv_c2w(rendered_poses)
    raw_norm = np.linalg.norm(raw[:, :3, 3], axis=1)
    rendered_norm = np.linalg.norm(rendered[:, :3, 3], axis=1)
    traces = np.trace(raw[:, :3, :3], axis1=1, axis2=2)
    rotation_degrees = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))
    return {
        "raw_translation_norm": float(raw_norm.max(initial=0.0)),
        "rendered_translation_norm": float(rendered_norm.max(initial=0.0)),
        "rotation_degrees": float(rotation_degrees.max(initial=0.0)),
    }
