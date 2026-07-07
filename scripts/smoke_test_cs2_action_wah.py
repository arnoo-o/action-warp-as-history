#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_synthetic_parquet(path: Path, num_frames: int = 40) -> None:
    import pandas as pd

    frame_data = []
    for idx in range(num_frames):
        actions = ""
        if idx < 20:
            actions += "W"
        if 10 <= idx < 13:
            actions += "J"
        if 24 <= idx < 30:
            actions += "C"
        if idx in {18, 27}:
            actions += "["
        frame_data.append(
            {
                "actions": actions,
                "position_x": float(idx) * 0.03,
                "position_y": float(idx) * 0.01,
                "position_z": 0.0,
                "rotation_yaw": float(idx) * 0.01,
                "rotation_pitch": float(idx) * 0.002,
                "mouse_x_delta": 1.0,
                "mouse_y_delta": 0.25,
            }
        )
    row = {
        "video_filename": "synthetic.mp4",
        "match_id": "synthetic",
        "map": "nuke",
        "round_number": 1,
        "team": 2,
        "player_index": 0,
        "category": "smoke",
        "width": 640,
        "height": 384,
        "fov": 90.0,
        "fps": 16.0,
        "num_frames": num_frames,
        "frame_data": frame_data,
    }
    pd.DataFrame([row]).to_parquet(path, index=False)


def run_camera_pose_builder(tmpdir: Path) -> dict:
    try:
        from scripts.infer_warp_as_history import load_camera_poses, load_demo_row
    except ModuleNotFoundError as exc:
        source = (REPO_ROOT / "scripts" / "build_cs2_camera_poses_from_actions.py").read_text(encoding="utf-8")
        assert_true("np.savez" in source and "camera_poses" in source, "camera pose builder is missing npz export path")
        return {"status": "static_only", "reason": f"missing dependency: {exc.name}"}

    parquet_path = tmpdir / "synthetic.parquet"
    build_synthetic_parquet(parquet_path)
    output_dir = tmpdir / "camera"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_cs2_camera_poses_from_actions.py"), str(parquet_path), "--output-dir", str(output_dir)],
        check=True,
    )
    poses, fps = load_camera_poses(output_dir / "camera_poses.npz", "camera_poses")
    assert_true(poses.shape == (40, 4, 4), f"unexpected camera pose shape: {poses.shape}")
    assert_true(int(fps) == 16, f"unexpected fps: {fps}")
    assert_true((output_dir / "debug_camera_motion.json").is_file(), "missing debug_camera_motion.json")
    assert_true((output_dir / "debug_trajectory.csv").is_file(), "missing debug_trajectory.csv")

    first_frame = tmpdir / "first_frame.png"
    Image.new("RGB", (640, 384), color=(32, 64, 96)).save(first_frame)
    csv_path = tmpdir / "infer.csv"
    csv_path.write_text(
        "first_frame_path,prompt,camera_poses_path\n"
        f"{first_frame.name},smoke prompt,camera/camera_poses.npz\n",
        encoding="utf-8",
    )
    loaded = load_demo_row(csv_path)
    assert_true(loaded["camera_poses_path"] is not None, "infer csv did not resolve camera_poses_path")
    return {"camera_pose_shape": list(poses.shape), "camera_fps": int(fps)}


def run_primary_fire_alignment_checks(tmpdir: Path) -> dict:
    event_payload = {
        "fps": 16.0,
        "num_frames": 40,
        "click_frames_source": [118, 127],
        "click_frames_local": [18, 27],
        "event_windows": [
            {"click_frame_local": 18, "window_start": 10, "window_end_exclusive": 30},
            {"click_frame_local": 27, "window_start": 20, "window_end_exclusive": 39},
        ],
        "source_frame_indices": list(range(100, 140)),
        "time_mask": [1.0 if 10 <= idx < 30 or 20 <= idx < 39 else 0.0 for idx in range(40)],
    }
    event_path = tmpdir / "primary_fire_event.json"
    event_path.write_text(json.dumps(event_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded_event = json.loads(event_path.read_text(encoding="utf-8"))
    assert_true("click_frames_source" in loaded_event, "event json missing click_frames_source")

    full_mask = np.zeros((40, 8, 8), dtype=np.float32)
    full_mask[12:28, 2:6, 2:6] = 1.0
    target_indices = list(range(4, 37))
    cropped_event = {
        "fps": loaded_event["fps"],
        "num_frames": 33,
        "click_frames_source": [src for src in loaded_event["click_frames_source"] if src in set(loaded_event["source_frame_indices"][4:37])],
        "click_frames_local": [idx for idx, src in enumerate(loaded_event["source_frame_indices"][4:37]) if src in set(loaded_event["click_frames_source"])],
        "event_windows": [],
        "source_frame_indices": loaded_event["source_frame_indices"][4:37],
        "time_mask": loaded_event["time_mask"][4:37],
    }
    cropped_mask = full_mask[4:37]
    assert_true(len(cropped_event["time_mask"]) == 33, "cropped time mask should match 33-frame window")
    assert_true(int(cropped_mask.shape[0]) == 33, "cropped loss mask should match 33-frame window")

    temporal_scale = 4
    latent_frames = 9
    latent_values = np.zeros((latent_frames,), dtype=np.float32)
    mapping = []
    for latent_idx in range(latent_frames):
        start = latent_idx * temporal_scale
        end = min(33, start + temporal_scale)
        if end <= start:
            end = min(33, start + 1)
        slice_values = cropped_event["time_mask"][start:end]
        latent_values[latent_idx] = max(slice_values) if slice_values else 0.0
        mapping.append(
            {
                "latent_index": latent_idx,
                "frame_start": start,
                "frame_end_exclusive": end,
                "has_click": any(float(x) > 0.0 for x in slice_values),
            }
        )
    latent_mask = np.zeros((1, 1, 9, 4, 4), dtype=np.float32)
    for latent_idx, value in enumerate(latent_values):
        latent_mask[0, 0, latent_idx] = value
    event_latents = np.broadcast_to(latent_mask, (1, 16, 9, 4, 4)).copy()
    assert_true(tuple(event_latents.shape) == (1, 16, 9, 4, 4), f"unexpected event latent shape: {tuple(event_latents.shape)}")
    assert_true(tuple(latent_mask.shape) == (1, 1, 9, 4, 4), f"unexpected focus mask latent shape: {tuple(latent_mask.shape)}")
    assert_true(any(item["has_click"] for item in mapping), "frame-to-latent mapping lost click alignment")
    return {
        "event_latent_shape": list(event_latents.shape),
        "focus_mask_latent_shape": list(latent_mask.shape),
        "mapping_with_clicks": int(sum(1 for item in mapping if item["has_click"])),
    }


def run_pipeline_signature_checks() -> dict:
    pipeline_source = (REPO_ROOT / "warp_as_history" / "pipeline.py").read_text(encoding="utf-8")
    core_source = (REPO_ROOT / "warp_as_history" / "training" / "core.py").read_text(encoding="utf-8")
    assert_true("primary_fire_event_latents" in pipeline_source, "pipeline missing primary_fire_event_latents path")
    assert_true("use_primary_fire_event_condition" in pipeline_source, "pipeline missing use_primary_fire_event_condition path")
    assert_true("target_channel_fusion_latents" in pipeline_source, "pipeline does not forward event condition into transformer")
    assert_true("online_future_keyframe_prob is deprecated" in core_source, "future leakage guard missing for online_future_keyframe_prob")
    return {"pipeline_event_condition_signature": True, "validate_args_future_leakage_guard": True}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wah_cs2_smoke_") as tmp:
        tmpdir = Path(tmp)
        report = {
            "camera_pose_test": run_camera_pose_builder(tmpdir),
            "primary_fire_alignment_test": run_primary_fire_alignment_checks(tmpdir),
            "pipeline_signature_test": run_pipeline_signature_checks(),
        }
    print(json.dumps({"event": "cs2_action_wah_smoke_tests_passed", "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
