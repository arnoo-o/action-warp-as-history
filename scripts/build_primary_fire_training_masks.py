#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build primary-fire residual/time/loss masks from GT and warp videos.")
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--warp-video", type=Path, required=True)
    parser.add_argument("--warp-visibility-mask", type=Path, required=True)
    parser.add_argument("--primary-fire-event", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-fire-frames", type=int, default=16)
    parser.add_argument("--post-fire-frames", type=int, default=24)
    parser.add_argument("--residual-threshold", type=float, default=0.10)
    parser.add_argument("--visibility-threshold", type=float, default=0.35)
    parser.add_argument("--temporal-min-neighbors", type=int, default=2)
    return parser.parse_args()


def read_video(path: Path) -> tuple[np.ndarray, int]:
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    return np.asarray(frames), int(round(float(meta.get("fps") or 16)))


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame.astype(np.uint8))


def temporal_consistency(mask: np.ndarray, min_neighbors: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        start = max(0, t - 1)
        end = min(mask.shape[0], t + 2)
        support = mask[start:end].sum(axis=0)
        out[t] = (support >= float(min_neighbors)).astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_frames, fps = read_video(args.target_video.expanduser().resolve())
    warp_frames, _ = read_video(args.warp_video.expanduser().resolve())
    vis_frames, _ = read_video(args.warp_visibility_mask.expanduser().resolve())
    event_payload = json.loads(args.primary_fire_event.expanduser().resolve().read_text(encoding="utf-8"))
    if len(target_frames) != len(warp_frames) or len(target_frames) != len(vis_frames):
        raise ValueError("target, warp, and visibility videos must have the same frame count.")

    num_frames = len(target_frames)
    if event_payload.get("time_mask") is not None:
        time_mask = np.asarray(event_payload.get("time_mask", []), dtype=np.float32)
        if time_mask.shape[0] != num_frames:
            raise ValueError("primary_fire_event time_mask length must match video frame count.")
    else:
        click_frames_local = [int(x) for x in event_payload.get("click_frames_local", [])]
        time_mask = np.zeros((num_frames,), dtype=np.float32)
        for click in click_frames_local:
            start = max(0, int(click) - int(args.pre_fire_frames))
            end = min(num_frames, int(click) + int(args.post_fire_frames) + 1)
            time_mask[start:end] = 1.0

    target = target_frames.astype(np.float32) / 255.0
    warp = warp_frames.astype(np.float32) / 255.0
    visibility = vis_frames.astype(np.float32)
    if visibility.ndim == 4:
        visibility = visibility.mean(axis=3)
    visibility = visibility / 255.0
    residual = np.abs(target - warp).mean(axis=3)
    visible_mask = (visibility >= float(args.visibility_threshold)).astype(np.float32)
    residual_mask = (residual >= float(args.residual_threshold)).astype(np.float32) * visible_mask
    residual_mask = temporal_consistency(residual_mask, int(args.temporal_min_neighbors))
    loss_mask = residual_mask * time_mask[:, None, None]

    np.save(out_dir / "primary_fire_time_mask.npy", time_mask)
    np.save(out_dir / "primary_fire_residual_mask.npy", residual_mask.astype(np.float32))
    np.save(out_dir / "primary_fire_loss_mask.npy", loss_mask.astype(np.float32))

    residual_rgb = np.repeat((np.clip(residual, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=3)
    mask_rgb = np.repeat((np.clip(loss_mask, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=3)
    overlay = target_frames.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], (loss_mask * 255.0).astype(np.uint8))

    write_video(out_dir / "debug_primary_fire_residual.mp4", residual_rgb, fps=fps)
    write_video(out_dir / "debug_primary_fire_mask.mp4", mask_rgb, fps=fps)
    write_video(out_dir / "debug_primary_fire_overlay.mp4", overlay, fps=fps)

    (out_dir / "primary_fire_mask_report.json").write_text(
        json.dumps(
            {
                "target_video": str(args.target_video),
                "warp_video": str(args.warp_video),
                "warp_visibility_mask": str(args.warp_visibility_mask),
                "primary_fire_event": str(args.primary_fire_event),
                "fps": fps,
                "num_frames": num_frames,
                "time_mask_ratio": float(time_mask.mean()) if num_frames else 0.0,
                "residual_mask_ratio": float(residual_mask.mean()) if residual_mask.size else 0.0,
                "loss_mask_ratio": float(loss_mask.mean()) if loss_mask.size else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "primary_fire_masks_built",
                "output_dir": str(out_dir),
                "num_frames": num_frames,
                "fps": fps,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
