#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a reviewable fixed Minecraft interaction teacher pool from offline candidate NPZ files."
    )
    parser.add_argument("--candidate_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--support_threshold", type=float, default=0.25)
    parser.add_argument("--z_cap", type=float, default=3.0)
    parser.add_argument("--min_area_place", type=float, default=0.001)
    parser.add_argument("--min_area_mine_complete", type=float, default=0.001)
    parser.add_argument("--min_area_mine_active", type=float, default=0.0005)
    parser.add_argument("--max_area", type=float, default=0.25)
    parser.add_argument("--stage0_grid", type=int, nargs=3, default=[9, 6, 10], metavar=("T", "H", "W"))
    return parser.parse_args()


def robust_teacher(target, warp, valid, z_cap):
    residual = np.mean(np.abs(target.astype(np.float32) - warp.astype(np.float32)), axis=0, keepdims=True)
    teacher = np.zeros_like(residual, dtype=np.float32)
    for time_index in range(residual.shape[1]):
        values = residual[0, time_index][valid[0, time_index] > 0]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1.0e-6:
            scale = float(np.quantile(values, 0.90) - np.quantile(values, 0.50))
        if scale < 1.0e-6:
            continue
        teacher[0, time_index] = np.clip(
            np.maximum(residual[0, time_index] - median, 0.0) / (scale * float(z_cap)), 0.0, 1.0
        )
    padded = np.pad(residual, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    local_motion = sum(
        padded[:, :, y : y + residual.shape[2], x : x + residual.shape[3]]
        for y in range(3)
        for x in range(3)
    ) / 9.0
    motion_alignment = np.clip(residual / (residual + local_motion + 1.0e-6), 0.25, 1.0)
    return residual, teacher * motion_alignment * valid


def normalize_map(value, shape, *, conservative=False):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[None]
    if value.shape != shape:
        raise ValueError(f"Offline candidates must already share one latent grid: {value.shape} != {shape}")
    return (value >= 0.999).astype(np.float32) if conservative else np.clip(value, 0.0, 1.0)


def save_preview(path, residual, teacher):
    residual_2d = residual.max(axis=(0, 1))
    teacher_2d = teacher.max(axis=(0, 1))
    residual_2d = residual_2d / max(float(residual_2d.max()), 1.0e-6)
    rgb = np.stack([residual_2d, teacher_2d, np.zeros_like(teacher_2d)], axis=-1)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def adaptive_max_pool_3d(value, output_size):
    output = np.zeros((value.shape[0], *output_size), dtype=value.dtype)
    for t in range(output_size[0]):
        t0 = int(np.floor(t * value.shape[1] / output_size[0]))
        t1 = max(int(np.ceil((t + 1) * value.shape[1] / output_size[0])), t0 + 1)
        for y in range(output_size[1]):
            y0 = int(np.floor(y * value.shape[2] / output_size[1]))
            y1 = max(int(np.ceil((y + 1) * value.shape[2] / output_size[1])), y0 + 1)
            for x in range(output_size[2]):
                x0 = int(np.floor(x * value.shape[3] / output_size[2]))
                x1 = max(int(np.ceil((x + 1) * value.shape[3] / output_size[2])), x0 + 1)
                output[:, t, y, x] = value[:, t0:t1, y0:y1, x0:x1].max(axis=(1, 2, 3))
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.candidate_manifest.open("r", encoding="utf-8-sig", newline="")))
    output = []
    rejection_counts = Counter()
    recipe = {
        "support_threshold": args.support_threshold,
        "z_cap": args.z_cap,
        "max_area": args.max_area,
        "min_area": {
            "place": args.min_area_place,
            "mine_complete": args.min_area_mine_complete,
            "mine_active": args.min_area_mine_active,
        },
        "teacher": "per-time median/MAD residual with quantile fallback",
    }
    for row in rows:
        candidate_path = Path(row["teacher_candidate_path"])
        payload = np.load(candidate_path)
        target = payload["target_latents"]
        warp = payload["warp_latents"]
        if target.shape != warp.shape or target.ndim != 4:
            raise ValueError(f"Expected [C,T,H,W] matching latents in {candidate_path}")
        grid_shape = (1, *target.shape[1:])
        action = normalize_map(payload["action_mask"], grid_shape)
        visibility = normalize_map(payload["visibility"], grid_shape)
        world = normalize_map(payload["world_valid"], grid_shape, conservative=True)
        valid = action * visibility * world
        residual, teacher = robust_teacher(target, warp, valid, args.z_cap)
        support = (teacher > float(args.support_threshold)).astype(np.float32)
        stage0_teacher = adaptive_max_pool_3d(teacher, tuple(int(value) for value in args.stage0_grid))
        stage0_positive_tokens = int((stage0_teacher > float(args.support_threshold)).sum())
        valid_count = float(valid.sum())
        area = float((support * valid).sum() / max(valid_count, 1.0))
        action_type = str(row.get("action_type", ""))
        min_area = float(recipe["min_area"].get(action_type, args.min_area_place))
        reasons = []
        if valid_count <= 0:
            reasons.append("empty_valid_region")
        if float(support.sum()) <= 0:
            reasons.append("empty_stage0_positive_tokens")
        if stage0_positive_tokens <= 0 and "empty_stage0_positive_tokens" not in reasons:
            reasons.append("empty_stage0_positive_tokens")
        if area < min_area:
            reasons.append("teacher_too_small")
        if area > float(args.max_area):
            reasons.append("teacher_too_large")
        cache_payload = {
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "recipe": recipe,
            "event_id": row.get("event_id"),
            "history_type": row.get("history_type"),
        }
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
        npz_path = args.output_dir / f"{cache_key}.npz"
        preview_path = args.output_dir / f"{cache_key}_overlay.png"
        np.savez_compressed(
            npz_path,
            residual=residual.astype(np.float16),
            teacher=teacher.astype(np.float16),
            visibility=visibility.astype(np.float16),
            valid=valid.astype(np.float16),
        )
        save_preview(preview_path, residual, teacher)
        for reason in reasons:
            rejection_counts[reason] += 1
        output.append(
            {
                **row,
                "teacher_cache_path": npz_path.as_posix(),
                "teacher_overlay_path": preview_path.as_posix(),
                "teacher_cache_key": cache_key,
                "teacher_area_ratio": f"{area:.8f}",
                "stage0_positive_tokens": str(stage0_positive_tokens),
                "teacher_invalid_reasons": "|".join(reasons),
                "teacher_valid": str(not reasons).lower(),
                "review_status": "pending" if not reasons else "rejected",
            }
        )
    manifest_path = args.output_dir / "teacher_pool_review_manifest.csv"
    columns = sorted({key for row in output for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    (args.output_dir / "teacher_pool_audit.json").write_text(
        json.dumps(
            {
                "candidate_count": len(rows),
                "valid_for_review": sum(not row["teacher_invalid_reasons"] for row in output),
                "rejection_counts": dict(rejection_counts),
                "recipe": recipe,
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
