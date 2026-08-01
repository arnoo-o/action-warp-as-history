#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from warp_as_history.training.fixed_teacher import (
    is_negative_action,
    interaction_payload_hash,
    stable_json_hash,
    validate_fixed_artifact_hashes,
    validate_fixed_identity,
    validate_stage0_positive_tokens,
)


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
    parser.add_argument("--review_manifest_name", default="teacher_pool_review_manifest.csv")
    return parser.parse_args()


def robust_normalized_residual(target, source, valid, z_cap):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.ndim == 3 and target.ndim == 4:
        source = np.repeat(source[None], target.shape[0], axis=0)
    if source.shape != target.shape:
        raise ValueError(f"Dual residual sources must match target: {source.shape} != {target.shape}")
    residual = np.mean(np.abs(target - source), axis=-1)
    residual = residual * (np.asarray(valid, dtype=np.float32) > 0).astype(np.float32)
    normalized = np.zeros_like(residual, dtype=np.float32)
    for time_index in range(residual.shape[0]):
        values = residual[time_index][valid[time_index] > 0]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1.0e-6:
            scale = float(np.quantile(values, 0.90) - np.quantile(values, 0.50))
        if scale < 1.0e-6:
            continue
        normalized[time_index] = np.clip(
            np.maximum(residual[time_index] - median, 0.0) / (scale * float(z_cap)), 0.0, 1.0
        )
    return residual, normalized


def _max_filter(value, kernel_size):
    radius = int(kernel_size) // 2
    padded = np.pad(np.asarray(value, dtype=np.float32), ((radius, radius), (radius, radius)), mode="edge")
    return np.maximum.reduce(
        [padded[y : y + value.shape[0], x : x + value.shape[1]] for y in range(kernel_size) for x in range(kernel_size)]
    )


def _sobel_edge(rgb):
    gray = np.asarray(rgb, dtype=np.float32) @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    return np.clip(magnitude / max(float(np.quantile(magnitude, 0.99)), 1.0e-6), 0.0, 1.0)


def old_edge_weights(reference, warp):
    reference_edge = _sobel_edge(reference)
    old_edge = np.stack(
        [_max_filter(np.maximum(reference_edge, _sobel_edge(frame)), 5) for frame in warp],
        axis=0,
    )
    return old_edge, np.clip(1.0 - 0.75 * old_edge, 0.25, 1.0)


def robust_teacher(target, warp, reference, valid, z_cap):
    d_warp, warp_normalized = robust_normalized_residual(target, warp, valid, z_cap)
    d_raw, raw_normalized = robust_normalized_residual(target, reference, valid, z_cap)
    teacher = np.sqrt(np.clip(warp_normalized * raw_normalized, 0.0, 1.0))
    old_edge, edge_weight = old_edge_weights(reference, warp)
    teacher = teacher * edge_weight * valid
    if teacher.shape[0] > 1:
        previous = np.concatenate([teacher[:1], teacher[:-1]], axis=0)
        following = np.concatenate([teacher[1:], teacher[-1:]], axis=0)
        temporal_smoothing = 0.5 * teacher + 0.25 * previous + 0.25 * following
    else:
        temporal_smoothing = teacher
    return d_warp, d_raw, temporal_smoothing * valid, old_edge, edge_weight


def normalize_map(value, shape, *, conservative=False):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[None]
    if value.shape != shape:
        raise ValueError(f"Offline candidates must already share one latent grid: {value.shape} != {shape}")
    return (value >= 0.999).astype(np.float32) if conservative else np.clip(value, 0.0, 1.0)


def minecraft_static_hand_mask(height, width):
    """Return the normalized Minecraft hand/held-item staircase mask."""
    y, x = np.indices((int(height), int(width)))
    y_thresholds = [int(np.ceil(value * int(height))) for value in (0.95, 0.89, 0.80, 0.68, 0.56)]
    x_thresholds = [int(np.ceil(value * int(width))) for value in (0.62, 0.70, 0.78, 0.86, 0.94)]
    return (
        ((y >= y_thresholds[0]) & (x >= x_thresholds[0]))
        | ((y >= y_thresholds[1]) & (x >= x_thresholds[1]))
        | ((y >= y_thresholds[2]) & (x >= x_thresholds[2]))
        | ((y >= y_thresholds[3]) & (x >= x_thresholds[3]))
        | ((y >= y_thresholds[4]) & (x >= x_thresholds[4]))
    )


def binary_dilate(mask, kernel_size, iterations=1):
    result = np.asarray(mask, dtype=bool)
    radius = int(kernel_size) // 2
    for _ in range(int(iterations)):
        horizontal = np.zeros_like(result)
        padded = np.pad(result, ((0, 0), (radius, radius)), mode="constant")
        for offset in range(int(kernel_size)):
            horizontal |= padded[:, offset : offset + result.shape[1]]
        vertical = np.zeros_like(result)
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant")
        for offset in range(int(kernel_size)):
            vertical |= padded[offset : offset + result.shape[0], :]
        result = vertical
    return result


def seed_connected_components(candidate, seed, min_area, max_area):
    candidate = np.asarray(candidate, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    visited = np.zeros_like(candidate)
    selected = np.zeros_like(candidate)
    height, width = candidate.shape
    for start_y, start_x in np.argwhere(candidate & ~visited):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component = []
        touches_seed = False
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            touches_seed = touches_seed or bool(seed[y, x])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and candidate[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if touches_seed and int(min_area) <= len(component) <= int(max_area):
            yy, xx = zip(*component)
            selected[np.asarray(yy), np.asarray(xx)] = True
    return selected


def minecraft_hand_masks(reference_rgb, target_rgb, action_type):
    target = np.asarray(target_rgb, dtype=np.uint8)
    reference = np.asarray(reference_rgb, dtype=np.uint8)
    if target.ndim != 4 or reference.shape != target.shape[1:]:
        raise ValueError(f"Expected reference [H,W,3] and target [T,H,W,3], got {reference.shape}, {target.shape}")
    time, height, width, _ = target.shape
    static = minecraft_static_hand_mask(height, width)
    dynamic = np.zeros((time, height, width), dtype=bool)
    if str(action_type).lower() in {"place", "mine_active", "mine_complete"}:
        search = np.zeros((height, width), dtype=bool)
        search[int(np.ceil(0.40 * height)) :, int(np.ceil(0.50 * width)) :] = True
        seed = static
        maximum_area = int(np.floor(0.05 * height * width))
        reference_float = reference.astype(np.float32)
        for frame_index, frame in enumerate(target):
            residual = np.mean(np.abs(frame.astype(np.float32) - reference_float), axis=2)
            values = residual[search]
            if not values.size:
                continue
            threshold = max(20.0, float(np.quantile(values, 0.92)))
            connected = seed_connected_components(
                (residual >= threshold) & search,
                seed,
                min_area=20,
                max_area=maximum_area,
            )
            dynamic[frame_index] = binary_dilate(connected, 3, iterations=1)
    combined = dynamic | static[None]
    return combined, dynamic


def _area_resize(value, output_height, output_width):
    value = np.asarray(value, dtype=np.float32)
    output = np.zeros((int(output_height), int(output_width)), dtype=np.float32)
    for y in range(int(output_height)):
        y0 = int(np.floor(y * value.shape[0] / output_height))
        y1 = max(int(np.ceil((y + 1) * value.shape[0] / output_height)), y0 + 1)
        for x in range(int(output_width)):
            x0 = int(np.floor(x * value.shape[1] / output_width))
            x1 = max(int(np.ceil((x + 1) * value.shape[1] / output_width)), x0 + 1)
            output[y, x] = float(value[y0:y1, x0:x1].mean())
    return output


def rgb_teacher_to_latent(value, rgb_to_latent, latent_shape):
    value = np.asarray(value, dtype=np.float32)
    mapping = np.asarray(rgb_to_latent, dtype=np.int64).reshape(-1)
    if value.ndim != 3 or len(mapping) != value.shape[0]:
        raise ValueError(f"RGB teacher/mapping mismatch: {value.shape}, {mapping.shape}")
    latent_time, latent_height, latent_width = (int(value) for value in latent_shape)
    temporal = np.zeros((1, latent_time, latent_height, latent_width), dtype=np.float32)
    for frame_index, latent_index in enumerate(mapping):
        if not 0 <= int(latent_index) < latent_time:
            raise ValueError(f"RGB-to-latent mapping points outside latent timeline: {latent_index}")
        spatial = _area_resize(value[frame_index], latent_height, latent_width)
        temporal[0, int(latent_index)] = np.maximum(temporal[0, int(latent_index)], spatial)
    return temporal


def save_preview(path, residual, teacher):
    residual_2d = residual.max(axis=0) if residual.ndim == 3 else residual.max(axis=(0, 1))
    teacher_2d = teacher.max(axis=0) if teacher.ndim == 3 else teacher.max(axis=(0, 1))
    residual_2d = residual_2d / max(float(residual_2d.max()), 1.0e-6)
    rgb = np.stack([residual_2d, teacher_2d, np.zeros_like(teacher_2d)], axis=-1)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _map_image(value, size, *, color):
    value = np.asarray(value, dtype=np.float32)
    value = value / max(float(value.max()), 1.0e-6)
    rgb = np.zeros((*value.shape, 3), dtype=np.float32)
    rgb[..., color] = value
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)


def review_frame_latent_pairs(payload, row):
    target = np.asarray(payload["target_rgb"])
    mapping = np.asarray(payload["rgb_frame_to_latent_index"], dtype=np.int64).reshape(-1)
    if len(mapping) != len(target):
        raise ValueError(
            f"rgb_frame_to_latent_index length {len(mapping)} != target RGB frames {len(target)}"
        )
    event = int(row.get("event_local_frame", 0) or 0)
    if is_negative_action(row.get("action_type", "none")):
        frame_indices = np.linspace(0, len(target) - 1, num=min(7, len(target)), dtype=np.int64).tolist()
    else:
        start = max(event, 0)
        frame_indices = list(range(start, min(start + 7, len(target))))
    return [(frame_index, int(mapping[frame_index])) for frame_index in frame_indices]


def save_review_contact_sheet(
    path,
    payload,
    teacher,
    d_warp,
    d_raw,
    old_edge_rgb,
    edge_weight_rgb,
    hand_mask_rgb,
    row,
    stage0_positive_tokens,
    reasons,
):
    target = np.asarray(payload["target_rgb"], dtype=np.uint8)
    warp = np.asarray(payload["warp_rgb"], dtype=np.uint8)
    reference = np.asarray(
        payload["reference_rgb"] if "reference_rgb" in payload else payload["target_rgb"][0],
        dtype=np.uint8,
    )
    visibility = np.asarray(payload["visibility_rgb"], dtype=np.float32)
    world = np.asarray(payload["world_valid_rgb"], dtype=np.float32)
    event = int(row.get("event_local_frame", 0) or 0)
    frame_latent_pairs = review_frame_latent_pairs(payload, row)
    panel_size = (240, 135)
    labels = (
        "reference RGB", "target RGB", "event-local warp RGB", "D_warp", "D_raw",
        "old edge", "edge weight", "fused teacher", "teacher overlay",
        "visibility / world", "hand / held-item mask",
    )
    header_height = 120
    row_height = panel_size[1] + 24
    sheet = Image.new(
        "RGB",
        (panel_size[0] * len(labels), header_height + row_height * len(frame_latent_pairs)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    summary = (
        f"event_id={row.get('event_id')}  action={row.get('action_type')}  block={row.get('block_id', row.get('object_id'))}\n"
        f"history={row.get('history_type')}  event_local={event}  area={row.get('teacher_area_ratio')}  "
        f"stage0_positive_tokens={stage0_positive_tokens}\n"
        f"reference={row.get('reference_frame_index')}  target_indices={row.get('target_indices')}\n"
        f"telemetry_source={row.get('telemetry_source_event_frame')}  "
        f"visual_start={row.get('visual_start_source_frame')}  reference_source={row.get('reference_source_frame')}\n"
        f"place_click={row.get('place_click_source_frame')}  place_stat={row.get('place_stat_source_frame')}  "
        f"click_to_stat_delay={row.get('place_click_to_stat_delay')}\n"
        f"teacher_source_frames={row.get('teacher_rgb_source_frames')}  "
        f"teacher_resampled={row.get('teacher_resampled_indices')}\n"
        f"hand_area={row.get('hand_mask_area_ratio')}  dynamic_hand_area={row.get('dynamic_hand_mask_area_ratio')}\n"
        f"invalid_reasons={'|'.join(reasons) or 'none'}"
    )
    draw.multiline_text((8, 8), summary, fill="black", font=font, spacing=3)
    for column, label in enumerate(labels):
        draw.text((column * panel_size[0] + 4, header_height - 18), label, fill="black", font=font)
    for row_number, (frame_index, latent_index) in enumerate(frame_latent_pairs):
        target_image = Image.fromarray(target[frame_index], mode="RGB").resize(panel_size)
        warp_image = Image.fromarray(warp[frame_index], mode="RGB").resize(panel_size)
        reference_image = Image.fromarray(reference, mode="RGB").resize(panel_size)
        d_warp_image = _map_image(d_warp[frame_index], panel_size, color=0)
        d_raw_image = _map_image(d_raw[frame_index], panel_size, color=0)
        old_edge_image = _map_image(old_edge_rgb[frame_index], panel_size, color=0)
        edge_weight_image = _map_image(edge_weight_rgb[frame_index], panel_size, color=1)
        teacher_image = _map_image(teacher[frame_index], panel_size, color=1)
        overlay = Image.blend(target_image, _map_image(teacher[frame_index], panel_size, color=0), 0.35)
        valid_rgb = np.stack(
            [
                np.zeros_like(visibility[frame_index]),
                visibility[frame_index],
                world[frame_index],
            ],
            axis=-1,
        )
        valid_image = Image.fromarray((np.clip(valid_rgb, 0.0, 1.0) * 255).astype(np.uint8)).resize(
            panel_size, Image.Resampling.NEAREST
        )
        hand_image = _map_image(np.asarray(hand_mask_rgb[frame_index], dtype=np.float32), panel_size, color=0)
        panels = (
            reference_image, target_image, warp_image, d_warp_image, d_raw_image,
            old_edge_image, edge_weight_image, teacher_image, overlay, valid_image, hand_image,
        )
        top = header_height + row_number * row_height
        draw.text((4, top), f"RGB frame {frame_index} / latent {latent_index}", fill="black", font=font)
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * panel_size[0], top + 20))
    sheet.save(path)


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


def fixed_teacher_statistics(
    teacher,
    action_mask,
    visibility,
    world,
    support_threshold,
    stage0_grid,
    *,
    action_type,
):
    teacher = np.asarray(teacher, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)
    world = np.asarray(world, dtype=np.float32)
    valid_action_region = action_mask * visibility * world
    support = (teacher > float(support_threshold)).astype(np.float32)
    denominator = float(valid_action_region.sum())
    area_ratio = float((support * valid_action_region).sum() / max(denominator, 1.0))
    visibility_denominator = float((action_mask * world).sum())
    visibility_ratio = float(valid_action_region.sum() / max(visibility_denominator, 1.0))
    stage0_teacher = adaptive_max_pool_3d(teacher, tuple(int(value) for value in stage0_grid))
    positive_tokens = int((stage0_teacher > float(support_threshold)).sum())
    if is_negative_action(action_type):
        area_ratio = 0.0
        positive_tokens = 0
    return {
        "valid_action_region": valid_action_region,
        "teacher_support": support,
        "teacher_area_ratio": area_ratio,
        "teacher_visibility_ratio": visibility_ratio,
        "stage0_positive_tokens": positive_tokens,
    }


def rejected_manifest_row(row, reasons):
    return {
        **row,
        "teacher_cache_path": "",
        "teacher_overlay_path": "",
        "teacher_cache_key": "",
        "teacher_area_ratio": "0.00000000",
        "hand_mask_area_ratio": "0.00000000",
        "dynamic_hand_mask_area_ratio": "0.00000000",
        "stage0_positive_tokens": "0",
        "teacher_invalid_reasons": "|".join(str(reason) for reason in reasons),
        "teacher_valid": "false",
        "review_status": "rejected",
        "review_note": "",
        "overfit_selected": "false",
    }


def write_review_index(path, rows):
    groups = (
        "place|first",
        "place|later",
        "mine_active|first",
        "mine_active|later",
        "mine_complete|first",
        "mine_complete|later",
        "negative|first",
        "negative|later",
    )
    grouped = {name: [] for name in groups}
    for row in rows:
        if not str(row.get("teacher_overlay_path", "") or "").strip():
            continue
        action = "negative" if is_negative_action(row.get("action_type")) else str(row.get("action_type", ""))
        key = f"{action}|{row.get('history_type', '')}"
        if key in grouped:
            grouped[key].append(row)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Minecraft teacher review</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f4f1e8;color:#1d261e}",
        "section{margin:28px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #c7c2b5;padding:10px}.card img{width:100%;height:auto}",
        "code{font-size:12px;overflow-wrap:anywhere}.rejected{border-color:#b64632}</style></head><body>",
        "<h1>Minecraft fixed teacher review</h1>",
        "<p>This page shows all candidates with review images. Automated rejected rows remain visible with their warning.</p>",
    ]
    for group in groups:
        entries = grouped[group]
        parts.append(f"<section><h2>{html.escape(group)} ({len(entries)})</h2><div class='grid'>")
        for row in entries:
            overlay = str(row.get("teacher_overlay_path", "") or "")
            image_markup = "<p>No preview generated.</p>"
            if overlay:
                overlay_path = Path(overlay)
                try:
                    overlay = overlay_path.resolve().relative_to(path.parent.resolve()).as_posix()
                except ValueError:
                    overlay = overlay_path.as_posix()
                image_markup = f"<img loading='lazy' src='{html.escape(overlay, quote=True)}'>"
            css_class = "card rejected" if row.get("review_status") == "rejected" else "card"
            metadata = "<br>".join(
                f"{html.escape(label)}: <code>{html.escape(str(row.get(field, '')))}</code>"
                for label, field in (
                    ("event_id", "event_id"),
                    ("review_status", "review_status"),
                    ("action_type", "action_type"),
                    ("block_id", "block_id"),
                    ("history_type", "history_type"),
                    ("teacher_area_ratio", "teacher_area_ratio"),
                    ("stage0_positive_tokens", "stage0_positive_tokens"),
                    ("hand_mask_area_ratio", "hand_mask_area_ratio"),
                    ("dynamic_hand_mask_area_ratio", "dynamic_hand_mask_area_ratio"),
                    ("teacher_invalid_reasons", "teacher_invalid_reasons"),
                )
            )
            parts.append(f"<article class='{css_class}'>{image_markup}<p>{metadata}</p></article>")
        parts.append("</div></section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.candidate_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
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
        "teacher": "RGB dual residual per-frame median/MAD with quantile fallback",
        "old_edge_weight": "dilated Sobel, clip(1-0.75*edge,0.25,1)",
        "temporal_smoothing": "0.5 current + 0.25 previous + 0.25 following",
        "hand_mask": "small_normalized_static+seed_connected_dynamic_all_positive_v2",
        "teacher_rgb_frames": 7,
        "spatial_pooling": "area_average",
        "temporal_pooling": "saved_rgb_to_latent_mapping_pixelwise_max",
    }
    for row in rows:
        precheck_reasons = []
        candidate_error = str(row.get("candidate_error", "") or "").strip()
        candidate_path_text = str(row.get("teacher_candidate_path", "") or "").strip()
        training_cache_text = str(row.get("training_cache_path", "") or "").strip()
        if candidate_error:
            precheck_reasons.append(f"candidate_error:{candidate_error}")
        if not candidate_path_text:
            precheck_reasons.append("missing_teacher_candidate_path")
        candidate_path = Path(candidate_path_text) if candidate_path_text else None
        if candidate_path is not None and not candidate_path.is_file():
            precheck_reasons.append(f"candidate_file_missing:{candidate_path}")
        if not training_cache_text:
            precheck_reasons.append("missing_training_cache_path")
        training_cache_path = Path(training_cache_text) if training_cache_text else None
        if training_cache_path is not None and not training_cache_path.is_file():
            precheck_reasons.append(f"training_cache_file_missing:{training_cache_path}")
        if precheck_reasons:
            rejection_counts.update(precheck_reasons)
            output.append(rejected_manifest_row(row, precheck_reasons))
            continue
        try:
            payload = np.load(candidate_path)
        except (OSError, ValueError) as exc:
            reason = f"candidate_load_error:{type(exc).__name__}:{exc}"
            rejection_counts[reason] += 1
            output.append(rejected_manifest_row(row, [reason]))
            continue
        target = payload["target_latents"]
        warp = payload["warp_latents"]
        reference = payload["reference_latents"] if "reference_latents" in payload else None
        action_type = str(row.get("action_type", "none") or "none").strip().lower()
        is_negative = action_type in {"", "none", "negative"}
        if target.shape != warp.shape or target.ndim != 4:
            payload.close()
            raise ValueError(f"Expected [C,T,H,W] matching latents in {candidate_path}")
        if reference is None:
            if is_negative:
                reference = target[:, :1].copy()
            else:
                reason = "missing_reference_latents"
                rejection_counts[reason] += 1
                payload.close()
                output.append(rejected_manifest_row(row, [reason]))
                continue
        required_rgb = ("target_rgb", "warp_rgb", "visibility_rgb", "world_valid_rgb", "rgb_frame_to_latent_index")
        missing_rgb = [name for name in required_rgb if name not in payload]
        if missing_rgb:
            reason = f"missing_rgb_teacher_inputs:{','.join(missing_rgb)}"
            rejection_counts[reason] += 1
            payload.close()
            output.append(rejected_manifest_row(row, [reason]))
            continue
        target_rgb = np.asarray(payload["target_rgb"], dtype=np.uint8)
        warp_rgb = np.asarray(payload["warp_rgb"], dtype=np.uint8)
        visibility_rgb = np.clip(np.asarray(payload["visibility_rgb"], dtype=np.float32), 0.0, 1.0)
        world_valid_rgb = np.clip(np.asarray(payload["world_valid_rgb"], dtype=np.float32), 0.0, 1.0)
        if target_rgb.shape != warp_rgb.shape or target_rgb.shape[:3] != visibility_rgb.shape:
            raise ValueError(
                f"RGB teacher inputs mismatch target={target_rgb.shape} warp={warp_rgb.shape} "
                f"visibility={visibility_rgb.shape}"
            )
        if world_valid_rgb.shape != visibility_rgb.shape:
            raise ValueError(f"world_valid_rgb mismatch: {world_valid_rgb.shape} != {visibility_rgb.shape}")
        if "reference_rgb" in payload:
            reference_rgb = payload["reference_rgb"]
        elif is_negative:
            reference_rgb = payload["target_rgb"][0]
        else:
            reason = "missing_reference_rgb"
            rejection_counts[reason] += 1
            payload.close()
            output.append(rejected_manifest_row(row, [reason]))
            continue
        hand_mask_rgb, dynamic_hand_mask_rgb = minecraft_hand_masks(
            reference_rgb,
            target_rgb,
            action_type,
        )
        rgb_to_latent = np.asarray(payload["rgb_frame_to_latent_index"], dtype=np.int64)
        action_time_mask_rgb = np.zeros_like(visibility_rgb, dtype=np.float32)
        if not is_negative:
            action_time_mask_rgb[: min(7, len(action_time_mask_rgb))] = 1.0
        valid_rgb = (
            action_time_mask_rgb
            * visibility_rgb
            * world_valid_rgb
            * (1.0 - hand_mask_rgb.astype(np.float32))
        )
        hand_mask_area_ratio = float(hand_mask_rgb.mean())
        dynamic_hand_mask_area_ratio = float(
            dynamic_hand_mask_rgb.reshape(dynamic_hand_mask_rgb.shape[0], -1).mean(axis=1).max()
        )
        payload_json = json.loads(str(payload["interaction_payload_json"].item()))
        event_valid = float(payload_json.get("event_valid", 1.0))
        valid_rgb *= event_valid
        identity = json.loads(str(payload["candidate_identity_json"].item()))
        expected_candidate_key = str(row.get("candidate_cache_key", ""))
        reasons = []
        validate_fixed_identity(row, identity, str(row.get("candidate_config_hash", "")))
        validate_fixed_artifact_hashes(
            row,
            candidate_path=candidate_path,
            training_cache_path=training_cache_path,
        )
        if interaction_payload_hash(payload_json) != str(row.get("interaction_payload_hash", "")):
            raise ValueError(
                f"Candidate payload hash mismatch event_id={row.get('event_id')} "
                f"history_type={row.get('history_type')}"
            )
        if is_negative:
            d_warp = np.mean(np.abs(target_rgb.astype(np.float32) - warp_rgb.astype(np.float32)), axis=-1)
            d_raw = np.mean(np.abs(target_rgb.astype(np.float32) - reference_rgb.astype(np.float32)), axis=-1)
            teacher_rgb = np.zeros_like(d_warp, dtype=np.float32)
            old_edge_rgb, edge_weight_rgb = old_edge_weights(reference_rgb, warp_rgb)
        else:
            d_warp, d_raw, teacher_rgb, old_edge_rgb, edge_weight_rgb = robust_teacher(
                target_rgb, warp_rgb, reference_rgb, valid_rgb, args.z_cap
            )
        teacher = rgb_teacher_to_latent(teacher_rgb, rgb_to_latent, target.shape[1:])
        teacher_action = rgb_teacher_to_latent(action_time_mask_rgb, rgb_to_latent, target.shape[1:])
        visibility = rgb_teacher_to_latent(visibility_rgb, rgb_to_latent, target.shape[1:])
        effective_world = rgb_teacher_to_latent(
            world_valid_rgb * (1.0 - hand_mask_rgb.astype(np.float32)),
            rgb_to_latent,
            target.shape[1:],
        )
        world = rgb_teacher_to_latent(world_valid_rgb, rgb_to_latent, target.shape[1:])
        valid = rgb_teacher_to_latent(valid_rgb, rgb_to_latent, target.shape[1:])
        statistics = fixed_teacher_statistics(
            teacher,
            teacher_action,
            visibility,
            effective_world,
            args.support_threshold,
            args.stage0_grid,
            action_type=action_type,
        )
        support = statistics["teacher_support"]
        stage0_positive_tokens = statistics["stage0_positive_tokens"]
        valid_count = float(statistics["valid_action_region"].sum())
        area = statistics["teacher_area_ratio"]
        min_area = float(recipe["min_area"].get(action_type, args.min_area_place))
        if is_negative:
            target_indices = [int(value) for value in identity.get("target_indices", [])]
            if float(np.asarray(payload["world_valid"]).sum()) <= 0:
                reasons.append("empty_world_valid")
            if len(target_indices) != int(row.get("window_frames", 33) or 33):
                reasons.append("invalid_target_window")
            if target_indices and target_indices != list(range(target_indices[0], target_indices[0] + len(target_indices))):
                reasons.append("non_contiguous_frames")
            if identity.get("history_type") not in {"first", "later"}:
                reasons.append("invalid_history_type")
            for flag, reason in (
                ("no_interaction_event_verified", "interaction_event_in_negative_window"),
                ("gui_closed_verified", "gui_open"),
                ("frame_contiguous_verified", "non_contiguous_frames"),
                ("source_mapping_valid", "invalid_source_mapping"),
            ):
                if str(row.get(flag, "false")).strip().lower() != "true":
                    reasons.append(reason)
            if not identity.get("source_segment_id"):
                reasons.append("missing_source_segment_id")
            if str(row.get("candidate_error", "")).strip():
                reasons.append("candidate_error")
            if str(row.get("metadata_filter_status", "passed")).strip().lower() == "rejected":
                reasons.append("metadata_filter_rejected")
        else:
            if dynamic_hand_mask_area_ratio > 0.05:
                reasons.append("excessive_dynamic_hand_mask")
            if valid_count <= 0:
                reasons.append("empty_valid_region")
            try:
                validate_stage0_positive_tokens(
                    action_type,
                    stage0_positive_tokens,
                    event_id=row.get("event_id", ""),
                    history_type=row.get("history_type", ""),
                )
            except RuntimeError:
                reasons.append("empty_stage0_positive_tokens")
            if area < min_area:
                reasons.append("teacher_too_small")
            if area > float(args.max_area):
                reasons.append("teacher_too_large")
        cache_payload = {
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": str(row.get("candidate_npz_sha256", "")),
            "recipe": recipe,
            "event_id": row.get("event_id"),
            "history_type": row.get("history_type"),
        }
        cache_key = stable_json_hash(cache_payload)
        npz_path = args.output_dir / f"{cache_key}.npz"
        preview_path = args.output_dir / f"{cache_key}_overlay.png"
        np.savez_compressed(
            npz_path,
            residual=d_warp.astype(np.float16),
            d_warp=d_warp.astype(np.float16),
            d_raw=d_raw.astype(np.float16),
            teacher=teacher.astype(np.float16),
            teacher_rgb=teacher_rgb.astype(np.float16),
            d_warp_rgb=d_warp.astype(np.float16),
            d_raw_rgb=d_raw.astype(np.float16),
            old_edge_rgb=old_edge_rgb.astype(np.float16),
            edge_weight_rgb=edge_weight_rgb.astype(np.float16),
            hand_mask_rgb=hand_mask_rgb.astype(np.uint8),
            valid_rgb=valid_rgb.astype(np.float16),
            visibility=visibility.astype(np.float16),
            world_valid=world.astype(np.float16),
            valid=valid.astype(np.float16),
            candidate_cache_key=np.asarray(expected_candidate_key),
            candidate_config_hash=np.asarray(str(row.get("candidate_config_hash", ""))),
            candidate_npz_sha256=np.asarray(str(row.get("candidate_npz_sha256", ""))),
            training_cache_sha256=np.asarray(str(row.get("training_cache_sha256", ""))),
            candidate_identity_json=np.asarray(json.dumps(identity, ensure_ascii=False, sort_keys=True)),
        )
        row["teacher_area_ratio"] = f"{area:.8f}"
        row["hand_mask_area_ratio"] = f"{hand_mask_area_ratio:.8f}"
        row["dynamic_hand_mask_area_ratio"] = f"{dynamic_hand_mask_area_ratio:.8f}"
        save_review_contact_sheet(
            preview_path,
            payload,
            teacher_rgb,
            d_warp,
            d_raw,
            old_edge_rgb,
            edge_weight_rgb,
            hand_mask_rgb,
            row,
            stage0_positive_tokens,
            reasons,
        )
        save_preview(args.output_dir / f"{cache_key}_summary.png", d_warp, teacher_rgb)
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
                "hand_mask_area_ratio": f"{hand_mask_area_ratio:.8f}",
                "dynamic_hand_mask_area_ratio": f"{dynamic_hand_mask_area_ratio:.8f}",
                "teacher_invalid_reasons": "|".join(reasons),
                "teacher_valid": str(not reasons).lower(),
                "review_status": "pending" if not reasons else "rejected",
                "review_note": "",
                "overfit_selected": "false",
                "teacher_support_threshold": str(float(args.support_threshold)),
                "stage0_grid_t": str(int(args.stage0_grid[0])),
                "stage0_grid_h": str(int(args.stage0_grid[1])),
                "stage0_grid_w": str(int(args.stage0_grid[2])),
            }
        )
        payload.close()
    manifest_path = args.output_dir / str(args.review_manifest_name)
    columns = sorted({key for row in output for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    review_index_path = args.output_dir / "review_index.html"
    write_review_index(review_index_path, output)
    combination_counts = Counter()
    reviewable_action_counts = Counter()
    reviewable_combination_counts = Counter()
    for row in output:
        action = "negative" if is_negative_action(row.get("action_type")) else str(row.get("action_type", ""))
        combination_counts[f"{action}|{row.get('history_type', '')}"] += 1
        if str(row.get("teacher_overlay_path", "") or "").strip():
            reviewable_action_counts[action] += 1
            reviewable_combination_counts[f"{action}|{row.get('history_type', '')}"] += 1
    (args.output_dir / "teacher_pool_audit.json").write_text(
        json.dumps(
            {
                "candidate_count": len(rows),
                "valid_for_review": sum(not row["teacher_invalid_reasons"] for row in output),
                "pending": sum(row.get("review_status") == "pending" for row in output),
                "rejected": sum(row.get("review_status") == "rejected" for row in output),
                "rejection_counts": dict(rejection_counts),
                "action_history_counts": dict(combination_counts),
                "reviewable_action_counts": dict(reviewable_action_counts),
                "reviewable_action_history_counts": dict(reviewable_combination_counts),
                "recipe": recipe,
                "manifest": str(manifest_path),
                "review_index": str(review_index_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
