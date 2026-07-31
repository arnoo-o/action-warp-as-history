#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import copy
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from warp_as_history.camera_warp import (
    CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
    CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
    CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
    CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
    CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
    CAMERA_CONTROL_PI3_PIXEL_LIMIT,
    CAMERA_CONTROL_PROMPT_TRIGGER,
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
    se3_inverse,
)
from warp_as_history.minecraft_camera import (
    POSE_CONVENTION,
    effective_translation_scale,
    pose_motion_statistics,
    vpt_rows_to_relative_opencv_c2w,
)
from warp_as_history.minecraft_sampling import (
    build_interaction_event_window as _build_interaction_event_window,
)
from warp_as_history.training import core as opt
from warp_as_history.training.utils import detach_tree
from helios.modules.interaction_conditioning import (
    align_interaction_signals_to_grid,
    interaction_action_id,
    interaction_block_id,
    mine_progress_for_source_frames,
)
from warp_as_history.training.fixed_teacher import (
    FixedTeacherIntegrityError,
    action_history_key,
    canonical_history_type,
    interaction_payload_hash,
    parse_index_sequence,
    validate_fixed_artifact_hashes,
    validate_fixed_identity,
    validate_stage0_positive_tokens,
)


ONLINE_VIDEO_COLUMNS = ("video", "video_url", "url", "video_path", "path")
ONLINE_PROMPT_COLUMNS = ("prompt", "prompts", "caption", "text")
ONLINE_INTERACTION_COLUMNS = ("interaction_history_path", "action_history_path", "frame_action_summary_path")
ONLINE_PRIMARY_FIRE_EVENT_COLUMNS = ("primary_fire_event_path",)
ONLINE_PRIMARY_FIRE_MASK_COLUMNS = ("primary_fire_loss_mask_path",)
ONLINE_INTERACTION_EVENT_COLUMNS = ("interaction_event_path", "mc_event_path", "primary_fire_event_path")
ONLINE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PRIMARY_FIRE_CHAR = "["
MC_TRAINING_CATEGORIES = ("place", "mine", "movement", "other", "negative")
MC_HUD_SOURCE_SIZE = (640, 360)
MC_HUD_RECTS = (
    (226, 316, 316, 335),
    (324, 414, 316, 335),
    (224, 416, 330, 360),
    (313, 328, 324, 347),
)


def data_value_present(value):
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return False
    return bool(str(value).strip())


def first_present_data_value(*values, default=None):
    for value in values:
        if data_value_present(value):
            return value
    return default


def minecraft_world_valid_mask(*, height=360, width=640):
    """Build the Minecraft world mask using the exact video fit and nearest sampling."""
    source_width, source_height = MC_HUD_SOURCE_SIZE
    mask = np.ones((source_height, source_width), dtype=np.uint8) * 255
    for x1, x2, y1, y2 in MC_HUD_RECTS:
        mask[y1:y2, x1:x2] = 0
    source = Image.fromarray(mask, mode="L")
    return ImageOps.fit(
        source,
        (int(width), int(height)),
        method=Image.Resampling.NEAREST,
        centering=(0.5, 0.5),
    )


def _minecraft_hud_rect_masks(*, height, width):
    source_width, source_height = MC_HUD_SOURCE_SIZE
    masks = []
    for x1, x2, y1, y2 in MC_HUD_RECTS:
        rect = np.zeros((source_height, source_width), dtype=np.uint8)
        rect[y1:y2, x1:x2] = 255
        masks.append(
            ImageOps.fit(
                Image.fromarray(rect, mode="L"),
                (int(width), int(height)),
                method=Image.Resampling.NEAREST,
                centering=(0.5, 0.5),
            )
        )
    return masks


def fill_minecraft_hud_for_pi3(frame):
    """Fill only a Pi3 input copy; target/VAE pixels remain untouched."""
    original = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    result = original.copy()
    height, width = result.shape[:2]
    for rect_mask in _minecraft_hud_rect_masks(height=height, width=width):
        region = np.asarray(rect_mask, dtype=np.uint8) > 0
        ys, xs = np.nonzero(region)
        if not len(ys):
            continue
        source_y = max(0, int(ys.min()) - 1)
        source_row = original[source_y]
        for y in np.unique(ys):
            columns = np.flatnonzero(region[y])
            result[y, columns] = source_row[columns]
    return Image.fromarray(result, mode="RGB")


def multiply_mask_frames(mask_frames, world_valid_mask):
    world = np.asarray(world_valid_mask.convert("L"), dtype=np.float32) / 255.0
    result = []
    for frame in mask_frames:
        values = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
        if values.shape != world.shape:
            raise ValueError(f"Mask/world shape mismatch: {values.shape} != {world.shape}.")
        result.append(Image.fromarray(np.rint(values * world * 255.0).astype(np.uint8), mode="L"))
    return result


def clear_minecraft_hud_geometry(geometry, world_valid_mask):
    """Remove HUD pixels from every Pi3 geometry representation used by rendering."""
    keyframes = geometry.get("keyframe_geometries")
    geometries = list(keyframes) if keyframes is not None else [geometry]
    for item in geometries:
        render_height = int(item["render_height"])
        render_width = int(item["render_width"])
        valid_image = world_valid_mask.resize(
            (render_width, render_height),
            resample=Image.Resampling.NEAREST,
        )
        world_valid = np.asarray(valid_image, dtype=np.uint8) > 0
        valid = np.asarray(item["valid_mask"], dtype=bool).copy()
        valid &= world_valid
        item["valid_mask"] = valid
        if "conf_map" in item:
            conf = np.asarray(item["conf_map"], dtype=np.float32).copy()
            conf[~world_valid] = 0.0
            item["conf_map"] = conf
        if "depth_map" in item:
            depth = np.asarray(item["depth_map"], dtype=np.float32).copy()
            depth[~world_valid] = 0.0
            item["depth_map"] = depth
        if "point_map_world" in item:
            points = np.asarray(item["point_map_world"], dtype=np.float32).copy()
            points[~world_valid] = 0.0
            item["point_map_world"] = points
        item["minecraft_world_valid_mask"] = world_valid
    if keyframes:
        latest = keyframes[-1]
        for name in ("valid_mask", "conf_map", "depth_map", "point_map_world"):
            if name in latest:
                geometry[name] = latest[name]
    geometry["minecraft_hud_mask_applied"] = True
    return geometry


def _safe_debug_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _save_mask_debug(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("L").save(path)


def _online_infer_column(columns, requested, candidates, label):
    if requested:
        if requested not in columns:
            raise KeyError(f"Requested online {label} column {requested!r} is missing from CSV header {list(columns)}.")
        return requested
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"Could not infer online {label} column from CSV header {list(columns)}.")


def add_online_prompt_trigger(prompt, trigger=None):
    prompt = str(prompt or "").strip()
    trigger = str(CAMERA_CONTROL_PROMPT_TRIGGER if trigger is None else trigger).strip()
    if not trigger:
        return prompt
    if prompt.startswith(trigger):
        return prompt
    return f"{trigger} {prompt}".strip()


def normalize_online_training_dataframe(df, exact_args):
    columns = list(df.columns)
    video_column = _online_infer_column(
        columns,
        str(getattr(exact_args, "online_video_column", "") or ""),
        ONLINE_VIDEO_COLUMNS,
        "video",
    )
    prompt_column = _online_infer_column(
        columns,
        str(getattr(exact_args, "online_prompt_column", "") or ""),
        ONLINE_PROMPT_COLUMNS,
        "prompt",
    )
    prompt_trigger = str(getattr(exact_args, "online_prompt_trigger", CAMERA_CONTROL_PROMPT_TRIGGER) or "")
    event_column = _online_optional_column(columns, "", ONLINE_PRIMARY_FIRE_EVENT_COLUMNS)
    loss_mask_column = _online_optional_column(columns, "", ONLINE_PRIMARY_FIRE_MASK_COLUMNS)
    interaction_event_column = _online_optional_column(columns, "", ONLINE_INTERACTION_EVENT_COLUMNS)
    rows = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        base = row.to_dict()
        raw_prompt = str(base.get(prompt_column, ""))
        base["id"] = str(base.get("id") or f"online_{row_index:06d}")
        base["online_row_index"] = int(row_index)
        base["video_path"] = str(base[video_column])
        base["prompt_raw"] = raw_prompt
        if str(getattr(exact_args, "training_profile", "joint")) == "interaction":
            base["prompt"] = "Minecraft first-person gameplay."
        else:
            base["prompt"] = add_online_prompt_trigger(raw_prompt, prompt_trigger)
        base["training_category"] = canonical_training_category(
            base.get("category", base.get("action_type", "movement"))
        )
        if event_column:
            base["primary_fire_event_path"] = base.get(event_column, "")
        if loss_mask_column:
            base["primary_fire_loss_mask_path"] = base.get(loss_mask_column, "")
        if interaction_event_column:
            base["interaction_event_path"] = base.get(interaction_event_column, "")
        rows.append(base)
    normalized = df.__class__(rows)
    meta = {
        "video_column": video_column,
        "prompt_column": prompt_column,
        "prompt_trigger": prompt_trigger,
        "primary_fire_event_column": event_column,
        "primary_fire_loss_mask_column": loss_mask_column,
        "interaction_event_column": interaction_event_column,
        "rows": len(rows),
    }
    return normalized, meta


def canonical_training_category(value):
    """Keep manifest labels separate from the Router action vocabulary."""
    category = str(value or "movement").strip().lower()
    if category in MC_TRAINING_CATEGORIES:
        return category
    return "other"


def canonical_interaction_action(value):
    action = str(value or "none").strip().lower()
    if action == "mine":
        # Inventory/stat deltas identify the completed block removal.  Datasets
        # with an explicit held-mining signal may still provide mine_active.
        return "mine_complete"
    if action in {"place", "mine_active", "mine_complete"}:
        return action
    return "none"


def event_alignment_from_row(row, source_indices, *, source_fps=None, target_fps=None):
    """Map a source-FPS event to the decoded target-FPS timeline by timestamp."""
    if not source_indices:
        raise ValueError("Cannot align an event without decoded source frame indices.")
    source_start = int(row.get("source_frame_start", 0) or 0)
    if str(row.get("event_source_frame", "")).strip():
        source_event_frame = int(row["event_source_frame"])
        segment_event_frame = source_event_frame - source_start
    else:
        segment_event_frame = int(row.get("event_local_frame", row.get("event_frame", 0)) or 0)
        source_event_frame = source_start + segment_event_frame
    distances = [abs(int(frame) - segment_event_frame) for frame in source_indices]
    resampled_event_frame = int(np.argmin(distances))
    source_fps = float(source_fps if source_fps is not None else row.get("fps", 0.0) or 0.0)
    target_fps = float(target_fps or 0.0)
    return {
        "source_event_frame": int(source_event_frame),
        "segment_event_frame": int(segment_event_frame),
        "source_event_time_ms": None
        if source_fps <= 0.0
        else 1000.0 * float(segment_event_frame) / source_fps,
        "resampled_event_frame": resampled_event_frame,
        "resampled_event_time_ms": None
        if target_fps <= 0.0
        else 1000.0 * float(resampled_event_frame) / target_fps,
    }


def resampled_event_frame_from_row(row, source_indices):
    alignment = event_alignment_from_row(row, source_indices)
    return alignment["resampled_event_frame"], alignment["segment_event_frame"]


def reverse_event_frame(event_frame, num_frames):
    return int(num_frames) - 1 - int(event_frame)


def reverse_temporal_event_payload(payload, num_frames):
    """Reverse every frame-indexed field used by interaction conditioning."""
    if payload is None:
        return None
    reversed_payload = dict(payload)
    total = int(num_frames)
    if "event_frame" in reversed_payload:
        reversed_payload["event_frame"] = reverse_event_frame(reversed_payload["event_frame"], total)
    if "click_frames_local" in reversed_payload:
        reversed_payload["click_frames_local"] = [
            reverse_event_frame(value, total) for value in reversed(reversed_payload["click_frames_local"])
        ]
    for key in ("primary_fire_time_mask", "time_mask", "source_frame_indices"):
        if key in reversed_payload and reversed_payload[key] is not None:
            reversed_payload[key] = list(reversed(reversed_payload[key]))
    reversed_events = []
    for event in reversed_payload.get("events", []) or []:
        updated = dict(event)
        updated["event_frame"] = reverse_event_frame(updated.get("event_frame", 0), total)
        if updated.get("event_end_frame") is not None:
            original_start = int(event.get("event_frame", 0))
            original_end = int(updated["event_end_frame"])
            updated["event_frame"] = reverse_event_frame(original_end, total)
            updated["event_end_frame"] = reverse_event_frame(original_start, total)
        reversed_events.append(updated)
    if "events" in reversed_payload:
        reversed_payload["events"] = list(reversed(reversed_events))
    reversed_windows = []
    for window in reversed_payload.get("event_windows", []) or []:
        updated = dict(window)
        start = int(window.get("window_start", 0))
        end = int(window.get("window_end_exclusive", start))
        updated["window_start"] = total - end
        updated["window_end_exclusive"] = total - start
        if "click_frame_local" in updated:
            updated["click_frame_local"] = reverse_event_frame(updated["click_frame_local"], total)
        reversed_windows.append(updated)
    if "event_windows" in reversed_payload:
        reversed_payload["event_windows"] = list(reversed(reversed_windows))
    return reversed_payload


def build_interaction_event_window(
    event_frame,
    *,
    num_source_frames,
    window_size,
    rng,
    local_min,
    local_max,
    require_later=True,
):
    """Build a positive 33-frame window with a real pre-event baseline."""
    return _build_interaction_event_window(
        event_frame,
        num_source_frames=num_source_frames,
        window_size=window_size,
        rng=rng,
        local_min=local_min,
        local_max=local_max,
        require_later=require_later,
    )


def _online_optional_column(columns, requested, candidates):
    if requested:
        return requested if requested in columns else None
    for name in candidates:
        if name in columns:
            return name
    return None


def resolve_optional_data_path(value, data_root):
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return Path(data_root) / path


def _normalize_frame_to_latent_mapping(num_frames, latent_frames, temporal_scale):
    mapping = []
    for latent_idx in range(int(latent_frames)):
        start = int(latent_idx) * int(temporal_scale)
        end = min(int(num_frames), start + int(temporal_scale))
        if end <= start:
            end = min(int(num_frames), start + 1)
        mapping.append(
            {
                "latent_index": int(latent_idx),
                "frame_start": int(start),
                "frame_end_exclusive": int(end),
            }
        )
    return mapping


def rgb_frame_to_latent_indices(num_frames, latent_frames, temporal_scale):
    """Invert the shared VAE temporal coverage map without linear interpolation."""
    result = [-1] * int(num_frames)
    for item in _normalize_frame_to_latent_mapping(num_frames, latent_frames, temporal_scale):
        for frame_index in range(int(item["frame_start"]), int(item["frame_end_exclusive"])):
            result[frame_index] = int(item["latent_index"])
    if result and result[-1] < 0:
        last = max(int(latent_frames) - 1, 0)
        result = [last if value < 0 else value for value in result]
    return result


def _load_interaction_history_store(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frame_summaries = payload.get("frame_summaries", [])
    frame_features = payload.get("frame_features", [])
    if not isinstance(frame_summaries, list):
        raise ValueError(f"Interaction history payload at {path} must contain a list field 'frame_summaries'.")
    if frame_features is not None and not isinstance(frame_features, list):
        raise ValueError(f"Interaction history payload at {path} must contain a list field 'frame_features'.")
    summaries = {}
    for item in frame_summaries:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame")
        summary = str(item.get("summary") or "").strip()
        if frame is None or not summary:
            continue
        summaries[int(frame)] = summary
    features = {}
    for item in frame_features or []:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame")
        if frame is None:
            continue
        features[int(frame)] = dict(item)
    return {
        "path": str(path),
        "fps": float(payload.get("fps", 0.0) or 0.0),
        "frame_summaries": summaries,
        "frame_features": features,
        "meta": payload.get("meta", {}),
    }


def _summarize_interaction_history(store, indices, *, fallback_indices=None, max_items=8):
    if not store:
        return ""
    summaries = store.get("frame_summaries", {})
    ordered = list(indices or [])
    if not ordered and fallback_indices:
        ordered = list(fallback_indices)
    if not ordered:
        return ""

    unique_segments = []
    last_summary = None
    for idx in ordered:
        summary = str(summaries.get(int(idx), "")).strip()
        if not summary or summary == last_summary:
            continue
        unique_segments.append(summary)
        last_summary = summary
        if len(unique_segments) >= int(max_items):
            break
    return " ; ".join(unique_segments)


def summarize_multiscale_interaction_history(store, history_indices, target_indices, *, max_items=8):
    ordered_history = list(history_indices or [])
    if not ordered_history:
        return {
            "long_term": "",
            "mid_term": "",
            "short_term": "",
            "merged": "",
        }

    total = len(ordered_history)
    short_count = max(1, min(total, 4))
    mid_count = max(1, min(total - short_count, 6)) if total > short_count else 0
    long_end = max(0, total - short_count - mid_count)
    long_indices = ordered_history[:long_end]
    mid_indices = ordered_history[long_end : total - short_count]
    short_indices = ordered_history[total - short_count :]

    long_term = _summarize_interaction_history(store, long_indices, max_items=max_items)
    mid_term = _summarize_interaction_history(store, mid_indices, max_items=max_items)
    short_term = _summarize_interaction_history(store, short_indices, max_items=max_items)
    merged = _summarize_interaction_history(store, ordered_history, max_items=max_items)
    return {
        "long_term": long_term,
        "mid_term": mid_term,
        "short_term": short_term,
        "merged": merged,
    }


def compose_action_conditioned_prompt(base_prompt, interaction_memory):
    prompt = str(base_prompt or "").strip()
    if isinstance(interaction_memory, dict):
        long_term = str(interaction_memory.get("long_term") or "").strip()
        mid_term = str(interaction_memory.get("mid_term") or "").strip()
        short_term = str(interaction_memory.get("short_term") or "").strip()
        merged = str(interaction_memory.get("merged") or "").strip()
    else:
        long_term = ""
        mid_term = ""
        short_term = ""
        merged = str(interaction_memory or "").strip()

    segments = []
    if long_term:
        segments.append(f"Long-term interaction memory: {long_term}")
    if mid_term:
        segments.append(f"Mid-term interaction memory: {mid_term}")
    if short_term:
        segments.append(f"Short-term interaction memory: {short_term}")
    if not segments and merged:
        segments.append(f"Historical player interactions: {merged}")
    if not segments:
        return prompt
    return f"{prompt} {' '.join(segment + '.' for segment in segments)}".strip()


def summarize_multiscale_action_pseudo_history(store, history_indices, target_indices):
    frame_features = {} if not store else dict(store.get("frame_features", {}) or {})
    ordered_history = list(history_indices or [])

    empty = {
        "long_term": {},
        "mid_term": {},
        "short_term": {},
        "merged": {},
    }
    if not ordered_history or not frame_features:
        return empty

    total = len(ordered_history)
    short_count = max(1, min(total, 4))
    mid_count = max(1, min(total - short_count, 6)) if total > short_count else 0
    long_end = max(0, total - short_count - mid_count)
    slices = {
        "long_term": ordered_history[:long_end],
        "mid_term": ordered_history[long_end : total - short_count],
        "short_term": ordered_history[total - short_count :],
        "merged": ordered_history,
    }

    feature_keys = (
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "jump",
        "crouch",
        "reload",
        "primary_fire",
        "secondary_fire",
        "use",
        "mouse_dx",
        "mouse_dy",
        "yaw_delta",
        "pitch_delta",
        "speed",
    )

    def aggregate(indices):
        collected = [frame_features.get(int(idx)) for idx in indices if int(idx) in frame_features]
        collected = [item for item in collected if item]
        if not collected:
            return {}
        result = {}
        for key in feature_keys:
            values = []
            for item in collected:
                value = item.get(key)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            if values:
                result[key] = sum(values) / float(len(values))
        result["count"] = float(len(collected))
        return result

    return {name: aggregate(indices) for name, indices in slices.items()}


def extract_primary_fire_click_frames(store):
    frame_features = {} if not store else dict(store.get("frame_features", {}) or {})
    if not frame_features:
        return []
    click_frames = []
    prev_pressed = False
    for frame_idx in sorted(int(idx) for idx in frame_features.keys()):
        action_text = str(frame_features.get(frame_idx, {}).get("actions_raw", "") or "")
        pressed = PRIMARY_FIRE_CHAR in action_text
        if pressed and not prev_pressed:
            click_frames.append(int(frame_idx))
        prev_pressed = pressed
    return click_frames


def build_primary_fire_click_supervision(store, target_indices, *, radius_frames=12):
    target_indices = [int(idx) for idx in target_indices]
    click_frames = extract_primary_fire_click_frames(store)
    hit_frames = [frame for frame in click_frames if frame in set(target_indices)]
    temporal_mask = [
        1.0 if any(abs(int(frame_idx) - int(click_frame)) <= int(radius_frames) for click_frame in click_frames) else 0.0
        for frame_idx in target_indices
    ]
    return {
        "click_frames": click_frames,
        "target_click_frames": hit_frames,
        "target_has_click": bool(hit_frames),
        "temporal_mask": temporal_mask,
        "radius_frames": int(radius_frames),
    }


def build_primary_fire_focus_mask_frames(target_frames, warp_frames, supervision, *, residual_threshold=0.08):
    temporal_mask = list(supervision.get("temporal_mask", []) or [])
    if not temporal_mask or len(target_frames) != len(warp_frames):
        return None, {}
    mask_frames = []
    active_frames = 0
    active_pixels = 0.0
    total_pixels = 0.0
    for weight, target_frame, warp_frame in zip(temporal_mask, target_frames, warp_frames):
        target_np = np.asarray(target_frame.convert("RGB"), dtype=np.float32) / 255.0
        warp_np = np.asarray(warp_frame.convert("RGB"), dtype=np.float32) / 255.0
        residual = np.abs(target_np - warp_np).mean(axis=2)
        mask = (residual >= float(residual_threshold)).astype(np.float32)
        if float(weight) <= 0.0:
            mask *= 0.0
        if mask.any():
            active_frames += 1
        active_pixels += float(mask.sum())
        total_pixels += float(mask.size)
        mask_frames.append(Image.fromarray((mask * 255.0).astype(np.uint8), mode="L"))
    stats = {
        "residual_threshold": float(residual_threshold),
        "active_frame_count": int(active_frames),
        "active_pixel_ratio": 0.0 if total_pixels <= 0 else float(active_pixels / total_pixels),
    }
    return mask_frames, stats


def online_mask_frames_to_latent_mask(
    mask_frames,
    *,
    target_latents,
    num_frames,
    temporal_scale,
    device,
    interpolation_mode="trilinear",
):
    if not mask_frames:
        return None
    mask = np.stack([np.asarray(frame.convert("L"), dtype=np.float32) / 255.0 for frame in mask_frames], axis=0)
    if mask.shape[0] < int(num_frames):
        raise ValueError(f"Focus mask produced {mask.shape[0]} frames, need at least {int(num_frames)}.")
    sampled_ids = np.arange(int(target_latents.shape[2]), dtype=np.int64) * int(temporal_scale)
    sampled_ids = np.clip(sampled_ids, 0, mask.shape[0] - 1)
    sampled = torch.from_numpy(mask[sampled_ids]).to(device=device, dtype=torch.float32)
    sampled = sampled.unsqueeze(0).unsqueeze(0)
    interpolate_kwargs = {
        "size": (int(target_latents.shape[2]), int(target_latents.shape[3]), int(target_latents.shape[4])),
        "mode": str(interpolation_mode),
    }
    if str(interpolation_mode) != "nearest":
        interpolate_kwargs["align_corners"] = False
    sampled = torch.nn.functional.interpolate(sampled, **interpolate_kwargs)
    return sampled.clamp_(0.0, 1.0)


def load_primary_fire_event_payload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_interaction_events(payload):
    events = []
    for event in list(payload.get("events", payload.get("selected_events", [])) or []):
        action_type = canonical_interaction_action(event.get("action_type", event.get("category", "none")))
        frame = event.get("event_frame", event.get("local_frame", event.get("frame")))
        if frame is None:
            continue
        events.append(
            {
                "event_id": event.get("event_id"),
                "event_frame": int(frame),
                "action_type": action_type,
                "object_id": event.get("object_id"),
                "block_id": event.get("block_id", event.get("object_id")),
                "event_end_frame": event.get(
                    "event_end_frame",
                    event.get("end_frame", event.get("mine_end_frame")),
                ),
                "action_start_frame": event.get("action_start_frame", frame),
                "action_end_frame": event.get(
                    "action_end_frame",
                    event.get("event_end_frame", event.get("end_frame", event.get("mine_end_frame"))),
                ),
                "complete_frame": event.get("complete_frame"),
            }
        )
    if not events:
        click_frames = payload.get("click_frames_local", payload.get("click_frames_source", payload.get("click_frames", [])))
        for frame in click_frames:
            events.append(
                {
                    "event_frame": int(frame),
                    "action_type": "primary_fire",
                    "object_id": "primary_fire",
                    "block_id": "primary_fire",
                }
            )
    return sorted(events, key=lambda item: int(item["event_frame"]))


def choose_online_interaction_window(rng, num_frames, window_size, interaction_payload):
    events = canonical_interaction_events(interaction_payload or {})
    if not events:
        return None
    event = rng.choice(events)
    event_offset = rng.randrange(max(int(window_size), 1))
    start = min(max(int(event["event_frame"]) - event_offset, 0), int(num_frames) - int(window_size))
    return list(range(start, start + int(window_size)))


def crop_interaction_payload(interaction_payload, target_indices):
    target_indices = [int(index) for index in target_indices]
    index_to_local = {source: local for local, source in enumerate(target_indices)}
    events = [
        event
        for event in canonical_interaction_events(interaction_payload or {})
        if int(event["event_frame"]) in index_to_local
    ]
    if not events:
        return {
            "event_frame": 0,
            "action_type": "none",
            "object_id": None,
            "block_id": None,
            "event_valid": 0.0,
            "num_frames": len(target_indices),
        }
    event = events[0]
    local_event = int(index_to_local[int(event["event_frame"])])
    source_action_start = int(event.get("action_start_frame", event["event_frame"]))
    end_source = event.get("action_end_frame", event.get("event_end_frame"))
    if event["action_type"] == "mine_active" and end_source is not None:
        end_local = next(
            (
                local
                for local, source in enumerate(target_indices)
                if int(source) >= int(end_source)
            ),
            len(target_indices),
        )
        end_local = max(local_event + 1, end_local)
    else:
        end_local = len(target_indices)
    time_mask = [0.0] * len(target_indices)
    for local in range(local_event, min(end_local, len(time_mask))):
        time_mask[local] = 1.0
    progress_curve = [0.0] * len(target_indices)
    if event["action_type"] == "mine_active":
        complete_source = int(event.get("complete_frame") or end_source or (source_action_start + 1))
        progress_curve = mine_progress_for_source_frames(
            target_indices, source_action_start, complete_source
        )
    elif event["action_type"] == "mine_complete":
        for local in range(local_event, len(progress_curve)):
            progress_curve[local] = 1.0
    return {
        **event,
        "event_frame_source": int(event["event_frame"]),
        "event_frame": local_event,
        "event_valid": 1.0,
        "num_frames": len(target_indices),
        "time_mask": time_mask,
        "frame_action_mask": time_mask,
        "frame_progress_curve": progress_curve,
        "action_start_frame": local_event,
        "action_end_frame": min(end_local - 1, len(target_indices) - 1),
        "source_action_start_frame": source_action_start,
        "source_complete_frame": event.get("complete_frame", end_source),
    }


def interaction_payload_tensors(payload, device):
    payload = payload or {}
    return {
        "action_ids": torch.tensor([interaction_action_id(payload.get("action_type"))], device=device, dtype=torch.long),
        "block_ids": torch.tensor(
            [interaction_block_id(payload.get("block_id", payload.get("object_id")))],
            device=device,
            dtype=torch.long,
        ),
        "event_frames": torch.tensor([float(payload.get("event_frame", 0))], device=device),
        "total_frames": torch.tensor([float(payload.get("num_frames", 1))], device=device),
        "event_valid": torch.tensor([float(payload.get("event_valid", 0.0))], device=device),
        "frame_action_mask": torch.tensor(
            [list(payload.get("frame_action_mask", payload.get("time_mask", [])))],
            device=device,
            dtype=torch.float32,
        ),
        "frame_progress_curve": torch.tensor(
            [list(payload.get("frame_progress_curve", [0.0] * int(payload.get("num_frames", 1))))],
            device=device,
            dtype=torch.float32,
        ),
    }


def build_residual_teacher_components(
    target_latents,
    warp_latents,
    visibility,
    world_valid_mask=None,
    interaction_payload=None,
    teacher_support_threshold=0.25,
    teacher_min_area_by_action=None,
    teacher_max_area=0.25,
    min_valid_pixels=8,
    min_visibility_ratio=0.05,
    z_cap=3.0,
    camera_rotation_degrees=None,
    max_camera_rotation_degrees=None,
):
    """Build a detached latent-grid residual teacher and per-sample validity audit."""
    warp = warp_latents.to(device=target_latents.device, dtype=target_latents.dtype)
    if warp.shape[2:] != target_latents.shape[2:]:
        warp = torch.nn.functional.interpolate(
            warp.float(), size=target_latents.shape[2:], mode="trilinear", align_corners=False
        ).to(target_latents)
    raw_residual = (target_latents.float() - warp.float()).abs().mean(dim=1, keepdim=True)
    if visibility is not None:
        visible = torch.nn.functional.interpolate(
            visibility.float(), size=raw_residual.shape[2:], mode="nearest"
        )
        visible = visible.clamp(0.0, 1.0)
    else:
        visible = torch.ones_like(raw_residual)
    if world_valid_mask is not None:
        world_valid = torch.nn.functional.interpolate(
            world_valid_mask.float(), size=raw_residual.shape[2:], mode="nearest"
        )
        world_valid = (world_valid > 0.5).to(raw_residual)
    else:
        world_valid = torch.ones_like(raw_residual)
    if interaction_payload and float(interaction_payload.get("event_valid", 0.0)) > 0.0:
        source_frames = max(int(interaction_payload.get("num_frames", target_latents.shape[2])), 1)
        frame_mask = list(
            interaction_payload.get("frame_action_mask", interaction_payload.get("time_mask", []))
        )
        if len(frame_mask) != source_frames:
            event_frame = int(interaction_payload.get("event_frame", 0))
            frame_mask = [0.0] * max(event_frame, 0) + [1.0] * max(source_frames - event_frame, 0)
            frame_mask = frame_mask[:source_frames]
        temporal_values = []
        latent_frames = int(raw_residual.shape[2])
        for latent_index in range(latent_frames):
            start = int(math.floor(latent_index * source_frames / latent_frames))
            end = max(int(math.ceil((latent_index + 1) * source_frames / latent_frames)), start + 1)
            temporal_values.append(max(float(value) for value in frame_mask[start:end]))
        temporal_mask = torch.as_tensor(
            temporal_values, device=raw_residual.device, dtype=torch.float32
        ).view(1, 1, latent_frames, 1, 1)
    else:
        temporal_mask = torch.zeros_like(raw_residual[:, :, :, :1, :1])

    hand_valid = torch.ones_like(raw_residual[:, :, :1])
    hand_y = int(round(hand_valid.shape[-2] * 0.58))
    hand_x = int(round(hand_valid.shape[-1] * 0.65))
    hand_valid[..., hand_y:, hand_x:] = 0.0
    base_valid = visible * world_valid * hand_valid
    teacher_score = torch.zeros_like(raw_residual)
    epsilon = 1.0e-6
    for batch_index in range(raw_residual.shape[0]):
        for time_index in range(raw_residual.shape[2]):
            valid_now = base_valid[batch_index, 0, time_index] > 0.0
            values = raw_residual[batch_index, 0, time_index][valid_now]
            if values.numel() < int(min_valid_pixels):
                continue
            median = values.median()
            mad = (values - median).abs().median()
            scale = 1.4826 * mad
            if float(scale) <= epsilon:
                q50 = torch.quantile(values, 0.50)
                q90 = torch.quantile(values, 0.90)
                scale = q90 - q50
            if float(scale) <= epsilon:
                continue
            positive = (raw_residual[batch_index, 0, time_index] - median).clamp_min(0.0)
            teacher_score[batch_index, 0, time_index] = (
                positive / (scale + epsilon) / max(float(z_cap), epsilon)
            ).clamp(0.0, 1.0)

    # A weak, continuous crosshair prior rejects broad camera residuals without hard-coding a box.
    grid_y = torch.linspace(-1.0, 1.0, raw_residual.shape[-2], device=raw_residual.device)
    grid_x = torch.linspace(-1.0, 1.0, raw_residual.shape[-1], device=raw_residual.device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    crosshair_prior = 0.5 + 0.5 * torch.exp(-((xx.square() + yy.square()) / 0.5))
    teacher_score = teacher_score * crosshair_prior.view(1, 1, 1, *crosshair_prior.shape)
    local_motion = torch.nn.functional.avg_pool3d(
        raw_residual, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    )
    local_residual_suppression = (
        raw_residual / (raw_residual + local_motion + epsilon)
    ).clamp(0.25, 1.0)
    teacher_score = teacher_score * local_residual_suppression
    if teacher_score.shape[2] > 1:
        previous = torch.cat([teacher_score[:, :, :1], teacher_score[:, :, :-1]], dim=2)
        following = torch.cat([teacher_score[:, :, 1:], teacher_score[:, :, -1:]], dim=2)
        temporal_smoothing = 0.5 * teacher_score + 0.25 * previous + 0.25 * following
        teacher_score = temporal_smoothing
    else:
        temporal_smoothing = teacher_score

    valid_action_region = temporal_mask * base_valid
    teacher = (teacher_score * valid_action_region).detach()
    support = ((teacher_score > float(teacher_support_threshold)) & (valid_action_region > 0)).to(teacher)
    support_weighted = support * valid_action_region
    valid_denominator = valid_action_region.flatten(1).sum(dim=1)
    area_ratio = support_weighted.flatten(1).sum(dim=1) / valid_denominator.clamp_min(1.0)
    visibility_ratio = (visible * temporal_mask).flatten(1).sum(dim=1) / (
        temporal_mask.expand_as(visible).flatten(1).sum(dim=1).clamp_min(1.0)
    )
    action_type = str((interaction_payload or {}).get("action_type", "none"))
    minimums = {
        "place": 0.001,
        "mine_complete": 0.001,
        "mine_active": 0.0001,
    }
    minimums.update(dict(teacher_min_area_by_action or {}))
    min_area = float(minimums.get(action_type, 0.001))
    teacher_valid = (
        (valid_denominator >= float(min_valid_pixels))
        & (visibility_ratio >= float(min_visibility_ratio))
        & (area_ratio >= min_area)
        & (area_ratio <= float(teacher_max_area))
    )
    if max_camera_rotation_degrees is not None and camera_rotation_degrees is not None:
        rotation_ok = torch.as_tensor(
            [float(camera_rotation_degrees) <= float(max_camera_rotation_degrees)],
            device=teacher_valid.device,
            dtype=torch.bool,
        )
        teacher_valid = teacher_valid & rotation_ok
    invalid_reason = []
    for index in range(raw_residual.shape[0]):
        reasons = []
        if float(valid_denominator[index]) < float(min_valid_pixels):
            reasons.append("insufficient_valid_pixels")
        if float(visibility_ratio[index]) < float(min_visibility_ratio):
            reasons.append("low_visibility")
        if float(area_ratio[index]) < min_area:
            reasons.append("teacher_too_small")
        if float(area_ratio[index]) > float(teacher_max_area):
            reasons.append("teacher_too_large")
        if (
            max_camera_rotation_degrees is not None
            and camera_rotation_degrees is not None
            and float(camera_rotation_degrees) > float(max_camera_rotation_degrees)
        ):
            reasons.append("camera_rotation_exceeds_threshold")
        invalid_reason.append(reasons)
    return {
        "raw_residual": raw_residual.detach(),
        "visibility": None if visibility is None else visibility.detach(),
        "temporal_mask": temporal_mask.detach(),
        "valid_action_region": valid_action_region.detach(),
        "teacher_score": teacher_score.detach(),
        "local_residual_suppression": local_residual_suppression.detach(),
        "temporal_smoothing": temporal_smoothing.detach(),
        "teacher_support": support.detach(),
        "teacher_area_ratio": area_ratio.detach(),
        "teacher_visibility_ratio": visibility_ratio.detach(),
        "teacher_valid": teacher_valid.detach(),
        "teacher_invalid_reasons": invalid_reason,
        "clean_teacher_mask": teacher.detach(),
    }


def build_residual_teacher_map(target_latents, warp_latents, visibility, world_valid_mask=None, interaction_payload=None):
    """Return the clean teacher map; target-derived tensors are never inference inputs."""
    return build_residual_teacher_components(
        target_latents,
        warp_latents,
        visibility,
        world_valid_mask,
        interaction_payload,
    )["clean_teacher_mask"]


def load_primary_fire_loss_mask_frames(path):
    array = np.load(path)
    if array.ndim != 3:
        raise ValueError(f"Expected primary_fire_loss_mask.npy with shape [T,H,W], got {array.shape}")
    array = np.clip(array.astype(np.float32), 0.0, 1.0)
    return [Image.fromarray((frame * 255.0).astype(np.uint8), mode="L") for frame in array]


def crop_primary_fire_event_payload(event_payload, target_indices):
    target_indices = [int(idx) for idx in target_indices]
    source_frame_indices_full = [int(x) for x in event_payload.get("source_frame_indices", [])]
    if source_frame_indices_full and max(target_indices, default=-1) >= len(source_frame_indices_full):
        raise ValueError("target_indices exceed primary_fire_event source_frame_indices length.")
    source_frame_indices = (
        [source_frame_indices_full[idx] for idx in target_indices]
        if source_frame_indices_full
        else [int(idx) for idx in target_indices]
    )
    time_mask_full = list(event_payload.get("time_mask", []))
    if time_mask_full:
        time_mask = [float(time_mask_full[idx]) for idx in target_indices]
    else:
        click_frames_source = {int(x) for x in event_payload.get("click_frames_source", [])}
        time_mask = [1.0 if int(src) in click_frames_source else 0.0 for src in source_frame_indices]
    click_frames_source = [int(x) for x in event_payload.get("click_frames_source", []) if int(x) in set(source_frame_indices)]
    click_frames_local = [idx for idx, src in enumerate(source_frame_indices) if int(src) in set(click_frames_source)]
    event_windows = []
    for window in event_payload.get("event_windows", []) or []:
        start = int(window.get("window_start", 0))
        end = int(window.get("window_end_exclusive", 0))
        overlap = [idx for idx in target_indices if start <= idx < end]
        if overlap:
            event_windows.append(
                {
                    "window_start": max(0, start - target_indices[0]),
                    "window_end_exclusive": min(len(target_indices), end - target_indices[0]),
                    "click_frame_local": int(window.get("click_frame_local", -1)) - int(target_indices[0]),
                }
            )
    return {
        "fps": float(event_payload.get("fps", 0.0) or 0.0),
        "num_frames": int(len(target_indices)),
        "click_frames_source": click_frames_source,
        "click_frames_local": click_frames_local,
        "event_windows": event_windows,
        "source_frame_indices": source_frame_indices,
        "time_mask": time_mask,
    }


def crop_mask_frames(mask_frames, target_indices):
    if not mask_frames:
        return None
    return [mask_frames[int(idx)] for idx in target_indices]


def choose_online_primary_fire_window(rng, num_frames, window_size, event_payload):
    if event_payload is None:
        return None
    windows = list(event_payload.get("event_windows", []) or [])
    if not windows:
        return None
    chosen = rng.choice(windows)
    click_local = int(chosen.get("click_frame_local", 0))
    pre = min(window_size // 2, click_local)
    start_min = max(0, click_local - window_size + 1)
    start_max = min(max(0, num_frames - window_size), click_local)
    preferred = max(0, click_local - pre)
    start = min(max(preferred, start_min), start_max)
    if start_max > start_min:
        jitter = min(4, start_max - start_min)
        start = max(start_min, min(start_max, start + rng.randint(-jitter, jitter)))
    return list(range(int(start), int(start) + int(window_size)))


def choose_online_movement_window(rng, num_frames, window_size, event_payload):
    latest_start = max(0, int(num_frames) - int(window_size))
    if latest_start <= 0:
        return list(range(0, min(int(window_size), int(num_frames))))
    event_mask = np.zeros((int(num_frames),), dtype=np.float32)
    if event_payload is not None:
        for idx, value in enumerate(list(event_payload.get("time_mask", []) or [])[: int(num_frames)]):
            event_mask[idx] = float(value)
    candidates = []
    for start in range(0, latest_start + 1):
        score = float(event_mask[start : start + int(window_size)].mean())
        if score <= 0.05:
            candidates.append(start)
    if not candidates:
        candidates = list(range(0, latest_start + 1))
    start = rng.choice(candidates)
    return list(range(int(start), int(start) + int(window_size)))


def build_primary_fire_event_latents(*, event_payload, target_indices, target_latents, temporal_scale, device):
    target_indices = [int(idx) for idx in target_indices]
    latent_frames = int(target_latents.shape[2])
    latent_height = int(target_latents.shape[3])
    latent_width = int(target_latents.shape[4])
    target_channels = int(target_latents.shape[1])

    source_frame_indices = [int(x) for x in event_payload.get("source_frame_indices", target_indices)]
    time_mask = event_payload.get("time_mask")
    if time_mask is None:
        click_frames = {
            int(x) for x in event_payload.get("click_frames_source", event_payload.get("click_frames", []))
        }
        time_mask = [1.0 if int(idx) in click_frames else 0.0 for idx in source_frame_indices]
    if len(source_frame_indices) != len(time_mask):
        raise ValueError("primary_fire_event source_frame_indices and time_mask lengths must match.")
    target_frame_weights = np.asarray([float(weight) for weight in time_mask], dtype=np.float32)

    mapping = _normalize_frame_to_latent_mapping(len(target_indices), latent_frames, temporal_scale)
    latent_values = np.zeros(latent_frames, dtype=np.float32)
    for item in mapping:
        start = int(item["frame_start"])
        end = int(item["frame_end_exclusive"])
        latent_values[int(item["latent_index"])] = float(target_frame_weights[start:end].max()) if end > start else 0.0

    latent_mask = torch.from_numpy(latent_values).to(device=device, dtype=torch.float32).view(1, 1, latent_frames, 1, 1)
    latent_mask = latent_mask.expand(1, target_channels, latent_frames, latent_height, latent_width).contiguous()
    click_frames_source = set(int(x) for x in event_payload.get("click_frames_source", event_payload.get("click_frames", [])))
    for item in mapping:
        start = int(item["frame_start"])
        end = int(item["frame_end_exclusive"])
        item["source_frames"] = source_frame_indices[start:end]
        item["has_click"] = any(int(src) in click_frames_source for src in source_frame_indices[start:end])
    return latent_mask, mapping


def _online_is_uri(value):
    parsed = urlparse(str(value))
    return bool(parsed.scheme) and parsed.scheme not in {"", "file"}


def resolve_online_video_ref(value, data_root):
    text = str(value).strip()
    if _online_is_uri(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(data_root) / path
    return path


def _iter_online_image_files(path):
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in ONLINE_IMAGE_EXTS)


def _online_resample_indices(source_fps, target_fps, max_source_frames):
    source_fps = float(source_fps or 0.0)
    target_fps = float(target_fps or 0.0)
    max_source_frames = int(max_source_frames)
    if source_fps <= 0.0 or target_fps <= 0.0:
        return None
    if target_fps > source_fps + 1.0e-6:
        raise ValueError(f"Online target fps {target_fps:g} exceeds source fps {source_fps:g}.")
    count = max(1, int(np.floor((max_source_frames - 1) * target_fps / source_fps)) + 1)
    return np.rint(np.arange(count, dtype=np.float64) * source_fps / target_fps).astype(np.int64)


def load_online_video_frames(
    ref,
    *,
    height,
    width,
    frame_stride=1,
    max_video_frames=0,
    target_fps=0.0,
    source_fps=0.0,
    return_source_indices=False,
):
    frame_stride = max(1, int(frame_stride))
    max_video_frames = int(max_video_frames)
    frames = []
    source_indices = []
    selected_indices = None
    if isinstance(ref, Path) and ref.is_dir():
        image_files = _iter_online_image_files(ref)
        selected_indices = _online_resample_indices(source_fps, target_fps, len(image_files))
        selected_set = set(selected_indices.tolist()) if selected_indices is not None else None
        for src_idx, path in enumerate(image_files):
            if selected_set is not None and src_idx not in selected_set:
                continue
            if selected_set is None and src_idx % frame_stride != 0:
                continue
            frame = Image.open(path).convert("RGB")
            frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
            source_indices.append(int(src_idx))
            if max_video_frames > 0 and len(frames) >= max_video_frames:
                break
    else:
        reader = imageio.get_reader(str(ref))
        try:
            metadata = reader.get_meta_data()
            detected_fps = float(source_fps or metadata.get("fps") or 0.0)
            selected_cursor = 0
            use_target_fps = float(target_fps or 0.0) > 0.0
            if use_target_fps and detected_fps <= 0.0:
                raise ValueError(f"Could not determine source fps for {ref}.")
            if use_target_fps and float(target_fps) > detected_fps + 1.0e-6:
                raise ValueError(f"Online target fps {float(target_fps):g} exceeds source fps {detected_fps:g}.")
            for src_idx, array in enumerate(reader):
                if use_target_fps:
                    desired_source = int(round(selected_cursor * detected_fps / float(target_fps)))
                    while desired_source < src_idx:
                        selected_cursor += 1
                        desired_source = int(round(selected_cursor * detected_fps / float(target_fps)))
                    if desired_source != src_idx:
                        continue
                    selected_cursor += 1
                elif src_idx % frame_stride != 0:
                    continue
                frame = Image.fromarray(np.asarray(array)).convert("RGB")
                frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
                source_indices.append(int(src_idx))
                if max_video_frames > 0 and len(frames) >= max_video_frames:
                    break
        finally:
            reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from online training video {ref}.")
    if return_source_indices:
        return frames, source_indices
    return frames


def load_vpt_pose_rows(path, source_indices):
    wanted = set(int(index) for index in source_indices)
    rows_by_index = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            segment_frame = int(payload.get("segment_frame", line_index))
            if segment_frame in wanted:
                rows_by_index[segment_frame] = payload
            if len(rows_by_index) == len(wanted):
                break
    missing = [index for index in source_indices if int(index) not in rows_by_index]
    if missing:
        raise ValueError(f"VPT telemetry {path} is missing source frames {missing[:8]}.")
    required = ("xpos", "ypos", "zpos", "yaw", "pitch")
    rows = [rows_by_index[int(index)] for index in source_indices]
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError(f"VPT telemetry {path} lacks one of {required}.")
    return rows


def vpt_relative_camera_poses(pose_rows, source_index, target_indices, translation_scale=1.0):
    # Keep this public compatibility wrapper while sharing the implementation
    # with inference-side trajectory generation.
    from warp_as_history.minecraft_camera import vpt_rows_to_relative_opencv_c2w as convert

    return convert(
        pose_rows,
        source_index,
        target_indices,
        translation_scale=float(translation_scale),
    )


def online_pil_to_tensor(frame):
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def online_tensor_video_to_pil_frames(video):
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"Expected online warp video tensor [1, 3, T, H, W], got {tuple(video.shape)}.")
    arr = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in arr]


def pipeline_output_to_pil_frames(output):
    value = output.frames if hasattr(output, "frames") else output
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 5:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"Unexpected rollout output shape: {array.shape}")
    frames = []
    for frame in array:
        if frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {3, 4}:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = (
                np.clip(frame, 0.0, 1.0) * 255.0
                if float(np.max(frame)) <= 1.0
                else np.clip(frame, 0.0, 255.0)
            ).round().astype(np.uint8)
        frames.append(Image.fromarray(frame, mode="RGB"))
    return frames


def online_mask_tensor_to_pil_frames(mask):
    if mask.ndim != 5 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(f"Expected online visibility mask tensor [1, 1, T, H, W], got {tuple(mask.shape)}.")
    arr = mask[0, 0].detach().float().cpu().clamp(0.0, 1.0)
    arr = (arr * 255.0).round().to(torch.uint8).numpy()
    return [Image.fromarray(frame, mode="L") for frame in arr]


def subset_online_geometry(full_geometry, keyframe_indices):
    if not keyframe_indices:
        raise ValueError("Online warp rendering requires at least one keyframe.")
    keyframe_geometries = full_geometry["keyframe_geometries"]
    selected = [keyframe_geometries[int(idx)] for idx in keyframe_indices]
    latest = selected[-1]
    geometry = dict(full_geometry)
    geometry["intrinsic"] = latest["intrinsic"]
    geometry["keyframe_count"] = len(selected)
    geometry["keyframe_geometries"] = selected
    geometry["preserve_pi3x_keyframe_points"] = True
    geometry["render_height"] = latest["render_height"]
    geometry["render_width"] = latest["render_width"]
    geometry["source_pose"] = latest["source_pose"]
    geometry["source_rgb_u8"] = latest["source_rgb_u8"]
    return geometry


def online_relative_poses(full_geometry, source_pose, target_indices):
    keyframe_geometries = full_geometry["keyframe_geometries"]
    target_world = np.stack(
        [np.asarray(keyframe_geometries[int(idx)]["source_pose"], dtype=np.float32) for idx in target_indices],
        axis=0,
    )
    source_inv = se3_inverse(np.asarray(source_pose, dtype=np.float32)[None])[0]
    return np.einsum("ij,tjk->tik", source_inv.astype(np.float32, copy=False), target_world).astype(np.float32)


def online_renderer_config_from_args(args):
    return Pi3XWarpRendererConfig(
        pi3_pixel_limit=int(getattr(args, "online_pi3_pixel_limit", CAMERA_CONTROL_PI3_PIXEL_LIMIT)),
        conf_threshold=float(getattr(args, "online_pi3_conf_threshold", 0.1)),
        depth_edge_rtol=float(getattr(args, "online_pi3_depth_edge_rtol", 0.03)),
        mesh_samples_per_axis=int(getattr(args, "online_mesh_samples_per_axis", 4)),
        render_mode=str(getattr(args, "online_render_mode", CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)),
        target_fill_radius=int(getattr(args, "online_target_fill_radius", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)),
        target_fill_min_neighbors=int(
            getattr(args, "online_target_fill_min_neighbors", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS)
        ),
        mesh_break_mode=str(getattr(args, "online_mesh_break_mode", CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)),
        mesh_depth_rtol=float(getattr(args, "online_mesh_depth_rtol", CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL)),
        mesh_normal_tol_deg=float(
            getattr(args, "online_mesh_normal_tol_deg", CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG)
        ),
    )


class OnlineWarpTrainingCache:
    def __init__(self, rows, exact_args, device):
        self.rows = [row.to_dict() if hasattr(row, "to_dict") else dict(row) for row in rows]
        self.exact_args = exact_args
        self.device = torch.device(device)
        self.renderer = Pi3XWarpRenderer(online_renderer_config_from_args(exact_args))
        self.records = OrderedDict()
        self.memory_cache_size = max(1, int(getattr(self.exact_args, "online_warp_memory_cache_size", 2) or 2))
        self.disk_cache_dir = self._resolve_disk_cache_dir()
        self.interaction_histories = {}
        self._load_interaction_histories()
        if self.disk_cache_dir is not None:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            unique_sources = {
                self._video_source_digest(
                    resolve_online_video_ref(row["video_path"], getattr(self.exact_args, "data_root", "."))
                )
                for row in self.rows
            }
            print(
                json.dumps(
                    {
                        "event": "online_warp_cache_plan",
                        "rows": len(self.rows),
                        "unique_video_sources": len(unique_sources),
                        "cache_key": "video_content_direction_recipe_v5",
                        "conditioning_cache_scope": "video_content_recipe_conditioning_frame_indices",
                        "disk_cache_dir": str(self.disk_cache_dir),
                    }
                ),
                flush=True,
            )

    def _load_interaction_histories(self):
        columns = []
        if self.rows:
            columns = list(self.rows[0].keys())
        requested = str(getattr(self.exact_args, "online_interaction_column", "") or "")
        interaction_column = _online_optional_column(columns, requested, ONLINE_INTERACTION_COLUMNS)
        for row_index, row in enumerate(self.rows):
            raw_path = str(row.get(interaction_column, "")).strip() if interaction_column else ""
            if not raw_path:
                continue
            resolved = resolve_online_video_ref(raw_path, getattr(self.exact_args, "data_root", "."))
            if not isinstance(resolved, Path) or not resolved.is_file():
                raise FileNotFoundError(f"Missing interaction history file for row {row.get('id', row_index)}: {resolved}")
            self.interaction_histories[int(row_index)] = _load_interaction_history_store(resolved)

    def _resolve_disk_cache_dir(self):
        raw = str(getattr(self.exact_args, "online_warp_disk_cache_dir", "") or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(getattr(self.exact_args, "data_root", ".")) / path
        return path

    def _cache_payload(self):
        payload = {
            "cache_schema": 5,
            "height": int(self.exact_args.height),
            "width": int(self.exact_args.width),
            "frame_stride": int(getattr(self.exact_args, "online_frame_stride", 1)),
            "target_fps": float(getattr(self.exact_args, "online_target_fps", 0.0)),
            "use_vpt_camera_poses": bool(getattr(self.exact_args, "online_use_vpt_camera_poses", False)),
            "pose_convention": str(getattr(self.exact_args, "pose_convention", "opencv_c2w_relative")),
            "vpt_translation_scale": float(getattr(self.exact_args, "online_vpt_translation_scale", 0.1)),
            "direction_augmentation": bool(getattr(self.exact_args, "online_direction_augmentation", False)),
            "geometry_keyframe_stride": int(
                getattr(self.exact_args, "online_geometry_keyframe_stride", 1)
            ),
            "camera_keyframe_max_previous": int(
                getattr(self.exact_args, "online_max_history_frames", 19)
            ),
            "max_video_frames": int(getattr(self.exact_args, "online_max_video_frames", 0)),
            "pi3_pixel_limit": int(getattr(self.exact_args, "online_pi3_pixel_limit", CAMERA_CONTROL_PI3_PIXEL_LIMIT)),
            "pi3_conf_threshold": float(getattr(self.exact_args, "online_pi3_conf_threshold", 0.1)),
            "pi3_depth_edge_rtol": float(getattr(self.exact_args, "online_pi3_depth_edge_rtol", 0.03)),
            "mesh_samples_per_axis": int(getattr(self.exact_args, "online_mesh_samples_per_axis", 4)),
            "render_mode": str(getattr(self.exact_args, "online_render_mode", CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)),
            "target_fill_radius": int(
                getattr(self.exact_args, "online_target_fill_radius", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)
            ),
            "target_fill_min_neighbors": int(
                getattr(
                    self.exact_args,
                    "online_target_fill_min_neighbors",
                    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
                )
            ),
            "mesh_break_mode": str(getattr(self.exact_args, "online_mesh_break_mode", CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)),
            "mesh_depth_rtol": float(
                getattr(self.exact_args, "online_mesh_depth_rtol", CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL)
            ),
            "mesh_normal_tol_deg": float(
                getattr(self.exact_args, "online_mesh_normal_tol_deg", CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG)
            ),
            "pi3x_checkpoint": str(getattr(self.exact_args, "online_pi3x_checkpoint", "default")),
        }
        if bool(getattr(self.exact_args, "use_minecraft_hud_mask", False)):
            payload["minecraft_hud_mask"] = "mc_640x360_v1"
        return payload

    def _file_content_digest(self, path):
        path = Path(path)
        cache = getattr(self, "_source_content_digest_cache", None)
        if cache is None:
            cache = {}
            self._source_content_digest_cache = cache
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        if key not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            cache[key] = digest.hexdigest()
        return cache[key]

    def _video_source_payload(self, ref):
        if not isinstance(ref, Path):
            return {"kind": "uri", "value": str(ref)}

        path = ref.expanduser().resolve(strict=False)
        payload = {"kind": "directory" if path.is_dir() else "file", "path": str(path)}
        if path.is_file():
            stat = path.stat()
            payload.update(
                {
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "content_sha256": self._file_content_digest(path),
                }
            )
        elif path.is_dir():
            files = []
            for image_path in _iter_online_image_files(path):
                stat = image_path.stat()
                files.append(
                    {
                        "name": image_path.name,
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "content_sha256": self._file_content_digest(image_path),
                    }
                )
            payload["files_digest"] = hashlib.sha256(
                json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            payload["file_count"] = len(files)
        else:
            payload["missing"] = True
        return payload

    def _video_source_digest(self, ref):
        payload = self._video_source_payload(ref)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _geometry_cache_digest(self, ref, direction, source_fps=0.0):
        payload = {
            "source": self._video_source_payload(ref),
            "source_fps": float(source_fps or 0.0),
            "direction": str(direction),
            "config": self._cache_payload(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _geometry_cache_path(self, ref, direction, source_fps=0.0):
        if self.disk_cache_dir is None:
            return None
        digest = self._geometry_cache_digest(ref, direction, source_fps=source_fps)[:24]
        return self.disk_cache_dir / f"geometry_{direction}_{digest}.pt"

    def _conditioning_geometry_cache_path(self, ref, direction, frame_indices, source_fps=0.0):
        if self.disk_cache_dir is None:
            return None
        payload = {
            "base": self._geometry_cache_digest(ref, direction, source_fps=source_fps),
            "conditioning_frame_indices": [int(index) for index in frame_indices],
            "target_rgb_used": False,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return self.disk_cache_dir / f"conditioning_geometry_{direction}_{digest}.pt"

    def _record_cache_key(self, ref, direction, source_fps=0.0):
        return str(direction), self._geometry_cache_digest(ref, direction, source_fps=source_fps)

    def _load_frames(self, ref, direction):
        frames, source_indices = load_online_video_frames(
            ref,
            height=int(self.exact_args.height),
            width=int(self.exact_args.width),
            frame_stride=int(getattr(self.exact_args, "online_frame_stride", 1)),
            max_video_frames=int(getattr(self.exact_args, "online_max_video_frames", 0)),
            target_fps=float(getattr(self.exact_args, "online_target_fps", 0.0)),
            return_source_indices=True,
        )
        if direction == "reverse":
            frames = list(reversed(frames))
            source_indices = list(reversed(source_indices))
        return frames, source_indices

    def _release_record(self, record):
        if not record:
            return
        frames = record.get("frames")
        geometry = record.get("geometry")
        if isinstance(frames, list):
            frames.clear()
        if isinstance(geometry, dict):
            geometry.clear()
        record.clear()
        self.renderer._pi3x_runtime = None
        gc.collect()
        opt.clean_memory()

    def _load_geometry_from_disk(self, cache_path):
        if cache_path is None or not cache_path.is_file():
            return None
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("config", {}).get("cache_schema") != self._cache_payload()["cache_schema"]:
            print(json.dumps({"event": "online_warp_geometry_cache_rejected", "reason": "schema_mismatch", "path": str(cache_path)}), flush=True)
            return None
        geometry = payload.get("geometry")
        if not isinstance(geometry, dict):
            return None
        print(
            json.dumps(
                {
                    "event": "online_warp_geometry_cache_hit",
                    "cache": "disk",
                    "path": str(cache_path),
                }
            ),
            flush=True,
        )
        return geometry

    def _save_geometry_to_disk(self, cache_path, geometry, row_index, direction, ref, frames):
        if cache_path is None:
            return
        payload = {
            "geometry": geometry,
            "meta": {
                "row_index": int(row_index),
                "seq": str(self.rows[int(row_index)].get("id", row_index)),
                "direction": str(direction),
                "video": str(ref),
                "frames": int(len(frames)),
                "config": self._cache_payload(),
            },
        }
        torch.save(payload, cache_path)

    def _estimate_geometry(self, row_index, row, direction, ref, frames):
        use_hud_mask = bool(getattr(self.exact_args, "use_minecraft_hud_mask", False))
        geometry_stride = max(1, int(getattr(self.exact_args, "online_geometry_keyframe_stride", 1)))
        if geometry_stride > 1 and not bool(getattr(self.exact_args, "online_use_vpt_camera_poses", False)):
            raise ValueError("Sparse Pi3X geometry requires --online_use_vpt_camera_poses.")
        geometry_frame_indices = list(range(0, len(frames), geometry_stride))
        if geometry_frame_indices[-1] != len(frames) - 1:
            geometry_frame_indices.append(len(frames) - 1)
        geometry_frames = [frames[index] for index in geometry_frame_indices]
        pi3_frames = (
            [fill_minecraft_hud_for_pi3(frame) for frame in geometry_frames]
            if use_hud_mask
            else geometry_frames
        )
        tensors = [online_pil_to_tensor(frame).unsqueeze(0) for frame in pi3_frames]
        print(
            json.dumps(
                {
                    "event": "online_warp_estimate_geometry",
                    "row_index": int(row_index),
                    "seq": str(row["id"]),
                    "direction": direction,
                    "frames": len(frames),
                    "geometry_frames": len(geometry_frames),
                    "geometry_keyframe_stride": geometry_stride,
                    "video": str(ref),
                }
            ),
            flush=True,
        )
        try:
            geometry = self.renderer.estimate_keyframe_geometry(tensors, device=self.device)
            geometry["training_frame_indices"] = geometry_frame_indices
            if use_hud_mask:
                world_valid_mask = minecraft_world_valid_mask(
                    height=int(self.exact_args.height),
                    width=int(self.exact_args.width),
                )
                clear_minecraft_hud_geometry(geometry, world_valid_mask)
        finally:
            del tensors
            if use_hud_mask:
                del pi3_frames
            self.renderer._pi3x_runtime = None
            opt.clean_memory()
        return geometry

    def _estimate_conditioning_geometry(self, row_index, row, direction, ref, frames, frame_indices):
        """Estimate Pi3X geometry from conditioning frames only.

        Target RGB is deliberately unavailable here. Future camera poses remain
        valid controls, but future depth, points, and appearance cannot leak into
        the warp history.
        """
        selected_indices = [int(index) for index in frame_indices]
        if not selected_indices:
            raise ValueError("At least one conditioning frame is required for Pi3X geometry.")
        cache_path = self._conditioning_geometry_cache_path(
            ref,
            direction,
            selected_indices,
            source_fps=float(row.get("fps", 0.0) or 0.0),
        )
        cached = self._load_geometry_from_disk(cache_path)
        if cached is not None:
            return cached
        selected_frames = [frames[index] for index in selected_indices]
        use_hud_mask = bool(getattr(self.exact_args, "use_minecraft_hud_mask", False))
        pi3_frames = (
            [fill_minecraft_hud_for_pi3(frame) for frame in selected_frames]
            if use_hud_mask
            else selected_frames
        )
        tensors = [online_pil_to_tensor(frame).unsqueeze(0) for frame in pi3_frames]
        print(
            json.dumps(
                {
                    "event": "online_warp_estimate_conditioning_geometry",
                    "row_index": int(row_index),
                    "seq": str(row["id"]),
                    "direction": str(direction),
                    "conditioning_frame_indices": selected_indices,
                    "target_rgb_used": False,
                    "video": str(ref),
                }
            ),
            flush=True,
        )
        try:
            geometry = self.renderer.estimate_keyframe_geometry(tensors, device=self.device)
            geometry["training_frame_indices"] = selected_indices
            if use_hud_mask:
                world_valid_mask = minecraft_world_valid_mask(
                    height=int(self.exact_args.height),
                    width=int(self.exact_args.width),
                )
                clear_minecraft_hud_geometry(geometry, world_valid_mask)
        finally:
            del tensors
            if use_hud_mask:
                del pi3_frames
            self.renderer._pi3x_runtime = None
            opt.clean_memory()
        self._save_geometry_to_disk(cache_path, geometry, row_index, direction, ref, selected_frames)
        return geometry

    def _build_record(self, row_index, direction, ref=None):
        row_index = int(row_index)
        row = self.rows[row_index]
        if ref is None:
            ref = resolve_online_video_ref(row["video_path"], getattr(self.exact_args, "data_root", "."))
        frames, source_indices = self._load_frames(ref, direction)
        source_fps = float(row.get("fps", 0.0) or 0.0)
        pose_rows = None
        use_vpt_poses = bool(getattr(self.exact_args, "online_use_vpt_camera_poses", False))
        if use_vpt_poses:
            actions_path = resolve_online_video_ref(
                row.get("actions_path", ""),
                getattr(self.exact_args, "data_root", "."),
            )
            try:
                if not isinstance(actions_path, Path) or not actions_path.is_file():
                    raise FileNotFoundError(
                        f"Missing VPT telemetry for {row.get('id', row_index)}: {actions_path}"
                    )
                pose_rows = load_vpt_pose_rows(actions_path, source_indices)
            except (FileNotFoundError, ValueError) as exc:
                print(
                    json.dumps(
                        {
                            "event": "vpt_pose_fallback_pi3x",
                            "row_index": int(row_index),
                            "seq": str(row.get("id", row_index)),
                            "reason": str(exc),
                        }
                    ),
                    flush=True,
                )
        geometry = None
        if pose_rows is None:
            cache_path = self._geometry_cache_path(ref, direction, source_fps=source_fps)
            geometry = self._load_geometry_from_disk(cache_path)
            if geometry is None:
                geometry = self._estimate_geometry(row_index, row, direction, ref, frames)
                self._save_geometry_to_disk(cache_path, geometry, row_index, direction, ref, frames)
        return {
            "direction": direction,
            "frames": frames,
            "source_indices": source_indices,
            "pose_rows": pose_rows,
            "geometry": geometry,
            "row": row,
            "row_index": row_index,
            "video_ref": str(ref),
        }

    def _get_record(self, row_index, direction):
        row_index = int(row_index)
        row = self.rows[row_index]
        ref = resolve_online_video_ref(row["video_path"], getattr(self.exact_args, "data_root", "."))
        key = self._record_cache_key(ref, direction, source_fps=float(row.get("fps", 0.0) or 0.0))
        cached = self.records.get(key)
        if cached is not None:
            self.records.move_to_end(key)
            print(
                json.dumps(
                    {
                        "event": "online_warp_record_cache_hit",
                        "cache": "memory",
                        "row_index": int(row_index),
                        "seq": str(row.get("id", row_index)),
                        "direction": str(direction),
                        "video": str(ref),
                    }
                ),
                flush=True,
            )
            return cached

        record = self._build_record(row_index, direction, ref=ref)
        self.records[key] = record
        while len(self.records) > self.memory_cache_size:
            _old_key, old_record = self.records.popitem(last=False)
            self._release_record(old_record)
        return record

    def choose_direction(self, rng):
        if not bool(getattr(self.exact_args, "online_direction_augmentation", False)):
            return "forward"
        reverse_prob = float(getattr(self.exact_args, "online_direction_reverse_prob", 0.5))
        return "reverse" if rng.random() < reverse_prob else "forward"

    def _interaction_metadata_prefilter(
        self,
        row,
        *,
        category,
        requested_chunk_mode,
        event_local_frame,
    ):
        if str(getattr(self.exact_args, "training_profile", "joint")) != "interaction":
            return None
        if category not in {"place", "mine"}:
            return None
        max_rotation = float(getattr(self.exact_args, "interaction_max_metadata_rotation_deg", 0.0) or 0.0)
        if max_rotation > 0.0:
            rotation_value = first_present_data_value(row.get("cumulative_rotation"), default=None)
            if data_value_present(rotation_value) and float(rotation_value) > max_rotation:
                return f"metadata_rotation_exceeds_threshold:{float(rotation_value):.3f}"
        min_conf = float(getattr(self.exact_args, "interaction_min_telemetry_confidence", 0.0) or 0.0)
        if min_conf > 0.0:
            conf_value = first_present_data_value(row.get("telemetry_confidence"), default=None)
            if data_value_present(conf_value) and float(conf_value) < min_conf:
                return f"telemetry_confidence_below_threshold:{float(conf_value):.3f}"
        min_active_frames = int(getattr(self.exact_args, "interaction_min_mine_active_frames", 0) or 0)
        if (
            category == "mine"
            and str(row.get("action_type", "") or "").strip().lower() == "mine_active"
            and min_active_frames > 0
        ):
            active_frames = int(first_present_data_value(row.get("stable_active_frames"), default=0) or 0)
            if active_frames < min_active_frames:
                return f"mine_active_too_short:{active_frames}"
        if requested_chunk_mode == "interaction_first" and event_local_frame is not None:
            local_min = int(getattr(self.exact_args, "interaction_event_local_min", 6))
            local_max = int(getattr(self.exact_args, "interaction_event_local_max", 16))
            if not (local_min <= int(event_local_frame) <= local_max):
                return f"event_local_out_of_range:{int(event_local_frame)}"
        return None

    def sample_case(
        self,
        row_index,
        prepare_index,
        requested_category=None,
        requested_chunk_mode=None,
    ):
        row_index = int(row_index)
        row = self.rows[row_index]
        category = canonical_training_category(requested_category or row.get("training_category", row.get("category")))
        if category != canonical_training_category(row.get("training_category", row.get("category"))):
            raise ValueError(f"Sampler requested {category} for incompatible row {row.get('id', row_index)}.")
        rng = random.Random(
            opt.stable_seed_from_parts(int(self.exact_args.seed), "online_warp_training", row["id"], int(prepare_index))
        )
        direction = self.choose_direction(rng)
        prepared = self._get_record(row_index, direction)
        event_path = resolve_optional_data_path(row.get("primary_fire_event_path", ""), getattr(self.exact_args, "data_root", "."))
        full_event_payload = load_primary_fire_event_payload(event_path) if event_path is not None and event_path.is_file() else None
        loss_mask_path = resolve_optional_data_path(row.get("primary_fire_loss_mask_path", ""), getattr(self.exact_args, "data_root", "."))
        full_focus_mask_frames = (
            load_primary_fire_loss_mask_frames(loss_mask_path) if loss_mask_path is not None and loss_mask_path.is_file() else None
        )
        interaction_path = resolve_optional_data_path(
            row.get("interaction_event_path", ""), getattr(self.exact_args, "data_root", ".")
        )
        full_interaction_payload = (
            load_primary_fire_event_payload(interaction_path)
            if interaction_path is not None and interaction_path.is_file()
            else full_event_payload
        )
        frames = prepared["frames"]
        if direction == "reverse":
            full_event_payload = reverse_temporal_event_payload(full_event_payload, len(frames))
            if full_interaction_payload is not full_event_payload:
                full_interaction_payload = reverse_temporal_event_payload(full_interaction_payload, len(frames))
            if full_focus_mask_frames is not None:
                full_focus_mask_frames = list(reversed(full_focus_mask_frames))
        event_resampled_frame = None
        event_alignment = None
        if category in {"place", "mine"}:
            event_alignment = event_alignment_from_row(
                row,
                prepared["source_indices"],
                source_fps=float(row.get("fps", 0.0) or 0.0),
                target_fps=float(getattr(self.exact_args, "online_target_fps", 0.0) or 0.0),
            )
            event_resampled_frame = int(event_alignment["resampled_event_frame"])
            event_end_frame = None
            raw_event_end = first_present_data_value(
                row.get("event_end_frame"), row.get("mine_end_frame")
            )
            if data_value_present(raw_event_end):
                segment_end = int(raw_event_end)
                source_start = int(row.get("source_frame_start", 0) or 0)
                if segment_end >= source_start:
                    segment_end -= source_start
                event_end_frame = int(
                    np.argmin(
                        [abs(int(frame) - segment_end) for frame in prepared["source_indices"]]
                    )
                )
            action_type = canonical_interaction_action(row.get("action_type", category))
            if category == "mine" and not data_value_present(row.get("action_type")):
                action_type = "mine_complete"
            raw_action_end = first_present_data_value(
                row.get("action_end_frame"),
                row.get("event_end_frame"),
                row.get("mine_end_frame"),
            )
            if data_value_present(raw_action_end):
                source_start = int(row.get("source_frame_start", 0) or 0)
                segment_end = int(raw_action_end)
                if segment_end >= source_start:
                    segment_end -= source_start
                event_end_frame = int(
                    np.argmin([abs(int(frame) - segment_end) for frame in prepared["source_indices"]])
                ) + (1 if action_type == "mine_active" else 0)
            complete_frame = None
            raw_complete = first_present_data_value(row.get("complete_frame"))
            if data_value_present(raw_complete):
                source_start = int(row.get("source_frame_start", 0) or 0)
                complete_source = int(raw_complete)
                if complete_source >= source_start:
                    complete_source -= source_start
                complete_frame = int(
                    np.argmin(
                        [abs(int(frame) - complete_source) for frame in prepared["source_indices"]]
                    )
                )
            full_interaction_payload = {
                "events": [
                    {
                        "event_id": str(
                            row.get("event_id")
                            or (
                                f"{event_alignment['source_event_frame']}:"
                                f"{action_type}:"
                                f"{row.get('object_id')}"
                            )
                        ),
                        "event_frame": int(event_resampled_frame),
                        "action_type": action_type,
                        "object_id": first_present_data_value(row.get("object_id"), default="none"),
                        "block_id": first_present_data_value(
                            row.get("block_id"), row.get("object_id"), default="none"
                        ),
                        "event_end_frame": event_end_frame,
                        "action_start_frame": int(event_resampled_frame),
                        "action_end_frame": None if event_end_frame is None else int(event_end_frame),
                        "complete_frame": complete_frame,
                        "source_event_frame": event_alignment["source_event_frame"],
                        "source_event_time_ms": event_alignment["source_event_time_ms"],
                    }
                ]
            }
        elif category in {"movement", "other"}:
            # Movement is WAH camera supervision only.  It must never spend a
            # Router forward or be silently represented as a none-token event.
            full_interaction_payload = None
        elif category == "negative":
            full_interaction_payload = None
        geometry_stride = max(1, int(getattr(self.exact_args, "online_geometry_keyframe_stride", 1)))
        geometry_frame_indices = list(range(0, len(frames), geometry_stride))
        if geometry_frame_indices[-1] != len(frames) - 1:
            geometry_frame_indices.append(len(frames) - 1)
        n = len(frames)
        num_frames = int(self.exact_args.num_frames)
        if n < num_frames:
            raise ValueError(f"Online training video {prepared['video_ref']} has {n} frames, need {num_frames}.")
        target_indices = None
        event_local_frame = None
        chunk_mode = category
        first_chunk_prob = float(getattr(self.exact_args, "online_first_chunk_prob", 0.0) or 0.0)
        if category == "negative" and str(row.get("negative_window_start_frame", "")).strip():
            target_start = int(row["negative_window_start_frame"])
            target_indices = list(range(target_start, target_start + num_frames))
            if target_indices[-1] >= n:
                raise ValueError(f"Negative window exceeds resampled video: {row.get('id', row_index)}.")
            chunk_mode = (
                "first"
                if requested_chunk_mode == "interaction_first"
                else "generated"
                if requested_chunk_mode == "interaction_generated"
                else "later"
            )
        elif category in {"place", "mine"}:
            target_indices, event_local_frame = build_interaction_event_window(
                int(event_resampled_frame),
                num_source_frames=n,
                window_size=num_frames,
                rng=rng,
                local_min=int(getattr(self.exact_args, "interaction_event_local_min", 6)),
                local_max=int(getattr(self.exact_args, "interaction_event_local_max", 16)),
                require_later=requested_chunk_mode != "interaction_first",
            )
            chunk_mode = (
                "first"
                if requested_chunk_mode == "interaction_first"
                else "generated"
                if requested_chunk_mode == "interaction_generated"
                else "later"
            )
            filter_reason = self._interaction_metadata_prefilter(
                row,
                category=category,
                requested_chunk_mode=requested_chunk_mode,
                event_local_frame=event_local_frame,
            )
            if filter_reason is not None:
                raise ValueError(filter_reason)
        elif requested_chunk_mode == "camera_first":
            target_indices = list(range(num_frames))
            chunk_mode = "first"
        elif requested_chunk_mode in {"camera_later", "camera_rollout"}:
            max_chunk_index = (n - num_frames) // num_frames
            minimum_chunk_index = 2 if requested_chunk_mode == "camera_rollout" else 1
            if max_chunk_index < minimum_chunk_index:
                raise ValueError(
                    f"{requested_chunk_mode} requires at least {minimum_chunk_index + 1} chunks."
                )
            chunk_index = rng.randint(minimum_chunk_index, max_chunk_index)
            target_start = chunk_index * num_frames
            target_indices = list(range(target_start, target_start + num_frames))
            chunk_mode = "two_chunk_rollout" if requested_chunk_mode == "camera_rollout" else "later"
        elif (
            str(getattr(self.exact_args, "training_profile", "joint")) == "camera"
            and rng.random() < first_chunk_prob
        ):
            target_indices = list(range(num_frames))
            chunk_mode = "first"
        if target_indices is None:
            target_indices = choose_online_movement_window(rng, n, num_frames, full_event_payload)
        target_start = int(target_indices[0])
        if target_start <= 0 or requested_chunk_mode == "interaction_first":
            if requested_chunk_mode == "camera_first":
                chunk_mode = "first"
            if requested_chunk_mode not in {"camera_first", "interaction_first"}:
                chunk_mode = "first" if chunk_mode == "movement" else f"{chunk_mode}_first"
            source_idx = int(target_start)
            history_indices = []
            geometry_keyframe_frames = [source_idx]
            render_pose_indices = target_indices
            future_keyframe_indices = []
            drop_renderer_source = False
            keyframe_policy = "source_only"
            condition_frame = frames[source_idx]
        else:
            if requested_chunk_mode == "camera_later":
                chunk_mode = "later"
            elif requested_chunk_mode == "camera_rollout":
                chunk_mode = "two_chunk_rollout"
            elif requested_chunk_mode == "interaction_generated":
                chunk_mode = "generated"
            else:
                chunk_mode = "later" if chunk_mode == "movement" else f"{chunk_mode}_later"
            max_history = min(int(getattr(self.exact_args, "online_max_history_frames", 19)), target_start)
            history_len = rng.randint(1, max(1, max_history))
            history_indices = list(range(target_start - history_len, target_start))
            future_keyframe_indices = []
            keyframe_policy = "history_only"
            geometry_keyframe_frames = [
                frame_index
                for frame_index in geometry_frame_indices
                if history_indices[0] <= frame_index < target_start
            ]
            if requested_chunk_mode in {"camera_rollout", "interaction_generated"}:
                rollout_source = int(target_start) - (int(num_frames) - 1)
                if rollout_source < 0:
                    raise ValueError("Generated history requires a complete preceding chunk.")
                geometry_keyframe_frames = [max(rollout_source, 0)]
            if not geometry_keyframe_frames:
                geometry_keyframe_frames = [max(frame_index for frame_index in geometry_frame_indices if frame_index < target_start)]
            render_pose_indices = [geometry_keyframe_frames[-1], *target_indices]
            drop_renderer_source = True
            condition_frame = frames[geometry_keyframe_frames[-1]]

        preview_rotation = None
        max_rotation = float(getattr(self.exact_args, "interaction_max_camera_rotation_deg", 0.0) or 0.0)
        if (
            category in {"place", "mine"}
            and max_rotation > 0.0
            and prepared.get("pose_rows") is not None
        ):
            preview_poses = vpt_relative_camera_poses(
                prepared["pose_rows"],
                int(render_pose_indices[0]),
                render_pose_indices,
                translation_scale=1.0,
            )
            preview_rotation = float(
                pose_motion_statistics(preview_poses, preview_poses)["rotation_degrees"]
            )
            if preview_rotation > max_rotation:
                raise ValueError(f"camera_rotation_exceeds_threshold:{preview_rotation:.3f}")

        keyframe_indices = list(geometry_keyframe_frames)
        geometry = self._estimate_conditioning_geometry(
            row_index,
            row,
            direction,
            prepared["video_ref"],
            frames,
            geometry_keyframe_frames,
        )
        if prepared.get("pose_rows") is not None:
            pose_source_index = int(render_pose_indices[0])
            depth_scale = self.renderer.estimate_first_frame_depth_scale(geometry)
            raw_poses = vpt_relative_camera_poses(
                prepared["pose_rows"],
                pose_source_index,
                render_pose_indices,
                translation_scale=1.0,
            )
            effective_scale = effective_translation_scale(
                float(getattr(self.exact_args, "online_vpt_translation_scale", 0.1)),
                depth_scale,
                multiply_by_depth=bool(
                    getattr(self.exact_args, "camera_multiply_translation_by_depth", True)
                ),
            )
            poses = raw_poses.copy()
            poses[:, :3, 3] *= float(effective_scale)
            pose_source = "vpt_telemetry"
        else:
            full_pose_source = prepared["geometry"]["keyframe_geometries"][int(render_pose_indices[0])][
                "source_pose"
            ]
            poses = online_relative_poses(prepared["geometry"], full_pose_source, render_pose_indices)
            raw_poses = poses.copy()
            depth_scale = self.renderer.estimate_first_frame_depth_scale(geometry)
            effective_scale = 1.0
            pose_source = "pi3x"
        motion_stats = pose_motion_statistics(raw_poses, poses)
        rendered = self.renderer.render_from_geometry(
            geometry,
            poses,
            height=int(self.exact_args.height),
            width=int(self.exact_args.width),
            device=self.device,
            invisible_fill_mode=str(
                getattr(self.exact_args, "online_invisible_fill", CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL)
            ),
            render_mode=str(getattr(self.exact_args, "online_render_mode", CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)),
            target_fill_radius=int(
                getattr(self.exact_args, "online_target_fill_radius", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)
            ),
            target_fill_min_neighbors=int(
                getattr(
                    self.exact_args,
                    "online_target_fill_min_neighbors",
                    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
                )
            ),
            mesh_break_mode=str(getattr(self.exact_args, "online_mesh_break_mode", CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)),
        )
        warp_video = rendered["warp_video"]
        warp_mask = rendered["warp_visibility_mask"]
        if drop_renderer_source:
            warp_video = warp_video[:, :, 1:]
            warp_mask = warp_mask[:, :, 1:]
        warp_frames = online_tensor_video_to_pil_frames(warp_video)
        warp_mask_frames = online_mask_tensor_to_pil_frames(warp_mask)
        if len(warp_frames) != num_frames or len(warp_mask_frames) != num_frames:
            raise ValueError(
                f"Online warp rendered {len(warp_frames)} frames/{len(warp_mask_frames)} masks, need {num_frames}."
            )
        use_hud_mask = bool(getattr(self.exact_args, "use_minecraft_hud_mask", False))
        world_valid_mask = None
        visibility_before_hud = None
        if use_hud_mask:
            world_valid_mask = minecraft_world_valid_mask(
                height=int(self.exact_args.height),
                width=int(self.exact_args.width),
            )
            visibility_before_hud = [frame.copy() for frame in warp_mask_frames]
            warp_mask_frames = multiply_mask_frames(warp_mask_frames, world_valid_mask)
        interaction_memory = summarize_multiscale_interaction_history(
            self.interaction_histories.get(row_index),
            history_indices,
            target_indices,
            max_items=int(getattr(self.exact_args, "online_interaction_max_items", 8)),
        )
        event_payload = crop_primary_fire_event_payload(full_event_payload, target_indices) if full_event_payload is not None else None
        interaction_payload = (
            crop_interaction_payload(full_interaction_payload, target_indices)
            if category in {"place", "mine"}
            else None
        )
        if category == "negative":
            interaction_payload = {
                "event_id": f"negative:{row.get('id', row_index)}:{target_start}",
                "event_frame": 0,
                "action_type": "none",
                "object_id": None,
                "block_id": None,
                "event_valid": 0.0,
                "num_frames": int(num_frames),
                "time_mask": [0.0] * int(num_frames),
            }
        if category in {"place", "mine"}:
            if interaction_payload is None or float(interaction_payload.get("event_valid", 0.0)) != 1.0:
                raise RuntimeError(f"Positive {category} row lost its event window: {row.get('id', row_index)}.")
            if not (
                int(getattr(self.exact_args, "interaction_event_local_min", 6))
                <= int(interaction_payload["event_frame"])
                <= int(getattr(self.exact_args, "interaction_event_local_max", 16))
            ):
                raise RuntimeError(f"Positive event escaped the requested local range: {interaction_payload['event_frame']}.")
        focus_mask_frames = None
        focus_mask_stats = {}
        if event_payload is not None:
            primary_fire_supervision = {
                "click_frames": [int(x) for x in event_payload.get("click_frames_source", [])],
                "target_click_frames": [
                    int(x) for x in event_payload.get("click_frames_local", [])
                ],
                "target_has_click": bool(event_payload.get("click_frames_local")),
                "temporal_mask": list(event_payload.get("time_mask", [])),
            }
            if full_focus_mask_frames is not None:
                focus_mask_frames = crop_mask_frames(full_focus_mask_frames, target_indices)
                focus_mask_stats = {"source": "precomputed_primary_fire_loss_mask"}
        else:
            primary_fire_supervision = build_primary_fire_click_supervision(
                self.interaction_histories.get(row_index),
                target_indices,
                radius_frames=int(getattr(self.exact_args, "online_primary_fire_click_radius_frames", 12)),
            )
        if focus_mask_frames is None:
            focus_mask_frames, focus_mask_stats = build_primary_fire_focus_mask_frames(
                [frames[idx] for idx in target_indices],
                warp_frames,
                primary_fire_supervision,
                residual_threshold=float(getattr(self.exact_args, "online_primary_fire_residual_threshold", 0.08)),
            )
        seq = f"{row['id']}:{direction}:{chunk_mode}:{int(prepare_index)}"
        if world_valid_mask is not None:
            focus_mask_frames = multiply_mask_frames(focus_mask_frames, world_valid_mask)
            debug_dir = (
                Path(getattr(self.exact_args, "output_dir", "runs/warp_as_history_lora"))
                / "hud_debug"
                / _safe_debug_name(seq)
            )
            _save_mask_debug(debug_dir / "hud_valid_mask.png", world_valid_mask)
            fill_minecraft_hud_for_pi3(condition_frame).save(debug_dir / "pi3_input_hud_filled.png")
            _save_mask_debug(debug_dir / "visibility_before_hud.png", visibility_before_hud[0])
            _save_mask_debug(debug_dir / "visibility_after_hud.png", warp_mask_frames[0])
            focus_debug = np.maximum.reduce(
                [np.asarray(frame.convert("L"), dtype=np.uint8) for frame in focus_mask_frames]
            )
            _save_mask_debug(
                debug_dir / "focus_mask_after_hud.png",
                Image.fromarray(focus_debug, mode="L"),
            )
        result = {
            "condition_frame": condition_frame,
            "direction": direction,
            "history_indices": history_indices,
            "keyframe_indices": keyframe_indices,
            "keyframe_policy": keyframe_policy,
            "future_keyframe_indices": future_keyframe_indices,
            "metadata": {
                "chunk_mode": chunk_mode,
                "direction": direction,
                "future_keyframe_indices": future_keyframe_indices,
                "history_indices": history_indices,
                "keyframe_indices": keyframe_indices,
                "keyframe_policy": keyframe_policy,
                "render_pose_indices": render_pose_indices,
                "row_index": int(row_index),
                "seq": seq,
                "target_indices": target_indices,
                "target_start_frame": int(target_start),
                "chunk_index": int(target_start // num_frames),
                "requested_chunk_mode": requested_chunk_mode,
                "video": prepared["video_ref"],
                "warp_render_stats": rendered.get("warp_render_stats", {}),
                "interaction_history_text": interaction_memory.get("merged", ""),
                "interaction_memory": interaction_memory,
                "primary_fire_supervision": primary_fire_supervision,
                "primary_fire_event_payload": event_payload,
                "interaction_payload": interaction_payload,
                "focus_mask_stats": focus_mask_stats,
                "sample_window_type": chunk_mode,
                "training_category": category,
                "event_local_frame": event_local_frame,
                "source_fps": float(row.get("fps", 0.0) or 0.0),
                "target_fps": float(getattr(self.exact_args, "online_target_fps", 0.0) or 0.0),
                "source_event_frame": None if event_alignment is None else event_alignment["source_event_frame"],
                "source_event_time_ms": None
                if event_alignment is None
                else event_alignment["source_event_time_ms"],
                "resampled_event_frame": event_resampled_frame,
                "resampled_event_time_ms": None
                if event_alignment is None
                else event_alignment["resampled_event_time_ms"],
                "pose_source": pose_source,
                "pose_convention": POSE_CONVENTION,
                "raw_translation_norm": motion_stats["raw_translation_norm"],
                "median_scene_depth": float(depth_scale),
                "effective_translation_scale": float(effective_scale),
                "rendered_translation_norm": motion_stats["rendered_translation_norm"],
                "rotation_degrees": motion_stats["rotation_degrees"],
                "metadata_prefilter_rotation_degrees": preview_rotation,
            },
            "interaction_history_text": interaction_memory.get("merged", ""),
            "interaction_memory": interaction_memory,
            "primary_fire_supervision": primary_fire_supervision,
            "primary_fire_event_payload": event_payload,
            "interaction_payload": interaction_payload,
            "training_category": category,
            "prompt": row["prompt"],
            "prompt_base": row["prompt"],
            "prompt_raw": row.get("prompt_raw", row["prompt"]),
            "row": row,
            "seq": seq,
            "camera_pose_source": pose_source,
            "camera_translation_scale": float(effective_scale),
            "camera_raw_relative_poses": raw_poses.copy(),
            "target_frames": [frames[idx] for idx in target_indices],
            "target_indices": target_indices,
            "warp_frames": warp_frames,
            "warp_mask_frames": warp_mask_frames,
            "focus_mask_frames": focus_mask_frames,
            "world_valid_mask_frames": None
            if world_valid_mask is None
            else [world_valid_mask.copy() for _ in range(num_frames)],
        }
        if requested_chunk_mode in {"camera_rollout", "interaction_generated"}:
            previous_start = int(target_start) - (int(num_frames) - 1)
            previous_indices = list(range(previous_start, int(target_start) + 1))
            if previous_start < 0 or prepared.get("pose_rows") is None:
                raise ValueError("Two-chunk rollout requires an earlier VPT-controlled chunk.")
            result["rollout_source_frame"] = frames[previous_start]
            result["rollout_camera_poses"] = vpt_relative_camera_poses(
                prepared["pose_rows"],
                previous_start,
                previous_indices,
                translation_scale=1.0,
            )
            result["rollout_previous_indices"] = previous_indices
        del rendered, warp_video, warp_mask, geometry, poses
        self.renderer._pi3x_runtime = None
        opt.clean_memory()
        return result


def build_online_warp_training_cache(df, exact_args, device):
    rows = [row for _, row in df.iterrows()]
    return OnlineWarpTrainingCache(rows, exact_args, device)


def fixed_cache_rows_ready(rows, data_root="."):
    rows = [row.to_dict() if hasattr(row, "to_dict") else dict(row) for row in rows]
    if not rows:
        return False
    for row in rows:
        if str(row.get("review_status", "")).strip().lower() != "approved":
            return False
        for field in ("training_cache_path", "teacher_cache_path"):
            path = resolve_optional_data_path(row.get(field, ""), data_root)
            if path is None or not path.is_file():
                return False
    return True

def prompt_cache_key(exact_args, prompt):
    payload = {
        "base_model_path": str(exact_args.base_model_path),
        "prompt": str(prompt),
        "num_videos_per_prompt": 1,
        "max_sequence_length": 512,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_prompt_cached(pipe, prompt, exact_args, device, cache_dir, memory_cache):
    key = prompt_cache_key(exact_args, prompt)
    if key in memory_cache:
        cached = memory_cache[key]
        return cached["prompt_embeds"], "memory"

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            prompt_embeds = payload["prompt_embeds"].to(device=device, dtype=pipe.transformer.dtype)
            memory_cache[key] = {"prompt_embeds": prompt_embeds}
            return prompt_embeds, "disk"

    with torch.no_grad():
        prompt_embeds, _negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=opt.NEGATIVE_PROMPT,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
    prompt_embeds = prompt_embeds.to(pipe.transformer.dtype)
    memory_cache[key] = {"prompt_embeds": prompt_embeds.detach()}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_embeds": prompt_embeds.detach().cpu(),
                "meta": {
                    "prompt": str(prompt),
                    "base_model_path": str(exact_args.base_model_path),
                    "max_sequence_length": 512,
                },
            },
            cache_path,
        )
    return prompt_embeds, "encode"


def _restore_optional_attr(obj, name, had_value, old_value):
    if had_value:
        setattr(obj, name, old_value)
    elif hasattr(obj, name):
        delattr(obj, name)


def prepare_online_warp_item(
    pipe,
    row_index,
    exact_args,
    device,
    mean,
    std,
    keep_frames,
    cache_dir,
    memory_prompt_cache,
    prepare_index=0,
    requested_category=None,
    requested_chunk_mode=None,
):
    online_cache = getattr(exact_args, "online_warp_cache", None)
    if online_cache is None:
        raise ValueError("Training cache is missing.")
    case = online_cache.sample_case(
        row_index,
        prepare_index,
        requested_category=requested_category,
        requested_chunk_mode=requested_chunk_mode,
    )
    row = online_cache.rows[int(row_index)]
    seq = case["seq"]
    prompt_text = str(case["prompt"])
    first_frame = case["condition_frame"]
    target_frames = case["target_frames"]
    history_frames = case["warp_frames"]
    mask_frames = case["warp_mask_frames"]
    history_corruption = "clean"
    if (
        str(getattr(exact_args, "training_profile", "joint")) == "camera"
        and str(case.get("metadata", {}).get("chunk_mode")) == "later"
    ):
        history_corruption = ("clean", "latent_noise", "downsample", "photometric")[
            int(prepare_index) % 4
        ]
        if history_corruption == "photometric":
            factor_seed = random.Random(
                opt.stable_seed_from_parts(int(exact_args.seed), case["seq"], "history_photometric")
            )
            brightness = factor_seed.uniform(0.95, 1.05)
            contrast = factor_seed.uniform(0.95, 1.05)
            saturation = factor_seed.uniform(0.95, 1.05)
            history_frames = [
                ImageEnhance.Color(
                    ImageEnhance.Contrast(
                        ImageEnhance.Brightness(frame).enhance(brightness)
                    ).enhance(contrast)
                ).enhance(saturation)
                for frame in history_frames
            ]
    case["metadata"]["history_corruption"] = history_corruption

    had_extra_mask = hasattr(exact_args, "history_visibility_extra_mask_frames")
    old_extra_mask = getattr(exact_args, "history_visibility_extra_mask_frames", None)
    exact_args.history_visibility_extra_mask_frames = mask_frames
    loss_focus_mask_latents = None
    world_valid_mask_latents = None
    primary_fire_event_latents = None
    primary_fire_event_debug = None
    interaction_teacher_map = None
    interaction_conditioning = None
    interaction_teacher_components = None

    try:
        with torch.no_grad():
            target_latents = opt.encode_video_latents(pipe, target_frames, exact_args, device, mean, std).detach()
            cached_prompt_embeds, prompt_cache_status = encode_prompt_cached(
                pipe,
                prompt_text,
                exact_args,
                device,
                cache_dir,
                memory_prompt_cache,
            )
            rollout_history_frames = None
            if str(case.get("metadata", {}).get("chunk_mode")) in {"two_chunk_rollout", "generated"}:
                transformer_training = bool(pipe.transformer.training)
                pipe.transformer.eval()
                try:
                    rollout_output = pipe(
                        prompt=prompt_text,
                        image=case["rollout_source_frame"],
                        camera_poses=case["rollout_camera_poses"],
                        lora_path="current",
                        height=int(exact_args.height),
                        width=int(exact_args.width),
                        num_frames=int(exact_args.num_frames),
                        output_type="np",
                        generator=torch.Generator(device=device).manual_seed(
                            opt.stable_seed_from_parts(int(exact_args.seed), seq, "two_chunk_rollout")
                        ),
                        pyramid_num_inference_steps_list=list(
                            exact_args.pyramid_num_inference_steps_list
                        ),
                        camera_control_translation_scale=float(
                            getattr(exact_args, "online_vpt_translation_scale", 0.1)
                        ),
                        camera_control_translation_scale_use_first_frame_depth=bool(
                            getattr(exact_args, "camera_multiply_translation_by_depth", True)
                        ),
                        camera_control_warp_render_mode=str(exact_args.online_render_mode),
                        camera_control_mesh_samples_per_axis=int(
                            exact_args.online_mesh_samples_per_axis
                        ),
                        camera_keyframe_max_previous=int(exact_args.online_max_history_frames),
                        visible_token_threshold=float(exact_args.history_visible_token_threshold),
                        target_fps=float(exact_args.online_target_fps),
                        interaction_conditioning_mode="off",
                    )
                    rollout_history_frames = pipeline_output_to_pil_frames(rollout_output)
                    if len(rollout_history_frames) != int(exact_args.num_frames):
                        raise RuntimeError(
                            f"Two-chunk rollout generated {len(rollout_history_frames)} frames."
                        )
                finally:
                    pipe.transformer.train(transformer_training)
            prompt_embeds, image_latents, fake_image_latents, video_latents = opt.prepare_condition(
                pipe,
                first_frame,
                prompt_text,
                exact_args,
                device,
                mean,
                std,
                history_frames=history_frames,
                prompt_embeds_override=cached_prompt_embeds,
            )
            if history_corruption == "latent_noise":
                video_latents = (
                    video_latents
                    + torch.randn_like(video_latents) * 0.01
                )
            elif history_corruption == "downsample":
                original_size = video_latents.shape[2:]
                reduced_size = (
                    original_size[0],
                    max(1, original_size[1] // 2),
                    max(1, original_size[2] // 2),
                )
                video_latents = torch.nn.functional.interpolate(
                    video_latents.float(),
                    size=reduced_size,
                    mode="trilinear",
                    align_corners=False,
                )
                video_latents = torch.nn.functional.interpolate(
                    video_latents,
                    size=original_size,
                    mode="trilinear",
                    align_corners=False,
                ).to(image_latents)
            histories = opt.make_histories(
                pipe,
                image_latents,
                fake_image_latents,
                exact_args,
                device,
                video_latents=video_latents,
                seq=seq,
            )
            base_history_args = copy.copy(exact_args)
            base_history_args.use_warp_as_history = False
            if rollout_history_frames is None:
                base_histories = opt.make_histories(
                    pipe,
                    image_latents,
                    fake_image_latents,
                    base_history_args,
                    device,
                    video_latents=video_latents,
                    seq=f"{seq}:helios_base",
                )
            else:
                (
                    _rollout_prompt,
                    rollout_image_latents,
                    rollout_fake_image_latents,
                    rollout_video_latents,
                ) = opt.prepare_condition(
                    pipe,
                    case["rollout_source_frame"],
                    prompt_text,
                    base_history_args,
                    device,
                    mean,
                    std,
                    history_frames=rollout_history_frames,
                    prompt_embeds_override=cached_prompt_embeds,
                )
                base_histories = opt.make_histories(
                    pipe,
                    rollout_image_latents,
                    rollout_fake_image_latents,
                    base_history_args,
                    device,
                    video_latents=rollout_video_latents.detach(),
                    seq=f"{seq}:generated_rollout_history",
                )
            loss_focus_mask_latents = online_mask_frames_to_latent_mask(
                case.get("focus_mask_frames"),
                target_latents=target_latents,
                num_frames=int(exact_args.num_frames),
                temporal_scale=int(pipe.vae_scale_factor_temporal),
                device=device,
            )
            world_valid_mask_latents = online_mask_frames_to_latent_mask(
                case.get("world_valid_mask_frames"),
                target_latents=target_latents,
                num_frames=int(exact_args.num_frames),
                temporal_scale=int(pipe.vae_scale_factor_temporal),
                device=device,
                interpolation_mode="nearest",
            )
            visibility_latents = online_mask_frames_to_latent_mask(
                case.get("warp_mask_frames"),
                target_latents=target_latents,
                num_frames=int(exact_args.num_frames),
                temporal_scale=int(pipe.vae_scale_factor_temporal),
                device=device,
                interpolation_mode="nearest",
            )
            interaction_payload = case.get("interaction_payload")
            interaction_active = bool(
                interaction_payload
                and (
                    (
                        float(interaction_payload.get("event_valid", 0.0)) > 0.0
                        and str(case.get("training_category", "")) in {"place", "mine"}
                    )
                    or str(case.get("training_category", "")) == "negative"
                )
            )
            if str(getattr(exact_args, "interaction_conditioning_mode", "router")) == "router" and interaction_active:
                interaction_teacher_components = build_residual_teacher_components(
                    target_latents,
                    video_latents,
                    visibility_latents,
                    world_valid_mask_latents,
                    interaction_payload,
                    camera_rotation_degrees=case.get("metadata", {}).get("rotation_degrees"),
                    max_camera_rotation_degrees=float(
                        getattr(exact_args, "interaction_max_camera_rotation_deg", 0.0) or 0.0
                    )
                    or None,
                )
                fixed_teacher_path = resolve_optional_data_path(
                    row.get("teacher_cache_path", ""), getattr(exact_args, "data_root", ".")
                )
                if fixed_teacher_path is not None and fixed_teacher_path.is_file():
                    fixed_payload = np.load(fixed_teacher_path)
                    fixed_teacher = torch.as_tensor(
                        fixed_payload["teacher"], device=target_latents.device, dtype=torch.float32
                    )
                    if fixed_teacher.ndim == 4:
                        fixed_teacher = fixed_teacher.unsqueeze(0)
                    if fixed_teacher.ndim != 5:
                        raise ValueError(
                            f"Fixed teacher must have shape [B,1,T,H,W], got {tuple(fixed_teacher.shape)}"
                        )
                    if fixed_teacher.shape[2:] != target_latents.shape[2:]:
                        fixed_teacher = torch.nn.functional.interpolate(
                            fixed_teacher,
                            size=target_latents.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                    fixed_teacher = fixed_teacher.clamp(0.0, 1.0).detach()
                    interaction_teacher_components["clean_teacher_mask"] = fixed_teacher
                    interaction_teacher_components["teacher_valid"] = torch.tensor(
                        [str(row.get("review_status", "")).strip().lower() == "approved"],
                        device=target_latents.device,
                        dtype=torch.bool,
                    )
                    interaction_teacher_components["teacher_invalid_reasons"] = (
                        []
                        if bool(interaction_teacher_components["teacher_valid"].all())
                        else ["teacher_pool_not_approved"]
                    )
                    support = fixed_teacher > float(
                        getattr(exact_args, "interaction_teacher_support_threshold", 0.25)
                    )
                    interaction_teacher_components["teacher_area_ratio"] = support.float().mean(dim=(1, 2, 3, 4))
                interaction_teacher_map = interaction_teacher_components["clean_teacher_mask"]
                interaction_conditioning = {
                    "payload": interaction_payload_tensors(interaction_payload, device),
                    "warp_latents": video_latents.detach(),
                    "visibility": visibility_latents.detach(),
                    "world_valid": None
                    if world_valid_mask_latents is None
                    else world_valid_mask_latents.detach(),
                }
            if bool(getattr(exact_args, "use_primary_fire_event_condition", False)):
                event_payload = case.get("primary_fire_event_payload") or {
                    "click_frames": case.get("primary_fire_supervision", {}).get("click_frames", []),
                    "source_frame_indices": case.get("target_indices", []),
                    "time_mask": case.get("primary_fire_supervision", {}).get("temporal_mask", []),
                }
                primary_fire_event_latents, mapping = build_primary_fire_event_latents(
                    event_payload=event_payload,
                    target_indices=case.get("target_indices", []),
                    target_latents=target_latents,
                    temporal_scale=int(pipe.vae_scale_factor_temporal),
                    device=device,
                )
                primary_fire_event_debug = {
                    "frame_to_latent_mapping": mapping,
                    "source_frame_indices": list(event_payload.get("source_frame_indices", case.get("target_indices", []))),
                    "click_frames": list(event_payload.get("click_frames", [])),
                }
    finally:
        _restore_optional_attr(exact_args, "history_visibility_extra_mask_frames", had_extra_mask, old_extra_mask)

    item = {
        "seq": seq,
        "prompt": prompt_text,
        "prompt_raw": case.get("prompt_raw", prompt_text),
        "target_latents": target_latents,
        "prompt_embeds": prompt_embeds.detach(),
        "histories": detach_tree(histories),
        "base_histories": detach_tree(base_histories),
        "prompt_cache_status": prompt_cache_status,
        "training": case["metadata"],
        "interaction_memory": case.get("interaction_memory"),
        "primary_fire_supervision": case.get("primary_fire_supervision"),
        "primary_fire_time_mask": None
        if case.get("primary_fire_event_payload") is None
        else list(case["primary_fire_event_payload"].get("time_mask", [])),
        "primary_fire_event": case.get("primary_fire_event_payload"),
        "loss_focus_mask_latents": None if loss_focus_mask_latents is None else loss_focus_mask_latents.detach(),
        "world_valid_mask_latents": None
        if world_valid_mask_latents is None
        else world_valid_mask_latents.detach(),
        "primary_fire_event_latents": None
        if primary_fire_event_latents is None
        else primary_fire_event_latents.detach(),
        "primary_fire_event_debug": primary_fire_event_debug,
        "interaction_payload": case.get("interaction_payload"),
        "interaction_conditioning": detach_tree(interaction_conditioning),
        "interaction_teacher_map": None
        if interaction_teacher_map is None
        else interaction_teacher_map.detach(),
        "initial_teacher_map": None
        if interaction_teacher_map is None
        else interaction_teacher_map.detach(),
        "interaction_teacher_valid": bool(
            interaction_teacher_components is not None
            and bool(interaction_teacher_components["teacher_valid"].all().item())
        ),
        "interaction_teacher_area_ratio": None
        if interaction_teacher_components is None
        else float(interaction_teacher_components["teacher_area_ratio"].mean().item()),
        "interaction_teacher_visibility_ratio": None
        if interaction_teacher_components is None
        else float(interaction_teacher_components["teacher_visibility_ratio"].mean().item()),
        "interaction_teacher_invalid_reasons": []
        if interaction_teacher_components is None
        else list(interaction_teacher_components["teacher_invalid_reasons"]),
        "interaction_debug_inputs": None,
    }
    if interaction_active:
        event_frame = int((case.get("interaction_payload") or {}).get("event_frame", 0))
        debug_indices = sorted(
            {
                0,
                max(event_frame - 1, 0),
                event_frame,
                min(event_frame + 1, len(target_frames) - 1),
                len(target_frames) - 1,
            }
        )
        item["interaction_debug_inputs"] = {
            "frame_indices": debug_indices,
            "target_frames": [target_frames[index].copy() for index in debug_indices],
            "warp_frames": [history_frames[index].copy() for index in debug_indices],
            "visibility_frames": [mask_frames[index].copy() for index in debug_indices],
            "raw_residual": interaction_teacher_components["raw_residual"].detach(),
            "clean_teacher_mask": interaction_teacher_map.detach(),
        }
    if keep_frames:
        item["target_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in target_frames]
        item["history_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in history_frames]
    print(json.dumps({"event": "online_warp_item_prepared", **case["metadata"]}), flush=True)
    return item


def _fixed_tree_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _fixed_tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_fixed_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_fixed_tree_to_device(item, device) for item in value)
    return value


def load_fixed_teacher_training_item(row, exact_args, device, *, requested_category, requested_chunk_mode):
    row = dict(row)
    history_type = canonical_history_type(row.get("history_type"))
    requested_history = str(requested_chunk_mode or "").replace("interaction_", "")
    if requested_history and requested_history != history_type:
        raise FixedTeacherIntegrityError(
            f"Fixed teacher history mismatch event_id={row.get('event_id')} manifest={history_type} "
            f"requested={requested_history}"
        )
    expected_category = canonical_training_category(row.get("training_category", row.get("category")))
    if requested_category and canonical_training_category(requested_category) != expected_category:
        raise FixedTeacherIntegrityError(
            f"Fixed teacher category mismatch event_id={row.get('event_id')} "
            f"manifest={expected_category} requested={requested_category}"
        )
    data_root = getattr(exact_args, "data_root", ".")
    candidate_path = resolve_optional_data_path(row.get("teacher_candidate_path", ""), data_root)
    training_cache_path = resolve_optional_data_path(row.get("training_cache_path", ""), data_root)
    teacher_cache_path = resolve_optional_data_path(row.get("teacher_cache_path", ""), data_root)
    if candidate_path is None or not candidate_path.is_file():
        raise FixedTeacherIntegrityError(f"Missing fixed candidate cache: {candidate_path}")
    if training_cache_path is None or not training_cache_path.is_file():
        raise FixedTeacherIntegrityError(f"Missing fixed training cache: {training_cache_path}")
    if teacher_cache_path is None or not teacher_cache_path.is_file():
        raise FixedTeacherIntegrityError(f"Missing fixed teacher cache: {teacher_cache_path}")
    cached = torch.load(training_cache_path, map_location="cpu", weights_only=False)
    if int(cached.get("schema_version", 0)) not in {1, 2}:
        raise FixedTeacherIntegrityError(f"Unsupported fixed training cache schema: {training_cache_path}")
    expected_config_hash = str(getattr(exact_args, "fixed_teacher_config_hash", ""))
    identity = validate_fixed_identity(row, dict(cached.get("candidate_identity", {})), expected_config_hash)
    if interaction_payload_hash(cached.get("interaction_payload")) != identity["interaction_payload_hash"]:
        raise FixedTeacherIntegrityError(
            "Fixed training payload hash mismatch "
            f"event_id={identity['event_id']} history_type={history_type}"
        )
    with np.load(candidate_path) as candidate_payload:
        candidate_identity = json.loads(str(candidate_payload["candidate_identity_json"].item()))
        validate_fixed_identity(row, candidate_identity, expected_config_hash)
    teacher_payload = np.load(teacher_cache_path)
    teacher_identity = json.loads(str(teacher_payload["candidate_identity_json"].item()))
    validate_fixed_identity(row, teacher_identity, expected_config_hash)
    validate_fixed_artifact_hashes(
        row,
        candidate_path=candidate_path,
        training_cache_path=training_cache_path,
        teacher_payload=teacher_payload,
    )
    teacher_candidate_key = str(teacher_payload["candidate_cache_key"].item())
    teacher_config_hash = str(teacher_payload["candidate_config_hash"].item())
    if teacher_candidate_key != identity["candidate_cache_key"] or teacher_config_hash != expected_config_hash:
        raise FixedTeacherIntegrityError(
            "Fixed teacher cache identity mismatch "
            f"event_id={identity['event_id']} history_type={history_type}: "
            f"manifest_candidate={identity['candidate_cache_key']} teacher_candidate={teacher_candidate_key} "
            f"runtime_config={expected_config_hash} teacher_config={teacher_config_hash}"
        )
    teacher_array = np.asarray(teacher_payload["teacher"]).copy()
    teacher_payload.close()
    item = _fixed_tree_to_device(cached, device)
    target_latents = item["target_latents"]
    conditioning = item["interaction_conditioning"]
    teacher = torch.as_tensor(teacher_array, device=device, dtype=torch.float32)
    if teacher.ndim == 4:
        teacher = teacher.unsqueeze(0)
    if teacher.shape[2:] != target_latents.shape[2:]:
        raise FixedTeacherIntegrityError(
            f"Fixed teacher grid mismatch event_id={identity['event_id']}: "
            f"teacher={tuple(teacher.shape)} target={tuple(target_latents.shape)}"
        )
    aligned = align_interaction_signals_to_grid(
        conditioning["payload"],
        batch_size=target_latents.shape[0],
        temporal=teacher.shape[2],
        height=teacher.shape[3],
        width=teacher.shape[4],
        device=device,
        visibility=conditioning.get("visibility"),
        world_valid=conditioning.get("world_valid"),
        teacher=teacher,
    )
    action = aligned["action"]
    visibility = aligned["visibility"]
    world_valid = aligned["world_valid"]
    valid_action_region = action * visibility * world_valid
    support_threshold = float(getattr(exact_args, "interaction_teacher_support_threshold", 0.25))
    teacher = aligned["teacher"].detach()
    teacher_support = (teacher > support_threshold).to(teacher)
    valid_denominator = valid_action_region.flatten(1).sum(dim=1)
    teacher_area_ratio = (
        (teacher_support * valid_action_region).flatten(1).sum(dim=1)
        / valid_denominator.clamp_min(1.0)
    )
    visibility_denominator = (action * world_valid).flatten(1).sum(dim=1)
    teacher_visibility_ratio = (
        valid_action_region.flatten(1).sum(dim=1) / visibility_denominator.clamp_min(1.0)
    )
    stage0_grid = tuple(
        int(row.get(key, default))
        for key, default in (
            ("stage0_grid_t", 9),
            ("stage0_grid_h", 6),
            ("stage0_grid_w", 10),
        )
    )
    stage0_teacher = torch.nn.functional.adaptive_max_pool3d(teacher, stage0_grid)
    stage0_positive_tokens = int((stage0_teacher > support_threshold).sum().item())
    action_key = action_history_key(row)
    is_negative = action_key.startswith("negative|")
    validate_stage0_positive_tokens(
        action_key.split("|", 1)[0],
        stage0_positive_tokens,
        event_id=identity["event_id"],
        history_type=history_type,
    )
    if is_negative:
        teacher = torch.zeros_like(teacher)
        teacher_area_ratio = torch.zeros_like(teacher_area_ratio)
        teacher_visibility_ratio = torch.ones_like(teacher_visibility_ratio)
    training = dict(item.get("training", {}))
    for field in ("target_indices", "history_indices", "render_pose_indices"):
        if [int(value) for value in training.get(field, [])] != identity[field]:
            raise FixedTeacherIntegrityError(
                f"Fixed cached training index mismatch event_id={identity['event_id']} history_type={history_type} "
                f"field={field} manifest={identity[field]} cache={training.get(field)}"
            )
    if [int(value) for value in training.get("keyframe_indices", [])] != identity["geometry_keyframe_frames"]:
        raise FixedTeacherIntegrityError(
            f"Fixed cached geometry index mismatch event_id={identity['event_id']} history_type={history_type}"
        )
    training["fixed_history_type"] = history_type
    result = {
        "seq": str(item["seq"]),
        "prompt": "Minecraft first-person gameplay.",
        "prompt_raw": "Minecraft first-person gameplay.",
        "target_latents": target_latents,
        "prompt_embeds": item["prompt_embeds"],
        "histories": item["histories"],
        "base_histories": item.get("base_histories"),
        "prompt_cache_status": "fixed_teacher_cache",
        "training": training,
        "training_category": str(item.get("training_category", expected_category)),
        "interaction_payload": item.get("interaction_payload"),
        "interaction_conditioning": conditioning,
        "interaction_teacher_map": teacher,
        "initial_teacher_map": teacher,
        "interaction_teacher_valid": True,
        "interaction_teacher_area_ratio": float(teacher_area_ratio.mean().item()),
        "interaction_teacher_visibility_ratio": float(teacher_visibility_ratio.mean().item()),
        "interaction_teacher_stage0_positive_tokens": stage0_positive_tokens,
        "interaction_teacher_invalid_reasons": [],
        "world_valid_mask_latents": item.get("world_valid_mask_latents"),
        "loss_focus_mask_latents": None,
        "primary_fire_event_latents": None,
        "interaction_debug_inputs": None,
        "fixed_teacher_identity": identity,
    }
    return result


class LazyPreparedItems:
    def __init__(self, pipe, df, exact_args, device, mean, std, cache_dir):
        self.pipe = pipe
        self.rows = [row for _, row in df.iterrows()]
        self.exact_args = exact_args
        self.device = device
        self.mean = mean
        self.std = std
        self.cache_dir = cache_dir
        self.memory_prompt_cache = {}
        self.prompt_cache_status_counts = {}
        self.prepare_counter = 0
        self.fixed_cache_only = bool(getattr(self.exact_args, "fixed_cache_only", False))
        if getattr(self.exact_args, "online_warp_cache", None) is None and not self.fixed_cache_only:
            self.exact_args.online_warp_cache = build_online_warp_training_cache(df, self.exact_args, self.device)

    def __len__(self):
        return len(self.rows)

    def _remember_status(self, status):
        self.prompt_cache_status_counts[status] = self.prompt_cache_status_counts.get(status, 0) + 1

    def get(self, idx, requested_category=None, requested_chunk_mode=None, keep_frames=False):
        idx = int(idx)
        print(json.dumps({"event": "prepare_item_start", "index": idx, "seq": str(self.rows[idx]["id"])}), flush=True)
        if str(self.rows[idx].get("review_status", "")).strip().lower() == "approved":
            item = load_fixed_teacher_training_item(
                self.rows[idx],
                self.exact_args,
                self.device,
                requested_category=requested_category,
                requested_chunk_mode=requested_chunk_mode,
            )
            self._remember_status(item["prompt_cache_status"])
            return item
        self.prepare_counter += 1
        row_index = int(self.rows[idx]["online_row_index"]) if "online_row_index" in self.rows[idx] else idx
        item = prepare_online_warp_item(
            self.pipe,
            row_index,
            self.exact_args,
            self.device,
            self.mean,
            self.std,
            keep_frames=bool(keep_frames),
            cache_dir=self.cache_dir,
            memory_prompt_cache=self.memory_prompt_cache,
            prepare_index=self.prepare_counter,
            requested_category=requested_category,
            requested_chunk_mode=requested_chunk_mode,
        )
        self._remember_status(item["prompt_cache_status"])
        print(
            json.dumps(
                {
                    "event": "prepare_item_done",
                    "index": idx,
                    "seq": item["seq"],
                    "prompt_cache_status": item["prompt_cache_status"],
                }
            ),
            flush=True,
        )
        return item
