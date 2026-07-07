#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

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
from warp_as_history.training import core as opt
from warp_as_history.training.utils import detach_tree


ONLINE_VIDEO_COLUMNS = ("video", "video_url", "url", "video_path", "path")
ONLINE_PROMPT_COLUMNS = ("prompt", "prompts", "caption", "text")
ONLINE_INTERACTION_COLUMNS = ("interaction_history_path", "action_history_path", "frame_action_summary_path")
ONLINE_PRIMARY_FIRE_EVENT_COLUMNS = ("primary_fire_event_path",)
ONLINE_PRIMARY_FIRE_MASK_COLUMNS = ("primary_fire_loss_mask_path",)
ONLINE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PRIMARY_FIRE_CHAR = "["


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
    rows = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        base = row.to_dict()
        raw_prompt = str(base.get(prompt_column, ""))
        base["id"] = str(base.get("id") or f"online_{row_index:06d}")
        base["online_row_index"] = int(row_index)
        base["video_path"] = str(base[video_column])
        base["prompt_raw"] = raw_prompt
        base["prompt"] = add_online_prompt_trigger(raw_prompt, prompt_trigger)
        if event_column:
            base["primary_fire_event_path"] = base.get(event_column, "")
        if loss_mask_column:
            base["primary_fire_loss_mask_path"] = base.get(loss_mask_column, "")
        rows.append(base)
    normalized = df.__class__(rows)
    meta = {
        "video_column": video_column,
        "prompt_column": prompt_column,
        "prompt_trigger": prompt_trigger,
        "primary_fire_event_column": event_column,
        "primary_fire_loss_mask_column": loss_mask_column,
        "rows": len(rows),
    }
    return normalized, meta


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


def online_mask_frames_to_latent_mask(mask_frames, *, target_latents, num_frames, temporal_scale, device):
    if not mask_frames:
        return None
    mask = np.stack([np.asarray(frame.convert("L"), dtype=np.float32) / 255.0 for frame in mask_frames], axis=0)
    if mask.shape[0] < int(num_frames):
        raise ValueError(f"Focus mask produced {mask.shape[0]} frames, need at least {int(num_frames)}.")
    sampled_ids = np.arange(int(target_latents.shape[2]), dtype=np.int64) * int(temporal_scale)
    sampled_ids = np.clip(sampled_ids, 0, mask.shape[0] - 1)
    sampled = torch.from_numpy(mask[sampled_ids]).to(device=device, dtype=torch.float32)
    sampled = sampled.unsqueeze(0).unsqueeze(0)
    sampled = torch.nn.functional.interpolate(
        sampled,
        size=(int(target_latents.shape[2]), int(target_latents.shape[3]), int(target_latents.shape[4])),
        mode="trilinear",
        align_corners=False,
    )
    return sampled.clamp_(0.0, 1.0)


def load_primary_fire_event_payload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def load_online_video_frames(ref, *, height, width, frame_stride=1, max_video_frames=0):
    frame_stride = max(1, int(frame_stride))
    max_video_frames = int(max_video_frames)
    frames = []
    if isinstance(ref, Path) and ref.is_dir():
        for src_idx, path in enumerate(_iter_online_image_files(ref)):
            if src_idx % frame_stride != 0:
                continue
            frame = Image.open(path).convert("RGB")
            frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
            if max_video_frames > 0 and len(frames) >= max_video_frames:
                break
    else:
        reader = imageio.get_reader(str(ref))
        try:
            for src_idx, array in enumerate(reader):
                if src_idx % frame_stride != 0:
                    continue
                frame = Image.fromarray(np.asarray(array)).convert("RGB")
                frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
                if max_video_frames > 0 and len(frames) >= max_video_frames:
                    break
        finally:
            reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from online training video {ref}.")
    return frames


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
        return {
            "height": int(self.exact_args.height),
            "width": int(self.exact_args.width),
            "frame_stride": int(getattr(self.exact_args, "online_frame_stride", 1)),
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
        }

    def _geometry_cache_path(self, row_index, direction):
        if self.disk_cache_dir is None:
            return None
        row = self.rows[int(row_index)]
        payload = {
            "row_index": int(row_index),
            "row_id": str(row.get("id", row_index)),
            "direction": str(direction),
            "video_path": str(row["video_path"]),
            "config": self._cache_payload(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        return self.disk_cache_dir / f"{int(row_index):05d}_{direction}_{digest}.pt"

    def _load_frames(self, ref, direction):
        frames = load_online_video_frames(
            ref,
            height=int(self.exact_args.height),
            width=int(self.exact_args.width),
            frame_stride=int(getattr(self.exact_args, "online_frame_stride", 1)),
            max_video_frames=int(getattr(self.exact_args, "online_max_video_frames", 0)),
        )
        if direction == "reverse":
            frames = list(reversed(frames))
        return frames

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
        payload = torch.load(cache_path, map_location="cpu")
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
        tensors = [online_pil_to_tensor(frame).unsqueeze(0) for frame in frames]
        print(
            json.dumps(
                {
                    "event": "online_warp_estimate_geometry",
                    "row_index": int(row_index),
                    "seq": str(row["id"]),
                    "direction": direction,
                    "frames": len(frames),
                    "video": str(ref),
                }
            ),
            flush=True,
        )
        try:
            geometry = self.renderer.estimate_keyframe_geometry(tensors, device=self.device)
        finally:
            del tensors
            self.renderer._pi3x_runtime = None
            opt.clean_memory()
        return geometry

    def _build_record(self, row_index, direction):
        row_index = int(row_index)
        row = self.rows[row_index]
        ref = resolve_online_video_ref(row["video_path"], getattr(self.exact_args, "data_root", "."))
        frames = self._load_frames(ref, direction)
        cache_path = self._geometry_cache_path(row_index, direction)
        geometry = self._load_geometry_from_disk(cache_path)
        if geometry is None:
            geometry = self._estimate_geometry(row_index, row, direction, ref, frames)
            self._save_geometry_to_disk(cache_path, geometry, row_index, direction, ref, frames)
        return {
            "direction": direction,
            "frames": frames,
            "geometry": geometry,
            "row": row,
            "row_index": row_index,
            "video_ref": str(ref),
        }

    def _get_record(self, row_index, direction):
        key = (int(row_index), str(direction))
        cached = self.records.get(key)
        if cached is not None:
            self.records.move_to_end(key)
            print(
                json.dumps(
                    {
                        "event": "online_warp_record_cache_hit",
                        "cache": "memory",
                        "row_index": int(row_index),
                        "seq": str(self.rows[int(row_index)].get("id", row_index)),
                        "direction": str(direction),
                    }
                ),
                flush=True,
            )
            return cached

        record = self._build_record(row_index, direction)
        self.records[key] = record
        while len(self.records) > self.memory_cache_size:
            _old_key, old_record = self.records.popitem(last=False)
            self._release_record(old_record)
        return record

    def choose_direction(self, rng):
        if not bool(getattr(self.exact_args, "online_direction_augmentation", True)):
            return "forward"
        reverse_prob = float(getattr(self.exact_args, "online_direction_reverse_prob", 0.5))
        return "reverse" if rng.random() < reverse_prob else "forward"

    def sample_case(self, row_index, prepare_index):
        row_index = int(row_index)
        row = self.rows[row_index]
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
        frames = prepared["frames"]
        n = len(frames)
        num_frames = int(self.exact_args.num_frames)
        if n < num_frames:
            raise ValueError(f"Online training video {prepared['video_ref']} has {n} frames, need {num_frames}.")
        sample_fire_prob = float(getattr(self.exact_args, "online_primary_fire_window_probability", 0.6) or 0.6)
        target_indices = None
        chunk_mode = "movement"
        if full_event_payload is not None and rng.random() < sample_fire_prob:
            target_indices = choose_online_primary_fire_window(rng, n, num_frames, full_event_payload)
            chunk_mode = "primary_fire"
        if target_indices is None:
            target_indices = choose_online_movement_window(rng, n, num_frames, full_event_payload)
        target_start = int(target_indices[0])
        if target_start <= 0:
            chunk_mode = "first" if chunk_mode == "movement" else f"{chunk_mode}_first"
            source_idx = int(target_start)
            history_indices = []
            keyframe_indices = [source_idx]
            render_pose_indices = target_indices
            future_keyframe_indices = []
            drop_renderer_source = False
            keyframe_policy = "source_only"
            condition_frame = frames[source_idx]
        else:
            chunk_mode = "later" if chunk_mode == "movement" else f"{chunk_mode}_later"
            max_history = min(int(getattr(self.exact_args, "online_max_history_frames", 19)), target_start)
            history_len = rng.randint(1, max(1, max_history))
            history_indices = list(range(target_start - history_len, target_start))
            future_keyframe_indices = []
            keyframe_policy = "history_only"
            keyframe_indices = sorted(set(history_indices + future_keyframe_indices))
            render_pose_indices = [keyframe_indices[-1], *target_indices]
            drop_renderer_source = True
            condition_frame = frames[history_indices[-1]]

        geometry = subset_online_geometry(prepared["geometry"], keyframe_indices)
        poses = online_relative_poses(prepared["geometry"], geometry["source_pose"], render_pose_indices)
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
        interaction_memory = summarize_multiscale_interaction_history(
            self.interaction_histories.get(row_index),
            history_indices,
            target_indices,
            max_items=int(getattr(self.exact_args, "online_interaction_max_items", 8)),
        )
        event_payload = crop_primary_fire_event_payload(full_event_payload, target_indices) if full_event_payload is not None else None
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
                "video": prepared["video_ref"],
                "warp_render_stats": rendered.get("warp_render_stats", {}),
                "interaction_history_text": interaction_memory.get("merged", ""),
                "interaction_memory": interaction_memory,
                "primary_fire_supervision": primary_fire_supervision,
                "primary_fire_event_payload": event_payload,
                "focus_mask_stats": focus_mask_stats,
                "sample_window_type": chunk_mode,
            },
            "interaction_history_text": interaction_memory.get("merged", ""),
            "interaction_memory": interaction_memory,
            "primary_fire_supervision": primary_fire_supervision,
            "primary_fire_event_payload": event_payload,
            "prompt": row["prompt"],
            "prompt_base": row["prompt"],
            "prompt_raw": row.get("prompt_raw", row["prompt"]),
            "row": row,
            "seq": seq,
            "target_frames": [frames[idx] for idx in target_indices],
            "target_indices": target_indices,
            "warp_frames": warp_frames,
            "warp_mask_frames": warp_mask_frames,
            "focus_mask_frames": focus_mask_frames,
        }
        del rendered, warp_video, warp_mask, geometry, poses
        self.renderer._pi3x_runtime = None
        opt.clean_memory()
        return result


def build_online_warp_training_cache(df, exact_args, device):
    rows = [row for _, row in df.iterrows()]
    return OnlineWarpTrainingCache(rows, exact_args, device)

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
            payload = torch.load(cache_path, map_location="cpu")
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


def prepare_online_warp_item(pipe, row_index, exact_args, device, mean, std, keep_frames, cache_dir, memory_prompt_cache, prepare_index=0):
    online_cache = getattr(exact_args, "online_warp_cache", None)
    if online_cache is None:
        raise ValueError("Training cache is missing.")
    case = online_cache.sample_case(row_index, prepare_index)
    seq = case["seq"]
    prompt_text = str(case["prompt"])
    first_frame = case["condition_frame"]
    target_frames = case["target_frames"]
    history_frames = case["warp_frames"]
    mask_frames = case["warp_mask_frames"]

    had_extra_mask = hasattr(exact_args, "history_visibility_extra_mask_frames")
    old_extra_mask = getattr(exact_args, "history_visibility_extra_mask_frames", None)
    exact_args.history_visibility_extra_mask_frames = mask_frames
    loss_focus_mask_latents = None
    primary_fire_event_latents = None
    primary_fire_event_debug = None

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
            histories = opt.make_histories(
                pipe,
                image_latents,
                fake_image_latents,
                exact_args,
                device,
                video_latents=video_latents,
                seq=seq,
            )
            loss_focus_mask_latents = online_mask_frames_to_latent_mask(
                case.get("focus_mask_frames"),
                target_latents=target_latents,
                num_frames=int(exact_args.num_frames),
                temporal_scale=int(pipe.vae_scale_factor_temporal),
                device=device,
            )
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
        "prompt_cache_status": prompt_cache_status,
        "training": case["metadata"],
        "interaction_memory": case.get("interaction_memory"),
        "primary_fire_supervision": case.get("primary_fire_supervision"),
        "primary_fire_time_mask": None
        if case.get("primary_fire_event_payload") is None
        else list(case["primary_fire_event_payload"].get("time_mask", [])),
        "primary_fire_event": case.get("primary_fire_event_payload"),
        "loss_focus_mask_latents": None if loss_focus_mask_latents is None else loss_focus_mask_latents.detach(),
        "primary_fire_event_latents": None
        if primary_fire_event_latents is None
        else primary_fire_event_latents.detach(),
        "primary_fire_event_debug": primary_fire_event_debug,
    }
    if keep_frames:
        item["target_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in target_frames]
        item["history_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in history_frames]
    print(json.dumps({"event": "online_warp_item_prepared", **case["metadata"]}), flush=True)
    return item


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
        if getattr(self.exact_args, "online_warp_cache", None) is None:
            self.exact_args.online_warp_cache = build_online_warp_training_cache(df, self.exact_args, self.device)

    def __len__(self):
        return len(self.rows)

    def _remember_status(self, status):
        self.prompt_cache_status_counts[status] = self.prompt_cache_status_counts.get(status, 0) + 1

    def get(self, idx):
        idx = int(idx)
        print(json.dumps({"event": "prepare_item_start", "index": idx, "seq": str(self.rows[idx]["id"])}), flush=True)
        self.prepare_counter += 1
        row_index = int(self.rows[idx]["online_row_index"]) if "online_row_index" in self.rows[idx] else idx
        item = prepare_online_warp_item(
            self.pipe,
            row_index,
            self.exact_args,
            self.device,
            self.mean,
            self.std,
            keep_frames=False,
            cache_dir=self.cache_dir,
            memory_prompt_cache=self.memory_prompt_cache,
            prepare_index=self.prepare_counter,
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
