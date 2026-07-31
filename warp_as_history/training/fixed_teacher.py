from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


INTERACTION_ACTIONS = ("place", "mine_active", "mine_complete", "negative")
INTERACTION_HISTORIES = ("first", "later")
ACTION_HISTORY_KEYS = tuple(
    f"{action}|{history}"
    for action in INTERACTION_ACTIONS
    for history in INTERACTION_HISTORIES
)


class FixedTeacherIntegrityError(RuntimeError):
    pass


def canonical_pool_action(row):
    value = row.get("action_type", "")
    if isinstance(value, float) and math.isnan(value):
        value = "none"
    action = str(value or "").strip().lower()
    return "negative" if action in {"", "none", "negative"} else action


def is_negative_action(value):
    if isinstance(value, dict):
        return canonical_pool_action(value) == "negative"
    return str(value or "").strip().lower() in {"", "none", "negative"}


def canonical_history_type(value):
    value = str(value or "").strip().lower()
    if value not in INTERACTION_HISTORIES:
        raise ValueError(f"history_type must be first or later, got {value!r}")
    return value


def action_history_key(row):
    action = canonical_pool_action(row)
    history = canonical_history_type(row.get("history_type"))
    key = f"{action}|{history}"
    if key not in ACTION_HISTORY_KEYS:
        raise ValueError(f"Unsupported approved interaction pool key: {key}")
    return key


def parse_index_sequence(value, field_name):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [item for item in text.split(",") if item.strip()]
    if not isinstance(decoded, list):
        raise ValueError(f"{field_name} must encode a list, got {value!r}")
    return [int(item) for item in decoded]


def encode_index_sequence(values):
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def approved_row_id(row, fallback=None):
    return str(
        row.get("candidate_cache_key")
        or row.get("event_id")
        or row.get("sample_id")
        or row.get("id")
        or fallback
    )


def build_action_history_pools(rows, *, require_approved=True, per_pool_limit=None):
    pools = {key: [] for key in ACTION_HISTORY_KEYS}
    for position, row in enumerate(rows):
        if require_approved and str(row.get("review_status", "")).strip().lower() != "approved":
            continue
        key = action_history_key(row)
        if per_pool_limit is None or len(pools[key]) < int(per_pool_limit):
            pools[key].append(int(position))
    return pools


def validate_required_pools(pools, phase_plan, action_ratios):
    missing = []
    for phase_index, phase in enumerate(phase_plan):
        for action, action_ratio in action_ratios.items():
            if float(action_ratio) <= 0:
                continue
            for history, history_ratio in dict(phase["history"]).items():
                if float(history_ratio) <= 0:
                    continue
                key = f"{action}|{history}"
                if not pools.get(key):
                    missing.append(f"phase={phase_index}:{key}")
    if missing:
        raise ValueError(
            "Interaction curriculum requires non-empty approved action-by-history pools: "
            + ", ".join(missing)
        )


def stable_json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path):
    return file_sha256(path)


def model_artifact_fingerprint(path):
    path = Path(str(path or ""))
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path.resolve()), "sha256": file_sha256(path), "size": path.stat().st_size}
    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        entry = {
            "path": child.relative_to(path).as_posix(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if child.name in {"config.json", "model_index.json", "adapter_config.json"}:
            entry["sha256"] = file_sha256(child)
        files.append(entry)
    return {"path": str(path.resolve()), "files_hash": stable_json_hash(files), "file_count": len(files)}


def candidate_config_payload(args, wah_recipe):
    keys = (
        "base_model_path",
        "transformer_path",
        "camera_checkpoint",
        "height",
        "width",
        "num_frames",
        "online_target_fps",
        "online_frame_stride",
        "online_geometry_keyframe_stride",
        "online_max_history_frames",
        "online_render_mode",
        "online_mesh_samples_per_axis",
        "online_pi3_conf_threshold",
        "online_pi3_depth_edge_rtol",
        "use_minecraft_hud_mask",
        "direction_augmentation",
        "seed",
    )
    values = {key: str(getattr(args, key, None)) for key in keys}
    for model_key in ("base_model_path", "transformer_path", "camera_checkpoint"):
        values[f"{model_key}_fingerprint"] = model_artifact_fingerprint(getattr(args, model_key, None))
    values["wah_recipe"] = dict(wah_recipe or {})
    return values


def candidate_config_hash(args, wah_recipe):
    return stable_json_hash(candidate_config_payload(args, wah_recipe))


def fixed_identity_from_row(row):
    return {
        "event_id": str(row.get("event_id", "")),
        "history_type": canonical_history_type(row.get("history_type")),
        "target_indices": parse_index_sequence(row.get("target_indices"), "target_indices"),
        "history_indices": parse_index_sequence(row.get("history_indices"), "history_indices"),
        "geometry_keyframe_frames": parse_index_sequence(
            row.get("geometry_keyframe_frames"), "geometry_keyframe_frames"
        ),
        "render_pose_indices": parse_index_sequence(row.get("render_pose_indices"), "render_pose_indices"),
        "target_start_frame": int(row.get("target_start_frame", 0)),
        "event_local_frame": int(row.get("event_local_frame", 0) or 0),
        "chunk_mode": str(row.get("chunk_mode", "")),
        "direction": str(row.get("direction", "")),
        "source_segment_id": str(row.get("source_segment_id", row.get("segment_id", ""))),
        "candidate_cache_key": str(row.get("candidate_cache_key", "")),
        "candidate_config_hash": str(row.get("candidate_config_hash", "")),
    }


def validate_fixed_identity(row, cached_identity, expected_config_hash):
    expected = fixed_identity_from_row(row)
    differences = {}
    for key, expected_value in expected.items():
        actual_value = cached_identity.get(key)
        if actual_value != expected_value:
            differences[key] = {"manifest": expected_value, "cache": actual_value}
    if str(expected["candidate_config_hash"]) != str(expected_config_hash):
        differences["runtime_config_hash"] = {
            "manifest": expected["candidate_config_hash"],
            "runtime": str(expected_config_hash),
        }
    if differences:
        raise FixedTeacherIntegrityError(
            "Fixed teacher identity mismatch "
            f"event_id={expected['event_id']} history_type={expected['history_type']}: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    return expected


def validate_stage0_positive_tokens(action, positive_tokens, *, event_id="", history_type=""):
    positive_tokens = int(positive_tokens)
    if not is_negative_action(action) and positive_tokens <= 0:
        raise FixedTeacherIntegrityError(
            "Approved positive fixed teacher has no Stage 0 support "
            f"event_id={event_id} history_type={history_type}"
        )
    return positive_tokens


def validate_resume_contract(checkpoint_contract, current_contract, checkpoint_sampling, current_sampling):
    checkpoint_contract = dict(checkpoint_contract or {})
    current_contract = dict(current_contract or {})
    if checkpoint_contract != current_contract:
        raise ValueError(
            "Resume teacher-pool/sampling contract mismatch: "
            + json.dumps(
                {"checkpoint": checkpoint_contract, "current": current_contract},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    checkpoint_sampling = dict(checkpoint_sampling or {})
    current_sampling = dict(current_sampling or {})
    if checkpoint_sampling != current_sampling:
        raise ValueError(
            "Resume sampling_plan mismatch: "
            + json.dumps(
                {"checkpoint": checkpoint_sampling, "current": current_sampling},
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def restore_training_counters(resume_state, fallback_counts=None):
    resume_state = dict(resume_state or {})
    if resume_state.get("distribution_counts") is not None:
        distribution_counts = dict(resume_state.get("distribution_counts") or {})
    else:
        distribution_counts = dict(fallback_counts or {})
    global_step = int(resume_state.get("global_step", 0))
    effective_step = int(resume_state.get("effective_optimizer_step", global_step))
    if effective_step != global_step:
        raise ValueError(
            f"Checkpoint effective_optimizer_step={effective_step} does not match global_step={global_step}."
        )
    return {
        "distribution_counts": distribution_counts,
        "effective_optimizer_step": effective_step,
        "attempt_step": int(resume_state.get("attempt_step", 0)),
        "skipped_invalid_step": int(resume_state.get("skipped_invalid_step", 0)),
    }
