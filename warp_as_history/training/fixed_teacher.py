from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


INTERACTION_ACTIONS = ("place", "mine_active", "mine_complete", "negative")
INTERACTION_HISTORIES = ("first", "later", "multi")
ACTION_HISTORY_KEYS = tuple(
    f"{action}|{history}"
    for action in INTERACTION_ACTIONS
    for history in INTERACTION_HISTORIES
)


def interaction_action_ratios(mode):
    if str(mode) == "adapter_overfit":
        return {
            "place": 0.625,
            "mine_active": 0.1875,
            "mine_complete": 0.1875,
            "negative": 0.0,
        }
    return {
        "place": 0.40,
        "mine_active": 0.25,
        "mine_complete": 0.15,
        "negative": 0.20,
    }


def stratified_candidate_indices(rows, limit, action_ratios=None):
    rows = list(rows)
    limit = int(limit)
    if limit <= 0 or limit >= len(rows):
        return list(range(len(rows)))
    ratios = dict(action_ratios or interaction_action_ratios("joint_stage0"))
    pools = {action: [] for action in INTERACTION_ACTIONS}
    for index, row in enumerate(rows):
        action = canonical_pool_action(row)
        if action in pools:
            pools[action].append(index)
    available = {action: values for action, values in pools.items() if values and ratios.get(action, 0) > 0}
    if not available:
        raise ValueError("Candidate stratification found no supported interaction actions.")
    ratio_sum = sum(float(ratios[action]) for action in available)
    ideals = {action: limit * float(ratios[action]) / ratio_sum for action in available}
    counts = {action: min(len(available[action]), int(ideals[action])) for action in available}
    remaining = limit - sum(counts.values())
    while remaining > 0:
        candidates = [action for action in available if counts[action] < len(available[action])]
        if not candidates:
            break
        action = max(candidates, key=lambda name: (ideals[name] - counts[name], ratios[name], name))
        counts[action] += 1
        remaining -= 1
    selected = []
    for action in INTERACTION_ACTIONS:
        pool = available.get(action, [])
        count = counts.get(action, 0)
        if count <= 0:
            continue
        positions = [min(int((offset + 0.5) * len(pool) / count), len(pool) - 1) for offset in range(count)]
        selected.extend(pool[position] for position in positions)
    return selected


class FixedTeacherIntegrityError(RuntimeError):
    pass


def _identity_text(value, default=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return str(default)
    return str(value).strip()


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
    if value in {"generated", "generated_history", "multi_chunk"}:
        value = "multi"
    if value not in INTERACTION_HISTORIES:
        raise ValueError(f"history_type must be first, later or multi, got {value!r}")
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


def build_action_history_pools(
    rows,
    *,
    require_approved=True,
    require_overfit_selected=False,
    per_pool_limit=None,
):
    pools = {key: [] for key in ACTION_HISTORY_KEYS}
    for position, row in enumerate(rows):
        if require_approved and str(row.get("review_status", "")).strip().lower() != "approved":
            continue
        if require_overfit_selected and _identity_text(row.get("overfit_selected")).lower() not in {
            "1",
            "true",
            "yes",
        }:
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


def _json_identity_value(value):
    if isinstance(value, dict):
        return {str(key): _json_identity_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_identity_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def interaction_payload_hash(payload):
    return stable_json_hash(_json_identity_value(payload or {}))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path):
    return file_sha256(path)


def model_artifact_fingerprint(path):
    text = str(path or "").strip()
    if not text:
        return {"path": "", "exists": False}
    path = Path(text)
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
        "interaction_teacher_support_threshold",
        "flow_matching_train_exact_timestep_sampling",
        "event_aligned_interaction",
    )
    values = {key: str(getattr(args, key, None)) for key in keys}
    values["event_alignment_schema"] = "telemetry_action_then_visual_source_plus_one_v2"
    for model_key in ("base_model_path", "transformer_path", "camera_checkpoint"):
        values[f"{model_key}_fingerprint"] = model_artifact_fingerprint(getattr(args, model_key, None))
    values["wah_recipe"] = dict(wah_recipe or {})
    return values


def candidate_config_hash(args, wah_recipe):
    return stable_json_hash(candidate_config_payload(args, wah_recipe))


def fixed_identity_from_row(row):
    identity = {
        "event_id": _identity_text(row.get("event_id")),
        "action_type": "none" if is_negative_action(row.get("action_type")) else _identity_text(row.get("action_type")).lower(),
        "block_id": _identity_text(row.get("block_id", row.get("object_id", ""))),
        "object_id": _identity_text(row.get("object_id", row.get("block_id", ""))),
        "training_category": _identity_text(row.get("training_category", row.get("category", ""))).lower(),
        "interaction_payload_hash": _identity_text(row.get("interaction_payload_hash")),
        "source_video_digest": _identity_text(row.get("source_video_digest")),
        "history_type": canonical_history_type(row.get("history_type")),
        "target_indices": parse_index_sequence(row.get("target_indices"), "target_indices"),
        "history_indices": parse_index_sequence(row.get("history_indices"), "history_indices"),
        "geometry_keyframe_frames": parse_index_sequence(
            row.get("geometry_keyframe_frames"), "geometry_keyframe_frames"
        ),
        "render_pose_indices": parse_index_sequence(row.get("render_pose_indices"), "render_pose_indices"),
        "target_start_frame": int(row.get("target_start_frame", 0)),
        "event_local_frame": int(row.get("event_local_frame", 0) or 0),
        "chunk_mode": _identity_text(row.get("chunk_mode")),
        "direction": _identity_text(row.get("direction")),
        "source_segment_id": _identity_text(row.get("source_segment_id", row.get("segment_id", ""))),
        "candidate_cache_key": _identity_text(row.get("candidate_cache_key")),
        "candidate_config_hash": _identity_text(row.get("candidate_config_hash")),
    }
    if _identity_text(row.get("reference_frame_index")):
        identity["reference_frame_index"] = int(row.get("reference_frame_index"))
    for key in (
        "telemetry_source_event_frame",
        "visual_source_event_frame",
        "visual_effect_delay_source_frames",
    ):
        if _identity_text(row.get(key)):
            identity[key] = int(row.get(key))
    return identity


def _allowed_config_hashes(expected_config_hash):
    if isinstance(expected_config_hash, str):
        values = [expected_config_hash]
    else:
        values = list(expected_config_hash or [])
    return {str(value) for value in values if str(value)}


def validate_fixed_identity(row, cached_identity, expected_config_hash):
    expected = fixed_identity_from_row(row)
    differences = {}
    for key, expected_value in expected.items():
        actual_value = cached_identity.get(key)
        if actual_value != expected_value:
            differences[key] = {"manifest": expected_value, "cache": actual_value}
    allowed_hashes = _allowed_config_hashes(expected_config_hash)
    if str(expected["candidate_config_hash"]) not in allowed_hashes:
        differences["runtime_config_hash"] = {
            "manifest": expected["candidate_config_hash"],
            "allowed_manifest_hashes": sorted(allowed_hashes),
        }
    if differences:
        raise FixedTeacherIntegrityError(
            "Fixed teacher identity mismatch "
            f"event_id={expected['event_id']} history_type={expected['history_type']}: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    return expected


def fixed_artifact_hashes_from_row(row):
    hashes = {
        "candidate_npz_sha256": _identity_text(row.get("candidate_npz_sha256")),
        "training_cache_sha256": _identity_text(row.get("training_cache_sha256")),
        "base_training_cache_sha256": _identity_text(row.get("base_training_cache_sha256")),
    }
    history_variant = _identity_text(row.get("history_variant", "clean")).lower() or "clean"
    if history_variant == "damaged" and not hashes["base_training_cache_sha256"]:
        raise FixedTeacherIntegrityError(
            "Damaged fixed-cache row is missing base_training_cache_sha256 "
            f"event_id={_identity_text(row.get('event_id'))} "
            f"history_type={_identity_text(row.get('history_type'))}"
        )
    if not hashes["base_training_cache_sha256"]:
        hashes["base_training_cache_sha256"] = hashes["training_cache_sha256"]
    return hashes


def validate_fixed_artifact_hashes(
    row,
    *,
    candidate_path=None,
    training_cache_path=None,
    teacher_payload=None,
):
    expected = fixed_artifact_hashes_from_row(row)
    differences = {}
    for field, path in (
        ("candidate_npz_sha256", candidate_path),
        ("training_cache_sha256", training_cache_path),
    ):
        if not expected[field]:
            differences[field] = {"manifest": expected[field], "error": "missing_hash"}
        elif path is not None:
            actual = file_sha256(path)
            if actual != expected[field]:
                differences[field] = {"manifest": expected[field], "file": actual}
        if teacher_payload is not None and field in teacher_payload:
            cached = str(teacher_payload[field].item())
            teacher_expected = (
                expected["base_training_cache_sha256"]
                if field == "training_cache_sha256"
                else expected[field]
            )
            if cached != teacher_expected:
                differences[f"teacher_{field}"] = {
                    "manifest": teacher_expected,
                    "teacher": cached,
                }
    if differences:
        raise FixedTeacherIntegrityError(
            "Fixed teacher artifact hash mismatch "
            f"event_id={_identity_text(row.get('event_id'))} history_type={_identity_text(row.get('history_type'))}: "
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
