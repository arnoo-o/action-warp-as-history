#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ENABLE_OPTIONAL_ATTENTION = str(os.environ.get("WAH_ENABLE_OPTIONAL_ATTENTION", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not ENABLE_OPTIONAL_ATTENTION:
    os.environ.setdefault("XFORMERS_DISABLED", "1")


def _disable_broken_flash_attn_imports():
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def _false(*_args, **_kwargs):
        return False

    import_utils.is_flash_attn_2_available = _false
    import_utils.is_flash_attn_greater_or_equal = _false
    import_utils.is_flash_attn_greater_or_equal_2_10 = _false
    transformers_utils.is_flash_attn_2_available = _false
    transformers_utils.is_flash_attn_greater_or_equal = _false
    transformers_utils.is_flash_attn_greater_or_equal_2_10 = _false
    try:
        import diffusers.utils as diffusers_utils
        import diffusers.utils.import_utils as diffusers_import_utils

        diffusers_import_utils._xformers_available = False
        diffusers_import_utils._flash_attn_available = False
        diffusers_import_utils._flash_attn_3_available = False
        diffusers_import_utils.is_xformers_available = _false
        diffusers_import_utils.is_flash_attn_available = _false
        diffusers_import_utils.is_flash_attn_3_available = _false
        diffusers_import_utils.is_flash_attn_version = _false
        diffusers_utils.is_xformers_available = _false
        diffusers_utils.is_flash_attn_available = _false
        diffusers_utils.is_flash_attn_3_available = _false
        diffusers_utils.is_flash_attn_version = _false
    except Exception:
        pass

if not ENABLE_OPTIONAL_ATTENTION:
    _disable_broken_flash_attn_imports()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from helios.modules.interaction_conditioning import configure_interaction_trainability
from helios.modules.interaction_conditioning import align_interaction_signals_to_grid

from warp_as_history.training import core as opt
from warp_as_history.minecraft_recipe import minecraft_wah_recipe, recipe_mismatches
from warp_as_history.training.data import (
    LazyPreparedItems,
    build_online_warp_training_cache,
    fixed_cache_rows_ready,
    normalize_online_training_dataframe,
    resolve_optional_data_path,
    rgb_frame_to_latent_indices,
)
from warp_as_history.training.fixed_teacher import (
    ACTION_HISTORY_KEYS,
    FixedTeacherIntegrityError,
    approved_row_id,
    build_action_history_pools,
    candidate_config_hash,
    encode_index_sequence,
    file_sha256,
    interaction_action_ratios,
    interaction_payload_hash,
    manifest_sha256,
    model_artifact_fingerprint,
    restore_training_counters,
    stable_json_hash,
    stratified_candidate_indices,
    validate_resume_contract,
    validate_required_pools,
)
from warp_as_history.training.utils import (
    current_train_lr,
    release_cuda_cache,
    save_lora,
    scalar,
    StepCategorySampler,
    write_json,
    next_index_generator,
)

DEFAULT_HELIOS_MODEL = "checkpoints/helios-distilled"
NEUTRAL_MINECRAFT_PROMPT = "Minecraft first-person gameplay."


class InteractionJointSampler:
    """Deterministic action-by-history quota plan for configurable curricula."""

    ACTION_RATIOS = {
        "place": 0.50,
        "mine_active": 0.15,
        "mine_complete": 0.15,
        "negative": 0.20,
    }
    DEFAULT_STAGE0_PHASES = (
        {"steps": 500, "history": {"first": 0.40, "later": 0.60}},
        {"steps": 500, "history": {"first": 0.30, "later": 0.70}},
        {"steps": 500, "history": {"first": 0.25, "later": 0.75}},
    )
    DEFAULT_PILOT_PHASES = (
        {"steps": 100, "history": {"first": 0.40, "later": 0.60}},
        {"steps": 100, "history": {"first": 0.30, "later": 0.70}},
        {"steps": 100, "history": {"first": 0.25, "later": 0.75}},
    )

    def __init__(self, source_pools, total_steps, seed, *, phases, action_ratios=None):
        total_steps = int(total_steps)
        phase_plan = [dict(item) for item in phases]
        planned_steps = sum(int(item["steps"]) for item in phase_plan)
        if total_steps != planned_steps:
            raise ValueError(f"Interaction curriculum expected {planned_steps} effective steps, got {total_steps}.")
        self.samplers = []
        self.source_pools = {key: list(source_pools.get(key, [])) for key in ACTION_HISTORY_KEYS}
        self.seed = int(seed)
        self.action_ratios = dict(action_ratios or self.ACTION_RATIOS)
        self.phase_plan = phase_plan
        self.total_steps = total_steps
        self.offsets = []
        cursor = 0
        for phase_index, phase in enumerate(phase_plan):
            self.offsets.append(cursor)
            cursor += int(phase["steps"])
            history_ratios = dict(phase["history"])
            pools = {}
            ratios = {}
            for action, action_ratio in self.action_ratios.items():
                for history, history_ratio in history_ratios.items():
                    key = f"{action}|{history}"
                    pools[key] = list(self.source_pools[key])
                    ratios[key] = float(action_ratio) * float(history_ratio)
            validate_required_pools(self.source_pools, [phase], self.action_ratios)
            self.samplers.append(
                StepCategorySampler(pools, ratios, int(phase["steps"]), int(seed) + phase_index * 10007)
            )

    def sample(self, effective_step):
        effective_step = int(effective_step)
        phase = len(self.samplers) - 1
        for index, offset in enumerate(self.offsets):
            phase_steps = int(self.phase_plan[index]["steps"])
            if effective_step < offset + phase_steps:
                phase = index
                break
        local = effective_step - self.offsets[phase]
        return self.samplers[phase].sample(local)

    def sample_category(self, category, occurrence):
        category = str(category)
        pool = self.source_pools[category]
        return pool[int(occurrence) % len(pool)]

    def report(self, completed_steps):
        completed_steps = min(max(int(completed_steps), 0), self.total_steps)
        reports = []
        remaining = completed_steps
        for sampler, phase in zip(self.samplers, self.phase_plan):
            count = min(remaining, int(phase["steps"]))
            reports.append(sampler.report(count))
            remaining -= count
        actual = Counter()
        for report in reports:
            actual.update(report["actual_steps"])
        return {
            "target_effective_steps": self.total_steps,
            "completed_effective_steps": completed_steps,
            "joint_actual_steps": dict(actual),
            "phase_plan": self.phase_plan,
        }


def interaction_training_mode_default_steps(mode):
    mode = str(mode)
    if mode == "joint_pilot":
        return 200
    if mode == "router_overfit":
        return 200
    if mode == "adapter_overfit":
        return 300
    return 1500


def interaction_training_mode_phase_plan(mode, total_steps=None, phase_steps=None, first_ratios=None):
    mode = str(mode)
    if mode in {"joint_pilot", "joint_stage0"}:
        total_steps = int(total_steps or interaction_training_mode_default_steps(mode))
        if phase_steps is None:
            base, remainder = divmod(total_steps, 3)
            phase_steps = [base + (1 if index < remainder else 0) for index in range(3)]
        if first_ratios is None:
            first_ratios = [0.40, 0.30, 0.25]
        if len(phase_steps) != 3 or len(first_ratios) != 3:
            raise ValueError("Interaction Stage 0 curriculum requires exactly three phase steps and ratios.")
        if sum(int(value) for value in phase_steps) != total_steps:
            raise ValueError("--interaction_phase_steps must sum to the configured effective steps.")
        return tuple(
            {
                "steps": int(steps),
                "history": {"first": float(first), "later": 1.0 - float(first)},
            }
            for steps, first in zip(phase_steps, first_ratios)
        )
    return (
        {"steps": int(total_steps or interaction_training_mode_default_steps(mode)), "history": {"first": 0.50, "later": 0.50}},
    )


def training_total_steps(base_train_steps, bidirectional_train_steps, enable_bidirectional_training):
    base = max(int(base_train_steps), 0)
    bidirectional = max(int(bidirectional_train_steps), 0) if bool(enable_bidirectional_training) else 0
    return base + bidirectional


def training_resume_contract(args, *, camera_fingerprint=None):
    transformer_path = args.transformer_path or args.base_model_path
    return {
        "teacher_pool_manifest_hash": str(getattr(args, "teacher_pool_manifest_hash", "")),
        "approved_teacher_row_ids": list(getattr(args, "approved_teacher_row_ids", []) or []),
        "action_history_pool_sizes": dict(getattr(args, "action_history_pool_sizes", {}) or {}),
        "phase_plan": list(getattr(args, "resolved_interaction_phase_plan", []) or []),
        "sampler_seed": int(args.seed),
        "base_model_fingerprint": model_artifact_fingerprint(args.base_model_path),
        "transformer_fingerprint": model_artifact_fingerprint(transformer_path),
        "camera_checkpoint_fingerprint": (
            camera_fingerprint
            if camera_fingerprint is not None
            else model_artifact_fingerprint(args.camera_checkpoint)
        ),
        "interaction_lr": float(args.interaction_lr if args.interaction_lr is not None else 1.0e-4),
        "router_lr": float(args.router_lr),
        "teacher_support_threshold": float(args.interaction_teacher_support_threshold),
        "flow_matching_train_exact_timestep_sampling": str(
            args.flow_matching_train_exact_timestep_sampling
        ),
        "candidate_config_hash": str(getattr(args, "fixed_teacher_config_hash", "")),
    }


def training_stage_for_step(step, base_train_steps, enable_bidirectional_training):
    if bool(enable_bidirectional_training) and int(step) >= int(base_train_steps):
        return "bidirectional"
    return "base"


def should_compute_bidirectional_feedback(
    step,
    base_train_steps,
    enable_bidirectional_training,
    bidirectional_interval,
):
    if training_stage_for_step(step, base_train_steps, enable_bidirectional_training) != "bidirectional":
        return False
    stage_step = int(step) - int(base_train_steps)
    return stage_step % max(int(bidirectional_interval), 1) == 0


def build_minecraft_step_sampler(df, args):
    """Create quota-exact pools before expensive Pi3X/latent preparation."""
    source_pools = {
        "place": [],
        "mine": [],
        "mine_active": [],
        "mine_complete": [],
        "movement": [],
        "negative": [],
        "other": [],
    }
    for position, (_, row) in enumerate(df.iterrows()):
        category = str(row.get("training_category", row.get("category", "movement"))).strip().lower()
        action_type = str(row.get("action_type", "") or "").strip().lower()
        if category == "mine" and action_type in {"mine_active", "mine_complete"}:
            source_pools[action_type].append(int(position))
        source_pools[category if category in source_pools else "other"].append(int(position))
    profile = str(args.training_profile)
    if profile == "camera":
        pools = {
            "camera_first": list(source_pools["movement"]),
            "camera_later": list(source_pools["movement"]),
            "camera_rollout": list(source_pools["movement"]),
        }
        ratios = {"camera_first": 0.25, "camera_later": 0.55, "camera_rollout": 0.20}
    elif profile == "interaction":
        mode = str(getattr(args, "interaction_training_mode", "joint_stage0"))
        phase_plan = interaction_training_mode_phase_plan(
            mode,
            total_steps=args.max_steps,
            phase_steps=args.interaction_phase_steps,
            first_ratios=args.interaction_first_history_ratios,
        )
        action_ratios = interaction_action_ratios(mode)
        require_selected = mode in {"router_overfit", "adapter_overfit"}
        source_pools = build_action_history_pools(
            df.to_dict(orient="records"),
            require_approved=True,
            require_overfit_selected=require_selected,
        )
        if require_selected and not any(source_pools.values()):
            raise ValueError(
                f"{mode} requires manually approved rows with overfit_selected=true; "
                "select 16-32 clean samples in teacher_pool_review_manifest.csv."
            )
        validate_required_pools(source_pools, phase_plan, action_ratios)
        print(
            json.dumps(
                {
                    "event": "approved_action_history_pools",
                    "interaction_training_mode": mode,
                    "action_ratios": action_ratios,
                    "pool_sizes": {key: len(source_pools[key]) for key in ACTION_HISTORY_KEYS},
                }
            ),
            flush=True,
        )
        sampler = InteractionJointSampler(
            source_pools,
            args.max_steps,
            args.seed,
            phases=phase_plan,
            action_ratios=action_ratios,
        )
        return sampler, {
            "pool_sizes": {name: len(values) for name, values in source_pools.items()},
            "interaction_training_mode": mode,
            **sampler.report(0),
        }
    else:
        pools = {
            "place": source_pools["place"],
            "mine": source_pools["mine"],
            "negative": source_pools["negative"],
        }
        ratios = {
            "place": float(args.place_step_ratio),
            "mine": float(args.mine_step_ratio),
            "negative": float(args.other_step_ratio),
        }
    empty_requested = [name for name, ratio in ratios.items() if ratio > 0 and not pools.get(name)]
    if empty_requested:
        print(json.dumps({"event": "minecraft_sampler_pool_empty", "pools": empty_requested, "action": "renormalized"}), flush=True)
    sampler = StepCategorySampler(pools, ratios, args.max_steps, args.seed)
    return sampler, {
        "pool_sizes": {name: len(values) for name, values in source_pools.items()},
        "effective_pool_sizes": {name: len(values) for name, values in pools.items()},
        **sampler.report(0),
    }


def distribution_counts_from_losses(losses):
    counts = {
        "place": 0,
        "mine": 0,
        "mine_active": 0,
        "mine_complete": 0,
        "negative": 0,
        "camera_first": 0,
        "camera_later": 0,
        "camera_rollout": 0,
        "valid_place": 0,
        "valid_mine": 0,
        "movement": 0,
        "first": 0,
        "later": 0,
        "invalid_event_retries": 0,
        "pose_vpt": 0,
        "pose_pi3x": 0,
    }
    for record in losses:
        sampled = str(record.get("sampled_category", ""))
        if sampled in {
            "place",
            "mine",
            "negative",
            "camera_first",
            "camera_later",
            "camera_rollout",
        }:
            counts[sampled] += 1
        category = str(record.get("training_category", ""))
        if category == "movement":
            counts["movement"] += 1
        if category in {"place", "mine"} and bool(record.get("event_valid", False)):
            counts[f"valid_{category}"] += 1
        chunk_mode = str(record.get("chunk_mode", ""))
        counts["first" if chunk_mode == "first" else "later"] += 1
        pose_source = str(record.get("pose_source", ""))
        if pose_source == "vpt_telemetry":
            counts["pose_vpt"] += 1
        elif pose_source == "pi3x":
            counts["pose_pi3x"] += 1
        counts["invalid_event_retries"] += int(record.get("invalid_event_retries", 0) or 0)
    return counts


def interaction_teacher_cache_key(item):
    training = dict(item.get("training", {}) or {})
    target_indices = ",".join(str(int(value)) for value in training.get("target_indices", []))
    return (
        f"{int(training.get('row_index', -1))}:"
        f"{training.get('direction', 'forward')}:"
        f"{target_indices}"
    )


def checkpoint_model_path(value, *, label):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(str(path.absolute()))
    checkpoints_root = Path(str((REPO_ROOT / "checkpoints").absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"{label} must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}. Run `python scripts/check_models.py`.")
    return str(path)


def _json_text(obj):
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```"


def create_tensorboard_writer(args, out_dir):
    if not bool(args.tensorboard):
        return None, None
    log_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else out_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the tensorboard package. Install requirements.txt or run `pip install tensorboard`."
        ) from exc
    writer = SummaryWriter(log_dir=str(log_dir))
    print(json.dumps({"event": "tensorboard_enabled", "log_dir": str(log_dir)}), flush=True)
    return writer, log_dir


def tensorboard_add_scalar(writer, tag, value, step):
    if writer is None:
        return
    if isinstance(value, bool):
        value = float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(tag, value, int(step))


def tensorboard_log_record(writer, record, step):
    if writer is None:
        return
    tensorboard_add_scalar(writer, "train/loss", record.get("loss"), step)
    tensorboard_add_scalar(writer, "train/lr", record.get("lr"), step)
    tensorboard_add_scalar(writer, "train/grad_norm", record.get("grad_norm"), step)
    tensorboard_add_scalar(writer, "train/elapsed_s", record.get("elapsed_s"), step)
    skip = {
        "step",
        "seq",
        "loss",
        "lr",
        "optimizer",
        "adamw_weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
    }
    for key, value in record.items():
        if key in skip:
            continue
        tensorboard_add_scalar(writer, f"stats/{key}", value, step)


def write_interaction_debug_summary(out_dir, step, losses, *, window=150):
    recent = losses[-max(int(window), 1) :]
    if not recent:
        return
    summary = {
        "step": int(step),
        "window": int(min(len(recent), max(int(window), 1))),
        "records": len(recent),
        "training_stage": str(recent[-1].get("training_stage", "")),
        "sampled_category_counts": dict(Counter(str(item.get("sampled_category", "")) for item in recent)),
        "action_type_counts": dict(Counter(str(item.get("action_type", "")) for item in recent)),
    }
    numeric_keys = [
        "loss",
        "stage0_weighted_flow",
        "stage0_focus_flow",
        "stage0_background_flow",
        "interaction_router_loss",
        "router_weighted_ratio",
        "raw_gate_mean_stage0",
        "final_gate_mean_stage0",
        "raw_delta_rms_stage0",
        "final_injection_rms_stage0",
        "teacher_mask_area_ratio_stage0",
        "interaction_router_positive_bce_stage0",
        "interaction_router_negative_bce_stage0",
        "interaction_router_dice_stage0",
        "interaction_gate_inside_teacher_stage0",
        "interaction_gate_outside_teacher_stage0",
        "interaction_raw_delta_inside_teacher_stage0",
        "interaction_raw_delta_outside_teacher_stage0",
        "interaction_injection_inside_teacher_stage0",
        "interaction_injection_outside_teacher_stage0",
    ]
    for key in numeric_keys:
        values = [
            float(item[key])
            for item in recent
            if key in item and item[key] is not None and math.isfinite(float(item[key]))
        ]
        if values:
            summary[key] = {
                "mean": float(sum(values) / len(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
    write_json(Path(out_dir) / f"interaction_summary_step{int(step):04d}.json", summary)


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _row_text(row, *keys, default=""):
    for key in keys:
        value = row.get(key, None)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return str(default)


def _source_video_digest(row, exact_args, digest_cache):
    source_path = resolve_optional_data_path(row.get("video_path", ""), exact_args.data_root)
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(f"Cannot fingerprint candidate source video: {source_path}")
    cache_key = str(source_path.resolve())
    if cache_key not in digest_cache:
        digest_cache[cache_key] = file_sha256(source_path)
    return digest_cache[cache_key]


def _event_aligned_conflict(df, row_index, target_fps, window_frames):
    row = df.iloc[int(row_index)]
    segment = str(row.get("segment_id", row.get("source_segment_id", "")))
    current = row.get("event_source_frame", row.get("source_event_frame"))
    if not segment or current is None or pd.isna(current):
        return None
    current = int(current)
    source_fps = float(row.get("fps", target_fps) or target_fps)
    horizon = int(math.ceil(int(window_frames) * source_fps / max(float(target_fps), 1.0e-6)))
    current_action = str(row.get("action_type", "")).strip().lower()
    current_complete = row.get("complete_frame")
    for other_index, other in df.iterrows():
        if int(other_index) == int(row_index):
            continue
        if str(other.get("segment_id", other.get("source_segment_id", ""))) != segment:
            continue
        action = str(other.get("action_type", "")).strip().lower()
        if action not in {"place", "mine_active", "mine_complete"}:
            continue
        frame = other.get("event_source_frame", other.get("source_event_frame"))
        if frame is None or pd.isna(frame):
            continue
        frame = int(frame)
        same_mine_completion = (
            current_action == "mine_active"
            and action == "mine_complete"
            and current_complete is not None
            and not pd.isna(current_complete)
            and frame == int(current_complete)
        )
        if current < frame <= current + horizon and not same_mine_completion:
            return f"second_structural_event:{frame}:{action}"
    return None


def export_teacher_candidates(items, df, exact_args, output_dir, limit=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    source_digest_cache = {}
    event_aligned = bool(getattr(exact_args, "event_aligned_interaction", False))
    source_limit = int(math.ceil(int(limit) / 2.0)) if event_aligned and int(limit) > 0 else limit
    candidate_ratios = (
        {"place": 0.5, "mine_active": 0.25, "mine_complete": 0.25, "negative": 0.0}
        if event_aligned
        else interaction_action_ratios("joint_stage0")
    )
    row_indices = stratified_candidate_indices(
        df.to_dict(orient="records"),
        source_limit,
        candidate_ratios,
    )
    selected_actions = Counter(
        "negative"
        if _row_text(df.iloc[index], "action_type", default="none").lower() in {"", "none", "negative"}
        else _row_text(df.iloc[index], "action_type").lower()
        for index in row_indices
    )
    print(
        json.dumps(
            {
                "event": "teacher_candidate_stratified_selection",
                "source_rows": len(row_indices),
                "source_action_counts": dict(selected_actions),
                "requested_limit": int(limit),
            }
        ),
        flush=True,
    )
    for row_index in row_indices:
        row = df.iloc[row_index]
        action_type = str(row.get("action_type", "none") or "none").strip().lower()
        category = "negative" if action_type == "none" else "mine" if action_type.startswith("mine") else "place"
        histories = ["first", "later"]
        for history_type in histories:
            if event_aligned:
                conflict = _event_aligned_conflict(
                    df, row_index, exact_args.online_target_fps, exact_args.num_frames
                )
                if conflict:
                    manifest_rows.append(
                        {
                            **row.to_dict(),
                            "history_type": history_type,
                            "teacher_candidate_path": "",
                            "candidate_error": conflict,
                        }
                    )
                    continue
            try:
                item = items.get(
                    row_index,
                    requested_category=category,
                    requested_chunk_mode=(
                        f"interaction_event_{history_type}"
                        if event_aligned
                        else f"interaction_{history_type}"
                    ),
                    keep_frames=True,
                )
            except (RuntimeError, ValueError) as exc:
                manifest_rows.append(
                    {
                        **row.to_dict(),
                        "history_type": history_type,
                        "teacher_candidate_path": "",
                        "candidate_error": str(exc),
                    }
                )
                release_cuda_cache()
                continue
            conditioning = item.get("interaction_conditioning")
            if conditioning is None:
                raise RuntimeError(f"Candidate {row_index}:{history_type} has no interaction conditioning.")
            target = item["target_latents"].detach().float()
            aligned = align_interaction_signals_to_grid(
                conditioning["payload"],
                batch_size=target.shape[0],
                temporal=target.shape[2],
                height=target.shape[3],
                width=target.shape[4],
                device=target.device,
                visibility=conditioning.get("visibility"),
                world_valid=conditioning.get("world_valid"),
            )
            training = dict(item["training"])
            interaction_payload = item.get("interaction_payload") or {}
            payload_hash = interaction_payload_hash(interaction_payload)
            source_digest = _source_video_digest(row, exact_args, source_digest_cache)
            normalized_action = "none" if action_type in {"", "none", "negative"} else action_type
            object_id = _row_text(row, "object_id", "block_id")
            block_id = _row_text(row, "block_id", "object_id")
            training_category = _row_text(
                item,
                "training_category",
                default=_row_text(row, "training_category", "category", default=category),
            ).lower()
            identity = {
                "event_id": str(row.get("event_id", "")),
                "action_type": normalized_action,
                "block_id": block_id,
                "object_id": object_id,
                "training_category": training_category,
                "interaction_payload_hash": payload_hash,
                "source_video_digest": source_digest,
                "history_type": history_type,
                "target_indices": [int(value) for value in training.get("target_indices", [])],
                "reference_frame_index": int(
                    -1 if training.get("reference_frame_index") is None else training["reference_frame_index"]
                ),
                "history_indices": [int(value) for value in training.get("history_indices", [])],
                "geometry_keyframe_frames": [int(value) for value in training.get("keyframe_indices", [])],
                "render_pose_indices": [int(value) for value in training.get("render_pose_indices", [])],
                "target_start_frame": int(training.get("target_start_frame", 0)),
                "event_local_frame": int(training.get("event_local_frame", 0) or 0),
                "chunk_mode": str(training.get("chunk_mode", "")),
                "direction": str(training.get("direction", "")),
                "source_segment_id": str(row.get("segment_id", row.get("id", ""))),
                "candidate_config_hash": str(exact_args.fixed_teacher_config_hash),
            }
            candidate_name = stable_json_hash(identity)
            identity["candidate_cache_key"] = candidate_name
            candidate_path = output_dir / f"{candidate_name}.npz"
            training_cache_path = output_dir / f"{candidate_name}.pt"
            target_rgb = np.stack(
                [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in item.get("target_frames", [])]
            )
            warp_rgb = np.stack(
                [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in item.get("history_frames", [])]
            )
            rgb_to_latent = rgb_frame_to_latent_indices(
                len(target_rgb),
                int(target.shape[2]),
                int(getattr(exact_args, "vae_temporal_scale", 4)),
            )
            np.savez_compressed(
                candidate_path,
                target_latents=target[0].cpu().numpy().astype(np.float16),
                warp_latents=conditioning["warp_latents"][0].detach().float().cpu().numpy().astype(np.float16),
                reference_latents=item["reference_latents"][0].detach().float().cpu().numpy().astype(np.float16),
                action_mask=aligned["action"][0].cpu().numpy().astype(np.float16),
                visibility=aligned["visibility"][0].cpu().numpy().astype(np.float16),
                world_valid=aligned["world_valid"][0].cpu().numpy().astype(np.float16),
                target_rgb=target_rgb,
                warp_rgb=warp_rgb,
                reference_rgb=np.asarray(item["reference_frame"].convert("RGB"), dtype=np.uint8),
                interaction_payload_json=np.asarray(json.dumps(interaction_payload, ensure_ascii=False)),
                rgb_frame_to_latent_index=np.asarray(rgb_to_latent, dtype=np.int16),
                candidate_identity_json=np.asarray(json.dumps(identity, ensure_ascii=False, sort_keys=True)),
            )
            torch.save(
                {
                    "schema_version": 2,
                    "candidate_identity": identity,
                    "target_latents": _cpu_tree(item["target_latents"]),
                    "prompt_embeds": _cpu_tree(item["prompt_embeds"]),
                    "histories": _cpu_tree(item["histories"]),
                    "base_histories": _cpu_tree(item.get("base_histories")),
                    "interaction_conditioning": _cpu_tree(item["interaction_conditioning"]),
                    "interaction_payload": interaction_payload,
                    "world_valid_mask_latents": _cpu_tree(item.get("world_valid_mask_latents")),
                    "training": training,
                    "seq": str(item["seq"]),
                    "training_category": str(item.get("training_category", "")),
                },
                training_cache_path,
            )
            candidate_npz_sha256 = file_sha256(candidate_path)
            training_cache_sha256 = file_sha256(training_cache_path)
            manifest_rows.append(
                {
                    **row.to_dict(),
                    "history_type": history_type,
                    "teacher_candidate_path": candidate_path.as_posix(),
                    "training_cache_path": training_cache_path.as_posix(),
                    "target_indices": encode_index_sequence(identity["target_indices"]),
                    "reference_frame_index": identity["reference_frame_index"],
                    "target_start_frame": identity["target_start_frame"],
                    "event_local_frame": identity["event_local_frame"],
                    "history_indices": encode_index_sequence(identity["history_indices"]),
                    "geometry_keyframe_frames": encode_index_sequence(identity["geometry_keyframe_frames"]),
                    "render_pose_indices": encode_index_sequence(identity["render_pose_indices"]),
                    "chunk_mode": identity["chunk_mode"],
                    "direction": identity["direction"],
                    "source_segment_id": identity["source_segment_id"],
                    "candidate_cache_key": candidate_name,
                    "candidate_config_hash": identity["candidate_config_hash"],
                    "action_type": normalized_action,
                    "block_id": block_id,
                    "object_id": object_id,
                    "training_category": training_category,
                    "interaction_payload_hash": payload_hash,
                    "source_video_digest": source_digest,
                    "candidate_npz_sha256": candidate_npz_sha256,
                    "training_cache_sha256": training_cache_sha256,
                    "rgb_frame_to_latent_index": encode_index_sequence(rgb_to_latent),
                    "target_reference": str(row.get("video_path", "")),
                    "warp_reference": str(item.get("seq", "")),
                    "candidate_error": "",
                }
            )
            del item, target, aligned
            release_cuda_cache()
    manifest_path = output_dir / "teacher_candidate_manifest.csv"
    columns = sorted({key for row in manifest_rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_json(
        output_dir / "teacher_candidate_audit.json",
        {
            "rows": len(manifest_rows),
            "selected_source_rows": len(row_indices),
            "selected_source_action_counts": dict(selected_actions),
            "successful": sum(not row.get("candidate_error") for row in manifest_rows),
            "failed": sum(bool(row.get("candidate_error")) for row in manifest_rows),
            "manifest": str(manifest_path),
        },
    )
    return manifest_path


def save_training_state(
    path,
    *,
    transformer,
    optimizer,
    global_step,
    args,
    refined_teacher_cache,
    losses,
    distribution_counts=None,
    attempt_step=0,
    skipped_invalid_step=0,
    adapter_name=None,
    sampling_plan=None,
):
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }
    completed_step = max(int(global_step) - 1, 0)
    payload = {
        "training_state_version": 4,
        "trainable_state": trainable_state,
        "wah_lora_state": {
            key: value.detach().cpu()
            for key, value in opt.get_peft_model_state_dict(
                transformer, adapter_name=adapter_name
            ).items()
        },
        "interaction_state": {
            key: value.detach().cpu()
            for key, value in transformer.interaction_conditioning.state_dict().items()
        }
        if getattr(transformer, "interaction_conditioning", None) is not None
        else {},
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "current_stage": training_stage_for_step(
            completed_step,
            args.base_train_steps,
            args.enable_bidirectional_training,
        ),
        "base_train_steps": int(args.base_train_steps),
        "bidirectional_train_steps": int(args.bidirectional_train_steps),
        "base_completed_steps": min(int(global_step), int(args.base_train_steps)),
        "bidirectional_completed_steps": max(int(global_step) - int(args.base_train_steps), 0),
        "enable_bidirectional_training": bool(args.enable_bidirectional_training),
        "training_profile": str(args.training_profile),
        "wah_recipe": dict(getattr(transformer, "_wah_recipe", {}) or {}),
        "wah_initialization": dict(getattr(transformer, "_wah_initialization", {}) or {}),
        "distribution_counts": dict(distribution_counts or {}),
        "attempt_step": int(attempt_step),
        "skipped_invalid_step": int(skipped_invalid_step),
        "training_mode": str(args.interaction_training_mode),
        "interaction_active_stages": [0],
        "sampling_plan": dict(sampling_plan or {}),
        "phase_plan": list(getattr(args, "resolved_interaction_phase_plan", []) or []),
        "approved_teacher_row_ids": list(getattr(args, "approved_teacher_row_ids", []) or []),
        "teacher_pool_manifest_hash": str(getattr(args, "teacher_pool_manifest_hash", "")),
        "action_history_pool_sizes": dict(getattr(args, "action_history_pool_sizes", {}) or {}),
        "resume_contract": dict(getattr(args, "training_resume_contract", {}) or {}),
        "effective_optimizer_step": int(global_step),
        "config": vars(args).copy(),
        "launch_argv": list(sys.argv),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "refined_teacher_cache": {
            key: value.detach().cpu() for key, value in refined_teacher_cache.items()
        },
        "losses": list(losses),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_state(path, *, transformer, optimizer, device, adapter_name=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("training_state_version", 0)) < 1:
        print(
            json.dumps(
                {
                    "event": "legacy_checkpoint_no_training_state",
                    "path": str(path),
                    "resume_step": 0,
                }
            ),
            flush=True,
        )
        return {"global_step": 0, "refined_teacher_cache": {}, "losses": []}
    named_parameters = dict(transformer.named_parameters())
    wah_lora_state = dict(payload.get("wah_lora_state", {}))
    if wah_lora_state:
        expected_lora = opt.get_peft_model_state_dict(transformer, adapter_name=adapter_name)
        missing_lora = sorted(set(expected_lora) - set(wah_lora_state))
        unexpected_lora = sorted(set(wah_lora_state) - set(expected_lora))
        if missing_lora or unexpected_lora:
            raise ValueError(
                f"WAH LoRA checkpoint mismatch: missing={missing_lora[:10]}, unexpected={unexpected_lora[:10]}"
            )
        opt.set_peft_model_state_dict(transformer, wah_lora_state, adapter_name=adapter_name)
    interaction_state = dict(payload.get("interaction_state", {}))
    if interaction_state:
        result = transformer.interaction_conditioning.load_state_dict(interaction_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"Interaction checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
    checkpoint_trainable_state = dict(payload.get("trainable_state", {}))
    missing = []
    for name, value in checkpoint_trainable_state.items():
        parameter = named_parameters.get(name)
        if parameter is None:
            missing.append(name)
            continue
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(
                f"Resume checkpoint shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}."
            )
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if missing:
        raise ValueError(f"Resume checkpoint contains unknown trainable parameters: {missing[:10]}")
    default_initialized = [
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and name not in checkpoint_trainable_state
    ]
    if default_initialized:
        print(
            json.dumps(
                {
                    "event": "training_checkpoint_default_initialized",
                    "message": "Legacy checkpoint is missing coarse-to-fine parameters; using defaults.",
                    "missing_keys": default_initialized,
                }
            ),
            flush=True,
        )
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except ValueError as exc:
        if not default_initialized:
            raise
        print(
            json.dumps(
                {
                    "event": "legacy_optimizer_state_reset",
                    "message": str(exc),
                }
            ),
            flush=True,
        )
    else:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)
    rng = dict(payload.get("rng_state", {}) or {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {
        "global_step": int(payload.get("global_step", 0)),
        "current_stage": str(payload.get("current_stage", "base")),
        "base_train_steps": int(payload.get("base_train_steps", 0)),
        "bidirectional_train_steps": int(payload.get("bidirectional_train_steps", 0)),
        "enable_bidirectional_training": bool(payload.get("enable_bidirectional_training", False)),
        "base_completed_steps": int(payload.get("base_completed_steps", 0)),
        "bidirectional_completed_steps": int(payload.get("bidirectional_completed_steps", 0)),
        "refined_teacher_cache": {
            str(key): value.detach().cpu()
            for key, value in dict(payload.get("refined_teacher_cache", {})).items()
        },
        "losses": list(payload.get("losses", [])),
        "training_profile": str(payload.get("training_profile", "joint")),
        "training_mode": str(payload.get("training_mode", "joint_stage0")),
        "interaction_active_stages": list(payload.get("interaction_active_stages", [0])),
        "wah_recipe": dict(payload.get("wah_recipe", {}) or {}),
        "wah_initialization": dict(payload.get("wah_initialization", {}) or {}),
        "attempt_step": int(payload.get("attempt_step", 0)),
        "skipped_invalid_step": int(payload.get("skipped_invalid_step", 0)),
        "distribution_counts": dict(payload.get("distribution_counts", {}) or {}),
        "sampling_plan": dict(payload.get("sampling_plan", {}) or {}),
        "phase_plan": list(payload.get("phase_plan", []) or []),
        "approved_teacher_row_ids": list(payload.get("approved_teacher_row_ids", []) or []),
        "teacher_pool_manifest_hash": str(payload.get("teacher_pool_manifest_hash", "")),
        "action_history_pool_sizes": dict(payload.get("action_history_pool_sizes", {}) or {}),
        "resume_contract": dict(payload.get("resume_contract", {}) or {}),
        "effective_optimizer_step": int(payload.get("effective_optimizer_step", payload.get("global_step", 0))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the release Warp-as-History LoRA.")
    parser.add_argument("--base_model_path", default=DEFAULT_HELIOS_MODEL)
    parser.add_argument(
        "--transformer_path",
        default="",
        help="Optional transformer-only checkpoint. Defaults to --base_model_path.",
    )
    parser.add_argument("--data_root", default="data/training")
    parser.add_argument("--prompt_csv", default="data/training/training_data.csv")
    parser.add_argument("--output_dir", default="runs/warp_as_history_lora")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional manifest row limit for debugging. By default all rows are eligible for step-level sampling.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--base_train_steps", type=int, default=None)
    parser.add_argument("--bidirectional_train_steps", type=int, default=1500)
    parser.add_argument(
        "--enable_bidirectional_training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bidirectional_interval", type=int, default=8)
    parser.add_argument("--bidirectional_feedback_weight", type=float, default=0.5)
    parser.add_argument("--bidirectional_teacher_floor", type=float, default=0.5)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--camera_checkpoint",
        type=Path,
        default=None,
        help="Initialize camera/WAH LoRA weights for interaction-profile training.",
    )
    parser.add_argument(
        "--init_wah_lora_path",
        type=Path,
        default=Path("checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"),
    )
    parser.add_argument(
        "--require_init_wah_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wah_lora_lr", type=float, default=None)
    parser.add_argument("--interaction_lr", type=float, default=None)
    parser.add_argument("--router_lr", type=float, default=5.0e-5)
    parser.add_argument("--training_profile", choices=["camera", "interaction", "joint"], default="joint")
    parser.add_argument(
        "--interaction_training_mode",
        choices=["router_overfit", "adapter_overfit", "joint_pilot", "joint_stage0"],
        default="joint_stage0",
    )
    parser.add_argument("--place_step_ratio", type=float, default=0.5)
    parser.add_argument("--mine_step_ratio", type=float, default=0.3)
    parser.add_argument("--other_step_ratio", type=float, default=0.2)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine", "linear"], default="constant")
    parser.add_argument("--lr_schedule_final_ratio", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_attempt_steps", type=int, default=15000)
    parser.add_argument("--interaction_router_loss_scale", type=float, default=0.005)
    parser.add_argument("--interaction_focus_scale", type=float, default=1.0)
    parser.add_argument("--interaction_teacher_support_threshold", type=float, default=0.25)
    parser.add_argument("--interaction_max_metadata_rotation_deg", type=float, default=20.0)
    parser.add_argument("--interaction_max_camera_rotation_deg", type=float, default=20.0)
    parser.add_argument("--interaction_min_telemetry_confidence", type=float, default=0.0)
    parser.add_argument("--interaction_min_mine_active_frames", type=int, default=4)
    parser.add_argument(
        "--require_approved_teacher_pool",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--teacher_pool_review_column", default="review_status")
    parser.add_argument("--export_teacher_candidates_only", action="store_true")
    parser.add_argument("--teacher_candidate_output_dir", type=Path, default=None)
    parser.add_argument("--teacher_candidate_limit", type=int, default=0)
    parser.add_argument("--event_aligned_interaction", action="store_true")
    parser.add_argument("--interaction_phase_steps", type=int, nargs=3, default=None)
    parser.add_argument(
        "--interaction_first_history_ratios",
        type=float,
        nargs=3,
        default=[0.40, 0.30, 0.25],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--history_sizes", type=int, nargs=3, default=[16, 2, 1])
    parser.add_argument("--history_temporal_layout", choices=["long_mid_short", "short_mid_long"], default="long_mid_short")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--attention_backend", default="native")
    parser.add_argument("--flow_matching_stage_sampling", choices=["all", "fixed"], default="fixed")
    parser.add_argument("--flow_matching_stage_id", type=int, default=0)
    parser.add_argument("--flow_matching_train_exact_timestep_sampling", choices=["training_density", "first", "first_second_interval"], default="training_density")
    parser.add_argument("--history_positioning", choices=["none", "last_n", "last_n_same_order"], default="last_n_same_order")
    parser.add_argument("--history_position_count", type=int, default=9)
    parser.add_argument("--history_position_delta", type=int, default=0)
    parser.add_argument(
        "--warp_history_downsample_mode",
        choices=["short", "patch_mid"],
        default="short",
        help="Use patch_mid to train the efficient Warp-as-History LoRA; default preserves the release recipe.",
    )
    parser.add_argument("--add_noise_to_video_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--visible_token_drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible_token_mode", choices=["drop", "none"], default="drop")
    parser.add_argument("--visible_token_threshold", type=float, default=0.1)
    parser.add_argument("--direction_augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--direction_reverse_probability", type=float, default=0.5)
    parser.add_argument("--online_video_column", default="")
    parser.add_argument("--online_prompt_column", default="")
    parser.add_argument("--online_interaction_column", default="")
    parser.add_argument("--online_prompt_trigger", default="camctl23x.")
    parser.add_argument("--online_interaction_max_items", type=int, default=8)
    parser.add_argument("--online_interaction_pseudo_history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--online_interaction_pseudo_history_scale", type=float, default=0.0)
    parser.add_argument("--online_primary_fire_click_radius_frames", type=int, default=12)
    parser.add_argument("--online_primary_fire_residual_threshold", type=float, default=0.08)
    parser.add_argument("--primary_fire_focus_loss_scale", "--online_primary_fire_focus_loss_scale", dest="primary_fire_focus_loss_scale", type=float, default=3.0)
    parser.add_argument("--primary_fire_background_loss_scale", "--online_primary_fire_background_loss_scale", dest="primary_fire_background_loss_scale", type=float, default=1.0)
    parser.add_argument("--use_primary_fire_focus_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_primary_fire_event_condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--interaction_conditioning_mode",
        choices=["router", "binary", "off"],
        default="router",
        help="router is the default; binary preserves the old time-gate ablation.",
    )
    parser.add_argument("--interaction_adapter_rank", type=int, default=64)
    parser.add_argument("--interaction_semantic_dim", type=int, default=256)
    parser.add_argument("--interaction_stage_warp_scales", type=float, nargs=3, default=[1.0, 0.5, 0.25])
    parser.add_argument("--interaction_stage_adapter_scales", type=float, nargs=3, default=[1.0, 0.5, 0.25])
    parser.add_argument("--interaction_cross_stage_consistency_loss_scale", type=float, default=0.1)
    parser.add_argument("--interaction_router_temporal_loss_scale", type=float, default=1.0)
    parser.add_argument("--interaction_router_spatial_loss_scale", type=float, default=1.0)
    parser.add_argument("--interaction_router_negative_loss_scale", type=float, default=0.25)
    parser.add_argument("--interaction_router_sparsity_loss_scale", type=float, default=0.01)
    parser.add_argument("--interaction_debug_every", type=int, default=0)
    parser.add_argument("--use_minecraft_hud_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--online_frame_stride", type=int, default=1)
    parser.add_argument(
        "--online_target_fps",
        type=float,
        default=0.0,
        help="Timestamp-resample online videos to this fps. Use 16 for Minecraft; 0 keeps stride sampling.",
    )
    parser.add_argument(
        "--online_use_vpt_camera_poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use x/y/z/yaw/pitch from the row actions_path instead of Pi3X-estimated camera motion.",
    )
    parser.add_argument("--online_vpt_translation_scale", type=float, default=0.1)
    parser.add_argument(
        "--online_geometry_keyframe_stride",
        type=int,
        default=1,
        help="Estimate/cache Pi3X geometry every N target-fps frames; RGB targets remain full fps.",
    )
    parser.add_argument(
        "--minecraft_training_profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require 16 fps, VPT poses, HUD masking, and multi-chunk-compatible 33-frame windows.",
    )
    parser.add_argument("--online_primary_fire_window_probability", type=float, default=0.6)
    parser.add_argument("--online_max_video_frames", type=int, default=0)
    parser.add_argument("--online_warp_memory_cache_size", type=int, default=2)
    parser.add_argument("--online_warp_disk_cache_dir", default="auto")
    parser.add_argument("--online_first_chunk_prob", type=float, default=0.5)
    parser.add_argument("--interaction_event_local_min", type=int, default=6)
    parser.add_argument("--interaction_event_local_max", type=int, default=16)
    parser.add_argument("--online_max_history_frames", type=int, default=19)
    parser.add_argument("--online_future_keyframe_prob", type=float, default=0.0)
    parser.add_argument("--online_future_keyframes_min", type=int, default=0)
    parser.add_argument("--online_future_keyframes_max", type=int, default=0)
    parser.add_argument("--online_pi3_pixel_limit", type=int, default=255000)
    parser.add_argument("--online_pi3_conf_threshold", type=float, default=0.1)
    parser.add_argument("--online_pi3_depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--online_mesh_samples_per_axis", type=int, default=4)
    parser.add_argument("--online_render_mode", default="target_fill", choices=["splat", "target_fill"])
    parser.add_argument("--online_target_fill_radius", type=int, default=1)
    parser.add_argument("--online_target_fill_min_neighbors", type=int, default=4)
    parser.add_argument("--online_mesh_break_mode", default="depth_normal")
    parser.add_argument("--online_mesh_depth_rtol", type=float, default=0.03)
    parser.add_argument("--online_mesh_normal_tol_deg", type=float, default=5.0)
    parser.add_argument("--online_invisible_fill", default="mean_first_frame", choices=["mean_first_frame", "black"])
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0")
    parser.add_argument("--lora_adapter_name", default="warp_as_history")
    parser.add_argument("--save_every", type=int, default=150)
    parser.add_argument("--save_steps", type=int, nargs="*", default=[])
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard_log_dir", default="")
    parser.add_argument("--prompt_cache_dir", default="data/training/prompt_cache/helios_distilled_512")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_exact_args(args):
    if bool(args.minecraft_training_profile):
        if abs(float(args.online_target_fps) - 16.0) > 1.0e-6:
            raise ValueError("--minecraft_training_profile requires --online_target_fps 16.")
        if not bool(args.online_use_vpt_camera_poses):
            raise ValueError("--minecraft_training_profile requires --online_use_vpt_camera_poses.")
        if not bool(args.use_minecraft_hud_mask):
            raise ValueError("--minecraft_training_profile requires --use_minecraft_hud_mask.")
        if int(args.online_frame_stride) != 1:
            raise ValueError("--minecraft_training_profile forbids temporal frame stride; use --online_frame_stride 1.")
    exact = opt.parse_args([])
    exact.base_model_path = checkpoint_model_path(args.base_model_path, label="--base_model_path")
    exact.transformer_path = checkpoint_model_path(
        args.transformer_path or args.base_model_path,
        label="--transformer_path",
    )
    exact.data_root = args.data_root
    exact.prompt_csv = args.prompt_csv
    exact.output_dir = args.output_dir
    exact.limit = args.limit
    exact.use_warp_as_history = True
    exact.history_positioning = args.history_positioning
    exact.history_position_count = int(args.history_position_count)
    exact.history_position_delta = int(args.history_position_delta)
    exact.warp_history_downsample_mode = str(args.warp_history_downsample_mode)
    exact.add_noise_to_video_latents = bool(args.add_noise_to_video_latents)
    exact.image_noise_sigma_min = float(args.image_noise_sigma_min)
    exact.image_noise_sigma_max = float(args.image_noise_sigma_max)
    exact.video_noise_sigma_min = float(args.video_noise_sigma_min)
    exact.video_noise_sigma_max = float(args.video_noise_sigma_max)
    exact.history_visible_token_drop = bool(args.visible_token_drop)
    exact.visible_token_mode = str(args.visible_token_mode)
    exact.history_visible_token_threshold = float(args.visible_token_threshold)
    exact.history_invisible_token_mode = "none"
    exact.history_invisible_token_threshold = float(args.visible_token_threshold)
    exact.online_video_column = str(args.online_video_column)
    exact.online_prompt_column = str(args.online_prompt_column)
    exact.online_interaction_column = str(args.online_interaction_column)
    exact.online_prompt_trigger = str(args.online_prompt_trigger)
    exact.online_interaction_max_items = int(args.online_interaction_max_items)
    exact.online_interaction_pseudo_history = bool(args.online_interaction_pseudo_history)
    exact.online_interaction_pseudo_history_scale = float(args.online_interaction_pseudo_history_scale)
    exact.online_primary_fire_click_radius_frames = int(args.online_primary_fire_click_radius_frames)
    exact.online_primary_fire_residual_threshold = float(args.online_primary_fire_residual_threshold)
    exact.online_primary_fire_focus_loss_scale = float(args.primary_fire_focus_loss_scale)
    exact.online_primary_fire_background_loss_scale = float(args.primary_fire_background_loss_scale)
    exact.use_primary_fire_focus_loss = bool(args.use_primary_fire_focus_loss)
    exact.interaction_conditioning_mode = str(args.interaction_conditioning_mode)
    if str(args.training_profile) == "camera":
        exact.interaction_conditioning_mode = "off"
    exact.use_primary_fire_event_condition = bool(
        args.use_primary_fire_event_condition and args.interaction_conditioning_mode == "binary"
    )
    exact.interaction_adapter_rank = int(args.interaction_adapter_rank)
    exact.interaction_semantic_dim = int(args.interaction_semantic_dim)
    exact.interaction_stage_warp_scales = [float(value) for value in args.interaction_stage_warp_scales]
    exact.interaction_stage_adapter_scales = [float(value) for value in args.interaction_stage_adapter_scales]
    exact.interaction_cross_stage_consistency_loss_scale = float(
        args.interaction_cross_stage_consistency_loss_scale
    )
    exact.interaction_router_temporal_loss_scale = float(args.interaction_router_temporal_loss_scale)
    exact.interaction_router_spatial_loss_scale = float(args.interaction_router_spatial_loss_scale)
    exact.interaction_router_negative_loss_scale = float(args.interaction_router_negative_loss_scale)
    exact.interaction_router_sparsity_loss_scale = float(args.interaction_router_sparsity_loss_scale)
    exact.interaction_router_loss_scale = float(args.interaction_router_loss_scale)
    exact.interaction_focus_scale = float(args.interaction_focus_scale)
    exact.interaction_teacher_support_threshold = float(args.interaction_teacher_support_threshold)
    exact.interaction_training_mode = str(args.interaction_training_mode)
    exact.interaction_max_metadata_rotation_deg = float(args.interaction_max_metadata_rotation_deg)
    exact.interaction_max_camera_rotation_deg = float(args.interaction_max_camera_rotation_deg)
    exact.interaction_min_telemetry_confidence = float(args.interaction_min_telemetry_confidence)
    exact.interaction_min_mine_active_frames = int(args.interaction_min_mine_active_frames)
    exact.interaction_active_stages = [0]
    exact.event_aligned_interaction = bool(args.event_aligned_interaction)
    exact.base_train_steps = int(args.base_train_steps)
    exact.bidirectional_train_steps = int(args.bidirectional_train_steps)
    exact.enable_bidirectional_training = bool(args.enable_bidirectional_training)
    exact.bidirectional_interval = int(args.bidirectional_interval)
    exact.bidirectional_feedback_weight = float(args.bidirectional_feedback_weight)
    exact.bidirectional_teacher_floor = float(args.bidirectional_teacher_floor)
    exact.use_minecraft_hud_mask = bool(args.use_minecraft_hud_mask)
    exact.online_direction_augmentation = bool(args.direction_augmentation)
    exact.online_direction_reverse_prob = float(args.direction_reverse_probability)
    exact.online_frame_stride = int(args.online_frame_stride)
    exact.online_target_fps = float(args.online_target_fps)
    exact.online_use_vpt_camera_poses = bool(args.online_use_vpt_camera_poses)
    exact.online_vpt_translation_scale = float(args.online_vpt_translation_scale)
    exact.camera_multiply_translation_by_depth = True
    exact.online_geometry_keyframe_stride = max(1, int(args.online_geometry_keyframe_stride))
    exact.minecraft_training_profile = bool(args.minecraft_training_profile)
    exact.online_primary_fire_window_probability = float(args.online_primary_fire_window_probability)
    exact.online_max_video_frames = int(args.online_max_video_frames)
    exact.online_warp_memory_cache_size = int(args.online_warp_memory_cache_size)
    exact.online_warp_disk_cache_dir = str(args.online_warp_disk_cache_dir)
    exact.online_first_chunk_prob = float(args.online_first_chunk_prob)
    exact.interaction_event_local_min = int(args.interaction_event_local_min)
    exact.interaction_event_local_max = int(args.interaction_event_local_max)
    exact.training_profile = str(args.training_profile)
    exact.pose_convention = "opencv_c2w_relative"
    exact.online_max_history_frames = int(args.online_max_history_frames)
    exact.online_future_keyframe_prob = float(args.online_future_keyframe_prob)
    exact.online_future_keyframes_min = int(args.online_future_keyframes_min)
    exact.online_future_keyframes_max = int(args.online_future_keyframes_max)
    exact.online_pi3_pixel_limit = int(args.online_pi3_pixel_limit)
    exact.online_pi3_conf_threshold = float(args.online_pi3_conf_threshold)
    exact.online_pi3_depth_edge_rtol = float(args.online_pi3_depth_edge_rtol)
    exact.online_mesh_samples_per_axis = int(args.online_mesh_samples_per_axis)
    exact.online_render_mode = str(args.online_render_mode)
    exact.online_target_fill_radius = int(args.online_target_fill_radius)
    exact.online_target_fill_min_neighbors = int(args.online_target_fill_min_neighbors)
    exact.online_mesh_break_mode = str(args.online_mesh_break_mode)
    exact.online_mesh_depth_rtol = float(args.online_mesh_depth_rtol)
    exact.online_mesh_normal_tol_deg = float(args.online_mesh_normal_tol_deg)
    exact.online_invisible_fill = str(args.online_invisible_fill)
    exact.height = int(args.height)
    exact.width = int(args.width)
    exact.num_frames = int(args.num_frames)
    exact.num_latent_frames_per_chunk = int(args.num_latent_frames_per_chunk)
    exact.history_sizes = [int(x) for x in args.history_sizes]
    exact.history_temporal_layout = args.history_temporal_layout
    exact.pyramid_num_inference_steps_list = list(args.pyramid_num_inference_steps_list)
    exact.attention_backend = str(args.attention_backend)
    exact.is_amplify_first_chunk = False
    exact.seed = int(args.seed)
    exact.lora_rank = int(args.lora_rank)
    exact.lora_alpha = int(args.lora_alpha)
    exact.lora_dropout = float(args.lora_dropout)
    exact.lora_target_modules = args.lora_target_modules
    exact.lora_adapter_name = args.lora_adapter_name
    exact.flow_matching_mode = "train_exact"
    exact.flow_matching_stage_sampling = args.flow_matching_stage_sampling
    exact.flow_matching_stage_id = int(args.flow_matching_stage_id)
    exact.flow_matching_train_exact_timestep_sampling = str(args.flow_matching_train_exact_timestep_sampling)
    exact.flow_matching_use_dynamic_shifting = "off"
    exact.weighting_scheme = "none"
    exact.iters = int(args.max_steps)
    exact.lr = float(args.lr)
    exact.lr_schedule = args.lr_schedule
    exact.lr_schedule_final_ratio = float(args.lr_schedule_final_ratio)
    exact.gradient_checkpointing = bool(args.gradient_checkpointing)
    exact.overwrite = bool(args.overwrite)

    opt.validate_args(exact)
    return exact


def main():
    args = parse_args()
    if args.base_train_steps is None:
        args.base_train_steps = (
            1000
            if str(args.training_profile) == "camera"
            else interaction_training_mode_default_steps(args.interaction_training_mode)
            if str(args.training_profile) == "interaction"
            else 1500
        )
    if str(args.training_profile) == "camera":
        args.interaction_conditioning_mode = "off"
    if (
        str(args.training_profile) == "interaction"
        and args.camera_checkpoint is None
        and args.resume_from_checkpoint is None
    ):
        raise ValueError("--training_profile interaction requires --camera_checkpoint.")
    if args.max_steps is not None:
        args.base_train_steps = int(args.max_steps)
    if int(args.base_train_steps) < 0 or int(args.bidirectional_train_steps) < 0:
        raise ValueError("Training stage steps must be non-negative.")
    if int(args.bidirectional_interval) <= 0:
        raise ValueError("--bidirectional_interval must be positive.")
    args.max_steps = training_total_steps(
        args.base_train_steps,
        args.bidirectional_train_steps,
        args.enable_bidirectional_training,
    )
    if int(args.max_steps) <= 0:
        raise ValueError("Total training steps must be positive.")
    out_dir = Path(args.output_dir)
    loss_path = out_dir / "train_loss.json"
    if loss_path.exists() and not args.overwrite and args.resume_from_checkpoint is None:
        raise FileExistsError(f"{loss_path} exists. Use --overwrite to run again.")
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_cache_arg = str(args.online_warp_disk_cache_dir or "").strip()
    if disk_cache_arg.lower() == "auto":
        args.online_warp_disk_cache_dir = str(out_dir / "online_warp_cache")
    elif disk_cache_arg.lower() in {"", "none", "off", "disable", "disabled"}:
        args.online_warp_disk_cache_dir = ""
    exact_args = build_exact_args(args)
    tb_writer, tb_log_dir = create_tensorboard_writer(args, out_dir)

    opt.seed_global_rng(args.seed)
    device = torch.device("cuda")

    args.teacher_pool_manifest_hash = manifest_sha256(args.prompt_csv)
    df = pd.read_csv(args.prompt_csv)
    if args.limit is not None and int(args.limit) > 0:
        df = df.head(int(args.limit))
    df, training_meta = normalize_online_training_dataframe(df, exact_args)
    if df.empty:
        raise ValueError(f"No training rows loaded from {args.prompt_csv}")
    if str(args.training_profile) == "interaction":
        prompts = {str(value).strip() for value in df["prompt"].tolist()}
        if prompts != {NEUTRAL_MINECRAFT_PROMPT}:
            raise ValueError(f"Interaction training requires the exact neutral prompt, got {sorted(prompts)}.")
        if bool(args.require_approved_teacher_pool) and not bool(args.export_teacher_candidates_only):
            review_column = str(args.teacher_pool_review_column)
            if review_column not in df.columns:
                raise ValueError(
                    f"Interaction module validation requires an audited teacher pool column {review_column!r}."
                )
            approved = df[review_column].fillna("").astype(str).str.lower().eq("approved")
            df = df.loc[approved].reset_index(drop=True)
            if df.empty:
                raise ValueError("The audited teacher pool contains no review_status=approved rows.")
            if str(args.interaction_training_mode) in {"router_overfit", "adapter_overfit"}:
                if "overfit_selected" not in df.columns:
                    raise ValueError(
                        f"{args.interaction_training_mode} requires an overfit_selected column; "
                        "manually mark 16-32 approved samples with overfit_selected=true."
                    )
                selected = df["overfit_selected"].fillna("").astype(str).str.lower().isin(
                    ["1", "true", "yes"]
                )
                df = df.loc[selected].reset_index(drop=True)
                if df.empty:
                    raise ValueError(
                        f"{args.interaction_training_mode} requires manually approved rows with "
                        "overfit_selected=true; select 16-32 clean samples before training."
                    )
            positive_rows = df["action_type"].fillna("").astype(str).str.lower().isin(
                ["place", "mine_active", "mine_complete"]
            )
            if "stage0_positive_tokens" not in df.columns:
                raise ValueError("Audited teacher pool is missing stage0_positive_tokens.")
            invalid_stage0 = positive_rows & (
                pd.to_numeric(df["stage0_positive_tokens"], errors="coerce").fillna(0) <= 0
            )
            if bool(invalid_stage0.any()):
                raise ValueError(
                    f"Audited teacher pool contains {int(invalid_stage0.sum())} positive rows with empty Stage 0 support."
                )
            if "teacher_support_threshold" not in df.columns:
                raise ValueError("Audited teacher pool is missing teacher_support_threshold.")
            audited_thresholds = set(
                pd.to_numeric(df["teacher_support_threshold"], errors="raise").astype(float).tolist()
            )
            expected_threshold = float(args.interaction_teacher_support_threshold)
            if audited_thresholds != {expected_threshold}:
                raise ValueError(
                    "Audited teacher support threshold does not match training: "
                    f"manifest={sorted(audited_thresholds)} runtime={expected_threshold}"
                )
            for required_cache_column in ("teacher_cache_path", "training_cache_path"):
                if required_cache_column not in df.columns:
                    raise ValueError(f"Audited teacher pool is missing {required_cache_column}.")
            missing_teacher_cache = []
            for cache_column in ("teacher_cache_path", "training_cache_path"):
                for row_index, cache_value in df[cache_column].items():
                    cache_path = Path(str(cache_value))
                    if not cache_path.is_absolute():
                        cache_path = REPO_ROOT / cache_path
                    if not cache_path.is_file():
                        missing_teacher_cache.append((cache_column, int(row_index), str(cache_path)))
            if missing_teacher_cache:
                raise ValueError(
                    f"Audited teacher pool has missing fixed teacher files: {missing_teacher_cache[:5]}"
                )
    fixed_cache_only = (
        str(args.training_profile) == "interaction"
        and not bool(args.export_teacher_candidates_only)
        and fixed_cache_rows_ready(
            [row for _, row in df.iterrows()],
            getattr(exact_args, "data_root", "."),
        )
    )
    exact_args.fixed_cache_only = bool(fixed_cache_only)
    exact_args.online_warp_cache = (
        None if fixed_cache_only else build_online_warp_training_cache(df, exact_args, device)
    )
    if bool(args.export_teacher_candidates_only):
        step_sampler = None
        sampler_meta = {"mode": "teacher_candidate_export", "rows": len(df)}
        args.approved_teacher_row_ids = []
        args.action_history_pool_sizes = {}
        args.resolved_interaction_phase_plan = []
    else:
        step_sampler, sampler_meta = build_minecraft_step_sampler(df, args)
        args.approved_teacher_row_ids = sorted(
            approved_row_id(row, fallback=index)
            for index, row in enumerate(df.to_dict(orient="records"))
        )
        args.action_history_pool_sizes = dict(sampler_meta.get("pool_sizes", {}))
        args.resolved_interaction_phase_plan = list(sampler_meta.get("phase_plan", []))
    skipped_rows = []
    print(
        json.dumps(
            {
                "event": "online_warp_cache_config",
                "fixed_cache_only": bool(fixed_cache_only),
                "online_warp_cache_initialized": not bool(fixed_cache_only),
                "memory_cache_size": int(args.online_warp_memory_cache_size),
                "disk_cache_dir": str(args.online_warp_disk_cache_dir),
            }
        ),
        flush=True,
    )

    pipe = opt.load_pipeline(exact_args, device)
    pipe.transformer._wah_recipe = minecraft_wah_recipe(
        target_fps=args.online_target_fps,
        num_frames=args.num_frames,
        warp_history_downsample_mode=args.warp_history_downsample_mode,
        camera_warp_render_mode=args.online_render_mode,
        camera_control_translation_scale=args.online_vpt_translation_scale,
        camera_multiply_translation_by_depth=True,
        camera_mesh_samples_per_axis=args.online_mesh_samples_per_axis,
        camera_keyframe_max_previous=args.online_max_history_frames,
        visible_token_threshold=args.visible_token_threshold,
        amplify_first_chunk=False,
        history_sizes=args.history_sizes,
        history_positioning=args.history_positioning,
        pose_convention="opencv_c2w_relative",
        vae_temporal_scale=int(pipe.vae_scale_factor_temporal),
    )
    exact_args.vae_temporal_scale = int(pipe.vae_scale_factor_temporal)
    if args.resume_from_checkpoint is not None:
        manifest_config_hashes = sorted(
            set(str(value) for value in df["candidate_config_hash"].tolist())
        )
        if len(manifest_config_hashes) != 1:
            raise ValueError(f"Resume requires one candidate_config_hash, got {manifest_config_hashes}.")
        exact_args.fixed_teacher_config_hash = manifest_config_hashes[0]
    else:
        exact_args.fixed_teacher_config_hash = candidate_config_hash(args, pipe.transformer._wah_recipe)
    args.fixed_teacher_config_hash = str(exact_args.fixed_teacher_config_hash)
    args.training_resume_contract = training_resume_contract(args)
    mean, std = opt.latent_stats(pipe, device)

    serialized_train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "train_args": serialized_train_args,
        "exact_args": {
            key: value
            for key, value in vars(exact_args).items()
            if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
        },
        "rows": df.to_dict(orient="records"),
        "skipped_rows": skipped_rows,
        "training_data": training_meta,
        "step_sampler": sampler_meta,
        "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
        "tensorboard_log_dir": str(tb_log_dir) if tb_log_dir else "",
        "loss": "flow_matching_train_exact",
        "wah_recipe": dict(pipe.transformer._wah_recipe),
    }
    write_json(out_dir / "train_config.json", config)
    write_json(out_dir / "sampling_audit.json", sampler_meta)
    if tb_writer is not None:
        tb_writer.add_text("config/train_args", _json_text(serialized_train_args), 0)
        tb_writer.add_text("config/exact_args", _json_text(config["exact_args"]), 0)
        tb_writer.add_text("config/training_data", _json_text(training_meta), 0)

    items = LazyPreparedItems(
        pipe,
        df,
        exact_args,
        device,
        mean,
        std,
        args.prompt_cache_dir,
    )
    print(
        json.dumps(
            {
                "event": "prepared_items_ready",
                "num_items": len(items),
                "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
            }
        ),
        flush=True,
    )
    if bool(args.export_teacher_candidates_only):
        candidate_dir = args.teacher_candidate_output_dir or (out_dir / "teacher_candidates")
        manifest_path = export_teacher_candidates(
            items, df, exact_args, candidate_dir, limit=args.teacher_candidate_limit
        )
        print(json.dumps({"event": "teacher_candidates_exported", "manifest": str(manifest_path)}), flush=True)
        if tb_writer is not None:
            tb_writer.close()
        return

    opt.seed_global_rng(args.seed)
    adapter_name, lora_params, lora_stats = opt.setup_visible_lora(pipe.transformer, exact_args, "shared")
    pipe._wah_adapter_name = adapter_name
    peft_configs = dict(getattr(pipe.transformer, "peft_config", {}) or {})
    if len(peft_configs) != 1 or adapter_name not in peft_configs:
        raise RuntimeError(
            f"Training requires exactly one WAH LoRA adapter, got {sorted(peft_configs)}."
        )
    initialization = None
    if args.camera_checkpoint is not None and args.resume_from_checkpoint is None:
        camera_load = opt.load_visible_lora_state(
            pipe.transformer,
            args.camera_checkpoint,
            adapter_name,
            load_interaction=False,
        )
        print(
            json.dumps(
                {
                    "event": "camera_checkpoint_loaded",
                    "checkpoint": str(args.camera_checkpoint),
                    **camera_load,
                },
                default=str,
            ),
            flush=True,
        )
        checkpoint_recipe = camera_load.get("wah_recipe")
        if isinstance(checkpoint_recipe, dict):
            mismatches = recipe_mismatches(checkpoint_recipe, pipe.transformer._wah_recipe)
            if mismatches:
                raise ValueError(f"Camera checkpoint WAH recipe mismatch: {mismatches}")
        initialization = {
            "source": "camera_checkpoint",
            "path": str(args.camera_checkpoint),
            "official_parent": camera_load.get("wah_initialization"),
        }
        pipe.transformer._wah_initialization = dict(initialization)
    elif args.resume_from_checkpoint is None:
        init_path = Path(args.init_wah_lora_path)
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        if bool(args.require_init_wah_lora) or init_path.is_file():
            initialization = opt.load_initial_wah_lora(
                pipe.transformer,
                init_path,
                adapter_name,
                exact_args,
            )
            print(
                json.dumps(
                    {
                        "event": "wah_lora_initialized",
                        **initialization,
                    }
                ),
                flush=True,
            )
        else:
            initialization = {"source": "random", "path": None}
            pipe.transformer._wah_initialization = dict(initialization)
    opt.assert_lora_only_trainable(
        pipe.transformer,
        lora_params,
        allow_target_channel_fusion=str(args.interaction_conditioning_mode) == "binary",
        allow_interaction_conditioning=str(args.interaction_conditioning_mode) == "router",
    )
    print(json.dumps(lora_stats), flush=True)

    named_params = {id(param): name for name, param in pipe.transformer.named_parameters()}
    default_wah_lr = (
        1.0e-5
        if str(args.training_profile) == "camera"
        else 0.0
        if str(args.training_profile) == "interaction"
        else float(args.lr)
    )
    wah_lora_lr = float(args.wah_lora_lr if args.wah_lora_lr is not None else default_wah_lr)
    interaction_lr = float(args.interaction_lr if args.interaction_lr is not None else 1.0e-4)
    router_lr = float(args.router_lr)
    profile = str(args.training_profile)
    interaction_mode = str(getattr(args, "interaction_training_mode", "joint_stage0"))
    wah_params = []
    interaction_params = []
    for param in lora_params:
        name = named_params.get(id(param), "")
        is_interaction = name.startswith("interaction_conditioning.") or ".interaction_conditioning." in name
        should_train = (not is_interaction and profile in {"camera", "joint"}) or (
            is_interaction and profile in {"interaction", "joint"}
        )
        param.requires_grad_(should_train)
        if param.requires_grad:
            if is_interaction:
                interaction_params.append(param)
            else:
                wah_params.append(param)
    if profile in {"interaction", "joint"}:
        interaction_groups = configure_interaction_trainability(
            pipe.transformer.interaction_conditioning, interaction_mode
        )
        interaction_other_params = interaction_groups["interaction_semantic"]
        interaction_router_params = interaction_groups["interaction_router"]
        interaction_adapter_params = interaction_groups["interaction_adapter"]
        interaction_params = [
            *interaction_other_params,
            *interaction_router_params,
            *interaction_adapter_params,
        ]
    else:
        interaction_other_params = []
        interaction_router_params = []
        interaction_adapter_params = []
    if profile == "camera" and interaction_params:
        raise RuntimeError("Camera profile must not train interaction parameters.")
    if profile == "interaction" and wah_params:
        raise RuntimeError("Interaction profile must freeze the WAH LoRA by default.")
    if profile == "interaction" and not interaction_params:
        raise ValueError("interaction profile requires Router/Adapter trainable parameters.")
    trainable_params = [*wah_params, *interaction_params]
    groups = []
    if wah_params:
        groups.append({"params": wah_params, "lr": wah_lora_lr, "base_lr": wah_lora_lr, "name": "wah_lora"})
    if interaction_other_params:
        groups.append(
            {
                "params": interaction_other_params,
                "lr": interaction_lr,
                "base_lr": interaction_lr,
                "name": "interaction",
            }
        )
    if interaction_router_params:
        groups.append(
            {
                "params": interaction_router_params,
                "lr": router_lr,
                "base_lr": router_lr,
                "name": "interaction_router",
            }
        )
    if interaction_adapter_params and any(param.requires_grad for param in interaction_adapter_params):
        groups.append(
            {
                "params": [param for param in interaction_adapter_params if param.requires_grad],
                "lr": interaction_lr,
                "base_lr": interaction_lr,
                "name": "interaction_adapter",
            }
        )
    if not groups:
        raise ValueError(f"No optimizer parameters for training mode {interaction_mode!r}.")
    for group in groups:
        group_names = [named_params.get(id(param), "<unnamed>") for param in group["params"]]
        print(
            json.dumps(
                {
                    "event": "optimizer_group",
                    "name": group["name"],
                    "learning_rate": float(group["lr"]),
                    "parameter_count": int(sum(param.numel() for param in group["params"])),
                    "parameter_tensors": len(group["params"]),
                    "parameter_prefixes": group_names[:12],
                }
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    losses = []
    refined_teacher_cache = {}
    start_step = 0
    if args.resume_from_checkpoint is not None:
        resume_state = load_training_state(
            args.resume_from_checkpoint,
            transformer=pipe.transformer,
            optimizer=optimizer,
            device=device,
            adapter_name=adapter_name,
        )
        start_step = int(resume_state["global_step"])
        if str(resume_state.get("training_profile", args.training_profile)) != str(args.training_profile):
            raise ValueError("Resume checkpoint training_profile does not match the current profile.")
        if str(resume_state.get("training_mode", args.interaction_training_mode)) != str(
            args.interaction_training_mode
        ):
            raise ValueError("Resume checkpoint interaction_training_mode does not match the current mode.")
        if list(resume_state.get("interaction_active_stages", [0])) != [0]:
            raise ValueError("Resume checkpoint must use interaction_active_stages=[0].")
        resume_contract = dict(resume_state.get("resume_contract", {}) or {})
        if not resume_contract:
            raise ValueError(
                "Resume checkpoint is missing the fixed teacher/model/hyperparameter resume contract."
            )
        camera_fingerprint = (
            None
            if args.camera_checkpoint is not None
            else resume_contract.get("camera_checkpoint_fingerprint")
        )
        current_contract = training_resume_contract(
            args,
            camera_fingerprint=camera_fingerprint,
        )
        args.training_resume_contract = dict(current_contract)
        expected_sampling_plan = step_sampler.report(start_step)
        validate_resume_contract(
            resume_contract,
            current_contract,
            resume_state.get("sampling_plan", {}),
            expected_sampling_plan,
        )
        resume_recipe = resume_state.get("wah_recipe")
        if resume_recipe:
            mismatches = recipe_mismatches(resume_recipe, pipe.transformer._wah_recipe)
            if mismatches:
                raise ValueError(f"Resume checkpoint WAH recipe mismatch: {mismatches}")
        pipe.transformer._wah_initialization = dict(
            resume_state.get("wah_initialization", {}) or {}
        )
        resume_schedule = (
            int(resume_state.get("base_train_steps", args.base_train_steps)),
            int(resume_state.get("bidirectional_train_steps", args.bidirectional_train_steps)),
            bool(resume_state.get("enable_bidirectional_training", args.enable_bidirectional_training)),
        )
        configured_schedule = (
            int(args.base_train_steps),
            int(args.bidirectional_train_steps),
            bool(args.enable_bidirectional_training),
        )
        if resume_schedule != configured_schedule:
            raise ValueError(
                f"Resume schedule {resume_schedule} does not match configured schedule {configured_schedule}."
            )
        refined_teacher_cache = dict(resume_state["refined_teacher_cache"])
        losses = list(resume_state["losses"])
        print(
            json.dumps(
                {
                    "event": "training_resumed",
                    "checkpoint": str(args.resume_from_checkpoint),
                    "global_step": start_step,
                    "current_stage": resume_state.get("current_stage", "base"),
                    "base_completed_steps": resume_state.get("base_completed_steps", 0),
                    "bidirectional_completed_steps": resume_state.get("bidirectional_completed_steps", 0),
                }
            ),
            flush=True,
        )
    if start_step > int(args.max_steps):
        raise ValueError(f"Checkpoint step {start_step} exceeds configured total steps {args.max_steps}.")
    start_time = time.perf_counter()
    fallback_counts = distribution_counts_from_losses(losses)
    restored_counters = restore_training_counters(
        resume_state if args.resume_from_checkpoint is not None else {"global_step": start_step},
        fallback_counts=fallback_counts,
    )
    distribution_counts = Counter(restored_counters["distribution_counts"])
    effective_optimizer_step = restored_counters["effective_optimizer_step"]
    attempt_step = restored_counters["attempt_step"]
    skipped_invalid_step = restored_counters["skipped_invalid_step"]
    progress = tqdm(total=args.max_steps, initial=start_step, desc="train shared lora")
    while effective_optimizer_step < int(args.max_steps):
        if attempt_step >= int(args.max_attempt_steps):
            raise RuntimeError(
                f"Reached max_attempt_steps={args.max_attempt_steps} after "
                f"{effective_optimizer_step} effective optimizer steps; "
                f"skipped_invalid_step={skipped_invalid_step}."
            )
        step = int(effective_optimizer_step)
        attempt_step += 1
        sampled_class, item_idx = step_sampler.sample(step)
        if "|" in sampled_class:
            sampled_action, sampled_history = sampled_class.split("|", 1)
            item_idx = step_sampler.sample_category(sampled_class, step + attempt_step - 1)
        else:
            sampled_action, sampled_history = sampled_class, None
        item = None
        step_invalid_event_retries = 0
        max_prepare_retries = 32 if sampled_action in {"place", "mine_active", "mine_complete"} else 8
        last_prepare_error = None
        for retry in range(max_prepare_retries):
            try:
                requested_category = (
                    "movement"
                    if sampled_class.startswith("camera_")
                    else "mine"
                    if sampled_action in {"mine_active", "mine_complete"}
                    else sampled_action
                )
                requested_chunk_mode = (
                    sampled_class
                    if sampled_class.startswith("camera_")
                    else f"interaction_{sampled_history}"
                    if sampled_history is not None
                    else None
                )
                item = items.get(
                    item_idx,
                    requested_category=requested_category,
                    requested_chunk_mode=requested_chunk_mode,
                )
                if (
                    sampled_action in {"place", "mine_active", "mine_complete"}
                    and not bool(item.get("interaction_teacher_valid", False))
                    and retry < 4
                ):
                    distribution_counts["invalid_event_retries"] += 1
                    step_invalid_event_retries += 1
                    print(
                        json.dumps(
                            {
                                "event": "teacher_invalid_retry",
                                "effective_optimizer_step": step,
                                "category": sampled_class,
                                "seq": item.get("seq"),
                                "reasons": item.get("interaction_teacher_invalid_reasons"),
                            }
                        ),
                        flush=True,
                    )
                    item_idx = step_sampler.sample_category(sampled_class, step + attempt_step + retry)
                    continue
                break
            except (RuntimeError, ValueError) as exc:
                if isinstance(exc, FixedTeacherIntegrityError):
                    raise
                last_prepare_error = exc
                if (
                    sampled_action not in {"place", "mine_active", "mine_complete"}
                    and sampled_class != "camera_rollout"
                ):
                    raise
                if sampled_action in {"place", "mine_active", "mine_complete"}:
                    distribution_counts["invalid_event_retries"] += 1
                    step_invalid_event_retries += 1
                item_idx = step_sampler.sample_category(sampled_class, step + retry + 1)
                print(json.dumps({"event": "invalid_interaction_retry", "step": step, "category": sampled_class, "reason": str(exc)}), flush=True)
        if item is None:
            if sampled_action in {"place", "mine_active", "mine_complete"}:
                skipped_invalid_step += 1
                print(
                    json.dumps(
                        {
                            "event": "invalid_interaction_skip",
                            "attempt_step": attempt_step,
                            "effective_optimizer_step": effective_optimizer_step,
                            "sampled_class": sampled_class,
                            "retries": int(max_prepare_retries),
                            "reason": None if last_prepare_error is None else str(last_prepare_error),
                        }
                    ),
                    flush=True,
                )
                release_cuda_cache()
                continue
            raise RuntimeError("Failed to prepare a sampled training item.")
        is_positive_interaction = sampled_action in {"place", "mine_active", "mine_complete"}
        if is_positive_interaction and not bool(item.get("interaction_teacher_valid", False)):
            skipped_invalid_step += 1
            distribution_counts["invalid_event_retries"] += 1
            print(
                json.dumps(
                    {
                        "event": "teacher_invalid_skip",
                        "attempt_step": attempt_step,
                        "effective_optimizer_step": effective_optimizer_step,
                        "sampled_class": sampled_class,
                        "seq": item.get("seq"),
                        "teacher_area_ratio": item.get("interaction_teacher_area_ratio"),
                        "teacher_visibility_ratio": item.get("interaction_teacher_visibility_ratio"),
                        "reasons": item.get("interaction_teacher_invalid_reasons"),
                    }
                ),
                flush=True,
            )
            del item
            release_cuda_cache()
            continue
        category = str(item.get("training_category", item.get("training", {}).get("training_category", "other")))
        if sampled_action in distribution_counts:
            distribution_counts[sampled_action] += 1
        if category == "movement":
            distribution_counts["movement"] += 1
        if category in {"place", "mine"} and float((item.get("interaction_payload") or {}).get("event_valid", 0.0)) == 1.0:
            distribution_counts[f"valid_{category}"] += 1
        chunk_mode = str(item.get("training", {}).get("chunk_mode", ""))
        resolved_history_type = sampled_history or item.get("training", {}).get("fixed_history_type")
        distribution_counts["first" if resolved_history_type == "first" else "later"] += 1
        pose_source = str(item.get("training", {}).get("pose_source", ""))
        if pose_source == "vpt_telemetry":
            distribution_counts["pose_vpt"] += 1
        elif pose_source == "pi3x":
            distribution_counts["pose_pi3x"] += 1
        current_lr = current_train_lr(step, args.max_steps, args, exact_args)
        training_stage = training_stage_for_step(
            step,
            args.base_train_steps,
            args.enable_bidirectional_training,
        )
        compute_bidirectional_feedback = bool(
            args.interaction_conditioning_mode == "router"
            and should_compute_bidirectional_feedback(
                step,
                args.base_train_steps,
                args.enable_bidirectional_training,
                args.bidirectional_interval,
            )
        )
        initial_teacher_map = item.get("initial_teacher_map", item.get("interaction_teacher_map"))
        teacher_cache_key = interaction_teacher_cache_key(item)
        cached_refined_teacher = refined_teacher_cache.get(teacher_cache_key)
        if (
            training_stage == "bidirectional"
            and not compute_bidirectional_feedback
            and cached_refined_teacher is not None
        ):
            router_teacher_map = cached_refined_teacher.to(device=device)
        else:
            router_teacher_map = initial_teacher_map

        lr_scale = current_lr / max(float(args.lr), 1.0e-12)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * lr_scale
        optimizer.zero_grad(set_to_none=True)
        trainable_before = {
            name: param.detach().float().clone()
            for name, param in pipe.transformer.named_parameters()
            if param.requires_grad and name.startswith("interaction_conditioning.")
        }
        loss, stats, interaction_feedback = opt.flow_matching_loss(
            pipe,
            item["prompt_embeds"],
            item["target_latents"],
            item["histories"],
            exact_args,
            device,
            base_histories=item.get("base_histories"),
            loss_focus_mask=item.get("loss_focus_mask_latents"),
            world_valid_mask=item.get("world_valid_mask_latents"),
            target_channel_fusion_latents=item.get("primary_fire_event_latents"),
            interaction_conditioning=item.get("interaction_conditioning"),
            interaction_teacher_map=router_teacher_map,
            interaction_gate_override=(
                router_teacher_map.detach()
                if str(args.interaction_training_mode) == "adapter_overfit" and router_teacher_map is not None
                else None
            ),
            interaction_adapter_enabled=str(args.interaction_training_mode) != "router_overfit",
            compute_bidirectional_feedback=compute_bidirectional_feedback,
            bidirectional_feedback_weight=float(args.bidirectional_feedback_weight),
            bidirectional_teacher_floor=float(args.bidirectional_teacher_floor),
        )
        interaction_feedback = interaction_feedback or {}
        new_refined_teacher = interaction_feedback.get("refined_teacher_map")
        if new_refined_teacher is not None:
            refined_teacher_cache[teacher_cache_key] = new_refined_teacher.detach().cpu()
        active_refined_teacher = (
            new_refined_teacher
            if new_refined_teacher is not None
            else cached_refined_teacher.to(device=device)
            if cached_refined_teacher is not None and training_stage == "bidirectional"
            else None
        )
        if (
            item.get("interaction_debug_inputs") is not None
            and int(args.interaction_debug_every) > 0
            and (step == 0 or (step + 1) % int(args.interaction_debug_every) == 0)
        ):
            opt.save_interaction_debug(
                Path(args.output_dir) / "interaction_debug" / f"step_{step + 1:06d}",
                initial_teacher_map,
                getattr(pipe.transformer, "_last_interaction_debug", None),
                improvement_map=interaction_feedback.get("improvement_map"),
                refined_teacher_map=active_refined_teacher,
                input_debug=item.get("interaction_debug_inputs"),
            )
        loss.backward()
        router_grad_sq = 0.0
        adapter_grad_sq = 0.0
        for name, param in pipe.transformer.named_parameters():
            if param.grad is None or not name.startswith("interaction_conditioning."):
                continue
            value = float(param.grad.detach().float().square().sum().item())
            if ".router." in name or ".semantic_encoder." in name:
                router_grad_sq += value
            elif ".adapter." in name:
                adapter_grad_sq += value
        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.max_grad_norm))
        optimizer.step()
        router_update_sq = 0.0
        adapter_update_sq = 0.0
        for name, before in trainable_before.items():
            param = dict(pipe.transformer.named_parameters())[name]
            value = float((param.detach().float() - before).square().sum().item())
            if ".router." in name or ".semantic_encoder." in name:
                router_update_sq += value
            elif ".adapter." in name:
                adapter_update_sq += value
        stats["interaction_router_grad_norm"] = math.sqrt(router_grad_sq)
        stats["interaction_adapter_grad_norm"] = math.sqrt(adapter_grad_sq)
        stats["interaction_router_update_norm"] = math.sqrt(router_update_sq)
        stats["interaction_adapter_update_norm"] = math.sqrt(adapter_update_sq)
        pipe.transformer.set_adapter(adapter_name)

        record = {
            "step": int(step + 1),
            "effective_optimizer_step": int(step + 1),
            "attempt_step": int(attempt_step),
            "skipped_invalid_step": int(skipped_invalid_step),
            "training_stage": training_stage,
            "interaction_training_mode": str(args.interaction_training_mode),
            "base_completed_steps": min(int(step + 1), int(args.base_train_steps)),
            "bidirectional_completed_steps": max(int(step + 1) - int(args.base_train_steps), 0),
            "bidirectional_feedback_computed": bool(compute_bidirectional_feedback),
            "seq": item["seq"],
            "loss": float(loss.detach().cpu()),
            "lr": current_lr,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "lora_target_modules": opt.lora_target_modules(exact_args),
            "optimizer": "adamw",
            "adamw_weight_decay": 0.01,
            "warmup_steps": int(args.warmup_steps),
            "max_grad_norm": float(args.max_grad_norm),
            "grad_norm": scalar(grad_norm) if grad_norm is not None else None,
            "elapsed_s": time.perf_counter() - start_time,
            "sampled_category": sampled_action,
            "sampled_joint_class": sampled_class,
            "sampled_history_type": sampled_history,
            "training_category": category,
            "event_local_frame": item.get("training", {}).get("event_local_frame"),
            "source_fps": item.get("training", {}).get("source_fps"),
            "target_fps": item.get("training", {}).get("target_fps"),
            "source_event_frame": item.get("training", {}).get("source_event_frame"),
            "source_event_time_ms": item.get("training", {}).get("source_event_time_ms"),
            "resampled_event_frame": item.get("training", {}).get("resampled_event_frame"),
            "target_start_frame": item.get("training", {}).get("target_start_frame"),
            "event_valid": float((item.get("interaction_payload") or {}).get("event_valid", 0.0)),
            "action_type": str((item.get("interaction_payload") or {}).get("action_type", "none")),
            "block_id": (item.get("interaction_payload") or {}).get("block_id"),
            "progress_min": min(
                (item.get("interaction_payload") or {}).get("frame_progress_curve", [0.0]) or [0.0]
            ),
            "progress_max": max(
                (item.get("interaction_payload") or {}).get("frame_progress_curve", [0.0]) or [0.0]
            ),
            "source_action_start_frame": (item.get("interaction_payload") or {}).get(
                "source_action_start_frame"
            ),
            "source_complete_frame": (item.get("interaction_payload") or {}).get(
                "source_complete_frame"
            ),
            "teacher_valid": bool(item.get("interaction_teacher_valid", False)),
            "teacher_area_ratio": item.get("interaction_teacher_area_ratio"),
            "teacher_visibility_ratio": item.get("interaction_teacher_visibility_ratio"),
            "chunk_mode": chunk_mode,
            "chunk_index": item.get("training", {}).get("chunk_index"),
            "history_corruption": item.get("training", {}).get("history_corruption", "clean"),
            "pose_source": pose_source,
            "invalid_event_retries": step_invalid_event_retries,
            "wah_lora_lr": next((group["lr"] for group in optimizer.param_groups if group["name"] == "wah_lora"), 0.0),
            "interaction_lr": next((group["lr"] for group in optimizer.param_groups if group["name"] == "interaction"), 0.0),
        }
        for key, value in stats.items():
            record[key] = scalar(value)
        losses.append(record)
        tensorboard_log_record(tb_writer, record, step)

        do_log = args.log_every > 0 and ((step + 1) % args.log_every == 0 or step == 0)
        do_save = (step + 1) in set(int(value) for value in args.save_steps) or (
            args.save_every > 0 and (step + 1) % args.save_every == 0
        )
        if do_log:
            if (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "minecraft_sampling_progress",
                            "step": step + 1,
                            **step_sampler.report(step + 1),
                            "valid_positive_rate": {
                                "place": distribution_counts["valid_place"]
                                / max(distribution_counts["place"], 1),
                                "mine": distribution_counts["valid_mine"]
                                / max(distribution_counts["mine"], 1),
                            },
                            "counters": distribution_counts,
                        }
                    ),
                    flush=True,
                )
            print(json.dumps(record), flush=True)
            write_json(loss_path, losses)
            if tb_writer is not None:
                tb_writer.flush()
        if do_save:
            save_lora(pipe, out_dir, adapter_name, f"visible_lora_state_step{step + 1:04d}.pt")
            save_training_state(
                out_dir / f"training_state_step{step + 1:04d}.pt",
                transformer=pipe.transformer,
                optimizer=optimizer,
                global_step=step + 1,
                args=args,
                refined_teacher_cache=refined_teacher_cache,
                losses=losses,
                distribution_counts=distribution_counts,
                attempt_step=attempt_step,
                skipped_invalid_step=skipped_invalid_step,
                adapter_name=adapter_name,
                sampling_plan=step_sampler.report(step + 1),
            )
            write_interaction_debug_summary(out_dir, step + 1, losses, window=150)

        del loss, stats, item, interaction_feedback
        if grad_norm is not None:
            del grad_norm
        release_cuda_cache()
        effective_optimizer_step += 1
        progress.update(1)

    progress.close()
    save_lora(pipe, out_dir, adapter_name, "visible_lora_state.pt")
    save_training_state(
        out_dir / "training_state.pt",
        transformer=pipe.transformer,
        optimizer=optimizer,
        global_step=args.max_steps,
        args=args,
        refined_teacher_cache=refined_teacher_cache,
        losses=losses,
        distribution_counts=distribution_counts,
        attempt_step=attempt_step,
        skipped_invalid_step=skipped_invalid_step,
        adapter_name=adapter_name,
        sampling_plan=step_sampler.report(effective_optimizer_step),
    )
    write_json(loss_path, losses)
    distribution_report = {
        "training_profile": str(args.training_profile),
        "interaction_training_mode": str(args.interaction_training_mode),
        "requested": step_sampler.report(args.max_steps),
        "counters": distribution_counts,
        "valid_positive_rate": {
            "place": distribution_counts["valid_place"] / max(distribution_counts["place"], 1),
            "mine": distribution_counts["valid_mine"] / max(distribution_counts["mine"], 1),
        },
    }
    chunk_index_histogram = {}
    history_corruption_counts = {}
    event_local_frame_histogram = {}
    for record in losses:
        chunk_index = record.get("chunk_index")
        if chunk_index is not None:
            key = str(int(chunk_index))
            chunk_index_histogram[key] = chunk_index_histogram.get(key, 0) + 1
        corruption = str(record.get("history_corruption", "clean"))
        history_corruption_counts[corruption] = history_corruption_counts.get(corruption, 0) + 1
        event_local = record.get("event_local_frame")
        if event_local is not None:
            key = str(int(event_local))
            event_local_frame_histogram[key] = event_local_frame_histogram.get(key, 0) + 1
    distribution_report.update(
        {
            "first_chunk_steps": distribution_counts["camera_first"],
            "later_chunk_steps": distribution_counts["camera_later"],
            "two_chunk_rollout_steps": distribution_counts["camera_rollout"],
            "chunk_index_histogram": chunk_index_histogram,
            "history_corruption_counts": history_corruption_counts,
            "pose_source_vpt_steps": distribution_counts["pose_vpt"],
            "pose_source_pi3x_steps": distribution_counts["pose_pi3x"],
            "sampled_place_steps": distribution_counts["place"],
            "valid_place_steps": distribution_counts["valid_place"],
            "sampled_mine_steps": distribution_counts["mine"],
            "valid_mine_steps": distribution_counts["valid_mine"],
            "sampled_negative_steps": distribution_counts["negative"],
            "invalid_event_retries": distribution_counts["invalid_event_retries"],
            "event_local_frame_histogram": event_local_frame_histogram,
        }
    )
    write_json(out_dir / "training_distribution_report.json", distribution_report)
    if tb_writer is not None:
        tb_writer.add_text("summary/prompt_cache_status", _json_text(items.prompt_cache_status_counts), args.max_steps)
        tb_writer.flush()
        tb_writer.close()
    print(
        json.dumps(
            {
                "event": "prompt_cache_summary",
                "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
                "statuses": items.prompt_cache_status_counts,
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "event": "train_done",
                "output_dir": str(out_dir),
                "steps": int(args.max_steps),
                "num_items": len(items),
                "time_total_s": time.perf_counter() - start_time,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
