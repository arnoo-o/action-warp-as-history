#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from warp_as_history.training import core as opt
from warp_as_history.minecraft_recipe import minecraft_wah_recipe, recipe_mismatches
from warp_as_history.training.data import (
    LazyPreparedItems,
    build_online_warp_training_cache,
    normalize_online_training_dataframe,
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

    def __init__(self, source_pools, total_steps, seed, *, phases):
        total_steps = int(total_steps)
        phase_plan = [dict(item) for item in phases]
        planned_steps = sum(int(item["steps"]) for item in phase_plan)
        if total_steps != planned_steps:
            raise ValueError(f"Interaction curriculum expected {planned_steps} effective steps, got {total_steps}.")
        self.samplers = []
        self.source_pools = {key: list(value) for key, value in source_pools.items()}
        self.seed = int(seed)
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
            for action, action_ratio in self.ACTION_RATIOS.items():
                if not source_pools.get(action):
                    raise ValueError(f"Required interaction pool is empty: {action}.")
                for history, history_ratio in history_ratios.items():
                    key = f"{action}|{history}"
                    pools[key] = list(source_pools[action])
                    ratios[key] = float(action_ratio) * float(history_ratio)
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
        action = str(category).split("|", 1)[0]
        pool = self.source_pools[action]
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
        return 300
    if mode in {"router_overfit", "adapter_overfit"}:
        return 200
    return 1500


def interaction_training_mode_phase_plan(mode):
    mode = str(mode)
    if mode == "joint_pilot":
        return InteractionJointSampler.DEFAULT_PILOT_PHASES
    if mode == "joint_stage0":
        return InteractionJointSampler.DEFAULT_STAGE0_PHASES
    return (
        {"steps": interaction_training_mode_default_steps(mode), "history": {"first": 0.50, "later": 0.50}},
    )


def training_total_steps(base_train_steps, bidirectional_train_steps, enable_bidirectional_training):
    base = max(int(base_train_steps), 0)
    bidirectional = max(int(bidirectional_train_steps), 0) if bool(enable_bidirectional_training) else 0
    return base + bidirectional


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
        if mode in {"joint_pilot", "joint_stage0"}:
            sampler = InteractionJointSampler(
                source_pools,
                args.max_steps,
                args.seed,
                phases=interaction_training_mode_phase_plan(mode),
            )
        else:
            base_action = "place" if mode == "adapter_overfit" else "mine_active"
            pools = {base_action: list(source_pools[base_action]), "negative": list(source_pools["negative"])}
            ratios = {base_action: 0.80, "negative": 0.20}
            sampler = StepCategorySampler(pools, ratios, args.max_steps, args.seed)
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
):
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }
    completed_step = max(int(global_step) - 1, 0)
    payload = {
        "training_state_version": 3,
        "trainable_state": trainable_state,
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


def load_training_state(path, *, transformer, optimizer, device):
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
        "wah_recipe": dict(payload.get("wah_recipe", {}) or {}),
        "wah_initialization": dict(payload.get("wah_initialization", {}) or {}),
        "attempt_step": int(payload.get("attempt_step", 0)),
        "skipped_invalid_step": int(payload.get("skipped_invalid_step", 0)),
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
    parser.add_argument("--interaction_router_loss_scale", type=float, default=0.05)
    parser.add_argument("--interaction_focus_scale", type=float, default=1.0)
    parser.add_argument("--interaction_teacher_support_threshold", type=float, default=0.25)
    parser.add_argument("--interaction_max_metadata_rotation_deg", type=float, default=20.0)
    parser.add_argument("--interaction_max_camera_rotation_deg", type=float, default=20.0)
    parser.add_argument("--interaction_min_telemetry_confidence", type=float, default=0.0)
    parser.add_argument("--interaction_min_mine_active_frames", type=int, default=4)
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
    parser.add_argument("--interaction_debug_every", type=int, default=100)
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
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_steps", type=int, nargs="*", default=[300, 500, 750, 1000, 1500])
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
    if str(args.training_profile) == "interaction" and args.camera_checkpoint is None:
        raise ValueError("--training_profile interaction requires --camera_checkpoint.")
    if args.camera_checkpoint is not None and args.resume_from_checkpoint is not None:
        raise ValueError("--camera_checkpoint initializes a new stage and cannot be combined with --resume_from_checkpoint.")
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
    exact_args.online_warp_cache = build_online_warp_training_cache(df, exact_args, device)
    step_sampler, sampler_meta = build_minecraft_step_sampler(df, args)
    skipped_rows = []
    print(
        json.dumps(
            {
                "event": "online_warp_cache_config",
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

    opt.seed_global_rng(args.seed)
    adapter_name, lora_params, lora_stats = opt.setup_visible_lora(pipe.transformer, exact_args, "shared")
    pipe._wah_adapter_name = adapter_name
    peft_configs = dict(getattr(pipe.transformer, "peft_config", {}) or {})
    if len(peft_configs) != 1 or adapter_name not in peft_configs:
        raise RuntimeError(
            f"Training requires exactly one WAH LoRA adapter, got {sorted(peft_configs)}."
        )
    initialization = None
    if args.camera_checkpoint is not None:
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
    profile = str(args.training_profile)
    interaction_mode = str(getattr(args, "interaction_training_mode", "joint_stage0"))
    wah_params = []
    interaction_params = []
    interaction_router_params = []
    interaction_adapter_params = []
    interaction_other_params = []
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
                if ".router." in name:
                    interaction_router_params.append(param)
                elif ".adapter." in name:
                    interaction_adapter_params.append(param)
                else:
                    interaction_other_params.append(param)
            else:
                wah_params.append(param)
    if profile == "interaction":
        if interaction_mode == "router_overfit":
            for param in interaction_adapter_params:
                param.requires_grad_(False)
            interaction_params = [param for param in interaction_params if param.requires_grad]
        elif interaction_mode == "adapter_overfit":
            for param in interaction_router_params:
                param.requires_grad_(False)
            interaction_params = [param for param in interaction_params if param.requires_grad]
            interaction_router_params = []
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
                "lr": 5.0e-5,
                "base_lr": 5.0e-5,
                "name": "interaction_router",
            }
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
        )
        start_step = int(resume_state["global_step"])
        if str(resume_state.get("training_profile", args.training_profile)) != str(args.training_profile):
            raise ValueError("Resume checkpoint training_profile does not match the current profile.")
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
    distribution_counts = distribution_counts_from_losses(losses)
    effective_optimizer_step = int(start_step)
    attempt_step = int(resume_state.get("attempt_step", 0)) if args.resume_from_checkpoint is not None else 0
    skipped_invalid_step = (
        int(resume_state.get("skipped_invalid_step", 0)) if args.resume_from_checkpoint is not None else 0
    )
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
        for retry in range(8):
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
                if (
                    sampled_action not in {"place", "mine_active", "mine_complete"}
                    and sampled_class != "camera_rollout"
                ) or retry == 7:
                    raise
                if sampled_action in {"place", "mine_active", "mine_complete"}:
                    distribution_counts["invalid_event_retries"] += 1
                    step_invalid_event_retries += 1
                item_idx = step_sampler.sample_category(sampled_class, step + retry + 1)
                print(json.dumps({"event": "invalid_interaction_retry", "step": step, "category": sampled_class, "reason": str(exc)}), flush=True)
        if item is None:
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
        distribution_counts["first" if chunk_mode == "first" else "later"] += 1
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
        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.max_grad_norm))
        optimizer.step()
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
