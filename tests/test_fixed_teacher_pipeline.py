import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "run_h100_mc_interaction_retrain.sh"
TRAIN_PATH = ROOT / "scripts" / "train_warp_as_history_lora.py"
DATA_PATH = ROOT / "warp_as_history" / "training" / "data.py"

from warp_as_history.training.fixed_teacher import (
    ACTION_HISTORY_KEYS,
    FixedTeacherIntegrityError,
    build_action_history_pools,
    encode_index_sequence,
    restore_training_counters,
    validate_fixed_identity,
    validate_required_pools,
    validate_resume_contract,
    validate_stage0_positive_tokens,
)


def load_teacher_builder():
    path = ROOT / "scripts" / "build_minecraft_teacher_pool.py"
    spec = importlib.util.spec_from_file_location("teacher_pool_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_teacher_builder()


def identity_row(action, history, index):
    start = index * 40
    target = list(range(start, start + 33))
    history_indices = [] if history == "first" else list(range(max(0, start - 4), start))
    event_id = f"event-{action}-{history}-{index}"
    candidate_key = f"cache-{action}-{history}-{index}"
    return {
        "event_id": event_id,
        "action_type": "none" if action == "negative" else action,
        "category": "negative" if action == "negative" else "mine" if action.startswith("mine") else "place",
        "history_type": history,
        "review_status": "approved",
        "target_indices": encode_index_sequence(target),
        "history_indices": encode_index_sequence(history_indices),
        "geometry_keyframe_frames": encode_index_sequence([start] if history == "first" else [start - 1]),
        "render_pose_indices": encode_index_sequence(target if history == "first" else [start - 1, *target]),
        "target_start_frame": str(start),
        "event_local_frame": "7",
        "chunk_mode": f"interaction_{history}",
        "direction": "forward",
        "source_segment_id": f"segment-{index}",
        "candidate_cache_key": candidate_key,
        "candidate_config_hash": "config-hash",
    }


def cached_identity(row):
    return {
        "event_id": row["event_id"],
        "history_type": row["history_type"],
        "target_indices": json.loads(row["target_indices"]),
        "history_indices": json.loads(row["history_indices"]),
        "geometry_keyframe_frames": json.loads(row["geometry_keyframe_frames"]),
        "render_pose_indices": json.loads(row["render_pose_indices"]),
        "target_start_frame": int(row["target_start_frame"]),
        "event_local_frame": int(row["event_local_frame"]),
        "chunk_mode": row["chunk_mode"],
        "direction": row["direction"],
        "source_segment_id": row["source_segment_id"],
        "candidate_cache_key": row["candidate_cache_key"],
        "candidate_config_hash": row["candidate_config_hash"],
    }


class FixedTeacherPoolTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            identity_row(action, history, index)
            for index, (action, history) in enumerate(
                (key.split("|") for key in ACTION_HISTORY_KEYS)
            )
        ]
        self.pools = build_action_history_pools(self.rows)

    def test_place_first_only_contains_first(self):
        self.assertTrue(self.pools["place|first"])
        self.assertTrue(all(self.rows[index]["history_type"] == "first" for index in self.pools["place|first"]))

    def test_place_later_only_contains_later(self):
        self.assertTrue(self.pools["place|later"])
        self.assertTrue(all(self.rows[index]["history_type"] == "later" for index in self.pools["place|later"]))

    def test_all_action_history_pools_are_disjoint(self):
        flattened = [index for values in self.pools.values() for index in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        for key, values in self.pools.items():
            action, history = key.split("|")
            for index in values:
                actual_action = "negative" if self.rows[index]["action_type"] == "none" else self.rows[index]["action_type"]
                self.assertEqual((actual_action, self.rows[index]["history_type"]), (action, history))

    def test_required_empty_combination_fails_clearly(self):
        broken = dict(self.pools)
        broken["place|later"] = []
        with self.assertRaisesRegex(ValueError, r"place\|later"):
            validate_required_pools(
                broken,
                [{"steps": 10, "history": {"first": 0.5, "later": 0.5}}],
                {"place": 1.0},
            )

    def test_candidate_and_training_indices_match_exactly(self):
        row = self.rows[0]
        result = validate_fixed_identity(row, cached_identity(row), "config-hash")
        self.assertEqual(result["target_indices"], json.loads(row["target_indices"]))
        self.assertEqual(result["history_indices"], json.loads(row["history_indices"]))

    def test_prepare_index_cannot_change_approved_window(self):
        row = self.rows[1]
        first = validate_fixed_identity(row, cached_identity(row), "config-hash")
        second = validate_fixed_identity(row, cached_identity(row), "config-hash")
        self.assertEqual(first, second)
        source = DATA_PATH.read_text(encoding="utf-8")
        approved_branch = source.index('if str(self.rows[idx].get("review_status", ""))')
        counter_increment = source.index("self.prepare_counter += 1", approved_branch)
        fixed_return = source.index("return item", approved_branch)
        self.assertLess(fixed_return, counter_increment)

    def test_cache_key_and_config_hash_are_bound(self):
        row = self.rows[2]
        validate_fixed_identity(row, cached_identity(row), "config-hash")
        changed = cached_identity(row)
        changed["candidate_cache_key"] = "different-cache"
        with self.assertRaisesRegex(FixedTeacherIntegrityError, "candidate_cache_key"):
            validate_fixed_identity(row, changed, "config-hash")

    def test_index_or_config_mismatch_reports_event_and_history(self):
        row = self.rows[3]
        changed = cached_identity(row)
        changed["target_indices"] = changed["target_indices"][1:]
        with self.assertRaisesRegex(FixedTeacherIntegrityError, row["event_id"]):
            validate_fixed_identity(row, changed, "wrong-runtime-hash")

    def test_fixed_teacher_area_uses_action_valid_denominator(self):
        teacher = np.zeros((1, 1, 2, 2), dtype=np.float32)
        teacher[0, 0, 0, 0] = 1.0
        action = np.zeros_like(teacher)
        action[0, 0, 0, :] = 1.0
        stats = BUILDER.fixed_teacher_statistics(
            teacher,
            action,
            np.ones_like(teacher),
            np.ones_like(teacher),
            0.25,
            (1, 2, 2),
            action_type="place",
        )
        self.assertAlmostEqual(stats["teacher_area_ratio"], 0.5)

    def test_negative_allows_zero_stage0_tokens(self):
        self.assertEqual(validate_stage0_positive_tokens("none", 0), 0)
        stats = BUILDER.fixed_teacher_statistics(
            np.zeros((1, 1, 2, 2), dtype=np.float32),
            np.zeros((1, 1, 2, 2), dtype=np.float32),
            np.ones((1, 1, 2, 2), dtype=np.float32),
            np.ones((1, 1, 2, 2), dtype=np.float32),
            0.25,
            (1, 2, 2),
            action_type="none",
        )
        self.assertEqual(stats["stage0_positive_tokens"], 0)

    def test_positive_requires_stage0_tokens(self):
        with self.assertRaises(FixedTeacherIntegrityError):
            validate_stage0_positive_tokens("place", 0, event_id="event-x", history_type="first")


class NegativeCandidateReviewTest(unittest.TestCase):
    def test_negative_candidate_enters_pending_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            row = identity_row("negative", "first", 0)
            row.update(
                {
                    "window_frames": "33",
                    "no_interaction_event_verified": "true",
                    "gui_closed_verified": "true",
                    "frame_contiguous_verified": "true",
                    "source_mapping_valid": "true",
                    "metadata_filter_status": "passed",
                    "candidate_error": "",
                    "block_id": "",
                }
            )
            candidate = temp / "candidate.npz"
            row["teacher_candidate_path"] = str(candidate)
            latent_shape = (2, 3, 4, 4)
            identity = cached_identity(row)
            np.savez_compressed(
                candidate,
                target_latents=np.zeros(latent_shape, dtype=np.float16),
                warp_latents=np.zeros(latent_shape, dtype=np.float16),
                action_mask=np.zeros((1, 3, 4, 4), dtype=np.float16),
                visibility=np.ones((1, 3, 4, 4), dtype=np.float16),
                world_valid=np.ones((1, 3, 4, 4), dtype=np.float16),
                target_rgb=np.zeros((33, 8, 8, 3), dtype=np.uint8),
                warp_rgb=np.zeros((33, 8, 8, 3), dtype=np.uint8),
                candidate_identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
            )
            manifest = temp / "candidates.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(row))
                writer.writeheader()
                writer.writerow(row)
            output = temp / "teacher_pool"
            old_argv = sys.argv
            try:
                sys.argv = ["build_minecraft_teacher_pool.py", "--candidate_manifest", str(manifest), "--output_dir", str(output)]
                BUILDER.main()
            finally:
                sys.argv = old_argv
            with (output / "teacher_pool_review_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                reviewed = next(csv.DictReader(handle))
            self.assertEqual(reviewed["review_status"], "pending")
            self.assertEqual(reviewed["teacher_valid"], "true")
            self.assertEqual(reviewed["stage0_positive_tokens"], "0")


class LauncherAndResumeTest(unittest.TestCase):
    def test_launcher_mode_defaults_and_no_default_overwrite(self):
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        for mode, steps in (("router_overfit", 200), ("adapter_overfit", 300), ("joint_pilot", 200), ("joint_stage0", 1500)):
            self.assertIn(f"{mode}) DEFAULT_STEPS={steps}", source)
        self.assertIn('STEPS="${WAH_STEPS:-${DEFAULT_STEPS}}"', source)
        self.assertIn('--save_every 150', source)
        self.assertNotIn("--save_steps", source)
        self.assertEqual(source.count("--overwrite"), 1)
        self.assertIn('case "${WAH_OVERWRITE:-0}"', source)

    def test_resume_restores_plan_pool_hash_and_counters(self):
        contract = {
            "teacher_pool_manifest_hash": "manifest",
            "approved_teacher_row_ids": ["a", "b"],
            "action_history_pool_sizes": {"place|first": 2},
            "phase_plan": [{"steps": 10, "history": {"first": 1.0}}],
        }
        sampling = {"completed_effective_steps": 7}
        validate_resume_contract(contract, dict(contract), sampling, dict(sampling))
        restored = restore_training_counters(
            {
                "global_step": 7,
                "effective_optimizer_step": 7,
                "attempt_step": 11,
                "skipped_invalid_step": 4,
                "distribution_counts": {"place": 5, "negative": 2},
            },
            fallback_counts={"place": 999},
        )
        self.assertEqual(restored["distribution_counts"], {"place": 5, "negative": 2})
        self.assertEqual((restored["effective_optimizer_step"], restored["attempt_step"], restored["skipped_invalid_step"]), (7, 11, 4))
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            changed = dict(contract)
            changed["teacher_pool_manifest_hash"] = "changed"
            validate_resume_contract(contract, changed, sampling, sampling)

    def test_checkpoint_source_contains_required_resume_fields(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        for field in (
            '"distribution_counts"',
            '"sampling_plan"',
            '"phase_plan"',
            '"approved_teacher_row_ids"',
            '"teacher_pool_manifest_hash"',
            '"action_history_pool_sizes"',
            '"attempt_step"',
            '"skipped_invalid_step"',
            '"effective_optimizer_step"',
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
