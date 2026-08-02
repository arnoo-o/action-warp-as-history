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
PREPARATION_LAUNCHER_PATH = ROOT / "scripts" / "run_h100_mc_teacher_preparation.sh"
TRAIN_PATH = ROOT / "scripts" / "train_warp_as_history_lora.py"
DATA_PATH = ROOT / "warp_as_history" / "training" / "data.py"

from warp_as_history.training.fixed_teacher import (
    ACTION_HISTORY_KEYS,
    FixedTeacherIntegrityError,
    build_action_history_pools,
    encode_index_sequence,
    file_sha256,
    interaction_action_ratios,
    interaction_payload_hash,
    restore_training_counters,
    stratified_candidate_indices,
    validate_fixed_identity,
    validate_fixed_artifact_hashes,
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


def load_manifest_builder():
    path = ROOT / "scripts" / "build_minecraft_interaction_manifest.py"
    spec = importlib.util.spec_from_file_location("interaction_manifest_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST_BUILDER = load_manifest_builder()


def identity_row(action, history, index):
    start = index * 40
    target = list(range(start, start + 33))
    history_indices = [] if history == "first" else list(range(max(0, start - 4), start))
    event_id = f"event-{action}-{history}-{index}"
    candidate_key = f"cache-{action}-{history}-{index}"
    action_type = "none" if action == "negative" else action
    category = "negative" if action == "negative" else "mine" if action.startswith("mine") else "place"
    return {
        "event_id": event_id,
        "action_type": action_type,
        "block_id": "" if action == "negative" else "oak_planks",
        "object_id": "" if action == "negative" else "oak_planks",
        "category": category,
        "training_category": category,
        "interaction_payload_hash": interaction_payload_hash({}),
        "source_video_digest": f"source-digest-{index}",
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
        "overfit_selected": "true",
    }


def cached_identity(row):
    return {
        "event_id": row["event_id"],
        "action_type": row["action_type"],
        "block_id": row["block_id"],
        "object_id": row["object_id"],
        "training_category": row["training_category"],
        "interaction_payload_hash": row["interaction_payload_hash"],
        "source_video_digest": row["source_video_digest"],
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


def write_candidate_files(temp, row, *, rgb_to_latent=None):
    candidate = temp / f"{row['candidate_cache_key']}.npz"
    training = temp / f"{row['candidate_cache_key']}.pt"
    training.write_bytes(b"fixed training cache")
    latent_shape = (2, 3, 4, 4)
    mapping = rgb_to_latent or ([0] * 11 + [1] * 11 + [2] * 11)
    np.savez_compressed(
        candidate,
        target_latents=np.zeros(latent_shape, dtype=np.float16),
        warp_latents=np.zeros(latent_shape, dtype=np.float16),
        action_mask=np.zeros((1, 3, 4, 4), dtype=np.float16),
        visibility=np.ones((1, 3, 4, 4), dtype=np.float16),
        world_valid=np.ones((1, 3, 4, 4), dtype=np.float16),
        target_rgb=np.zeros((33, 8, 8, 3), dtype=np.uint8),
        warp_rgb=np.zeros((33, 8, 8, 3), dtype=np.uint8),
        interaction_payload_json=np.asarray(json.dumps({})),
        rgb_frame_to_latent_index=np.asarray(mapping, dtype=np.int16),
        candidate_identity_json=np.asarray(json.dumps(cached_identity(row), sort_keys=True)),
    )
    row["teacher_candidate_path"] = str(candidate)
    row["training_cache_path"] = str(training)
    row["candidate_npz_sha256"] = file_sha256(candidate)
    row["training_cache_sha256"] = file_sha256(training)
    return candidate, training


def run_teacher_builder(temp, rows):
    manifest = temp / "candidates.csv"
    columns = sorted({key for row in rows for key in row})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    output = temp / "teacher_pool"
    old_argv = sys.argv
    try:
        sys.argv = [
            "build_minecraft_teacher_pool.py",
            "--candidate_manifest",
            str(manifest),
            "--output_dir",
            str(output),
        ]
        BUILDER.main()
    finally:
        sys.argv = old_argv
    with (output / "teacher_pool_review_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reviewed = list(csv.DictReader(handle))
    return output, reviewed


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

    def test_action_block_and_payload_identity_mismatch_fails(self):
        row = self.rows[0]
        for field, replacement in (
            ("action_type", "mine_complete"),
            ("block_id", "stone"),
            ("interaction_payload_hash", "wrong-payload"),
        ):
            changed = cached_identity(row)
            changed[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                FixedTeacherIntegrityError, field
            ):
                validate_fixed_identity(row, changed, "config-hash")

    def test_overfit_pools_only_include_manually_selected_rows(self):
        rows = [dict(self.rows[0]), dict(self.rows[1])]
        rows[0]["overfit_selected"] = "false"
        pools = build_action_history_pools(rows, require_overfit_selected=True)
        self.assertEqual(pools["place|first"], [])
        self.assertEqual(pools["place|later"], [1])

    def test_adapter_overfit_ratios_exclude_negative(self):
        ratios = interaction_action_ratios("adapter_overfit")
        self.assertEqual(ratios["negative"], 0.0)
        self.assertAlmostEqual(sum(ratios.values()), 1.0)
        self.assertTrue(all(ratios[action] > 0 for action in ("place", "mine_active", "mine_complete")))

    def test_candidate_limit_is_stratified_across_all_actions(self):
        rows = []
        for action, count in (
            ("place", 1000),
            ("mine_active", 500),
            ("mine_complete", 500),
            ("none", 300),
        ):
            rows.extend({"action_type": action} for _ in range(count))
        indices = stratified_candidate_indices(rows, 500)
        selected = [rows[index]["action_type"] for index in indices]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(selected.count("place"), 250)
        self.assertEqual(selected.count("mine_active"), 75)
        self.assertEqual(selected.count("mine_complete"), 75)
        self.assertEqual(selected.count("none"), 100)

    def test_artifact_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            row = dict(self.rows[0])
            candidate, training = write_candidate_files(temp, row)
            validate_fixed_artifact_hashes(
                row,
                candidate_path=candidate,
                training_cache_path=training,
            )
            row["training_cache_sha256"] = "0" * 64
            with self.assertRaisesRegex(FixedTeacherIntegrityError, "training_cache_sha256"):
                validate_fixed_artifact_hashes(
                    row,
                    candidate_path=candidate,
                    training_cache_path=training,
                )

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

    def test_global_edge_teacher_rejects_distributed_edge_fragments(self):
        teacher = np.zeros((3, 64, 64), dtype=np.float32)
        for y in range(3, 64, 8):
            for x in range(1, 58, 8):
                teacher[:, y : y + 2, x : x + 4] = 1.0
        statistics = BUILDER.global_edge_teacher_statistics(
            teacher,
            teacher.copy(),
            np.ones_like(teacher),
            0.25,
        )
        recipe = {
            "min_overlap": 0.30,
            "max_largest_component_ratio": 0.45,
            "min_components": 24,
            "min_grid_coverage": 0.50,
        }
        self.assertTrue(BUILDER.is_global_edge_teacher(statistics, recipe))
        self.assertGreaterEqual(statistics["teacher_significant_component_count"], 24)
        self.assertGreaterEqual(statistics["teacher_grid_coverage_ratio"], 0.50)

    def test_global_edge_teacher_keeps_compact_interaction_region(self):
        teacher = np.zeros((3, 64, 64), dtype=np.float32)
        teacher[:, 20:44, 20:44] = 1.0
        old_edge = np.zeros_like(teacher)
        old_edge[:, 20, 20:44] = 1.0
        old_edge[:, 43, 20:44] = 1.0
        old_edge[:, 20:44, 20] = 1.0
        old_edge[:, 20:44, 43] = 1.0
        statistics = BUILDER.global_edge_teacher_statistics(
            teacher,
            old_edge,
            np.ones_like(teacher),
            0.25,
        )
        recipe = {
            "min_overlap": 0.30,
            "max_largest_component_ratio": 0.45,
            "min_components": 24,
            "min_grid_coverage": 0.50,
        }
        self.assertFalse(BUILDER.is_global_edge_teacher(statistics, recipe))
        self.assertGreater(statistics["teacher_largest_component_ratio"], 0.90)

    def test_mining_spatial_filter_keeps_particles_but_rejects_global_noise(self):
        recipe = {
            "min_overlap": 0.30,
            "max_largest_component_ratio": 0.45,
            "min_components": 24,
            "min_grid_coverage": 0.50,
            "mining": {
                "active_dense_min_area": 0.20,
                "active_dense_min_grid_coverage": 0.90,
                "fragment_min_components": 200,
                "fragment_min_grid_coverage": 0.75,
                "fragment_min_thin_ratio": 0.10,
                "fragment_max_largest_component_ratio": 0.25,
            },
        }
        good_samples = (
            {
                "teacher_old_edge_overlap_ratio": 0.066,
                "teacher_largest_component_ratio": 0.167,
                "teacher_significant_component_count": 94,
                "teacher_grid_coverage_ratio": 0.625,
                "teacher_thin_support_ratio": 0.006,
            },
            {
                "teacher_old_edge_overlap_ratio": 0.074,
                "teacher_largest_component_ratio": 0.098,
                "teacher_significant_component_count": 140,
                "teacher_grid_coverage_ratio": 0.547,
                "teacher_thin_support_ratio": 0.013,
            },
        )
        for statistics in good_samples:
            with self.subTest(statistics=statistics):
                self.assertEqual(
                    BUILDER.teacher_spatial_rejection_reasons(
                        statistics, recipe, action_type="mine_complete", area_ratio=0.06
                    ),
                    [],
                )

        dense = {
            "teacher_old_edge_overlap_ratio": 0.001,
            "teacher_largest_component_ratio": 0.866,
            "teacher_significant_component_count": 87,
            "teacher_grid_coverage_ratio": 0.953,
            "teacher_thin_support_ratio": 0.003,
        }
        self.assertIn(
            "global_dense_teacher",
            BUILDER.teacher_spatial_rejection_reasons(
                dense, recipe, action_type="mine_active", area_ratio=0.239
            ),
        )
        fragmented = {
            "teacher_old_edge_overlap_ratio": 0.166,
            "teacher_largest_component_ratio": 0.095,
            "teacher_significant_component_count": 509,
            "teacher_grid_coverage_ratio": 0.844,
            "teacher_thin_support_ratio": 0.172,
        }
        self.assertIn(
            "global_fragmented_teacher",
            BUILDER.teacher_spatial_rejection_reasons(
                fragmented, recipe, action_type="mine_complete", area_ratio=0.034
            ),
        )

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

    def test_fixed_cache_mode_does_not_initialize_online_cache(self):
        source = DATA_PATH.read_text(encoding="utf-8")
        self.assertIn("self.fixed_cache_only", source)
        self.assertIn("and not self.fixed_cache_only", source)
        train_source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("None if fixed_cache_only else build_online_warp_training_cache", train_source)

    def test_windows_dataset_path_is_relocated_by_dataset_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "wah_mc_training"
            target = root / "segments" / "sample.jsonl"
            target.parent.mkdir(parents=True)
            target.write_text("{}\n", encoding="utf-8")
            resolved = MANIFEST_BUILDER.resolve_dataset_path(
                r"F:\video-gen\Warp-as-History\data\vpt_9x_100\wah_mc_training\segments\sample.jsonl",
                root,
            )
            self.assertEqual(Path(resolved), target.resolve())

    def test_online_candidate_preparation_binds_source_row(self):
        source = DATA_PATH.read_text(encoding="utf-8")
        function_start = source.index("def prepare_online_warp_item(")
        row_binding = source.index("row = online_cache.rows[int(row_index)]", function_start)
        teacher_lookup = source.index('row.get("teacher_cache_path", "")', function_start)
        self.assertLess(row_binding, teacher_lookup)

    def test_teacher_preparation_launcher_exports_repo_pythonpath(self):
        launcher = (ROOT / "scripts" / "run_h100_mc_teacher_preparation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"', launcher)


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
            write_candidate_files(temp, row)
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

    def test_failed_candidate_is_rejected_and_next_row_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            failed = identity_row("place", "first", 0)
            failed.update(
                {
                    "candidate_error": "metadata rotation rejected",
                    "teacher_candidate_path": "",
                    "training_cache_path": "",
                }
            )
            good = identity_row("negative", "later", 1)
            good.update(
                {
                    "window_frames": "33",
                    "no_interaction_event_verified": "true",
                    "gui_closed_verified": "true",
                    "frame_contiguous_verified": "true",
                    "source_mapping_valid": "true",
                    "metadata_filter_status": "passed",
                    "candidate_error": "",
                }
            )
            write_candidate_files(temp, good)
            _, reviewed = run_teacher_builder(temp, [failed, good])
            self.assertEqual(len(reviewed), 2)
            self.assertEqual(reviewed[0]["review_status"], "rejected")
            self.assertIn("candidate_error:metadata rotation rejected", reviewed[0]["teacher_invalid_reasons"])
            self.assertEqual(reviewed[1]["review_status"], "pending")
            review_html = (temp / "teacher_pool" / "review_index.html").read_text(encoding="utf-8")
            self.assertNotIn("No preview generated", review_html)
            self.assertEqual(review_html.count("<article"), 1)

    def test_missing_candidate_file_does_not_stop_builder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            missing = identity_row("place", "first", 0)
            missing.update(
                {
                    "candidate_error": "",
                    "teacher_candidate_path": str(temp / "missing.npz"),
                    "training_cache_path": str(temp / "missing.pt"),
                }
            )
            _, reviewed = run_teacher_builder(temp, [missing])
            self.assertEqual(reviewed[0]["teacher_valid"], "false")
            self.assertIn("candidate_file_missing", reviewed[0]["teacher_invalid_reasons"])

    def test_review_frames_use_saved_rgb_to_latent_mapping(self):
        payload = {
            "target_rgb": np.zeros((33, 8, 8, 3), dtype=np.uint8),
            "rgb_frame_to_latent_index": np.asarray([0, 0, 0, 0, 0, 2, 1, 2, 0] + [1] * 24),
        }
        row = identity_row("place", "first", 0)
        row["event_local_frame"] = "7"
        self.assertEqual(
            BUILDER.review_frame_latent_pairs(payload, row),
            [(7, 2), (8, 0), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1)],
        )

    def test_static_hand_mask_matches_640_by_384_staircase(self):
        mask = BUILDER.minecraft_static_hand_mask(384, 640)
        for y, x in ((338, 269), (300, 333), (253, 397), (207, 461), (161, 531)):
            self.assertTrue(mask[y, x], (y, x))
        for y, x in ((337, 269), (299, 333), (252, 397), (206, 461), (160, 531)):
            self.assertFalse(mask[y, x], (y, x))
        central = mask[77:277, 192:448]
        self.assertFalse(mask[192, 320])
        self.assertLess(float(central.mean()), 0.05)

    def test_place_uses_static_mask_without_dynamic_expansion(self):
        reference = np.zeros((384, 640, 3), dtype=np.uint8)
        target = np.repeat(reference[None], 2, axis=0)
        target[:, 190:240, 440:490] = 255
        combined, dynamic = BUILDER.minecraft_hand_masks(reference, target, "place")
        self.assertFalse(dynamic.any())
        self.assertTrue(np.array_equal(combined[0], BUILDER.minecraft_static_hand_mask(384, 640)))

    def test_mining_masks_seed_connected_tool_but_not_background_motion(self):
        reference = np.zeros((384, 640, 3), dtype=np.uint8)
        target = np.repeat(reference[None], 1, axis=0)
        target[0, 195:230, 445:485] = 255
        target[0, 130:160, 280:320] = 255
        _, dynamic = BUILDER.minecraft_hand_masks(reference, target, "mine_active")
        self.assertTrue(dynamic[0, 205, 450])
        self.assertFalse(dynamic[0, 145, 300])

    def test_hand_mask_pixels_do_not_generate_teacher_or_tokens(self):
        static = BUILDER.minecraft_static_hand_mask(384, 640)
        target = np.zeros((1, 1, 384, 640), dtype=np.float32)
        target[0, 0, static] = 100.0
        valid = (~static)[None, None].astype(np.float32)
        _, _, teacher = BUILDER.robust_teacher(
            target,
            np.zeros_like(target),
            np.zeros_like(target),
            valid,
            3.0,
        )
        self.assertEqual(float(teacher[..., static].sum()), 0.0)
        stats = BUILDER.fixed_teacher_statistics(
            teacher,
            np.ones_like(valid),
            np.ones_like(valid),
            valid,
            0.25,
            (1, 6, 10),
            action_type="place",
        )
        self.assertEqual(stats["stage0_positive_tokens"], 0)

    def test_review_sheet_has_seven_rows_and_hand_mask_panel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.png"
            payload = {
                "target_rgb": np.zeros((33, 16, 16, 3), dtype=np.uint8),
                "warp_rgb": np.zeros((33, 16, 16, 3), dtype=np.uint8),
                "reference_rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                "visibility": np.ones((1, 9, 2, 2), dtype=np.float32),
                "world_valid": np.ones((1, 9, 2, 2), dtype=np.float32),
                "rgb_frame_to_latent_index": np.repeat(np.arange(9), 4)[:33],
            }
            row = identity_row("place", "first", 0)
            row.update({"event_local_frame": "0", "teacher_area_ratio": "0.1"})
            maps = np.zeros((1, 9, 2, 2), dtype=np.float32)
            BUILDER.save_review_contact_sheet(
                path,
                payload,
                maps,
                maps,
                maps,
                np.zeros((33, 16, 16), dtype=bool),
                row,
                1,
                [],
            )
            image = BUILDER.Image.open(path)
            self.assertEqual(image.width, 9 * 240)
            self.assertEqual(image.height, 120 + 7 * (135 + 24))
            self.assertIn('"hand / held-item mask"', (ROOT / "scripts" / "build_minecraft_teacher_pool.py").read_text())


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

    def test_teacher_preparation_launcher_stops_at_review(self):
        source = PREPARATION_LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", source)
        self.assertIn("--export_teacher_candidates_only", source)
        self.assertIn("build_minecraft_teacher_pool.py", source)
        self.assertIn("review_index.html", source)
        self.assertNotIn("router_overfit", source)
        self.assertNotIn("adapter_overfit", source)
        self.assertNotIn("joint_pilot", source)

    def test_resume_restores_plan_pool_hash_and_counters(self):
        contract = {
            "teacher_pool_manifest_hash": "manifest",
            "approved_teacher_row_ids": ["a", "b"],
            "action_history_pool_sizes": {"place|first": 2},
            "phase_plan": [{"steps": 10, "history": {"first": 1.0}}],
            "sampler_seed": 42,
            "base_model_fingerprint": {"sha256": "base"},
            "transformer_fingerprint": {"sha256": "transformer"},
            "camera_checkpoint_fingerprint": {"sha256": "camera"},
            "interaction_lr": 1.0e-4,
            "router_lr": 5.0e-5,
            "teacher_support_threshold": 0.25,
            "flow_matching_train_exact_timestep_sampling": "training_density",
            "candidate_config_hash": "config-hash",
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
        for field in (
            "sampler_seed",
            "base_model_fingerprint",
            "transformer_fingerprint",
            "camera_checkpoint_fingerprint",
            "interaction_lr",
            "router_lr",
            "teacher_support_threshold",
            "flow_matching_train_exact_timestep_sampling",
            "candidate_config_hash",
        ):
            changed = dict(contract)
            changed[field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
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
            '"resume_contract"',
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
