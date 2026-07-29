import ast
import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = REPO_ROOT / "warp_as_history" / "training" / "data.py"
PIPELINE_SOURCE = REPO_ROOT / "warp_as_history" / "pipeline.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLING = load_module(
    "minecraft_sampling_test",
    REPO_ROOT / "warp_as_history" / "minecraft_sampling.py",
)
CAMERA = load_module(
    "minecraft_camera_test",
    REPO_ROOT / "warp_as_history" / "minecraft_camera.py",
)


def load_data_functions(names):
    tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"), filename=str(DATA_SOURCE))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DATA_SOURCE), "exec"), namespace)
    return namespace


DATA = load_data_functions(
    {
        "event_alignment_from_row",
        "reverse_event_frame",
        "reverse_temporal_event_payload",
    }
)


class MinecraftStepSamplingAndAlignmentTest(unittest.TestCase):
    def test_ten_thousand_step_ratio_error_is_below_two_percent(self):
        sampler = SAMPLING.StepCategorySampler(
            {"place": list(range(11)), "mine": list(range(7)), "other": list(range(5))},
            {"place": 0.5, "mine": 0.3, "other": 0.2},
            total_steps=10_000,
            seed=17,
        )
        report = sampler.report(10_000)
        self.assertLess(abs(report["actual_ratios"]["place"] - 0.5), 0.02)
        self.assertLess(abs(report["actual_ratios"]["mine"] - 0.3), 0.02)
        self.assertLess(abs(report["actual_ratios"]["other"] - 0.2), 0.02)

    def test_positive_windows_always_keep_event_at_local_six_to_sixteen(self):
        rng = random.Random(23)
        for event_frame in range(20, 500):
            indices, local = SAMPLING.build_interaction_event_window(
                event_frame,
                num_source_frames=600,
                window_size=33,
                rng=rng,
                local_min=6,
                local_max=16,
                require_later=True,
            )
            self.assertIn(event_frame, indices)
            self.assertGreater(indices[0], 0)
            self.assertGreaterEqual(local, 6)
            self.assertLessEqual(local, 16)

    def test_timestamp_event_mapping_is_exact_after_resampling(self):
        source_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19]
        result = DATA["event_alignment_from_row"](
            {"source_frame_start": 100, "event_source_frame": 112},
            source_indices,
            source_fps=20,
            target_fps=16,
        )
        self.assertEqual(result["segment_event_frame"], 12)
        self.assertEqual(result["resampled_event_frame"], 10)
        self.assertAlmostEqual(result["source_event_time_ms"], 600.0)
        self.assertAlmostEqual(result["resampled_event_time_ms"], 625.0)

    def test_reverse_keeps_event_masks_and_poses_aligned(self):
        payload = {
            "event_frame": 7,
            "events": [{"event_frame": 7, "action_type": "place"}],
            "time_mask": [0.0] * 7 + [1.0] * 26,
            "source_frame_indices": list(range(33)),
        }
        reversed_payload = DATA["reverse_temporal_event_payload"](payload, 33)
        self.assertEqual(reversed_payload["event_frame"], 25)
        self.assertEqual(reversed_payload["events"][0]["event_frame"], 25)
        self.assertEqual(reversed_payload["time_mask"], list(reversed(payload["time_mask"])))
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 33, axis=0)
        poses[:, 0, 3] = np.arange(33)
        reversed_poses = CAMERA.relative_opencv_c2w(poses[::-1])
        self.assertAlmostEqual(float(reversed_poses[0, 0, 3]), 0.0)
        self.assertLess(float(reversed_poses[-1, 0, 3]), 0.0)

    def test_movement_bypasses_router_and_first_chunk_is_not_global(self):
        source = DATA_SOURCE.read_text(encoding="utf-8")
        self.assertIn('elif category in {"movement", "other"}:', source)
        self.assertIn("full_interaction_payload = None", source)
        self.assertNotIn("if rng.random() < first_chunk_prob:\n            target_indices = range(33)", source)

    def test_pi3_geometry_receives_only_conditioning_frames(self):
        source = DATA_SOURCE.read_text(encoding="utf-8")
        self.assertIn("def _estimate_conditioning_geometry", source)
        self.assertIn('"target_rgb_used": False', source)
        first_branch = source[source.index("if target_start <= 0:") : source.index("else:", source.index("if target_start <= 0:"))]
        self.assertIn("geometry_keyframe_frames = [source_idx]", first_branch)
        self.assertIn("history_indices[0] <= frame_index < target_start", source)

    def test_training_and_inference_share_pose_convention(self):
        training_source = DATA_SOURCE.read_text(encoding="utf-8")
        inference_source = (REPO_ROOT / "scripts" / "learn_vpt_camera_controller.py").read_text(encoding="utf-8")
        self.assertIn("warp_as_history.minecraft_camera", training_source)
        self.assertIn("warp_as_history.minecraft_camera", inference_source)

    def test_place_event_is_consumed_once_across_two_chunks(self):
        payload = {"event_frame": 6, "event_valid": 1.0, "action_type": "place", "block_id": "oak_log"}
        chunk_zero = SAMPLING.interaction_payload_for_chunk(
            payload,
            chunk_index=0,
            window_frames=33,
            consumed_event_frames=[],
        )
        self.assertEqual(chunk_zero["event_frame"], 6)
        chunk_one = SAMPLING.interaction_payload_for_chunk(
            payload,
            chunk_index=1,
            window_frames=33,
            consumed_event_frames=[6],
        )
        self.assertIsNone(chunk_one)
        pipeline = PIPELINE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('state["prev_history_latent_window"] = current_history', pipeline)
        self.assertIn("consumed_interaction_event_frames", pipeline)


if __name__ == "__main__":
    unittest.main()
