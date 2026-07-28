import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = REPO_ROOT / "warp_as_history" / "training" / "data.py"
SYMBOLS = {
    "_online_resample_indices",
    "load_vpt_pose_rows",
    "vpt_relative_camera_poses",
}


def load_symbols():
    tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"), filename=str(DATA_SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SYMBOLS
    ]
    namespace = {"np": np, "Path": Path, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DATA_SOURCE), "exec"), namespace)
    return namespace


MC = load_symbols()


class Minecraft16FpsCameraTrainingTest(unittest.TestCase):
    def test_twenty_fps_is_timestamp_resampled_to_sixteen(self):
        indices = MC["_online_resample_indices"](20.0, 16.0, 20)
        self.assertEqual(indices.tolist(), [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19])
        self.assertEqual(len(indices), 16)

    def test_vpt_pose_rows_follow_resampled_source_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.jsonl"
            rows = [
                {"segment_frame": index, "xpos": index, "ypos": 64, "zpos": 0, "yaw": 0, "pitch": 0}
                for index in range(6)
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            selected = MC["load_vpt_pose_rows"](path, [0, 2, 5])
            self.assertEqual([row["segment_frame"] for row in selected], [0, 2, 5])

    def test_vpt_motion_maps_to_camera_translation_and_rotation(self):
        rows = [
            {"xpos": 0, "ypos": 64, "zpos": 0, "yaw": 0, "pitch": 0},
            {"xpos": -2, "ypos": 65, "zpos": 3, "yaw": 90, "pitch": 10},
        ]
        poses = MC["vpt_relative_camera_poses"](rows, 0, [0, 1], translation_scale=0.5)
        self.assertEqual(poses.shape, (2, 4, 4))
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1.0e-6)
        np.testing.assert_allclose(poses[1, :3, 3], [1.0, 0.5, 1.5], atol=1.0e-6)
        self.assertFalse(np.allclose(poses[1, :3, :3], np.eye(3)))

    def test_upsampling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds source fps"):
            MC["_online_resample_indices"](16.0, 20.0, 100)


if __name__ == "__main__":
    unittest.main()
