import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = REPO_ROOT / "warp_as_history" / "training" / "data.py"


def load_cache_class():
    tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"), filename=str(DATA_SOURCE))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OnlineWarpTrainingCache"
    )
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "_iter_online_image_files": lambda path: sorted(Path(path).iterdir()),
        "CAMERA_CONTROL_PI3_PIXEL_LIMIT": 255_000,
        "CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE": "mesh",
        "CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS": 0,
        "CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS": 1,
        "CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE": "depth_normal",
        "CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL": 0.03,
        "CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG": 35.0,
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(DATA_SOURCE), "exec"), namespace)
    return namespace["OnlineWarpTrainingCache"]


OnlineWarpTrainingCache = load_cache_class()


def make_cache(cache_dir):
    cache = OnlineWarpTrainingCache.__new__(OnlineWarpTrainingCache)
    cache.disk_cache_dir = Path(cache_dir)
    cache.exact_args = SimpleNamespace(
        height=360,
        width=640,
        online_frame_stride=1,
        online_max_video_frames=0,
        online_pi3_pixel_limit=255_000,
        online_pi3_conf_threshold=0.1,
        online_pi3_depth_edge_rtol=0.03,
        online_mesh_samples_per_axis=4,
        online_render_mode="mesh",
        online_target_fill_radius=0,
        online_target_fill_min_neighbors=1,
        online_mesh_break_mode="depth_normal",
        online_mesh_depth_rtol=0.03,
        online_mesh_normal_tol_deg=35.0,
        use_minecraft_hud_mask=True,
    )
    return cache


class OnlineWarpCacheDedupTest(unittest.TestCase):
    def test_rows_sharing_video_share_geometry_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "segment.mp4"
            video.write_bytes(b"same video")
            cache = make_cache(root / "cache")

            first = cache._geometry_cache_path(video, "forward")
            second = cache._geometry_cache_path(Path(str(video)), "forward")

            self.assertEqual(first, second)
            self.assertNotRegex(first.name, r"^\d{5}_")

    def test_direction_config_and_content_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "segment.mp4"
            video.write_bytes(b"version one")
            cache = make_cache(root / "cache")

            forward = cache._geometry_cache_path(video, "forward")
            reverse = cache._geometry_cache_path(video, "reverse")
            cache.exact_args.online_frame_stride = 2
            changed_config = cache._geometry_cache_path(video, "forward")
            cache.exact_args.online_frame_stride = 1
            video.write_bytes(b"version two with a different size")
            changed_content = cache._geometry_cache_path(video, "forward")

            self.assertNotEqual(forward, reverse)
            self.assertNotEqual(forward, changed_config)
            self.assertNotEqual(forward, changed_content)

    def test_different_paths_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_video = root / "one" / "segment.mp4"
            second_video = root / "two" / "segment.mp4"
            first_video.parent.mkdir()
            second_video.parent.mkdir()
            first_video.write_bytes(b"same bytes")
            second_video.write_bytes(b"same bytes")
            cache = make_cache(root / "cache")

            first = cache._geometry_cache_path(first_video, "forward")
            second = cache._geometry_cache_path(second_video, "forward")

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
