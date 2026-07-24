import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = REPO_ROOT / "warp_as_history" / "training" / "data.py"
LOSS_MASKS_SOURCE = REPO_ROOT / "warp_as_history" / "training" / "loss_masks.py"
HUD_SYMBOLS = {
    "MC_HUD_SOURCE_SIZE",
    "MC_HUD_RECTS",
    "minecraft_world_valid_mask",
    "_minecraft_hud_rect_masks",
    "fill_minecraft_hud_for_pi3",
    "multiply_mask_frames",
    "clear_minecraft_hud_geometry",
}


def load_hud_symbols_from_data():
    tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"), filename=str(DATA_SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = {target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)}
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            if names & HUD_SYMBOLS:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HUD_SYMBOLS:
            selected.append(node)
    namespace = {"np": np, "Image": Image, "ImageOps": ImageOps}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DATA_SOURCE), "exec"), namespace)
    return namespace


HUD = load_hud_symbols_from_data()
LOSS_SPEC = importlib.util.spec_from_file_location("wah_loss_masks_test", LOSS_MASKS_SOURCE)
LOSS_MODULE = importlib.util.module_from_spec(LOSS_SPEC)
LOSS_SPEC.loader.exec_module(LOSS_MODULE)
valid_element_normalized_loss = LOSS_MODULE.valid_element_normalized_loss


class MinecraftHudMaskTest(unittest.TestCase):
    def test_rectangles_are_invalid_and_world_is_valid(self):
        mask = np.asarray(HUD["minecraft_world_valid_mask"](), dtype=np.uint8)
        self.assertEqual(mask.shape, (360, 640))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        for x1, x2, y1, y2 in HUD["MC_HUD_RECTS"]:
            self.assertTrue(np.all(mask[y1:y2, x1:x2] == 0))
        self.assertEqual(int(mask[100, 100]), 255)
        self.assertEqual(int(mask[180, 320]), 255)  # Crosshair remains valid.
        self.assertEqual(int(mask[300, 600]), 255)  # Hand area remains valid.
        resized = np.asarray(HUD["minecraft_world_valid_mask"](height=384, width=640))
        self.assertEqual(resized.shape, (384, 640))
        self.assertTrue(set(np.unique(resized)).issubset({0, 255}))

    def test_pi3_fill_does_not_mutate_target(self):
        rows = np.arange(360, dtype=np.uint16)[:, None, None]
        frame_array = np.broadcast_to(rows, (360, 640, 3)).astype(np.uint8).copy()
        target = Image.fromarray(frame_array, mode="RGB")
        target_before = np.asarray(target).copy()
        filled = np.asarray(HUD["fill_minecraft_hud_for_pi3"](target))
        self.assertTrue(np.array_equal(np.asarray(target), target_before))
        rects = HUD["MC_HUD_RECTS"]
        for rect_index, (x1, x2, y1, y2) in enumerate(rects):
            region = np.zeros((360, 640), dtype=bool)
            region[y1:y2, x1:x2] = True
            for later_rect in rects[rect_index + 1 :]:
                lx1, lx2, ly1, ly2 = later_rect
                region[ly1:ly2, lx1:lx2] = False
            ys, xs = np.nonzero(region)
            self.assertTrue(np.all(filled[ys, xs] == target_before[y1 - 1, xs]))

    def test_hud_target_changes_do_not_change_valid_loss(self):
        valid = np.asarray(HUD["minecraft_world_valid_mask"](), dtype=np.float32) / 255.0
        valid = valid[None, None]
        target_a = np.zeros((1, 3, 360, 640), dtype=np.float32)
        target_b = target_a.copy()
        target_b[:, :, valid[0, 0] == 0] = 100.0
        loss_a = valid_element_normalized_loss(target_a**2, valid)
        loss_b = valid_element_normalized_loss(target_b**2, valid)
        self.assertEqual(float(loss_a), float(loss_b))
        target_b[:, :, 100, 100] = 1.0
        self.assertGreater(float(valid_element_normalized_loss(target_b**2, valid)), float(loss_a))

    def test_visibility_focus_and_pi3_geometry_are_cleared(self):
        world_image = HUD["minecraft_world_valid_mask"]()
        ones = Image.fromarray(np.full((360, 640), 255, dtype=np.uint8), mode="L")
        masked = np.asarray(HUD["multiply_mask_frames"]([ones], world_image)[0], dtype=np.uint8)
        self.assertTrue(np.array_equal(masked, np.asarray(world_image)))

        geometry = {
            "render_height": 360,
            "render_width": 640,
            "valid_mask": np.ones((360, 640), dtype=bool),
            "conf_map": np.ones((360, 640), dtype=np.float32),
            "depth_map": np.ones((360, 640), dtype=np.float32),
            "point_map_world": np.ones((360, 640, 3), dtype=np.float32),
        }
        HUD["clear_minecraft_hud_geometry"](geometry, world_image)
        invalid = np.asarray(world_image) == 0
        self.assertFalse(geometry["valid_mask"][invalid].any())
        self.assertFalse(geometry["conf_map"][invalid].any())
        self.assertFalse(geometry["depth_map"][invalid].any())
        self.assertFalse(geometry["point_map_world"][invalid].any())


if __name__ == "__main__":
    unittest.main()
