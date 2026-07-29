"""Shared, serializable Warp-as-History camera recipe for Minecraft."""
from __future__ import annotations


DEFAULT_MINECRAFT_WAH_RECIPE = {
    "schema_version": 1,
    "target_fps": 16.0,
    "num_frames": 33,
    "warp_history_downsample_mode": "short",
    "camera_warp_render_mode": "target_fill",
    "camera_control_translation_scale": 0.1,
    "camera_multiply_translation_by_depth": True,
    "camera_mesh_samples_per_axis": 4,
    "camera_keyframe_max_previous": 19,
    "visible_token_threshold": 0.1,
    "amplify_first_chunk": False,
    "history_sizes": [16, 2, 1],
    "pose_convention": "opencv_c2w_relative",
}


def minecraft_wah_recipe(**overrides):
    recipe = dict(DEFAULT_MINECRAFT_WAH_RECIPE)
    recipe.update({key: value for key, value in overrides.items() if value is not None})
    recipe["target_fps"] = float(recipe["target_fps"])
    recipe["num_frames"] = int(recipe["num_frames"])
    recipe["camera_control_translation_scale"] = float(recipe["camera_control_translation_scale"])
    recipe["camera_mesh_samples_per_axis"] = int(recipe["camera_mesh_samples_per_axis"])
    recipe["camera_keyframe_max_previous"] = int(recipe["camera_keyframe_max_previous"])
    recipe["visible_token_threshold"] = float(recipe["visible_token_threshold"])
    recipe["camera_multiply_translation_by_depth"] = bool(recipe["camera_multiply_translation_by_depth"])
    recipe["amplify_first_chunk"] = bool(recipe["amplify_first_chunk"])
    recipe["history_sizes"] = [int(value) for value in recipe["history_sizes"]]
    return recipe


def recipe_mismatches(expected, actual):
    expected = minecraft_wah_recipe(**dict(expected or {}))
    actual = minecraft_wah_recipe(**dict(actual or {}))
    return {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}


def format_recipe_warning(mismatches):
    if not mismatches:
        return ""
    details = ", ".join(
        f"{name}: checkpoint={values['expected']!r}, runtime={values['actual']!r}"
        for name, values in sorted(mismatches.items())
    )
    return f"WARNING: Minecraft WAH recipe mismatch. {details}"
