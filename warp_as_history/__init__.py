from .defaults import WAH_DEFAULT_LORA_PATH, WAH_NEGATIVE_PROMPT, WAH_PROMPT_TRIGGER

__all__ = [
    "Pi3XWarpRenderer",
    "Pi3XWarpRendererConfig",
    "WAH_NEGATIVE_PROMPT",
    "WAH_DEFAULT_LORA_PATH",
    "WAH_PROMPT_TRIGGER",
    "WarpAsHistoryPipeline",
    "WarpAsHistoryPipelineOutput",
    "default_pi3x_ckpt",
    "render_pi3x_camera_warp",
]


def __getattr__(name):
    if name in {"WarpAsHistoryPipeline", "WarpAsHistoryPipelineOutput"}:
        from .pipeline import WarpAsHistoryPipeline, WarpAsHistoryPipelineOutput

        return {
            "WarpAsHistoryPipeline": WarpAsHistoryPipeline,
            "WarpAsHistoryPipelineOutput": WarpAsHistoryPipelineOutput,
        }[name]
    if name in {
        "Pi3XWarpRenderer",
        "Pi3XWarpRendererConfig",
        "default_pi3x_ckpt",
        "render_pi3x_camera_warp",
    }:
        from .camera_warp import (
            Pi3XWarpRenderer,
            Pi3XWarpRendererConfig,
            default_pi3x_ckpt,
            render_pi3x_camera_warp,
        )

        return {
            "Pi3XWarpRenderer": Pi3XWarpRenderer,
            "Pi3XWarpRendererConfig": Pi3XWarpRendererConfig,
            "default_pi3x_ckpt": default_pi3x_ckpt,
            "render_pi3x_camera_warp": render_pi3x_camera_warp,
        }[name]
    raise AttributeError(name)
