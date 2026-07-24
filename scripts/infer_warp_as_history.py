#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TAEHV_ROOT = REPO_ROOT / "third_party" / "taehv"
if TAEHV_ROOT.is_dir() and str(TAEHV_ROOT) not in sys.path:
    sys.path.insert(0, str(TAEHV_ROOT))

DEFAULT_MODEL = "checkpoints/helios-distilled"
DEFAULT_WAH_LORA = "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"
DEFAULT_CAMERA_WARP_RENDER_MODE = "splat"
DEFAULT_CAMERA_PI3_PIXEL_LIMIT = 255000
DEFAULT_CAMERA_MESH_SAMPLES_PER_AXIS = 4
REALTIME_CAMERA_WARP_RENDER_MODE = "target_fill"
REALTIME_CAMERA_PI3_PIXEL_LIMIT = 130000
REALTIME_CAMERA_MESH_SAMPLES_PER_AXIS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Warp-as-History inference from a one-row demo CSV.")
    parser.add_argument(
        "csv_path",
        type=Path,
        help=(
            "CSV containing first_frame_path, prompt, camera_poses_path, "
            "warp_video_path, warp_visibility_mask_path, and optional primary_fire_event_path."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path. Defaults to runs/<csv_stem>.mp4.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--lora_path", default=DEFAULT_WAH_LORA)
    parser.add_argument("--camera_key", default="camera_poses")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Defaults to the warp video frame count or the number of frames in camera_poses.npz.",
    )
    parser.add_argument("--fps", type=int, default=0, help="Defaults to warp video fps, camera pose fps, or 16.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--no_lora", action="store_true", help="Run without loading a Warp-as-History LoRA.")
    parser.add_argument("--use_primary_fire_event_condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--interaction_conditioning_mode",
        choices=["router", "binary", "off"],
        default="router",
        help="router consumes event_frame/action_type/block_id; binary keeps the legacy time gate.",
    )
    parser.add_argument("--use_primary_fire_focus_loss", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--primary_fire_focus_loss_scale", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--primary_fire_background_loss_scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pyramid_num_inference_steps_list",
        type=int,
        nargs=3,
        default=None,
        metavar=("S0", "S1", "S2"),
        help="Override Helios pyramid inference steps, for example: 1 1 1.",
    )
    parser.add_argument(
        "--no_amplify_first_chunk",
        action="store_false",
        dest="amplify_first_chunk",
        default=True,
        help="Disable first-chunk amplification for faster inference.",
    )
    parser.add_argument(
        "--warp_history_downsample_mode",
        choices=["short", "patch_mid"],
        default="short",
        help="Use patch_mid with a LoRA trained using the efficient Warp-as-History recipe.",
    )
    parser.add_argument(
        "--camera_realtime_fast_warp",
        action="store_true",
        default=None,
        help=(
            "Use lower-quality realtime camera-warp settings. Defaults to on for "
            "--warp_history_downsample_mode patch_mid and off otherwise."
        ),
    )
    parser.add_argument(
        "--no_camera_realtime_fast_warp",
        action="store_false",
        dest="camera_realtime_fast_warp",
        help="Disable realtime camera-warp settings even when using patch_mid efficient inference.",
    )
    parser.add_argument(
        "--camera_warp_render_mode",
        choices=["splat", "target_fill"],
        default=None,
        help="Override camera warp render mode. Defaults to splat, or target_fill in realtime fast mode.",
    )
    parser.add_argument(
        "--camera_pi3_pixel_limit",
        type=int,
        default=None,
        help="Override Pi3/render pixel limit for camera warp. Defaults to 255000, or 130000 in realtime fast mode.",
    )
    parser.add_argument(
        "--camera_mesh_samples_per_axis",
        type=int,
        default=None,
        help="Override camera warp mesh samples per depth quad. Defaults to 4, or 2 in realtime fast mode.",
    )
    parser.add_argument(
        "--warp_debug_dir",
        type=Path,
        default=None,
        help="Optional directory where the pipeline writes warp.mp4 for the warp conditioning debug view.",
    )
    parser.add_argument(
        "--taehv_vae_mode",
        choices=["off", "decode", "full"],
        default=None,
        help=(
            "Optional TAEHV preview VAE mode. Defaults to off, or full when "
            "--taehv_checkpoint is provided. decode/full use TAEHV for faster display decoding."
        ),
    )
    parser.add_argument(
        "--taehv_checkpoint",
        type=Path,
        default=None,
        help="Path to a TAEHV checkpoint such as taew2_1.pth. Required unless --taehv_vae_mode off.",
    )
    parser.add_argument(
        "--enable_optional_attention",
        action="store_true",
        help=(
            "Let diffusers import optional attention packages such as xformers "
            "or flash-attn. By default the script uses native PyTorch attention."
        ),
    )
    parser.add_argument("--enable_xformers", dest="enable_optional_attention", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _field(row: dict[str, str], *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_csv_path(value: str, csv_path: Path, *, required: bool = False) -> Path | None:
    value = str(value or "").strip()
    if not value:
        if required:
            raise ValueError(f"Missing required path in {csv_path}")
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [csv_path.parent / raw, REPO_ROOT / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"Could not resolve {value!r} from {csv_path}")
    return candidates[0].resolve()


def load_demo_row(csv_path: Path) -> dict[str, Any]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing demo CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{csv_path} must contain exactly one data row, got {len(rows)}")
    row = rows[0]
    image_path = _resolve_csv_path(
        _field(row, "first_frame_path", "image_path", "image", "first_frame"),
        csv_path,
        required=True,
    )
    prompt = _field(row, "prompt", "prompts", "caption")
    prompt_path = _resolve_csv_path(_field(row, "prompt_path"), csv_path, required=False)
    if not prompt and prompt_path is not None:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{csv_path} must provide prompt or prompt_path")
    return {
        "csv_path": csv_path,
        "image_path": image_path,
        "prompt": prompt,
        "camera_poses_path": _resolve_csv_path(_field(row, "camera_poses_path", "camera_path"), csv_path),
        "warp_video_path": _resolve_csv_path(_field(row, "warp_video_path", "warp_path"), csv_path),
        "warp_visibility_mask_path": _resolve_csv_path(
            _field(
                row,
                "warp_visibility_mask_path",
                "warp_visibliry_mask_path",
                "visibility_mask_path",
                "warp_mask_path",
            ),
            csv_path,
        ),
        "primary_fire_event_path": _resolve_csv_path(_field(row, "primary_fire_event_path"), csv_path),
        "interaction_event_path": _resolve_csv_path(
            _field(row, "interaction_event_path", "mc_event_path", "primary_fire_event_path"), csv_path
        ),
    }


def load_interaction_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "event_frame" in payload and "action_type" in payload:
        return {
            "event_frame": int(payload["event_frame"]),
            "action_type": str(payload["action_type"]),
            "object_id": payload.get("object_id"),
            "block_id": payload.get("block_id", payload.get("object_id")),
            "event_valid": float(payload.get("event_valid", 1.0)),
        }
    events = list(payload.get("events", payload.get("selected_events", [])) or [])
    if events:
        event = events[0]
        action_type = str(event.get("action_type", event.get("category", "none")))
        if action_type == "mine":
            action_type = "mine_complete"
        return {
            "event_frame": int(event.get("event_frame", event.get("local_frame", event.get("frame", 0)))),
            "action_type": action_type,
            "object_id": event.get("object_id"),
            "block_id": event.get("block_id", event.get("object_id")),
            "event_valid": 1.0,
        }
    click_frames = payload.get("click_frames_local", payload.get("click_frames_source", payload.get("click_frames", [])))
    if click_frames:
        return {
            "event_frame": int(click_frames[0]),
            "action_type": "primary_fire",
            "object_id": "primary_fire",
            "block_id": "primary_fire",
            "event_valid": 1.0,
        }
    return None


def build_primary_fire_event_latents(
    *,
    path: Path,
    num_frames: int,
    latent_channels: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    temporal_scale: int,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_frame_indices = [int(x) for x in payload.get("source_frame_indices", list(range(int(num_frames))))]
    time_mask = payload.get("time_mask")
    if time_mask is None:
        click_frames = {int(x) for x in payload.get("click_frames_source", payload.get("click_frames", []))}
        time_mask = [1.0 if int(idx) in click_frames else 0.0 for idx in source_frame_indices]
    if len(source_frame_indices) != len(time_mask):
        raise ValueError("primary_fire_event source_frame_indices and time_mask lengths must match.")
    frame_weights = np.asarray([float(weight) for weight in time_mask[: int(num_frames)]], dtype=np.float32)
    latent_values = np.zeros(int(latent_frames), dtype=np.float32)
    mapping = []
    for latent_idx in range(int(latent_frames)):
        start = int(latent_idx) * int(temporal_scale)
        end = min(int(num_frames), start + int(temporal_scale))
        if end <= start:
            end = min(int(num_frames), start + 1)
        latent_values[int(latent_idx)] = float(frame_weights[start:end].max()) if end > start else 0.0
        mapping.append(
            {
                "latent_index": int(latent_idx),
                "frame_start": int(start),
                "frame_end_exclusive": int(end),
                "source_frames": source_frame_indices[start:end],
                "has_click": any(float(x) > 0.0 for x in frame_weights[start:end]),
            }
        )
    latents = latent_values.reshape(1, 1, int(latent_frames), 1, 1)
    latents = np.broadcast_to(latents, (1, int(latent_channels), int(latent_frames), int(latent_height), int(latent_width)))
    return latents.astype(np.float32), mapping


def load_camera_poses(path: Path, key: str) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera pose file: {path}")
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}. Available keys: {list(data.files)}")
        poses = np.asarray(data[key], dtype=np.float32)
        fps = int(round(float(data["fps"]))) if "fps" in data else 16
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera poses with shape [T, 4, 4], got {poses.shape}")
    return poses, fps


def load_video_frames(path: Path) -> tuple[list[np.ndarray], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing video file: {path}")
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        frames = [frame_to_uint8(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path} contains no frames")
    fps = int(round(float(meta.get("fps") or 16)))
    return frames, fps


def torch_dtype_from_arg(dtype: str, device: str):
    import torch

    if dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def unwrap_video_frames(value: Any) -> list[Any]:
    if hasattr(value, "frames"):
        value = value.frames
    if isinstance(value, np.ndarray):
        if value.ndim == 5:
            value = value[0]
        if value.ndim == 4:
            return [value[i] for i in range(value.shape[0])]
        if value.ndim == 3:
            return [value]
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
            nested = value[0]
            if not (isinstance(nested, np.ndarray) and nested.ndim == 3):
                return unwrap_video_frames(nested)
        return list(value)
    raise TypeError(f"Unsupported pipeline output type: {type(value)!r}")


def frame_to_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        arr = np.asarray(frame)
        if arr.ndim != 3:
            raise ValueError(f"Expected frame with shape [H, W, C], got {arr.shape}")
        if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.round().astype(np.uint8)
    return arr


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame_to_uint8(frame))


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(str(path.absolute()))
    checkpoints_root = Path(str((REPO_ROOT / "checkpoints").absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"--model_path must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {path}. Run `python scripts/check_models.py`.")
    return str(path)


def resolve_lora_path(lora_path: str | Path | None) -> str | None:
    if lora_path is None:
        return None
    value = str(lora_path).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(str(path.absolute()))
    allowed_roots = [
        Path(str((REPO_ROOT / "checkpoints").absolute())),
        Path(str((REPO_ROOT / "runs").absolute())),
    ]
    if not any(path.is_relative_to(root) for root in allowed_roots):
        roots_text = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"--lora_path must be under one of: {roots_text}; got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing LoRA checkpoint: {path}. Run `python scripts/check_models.py`.")
    return str(path)


def resolve_taehv_vae_mode_arg(mode: str | None, checkpoint: str | Path | None) -> str:
    if mode is not None:
        return str(mode).strip().lower()
    if checkpoint is not None and str(checkpoint).strip():
        return "full"
    return "off"


def validate_taehv_checkpoint_arg(mode: str, checkpoint: str | Path | None) -> Path | None:
    mode = str(mode or "off").strip().lower()
    if mode == "off":
        return None
    if checkpoint is None or not str(checkpoint).strip():
        raise ValueError(
            "TAEHV VAE mode is enabled but --taehv_checkpoint was not provided. "
            "Download taew2_1.pth from https://github.com/madebyollin/taehv and pass "
            "--taehv_checkpoint /path/to/taew2_1.pth."
        )
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing TAEHV checkpoint: {checkpoint_path}. Download it with:\n"
            "  mkdir -p checkpoints/taehv\n"
            "  wget -O checkpoints/taehv/taew2_1.pth "
            "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth\n"
            "Then pass --taehv_checkpoint checkpoints/taehv/taew2_1.pth."
        )
    return checkpoint_path


def validate_taehv_import_arg(mode: str) -> None:
    if str(mode or "off").strip().lower() == "off":
        return
    if importlib.util.find_spec("taehv") is None:
        raise ImportError(
            "TAEHV VAE was requested, but Python module 'taehv' is not importable. "
            "Use the vendored third_party/taehv module or put taehv.py on PYTHONPATH, then retry. "
            "Repository: https://github.com/madebyollin/taehv."
        )


def disable_diffusers_optional_attention() -> None:
    try:
        import diffusers.utils.import_utils as diffusers_import_utils
    except Exception:
        return
    for attr in (
        "_xformers_available",
        "_flash_attn_available",
        "_flash_attn_3_available",
        "_aiter_available",
        "_sageattention_available",
    ):
        if hasattr(diffusers_import_utils, attr):
            setattr(diffusers_import_utils, attr, False)
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils
    except Exception:
        return
    for module in (transformers_utils, transformers_import_utils):
        for name in (
            "is_flash_attn_2_available",
            "is_flash_attn_greater_or_equal",
            "is_flash_attn_greater_or_equal_2_10",
        ):
            if hasattr(module, name):
                setattr(module, name, lambda *args, **kwargs: False)


def main() -> None:
    args = parse_args()
    taehv_vae_mode = resolve_taehv_vae_mode_arg(args.taehv_vae_mode, args.taehv_checkpoint)
    taehv_checkpoint = args.taehv_checkpoint
    if taehv_vae_mode != "off":
        try:
            taehv_checkpoint = validate_taehv_checkpoint_arg(taehv_vae_mode, args.taehv_checkpoint)
            validate_taehv_import_arg(taehv_vae_mode)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
    if not args.enable_optional_attention:
        disable_diffusers_optional_attention()

    sample = load_demo_row(args.csv_path)
    csv_path = sample["csv_path"]
    image_path = sample["image_path"]
    prompt = sample["prompt"]
    output = args.output.expanduser().resolve() if args.output else (REPO_ROOT / "runs" / f"{csv_path.stem}.mp4")

    warp_video = None
    warp_visibility_mask = None
    camera_poses = None
    primary_fire_event_latents = None
    interaction_payload = load_interaction_payload(sample.get("interaction_event_path"))
    conditioning_type = ""
    conditioning_frames = 0
    conditioning_fps = 16
    if sample["warp_video_path"] is not None:
        warp_video, conditioning_fps = load_video_frames(sample["warp_video_path"])
        conditioning_type = "warp_video"
        conditioning_frames = len(warp_video)
        if sample["warp_visibility_mask_path"] is not None:
            warp_visibility_mask, mask_fps = load_video_frames(sample["warp_visibility_mask_path"])
            if len(warp_visibility_mask) != conditioning_frames:
                raise ValueError(
                    f"warp visibility mask has {len(warp_visibility_mask)} frames, "
                    f"but warp video has {conditioning_frames} frames"
                )
            if int(args.fps) <= 0:
                conditioning_fps = mask_fps or conditioning_fps
    elif sample["camera_poses_path"] is not None:
        camera_poses, conditioning_fps = load_camera_poses(sample["camera_poses_path"], args.camera_key)
        conditioning_type = "camera_poses"
        conditioning_frames = int(camera_poses.shape[0])
    else:
        raise ValueError(f"{csv_path} must provide either warp_video_path or camera_poses_path")

    fps = int(args.fps) if int(args.fps) > 0 else int(conditioning_fps)
    num_frames = int(args.num_frames) if int(args.num_frames) > 0 else int(conditioning_frames)
    if conditioning_type == "camera_poses" and num_frames > conditioning_frames:
        raise ValueError(f"--num_frames={num_frames} exceeds camera pose length {conditioning_frames}")

    import torch
    from warp_as_history import WarpAsHistoryPipeline

    device = args.device
    dtype = torch_dtype_from_arg(args.dtype, device)
    generator = torch.Generator(device=device).manual_seed(int(args.seed)) if device.startswith("cuda") else None

    model_path = resolve_model_path(args.model_path)
    lora_path = None if args.no_lora else resolve_lora_path(args.lora_path)
    pipe = WarpAsHistoryPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    if taehv_vae_mode != "off":
        try:
            pipe.install_taehv_vae(mode=taehv_vae_mode, checkpoint=taehv_checkpoint)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
    if args.camera_realtime_fast_warp is None:
        use_realtime_fast_warp = str(args.warp_history_downsample_mode) == "patch_mid"
    else:
        use_realtime_fast_warp = bool(args.camera_realtime_fast_warp)
    camera_warp_render_mode = (
        str(args.camera_warp_render_mode)
        if args.camera_warp_render_mode is not None
        else (REALTIME_CAMERA_WARP_RENDER_MODE if use_realtime_fast_warp else DEFAULT_CAMERA_WARP_RENDER_MODE)
    )
    camera_pi3_pixel_limit = (
        int(args.camera_pi3_pixel_limit)
        if args.camera_pi3_pixel_limit is not None
        else (REALTIME_CAMERA_PI3_PIXEL_LIMIT if use_realtime_fast_warp else DEFAULT_CAMERA_PI3_PIXEL_LIMIT)
    )
    camera_mesh_samples_per_axis = (
        int(args.camera_mesh_samples_per_axis)
        if args.camera_mesh_samples_per_axis is not None
        else (
            REALTIME_CAMERA_MESH_SAMPLES_PER_AXIS
            if use_realtime_fast_warp
            else DEFAULT_CAMERA_MESH_SAMPLES_PER_AXIS
        )
    )
    pipe_kwargs = {
        "prompt": prompt,
        "image": Image.open(image_path).convert("RGB"),
        "lora_path": lora_path,
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": num_frames,
        "generator": generator,
        "output_type": "np",
        "warp_history_downsample_mode": str(args.warp_history_downsample_mode),
        "is_amplify_first_chunk": bool(args.amplify_first_chunk),
        "camera_control_warp_render_mode": camera_warp_render_mode,
        "camera_control_pi3_pixel_limit": max(int(camera_pi3_pixel_limit), 1),
        "camera_control_mesh_samples_per_axis": max(int(camera_mesh_samples_per_axis), 1),
        "warp_debug_dir": args.warp_debug_dir,
        "warp_debug_fps": fps,
        "use_primary_fire_event_condition": bool(
            args.interaction_conditioning_mode == "binary"
            and args.use_primary_fire_event_condition
            and sample["primary_fire_event_path"] is not None
        ),
        "interaction_payload": interaction_payload if args.interaction_conditioning_mode == "router" else None,
        "interaction_conditioning_mode": str(args.interaction_conditioning_mode),
    }
    if args.pyramid_num_inference_steps_list is not None:
        pipe_kwargs["pyramid_num_inference_steps_list"] = list(args.pyramid_num_inference_steps_list)
    if conditioning_type == "warp_video":
        pipe_kwargs["warp_video"] = warp_video
        pipe_kwargs["warp_visibility_mask"] = warp_visibility_mask
    else:
        pipe_kwargs["camera_poses"] = camera_poses
    if (
        args.interaction_conditioning_mode == "binary"
        and bool(args.use_primary_fire_event_condition)
        and sample["primary_fire_event_path"] is not None
    ):
        latent_frames = ((int(num_frames) - 1) // int(pipe.vae_scale_factor_temporal)) + 1
        latent_height = int(args.height) // int(pipe.vae_scale_factor_spatial)
        latent_width = int(args.width) // int(pipe.vae_scale_factor_spatial)
        primary_fire_event_latents, event_mapping = build_primary_fire_event_latents(
            path=sample["primary_fire_event_path"],
            num_frames=int(num_frames),
            latent_channels=int(pipe.transformer.config.in_channels),
            latent_frames=int(latent_frames),
            latent_height=int(latent_height),
            latent_width=int(latent_width),
            temporal_scale=int(pipe.vae_scale_factor_temporal),
        )
        pipe_kwargs["primary_fire_event_latents"] = torch.from_numpy(primary_fire_event_latents).to(device=device, dtype=dtype)
        if args.warp_debug_dir is not None:
            debug_dir = Path(args.warp_debug_dir).expanduser().resolve()
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "primary_fire_event_latent_mapping.json").write_text(
                json.dumps({"mapping": event_mapping}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    result = pipe(**pipe_kwargs)
    frames = unwrap_video_frames(result)
    write_video(output, frames, fps=fps)
    print(
        json.dumps(
            {
                "event": "infer_done",
                "csv": str(csv_path),
                "conditioning_type": conditioning_type,
                "image": str(image_path),
                "output": str(output),
                "lora_path": lora_path,
                "warp_history_downsample_mode": str(args.warp_history_downsample_mode),
                "camera_realtime_fast_warp": bool(use_realtime_fast_warp),
                "camera_warp_render_mode": camera_warp_render_mode,
                "camera_pi3_pixel_limit": max(int(camera_pi3_pixel_limit), 1),
                "camera_mesh_samples_per_axis": max(int(camera_mesh_samples_per_axis), 1),
                "taehv_vae_mode": taehv_vae_mode,
                "taehv_checkpoint": str(taehv_checkpoint) if taehv_checkpoint is not None else None,
                "pyramid_num_inference_steps_list": args.pyramid_num_inference_steps_list,
                "amplify_first_chunk": bool(args.amplify_first_chunk),
                "frames": len(frames),
                "conditioning_frames": conditioning_frames,
                "num_frames": num_frames,
                "fps": fps,
                "use_primary_fire_event_condition": bool(
                    args.interaction_conditioning_mode == "binary"
                    and args.use_primary_fire_event_condition
                    and sample["primary_fire_event_path"] is not None
                ),
                "interaction_conditioning_mode": str(args.interaction_conditioning_mode),
                "interaction_payload": interaction_payload,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
