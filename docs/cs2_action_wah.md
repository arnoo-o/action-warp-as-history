# CS2 Action WAH Notes

## Camera Poses

- Use `scripts/build_cs2_camera_poses_from_actions.py` to convert CS2 parquet/action logs into `camera_poses.npz`.
- `jump` and `crouch` only affect `camera_poses`; they are not injected into prompts or extra training tokens.
- Output files:
  - `camera_poses.npz`
  - `debug_camera_motion.json`
  - `debug_trajectory.csv`
  - optional `debug_trajectory.png`

## Primary Fire Training

- Use `scripts/build_cs2_primary_fire_training_data.py` to build long clips around left-click rising edges.
- Training CSV now supports:
  - `primary_fire_event_path`
  - `primary_fire_loss_mask_path`
  - `warp_video_path`
  - `warp_visibility_mask_path`
- Use `scripts/build_primary_fire_training_masks.py` to generate:
  - `primary_fire_time_mask.npy`
  - `primary_fire_residual_mask.npy`
  - `primary_fire_loss_mask.npy`
  - debug residual/mask/overlay videos

## Training Flags

- `--use_primary_fire_focus_loss`
- `--primary_fire_focus_loss_scale`
- `--primary_fire_background_loss_scale`
- `--use_primary_fire_event_condition`
- `--online_primary_fire_window_probability`

If `primary_fire_loss_mask_path` is missing, training falls back to an online residual mask built from target vs warp frames.

## Inference Flags

- `scripts/infer_warp_as_history.py` supports `primary_fire_event_path` in the CSV.
- `--use_primary_fire_event_condition` enables event-latent conditioning at inference time.

## Leakage Policy

- Future GT frames are only used as targets or for loss-mask construction.
- Future GT content must not be injected into history latents.
- `online_future_keyframe_*` defaults are now zeroed and should be treated as deprecated.
- Image-memory / first-frame-memory branches are not part of the current CS2 primary-fire path.
