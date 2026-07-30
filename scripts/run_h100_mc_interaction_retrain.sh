#!/usr/bin/env bash
set -euo pipefail

ROOT="${WAH_ROOT:-/ephemeral/mdu/action-warp-as-history}"
PYTHON="${WAH_PYTHON:-/ephemeral/mdu/venvs/wah/bin/python}"
GPU="${WAH_GPU:-1}"
RUN_NAME="${WAH_RUN_NAME:-mc_interaction_stage0_281bc1a_1500}"
OUTPUT_DIR="runs/${RUN_NAME}"
LOG_FILE="runs/${RUN_NAME}.log"
STATUS_FILE="runs/${RUN_NAME}.status"
PID_FILE="runs/${RUN_NAME}.pid"

cd "${ROOT}"
mkdir -p runs
echo "$$" > "${PID_FILE}"
printf 'starting %s\n' "$(date -Is)" > "${STATUS_FILE}"

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    printf 'complete %s\n' "$(date -Is)" > "${STATUS_FILE}"
  else
    printf 'failed exit=%s %s\n' "${exit_code}" "$(date -Is)" > "${STATUS_FILE}"
  fi
}
trap on_exit EXIT

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" scripts/train_warp_as_history_lora.py \
  --base_model_path checkpoints/helios-distilled \
  --transformer_path checkpoints/helios-distilled \
  --data_root . \
  --prompt_csv data/vpt_9x_100/wah_mc_training/mc_interaction_training_samples.csv \
  --output_dir "${OUTPUT_DIR}" \
  --training_profile interaction \
  --base_train_steps 1500 \
  --camera_checkpoint runs/mc_camera_official_wah_c16cf2d_1000/visible_lora_state.pt \
  --interaction_conditioning_mode router \
  --interaction_lr 1e-4 \
  --warmup_steps 50 \
  --max_grad_norm 1.0 \
  --max_attempt_steps 15000 \
  --interaction_router_loss_scale 0.05 \
  --interaction_focus_scale 1.0 \
  --interaction_teacher_support_threshold 0.25 \
  --interaction_event_local_min 6 \
  --interaction_event_local_max 16 \
  --height 384 \
  --width 640 \
  --num_frames 33 \
  --online_frame_stride 1 \
  --online_target_fps 16 \
  --online_use_vpt_camera_poses \
  --minecraft_training_profile \
  --use_minecraft_hud_mask \
  --online_geometry_keyframe_stride 8 \
  --online_max_history_frames 19 \
  --online_warp_memory_cache_size 2 \
  --online_warp_disk_cache_dir runs/mc_interaction_geometry_cache_281bc1a \
  --online_render_mode target_fill \
  --warp_history_downsample_mode short \
  --history_positioning last_n_same_order \
  --prompt_cache_dir runs/mc_prompt_cache \
  --flow_matching_stage_sampling all \
  --save_steps 300 500 750 1000 1500 \
  --save_every 0 \
  --interaction_debug_every 100 \
  --log_every 1 \
  --no-direction_augmentation \
  --no-enable_bidirectional_training \
  --tensorboard \
  --overwrite \
  >> "${LOG_FILE}" 2>&1
