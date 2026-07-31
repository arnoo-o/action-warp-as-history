#!/usr/bin/env bash
set -euo pipefail

ROOT="${WAH_ROOT:-/ephemeral/mdu/action-warp-as-history}"
PYTHON="${WAH_PYTHON:-/ephemeral/mdu/venvs/wah/bin/python}"
GPU="${WAH_GPU:-1}"
MODE="${WAH_INTERACTION_MODE:-joint_pilot}"
STEPS="${WAH_STEPS:-300}"
RUN_NAME="${WAH_RUN_NAME:-mc_interaction_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="runs/${RUN_NAME}"
TRAIN_CSV="${WAH_TRAIN_CSV:-data/vpt_9x_100/wah_mc_training/teacher_pool/teacher_pool_review_manifest.csv}"
LOG_FILE="runs/${RUN_NAME}.log"
STATUS_FILE="runs/${RUN_NAME}.status"
PID_FILE="runs/${RUN_NAME}.pid"

cd "${ROOT}"
mkdir -p runs
if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Missing audited teacher pool: ${TRAIN_CSV}" >&2
  exit 2
fi
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
  --prompt_csv "${TRAIN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --training_profile interaction \
  --interaction_training_mode "${MODE}" \
  --base_train_steps "${STEPS}" \
  --camera_checkpoint runs/mc_camera_official_wah_c16cf2d_1000/visible_lora_state.pt \
  --interaction_conditioning_mode router \
  --interaction_lr 1e-4 \
  --warmup_steps 50 \
  --max_grad_norm 1.0 \
  --max_attempt_steps 15000 \
  --interaction_router_loss_scale 0.005 \
  --interaction_focus_scale 1.0 \
  --interaction_teacher_support_threshold 0.25 \
  --interaction_max_metadata_rotation_deg 20 \
  --interaction_max_camera_rotation_deg 20 \
  --interaction_min_mine_active_frames 4 \
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
  --flow_matching_stage_sampling fixed \
  --flow_matching_stage_id 0 \
  --save_every 150 \
  --interaction_debug_every 0 \
  --log_every 1 \
  --no-direction_augmentation \
  --no-enable_bidirectional_training \
  --tensorboard \
  --overwrite \
  >> "${LOG_FILE}" 2>&1
