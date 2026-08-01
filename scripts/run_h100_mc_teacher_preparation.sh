#!/usr/bin/env bash
set -euo pipefail

ROOT="${WAH_ROOT:-/ephemeral/mdu/action-warp-as-history}"
PYTHON="${WAH_PYTHON:-/ephemeral/mdu/venvs/wah/bin/python}"
GPU="${WAH_GPU:-1}"
DATA_ROOT="${WAH_DATA_ROOT:-data/vpt_9x_100/wah_mc_training}"
CAMERA_CHECKPOINT="${WAH_CAMERA_CHECKPOINT:-runs/mc_camera_official_wah_c16cf2d_1000/visible_lora_state.pt}"
WORKDIR="${WAH_TEACHER_WORKDIR:-data/vpt_9x_100/wah_mc_training/teacher_preparation}"
CANDIDATE_LIMIT="${WAH_CANDIDATE_LIMIT:-0}"

fail_missing() {
  printf 'ERROR: missing required %s: %s\n' "$1" "$2" >&2
  exit 2
}

require_file() {
  [[ -f "$1" ]] || fail_missing file "$1"
}

require_dir() {
  [[ -d "$1" ]] || fail_missing directory "$1"
}

require_executable() {
  [[ -x "$1" ]] || fail_missing executable "$1"
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
require_executable "${PYTHON}"
require_dir "${DATA_ROOT}"
require_file "${DATA_ROOT}/mc_training_samples.csv"
require_file "${DATA_ROOT}/mc_long_segments.csv"
require_file "${CAMERA_CHECKPOINT}"
require_dir "checkpoints/helios-distilled"
mkdir -p "${WORKDIR}/candidates" "${WORKDIR}/teacher_pool" "${WORKDIR}/candidate_export_run"

printf '[1/4] Building action pools\n'
"${PYTHON}" scripts/build_minecraft_interaction_manifest.py \
  --data_root "${DATA_ROOT}"

for required in \
  mc_interaction_training_samples.csv \
  place_pool.csv \
  mine_active_pool.csv \
  mine_complete_pool.csv \
  real_negative_pool.csv \
  mc_interaction_manifest_audit.json
do
  require_file "${DATA_ROOT}/${required}"
done

CANDIDATE_LIMIT_ARGS=()
if [[ "${CANDIDATE_LIMIT}" != "0" ]]; then
  CANDIDATE_LIMIT_ARGS+=(--teacher_candidate_limit "${CANDIDATE_LIMIT}")
fi

printf '[2/4] Exporting fixed teacher candidates (no optimizer or training)\n'
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" scripts/train_warp_as_history_lora.py \
  --base_model_path checkpoints/helios-distilled \
  --transformer_path checkpoints/helios-distilled \
  --data_root . \
  --prompt_csv "${DATA_ROOT}/mc_interaction_training_samples.csv" \
  --output_dir "${WORKDIR}/candidate_export_run" \
  --training_profile interaction \
  --interaction_training_mode joint_stage0 \
  --export_teacher_candidates_only \
  --teacher_candidate_output_dir "${WORKDIR}/candidates" \
  --event_aligned_interaction \
  --camera_checkpoint "${CAMERA_CHECKPOINT}" \
  --interaction_conditioning_mode router \
  --interaction_lr 1e-4 \
  --router_lr 5e-5 \
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
  --interaction_debug_every 0 \
  --no-direction_augmentation \
  --no-enable_bidirectional_training \
  "${CANDIDATE_LIMIT_ARGS[@]}"

require_file "${WORKDIR}/candidates/teacher_candidate_manifest.csv"
require_file "${WORKDIR}/candidates/teacher_candidate_audit.json"

printf '[3/4] Building fixed teacher pool and review materials\n'
"${PYTHON}" scripts/build_minecraft_teacher_pool.py \
  --candidate_manifest "${WORKDIR}/candidates/teacher_candidate_manifest.csv" \
  --output_dir "${WORKDIR}/teacher_pool" \
  --support_threshold 0.25

require_file "${WORKDIR}/teacher_pool/teacher_pool_review_manifest.csv"
require_file "${WORKDIR}/teacher_pool/teacher_pool_audit.json"

printf '[4/4] Verifying read-only review index\n'
require_file "${WORKDIR}/teacher_pool/review_index.html"
printf 'Teacher preparation complete. No training was started.\n'
printf 'Review index: %s\n' "${WORKDIR}/teacher_pool/review_index.html"
printf 'Review manifest: %s\n' "${WORKDIR}/teacher_pool/teacher_pool_review_manifest.csv"
