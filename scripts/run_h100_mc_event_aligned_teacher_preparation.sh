#!/usr/bin/env bash
set -euo pipefail

ROOT="${WAH_ROOT:-/ephemeral/mdu/action-warp-as-history}"
PYTHON="${WAH_PYTHON:-/ephemeral/mdu/venvs/wah/bin/python}"
GPU="${WAH_GPU:-1}"
DATA_ROOT="${WAH_DATA_ROOT:-data/vpt_9x_100/wah_mc_training}"
CAMERA_CHECKPOINT="${WAH_CAMERA_CHECKPOINT:-runs/mc_camera_official_wah_c16cf2d_1000/visible_lora_state.pt}"
WORKDIR="${WAH_TEACHER_WORKDIR:-${DATA_ROOT}/teacher_event_aligned_v1}"
CANDIDATE_LIMIT="${WAH_CANDIDATE_LIMIT:-32}"

fail_missing() { printf 'ERROR: missing required %s: %s\n' "$1" "$2" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || fail_missing executable "${PYTHON}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -d "${DATA_ROOT}" ]] || fail_missing directory "${DATA_ROOT}"
[[ -f "${DATA_ROOT}/mc_interaction_training_samples.csv" ]] || fail_missing file "${DATA_ROOT}/mc_interaction_training_samples.csv"
[[ -f "${CAMERA_CHECKPOINT}" ]] || fail_missing file "${CAMERA_CHECKPOINT}"
[[ -d checkpoints/helios-distilled ]] || fail_missing directory "${ROOT}/checkpoints/helios-distilled"
mkdir -p "${WORKDIR}/candidates" "${WORKDIR}/teacher_pool" "${WORKDIR}/candidate_export_run" "${WORKDIR}/geometry_cache"

printf '[1/2] Exporting %s event-aligned candidates; no optimizer is created\n' "${CANDIDATE_LIMIT}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" scripts/train_warp_as_history_lora.py \
  --base_model_path checkpoints/helios-distilled \
  --transformer_path checkpoints/helios-distilled \
  --data_root . \
  --prompt_csv "${DATA_ROOT}/mc_interaction_training_samples.csv" \
  --output_dir "${WORKDIR}/candidate_export_run" \
  --training_profile interaction \
  --interaction_training_mode joint_stage0 \
  --export_teacher_candidates_only \
  --event_aligned_interaction \
  --teacher_candidate_limit "${CANDIDATE_LIMIT}" \
  --teacher_candidate_output_dir "${WORKDIR}/candidates" \
  --camera_checkpoint "${CAMERA_CHECKPOINT}" \
  --interaction_conditioning_mode router \
  --interaction_router_loss_scale 0.005 \
  --interaction_teacher_support_threshold 0.25 \
  --height 384 --width 640 --num_frames 33 \
  --online_frame_stride 1 --online_target_fps 16 \
  --online_use_vpt_camera_poses --minecraft_training_profile --use_minecraft_hud_mask \
  --online_geometry_keyframe_stride 8 --online_max_history_frames 19 \
  --online_warp_memory_cache_size 2 \
  --online_warp_disk_cache_dir "${WORKDIR}/geometry_cache" \
  --online_render_mode target_fill \
  --warp_history_downsample_mode short \
  --history_positioning last_n_same_order \
  --prompt_cache_dir runs/mc_prompt_cache \
  --flow_matching_stage_sampling fixed --flow_matching_stage_id 0 \
  --interaction_debug_every 0 --no-direction_augmentation --no-enable_bidirectional_training

[[ -f "${WORKDIR}/candidates/teacher_candidate_manifest.csv" ]] || fail_missing file "${WORKDIR}/candidates/teacher_candidate_manifest.csv"

printf '[2/2] Building dual-residual region teachers and read-only review material\n'
"${PYTHON}" scripts/build_minecraft_teacher_pool.py \
  --candidate_manifest "${WORKDIR}/candidates/teacher_candidate_manifest.csv" \
  --output_dir "${WORKDIR}/teacher_pool" \
  --review_manifest_name teacher_region_review_manifest.csv \
  --support_threshold 0.25

[[ -f "${WORKDIR}/teacher_pool/teacher_region_review_manifest.csv" ]] || fail_missing file "${WORKDIR}/teacher_pool/teacher_region_review_manifest.csv"
[[ -f "${WORKDIR}/teacher_pool/review_index.html" ]] || fail_missing file "${WORKDIR}/teacher_pool/review_index.html"
printf 'STOP: review artifacts are ready; no training was started.\n'
printf 'Review index: %s\n' "${WORKDIR}/teacher_pool/review_index.html"
printf 'Review manifest: %s\n' "${WORKDIR}/teacher_pool/teacher_region_review_manifest.csv"
