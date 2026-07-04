#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

PROMPT_CSV="${PROMPT_CSV:-data/cs2_example/CS2-10k-sample/sample/data/85bf6db3-f7ee-578e-8ec4-3cab1930348c/cleaned/cs2_training.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/cs2_wah_lora}"
LOG_DIR="${LOG_DIR:-logs}"
LIMIT="${LIMIT:-129}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-1}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-640}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ONLINE_FRAME_STRIDE="${ONLINE_FRAME_STRIDE:-8}"
ONLINE_MAX_VIDEO_FRAMES="${ONLINE_MAX_VIDEO_FRAMES:-96}"
ONLINE_WARP_MEMORY_CACHE_SIZE="${ONLINE_WARP_MEMORY_CACHE_SIZE:-2}"
ONLINE_WARP_DISK_CACHE_DIR="${ONLINE_WARP_DISK_CACHE_DIR-auto}"
ONLINE_MAX_HISTORY_FRAMES="${ONLINE_MAX_HISTORY_FRAMES:-19}"
ONLINE_INTERACTION_PSEUDO_HISTORY_SCALE="${ONLINE_INTERACTION_PSEUDO_HISTORY_SCALE:-0.035}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-1e-4}"
WARP_HISTORY_DOWNSAMPLE_MODE="${WARP_HISTORY_DOWNSAMPLE_MODE:-short}"
WAH_ENABLE_OPTIONAL_ATTENTION="${WAH_ENABLE_OPTIONAL_ATTENTION:-1}"

mkdir -p "${REPO_ROOT}/${LOG_DIR}"
mkdir -p "${REPO_ROOT}/${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${REPO_ROOT}/${LOG_DIR}/cs2_wah_train_${TIMESTAMP}.log"

echo "[train] repo=${REPO_ROOT}"
echo "[train] python=${PYTHON_BIN}"
echo "[train] prompt_csv=${PROMPT_CSV}"
echo "[train] output_dir=${OUTPUT_DIR}"
echo "[train] log_path=${LOG_PATH}"
echo "[train] limit=${LIMIT} max_steps=${MAX_STEPS} frame_stride=${ONLINE_FRAME_STRIDE} max_video_frames=${ONLINE_MAX_VIDEO_FRAMES}"
echo "[train] online_warp_memory_cache_size=${ONLINE_WARP_MEMORY_CACHE_SIZE} online_warp_disk_cache_dir=${ONLINE_WARP_DISK_CACHE_DIR}"
echo "[train] lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} lr=${LR} pseudo_scale=${ONLINE_INTERACTION_PSEUDO_HISTORY_SCALE}"
echo "[train] warp_history_downsample_mode=${WARP_HISTORY_DOWNSAMPLE_MODE} optional_attention=${WAH_ENABLE_OPTIONAL_ATTENTION}"

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WAH_ENABLE_OPTIONAL_ATTENTION

CMD=(
  "${PYTHON_BIN}" scripts/train_warp_as_history_lora.py
  --base_model_path checkpoints/helios-distilled
  --transformer_path checkpoints/helios-distilled
  --data_root .
  --prompt_csv "${PROMPT_CSV}"
  --output_dir "${OUTPUT_DIR}"
  --limit "${LIMIT}"
  --max_steps "${MAX_STEPS}"
  --lr "${LR}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --online_frame_stride "${ONLINE_FRAME_STRIDE}"
  --online_max_video_frames "${ONLINE_MAX_VIDEO_FRAMES}"
  --online_warp_memory_cache_size "${ONLINE_WARP_MEMORY_CACHE_SIZE}"
  --online_warp_disk_cache_dir "${ONLINE_WARP_DISK_CACHE_DIR}"
  --online_max_history_frames "${ONLINE_MAX_HISTORY_FRAMES}"
  --online_interaction_pseudo_history_scale "${ONLINE_INTERACTION_PSEUDO_HISTORY_SCALE}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --warp_history_downsample_mode "${WARP_HISTORY_DOWNSAMPLE_MODE}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
  --overwrite
)

printf '[train] command='
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
