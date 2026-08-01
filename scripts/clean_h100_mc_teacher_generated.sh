#!/usr/bin/env bash
set -euo pipefail

DATA_REPO_ROOT="${WAH_DATA_REPO_ROOT:-/ephemeral/mdu/action-warp-as-history}"
CODE_REPO_ROOT="${WAH_CODE_REPO_ROOT:-/ephemeral/mdu/action-warp-as-history-teacher-prep}"
targets=(
  "${DATA_REPO_ROOT}/data/vpt_9x_100/wah_mc_training/teacher_preparation"
  "${CODE_REPO_ROOT}/runs/mc_interaction_geometry_cache_281bc1a"
)
allowed=(
  "$(realpath -m -- "${targets[0]}")"
  "$(realpath -m -- "${targets[1]}")"
)

for index in "${!targets[@]}"; do
  target="$(realpath -m -- "${targets[$index]}")"
  if [[ -z "${target}" || "${target}" == "/" || "${target}" != "${allowed[$index]}" ]]; then
    printf 'ERROR: unsafe cleanup target: %q\n' "${target}" >&2
    exit 2
  fi
  printf 'cleanup target: %s\n' "${target}"
  if [[ -e "${target}" ]]; then
    du -sh -- "${target}"
    rm -rf -- "${target}"
    printf 'deleted: %s\n' "${target}"
  else
    printf 'not present: %s\n' "${target}"
  fi
done
