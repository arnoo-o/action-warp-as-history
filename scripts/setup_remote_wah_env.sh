#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SOURCE_CHECKPOINTS_DIR="${SOURCE_CHECKPOINTS_DIR:-/root/autodl-tmp/Warp-as-History/checkpoints}"
TARGET_CHECKPOINTS_LINK="${REPO_ROOT}/checkpoints"
SOURCE_PI3_DIR="${SOURCE_PI3_DIR:-/root/autodl-tmp/Warp-as-History/third_party/Pi3}"
TARGET_PI3_DIR="${REPO_ROOT}/third_party/Pi3"

echo "[setup] repo: ${REPO_ROOT}"
echo "[setup] python: ${PYTHON_BIN}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[setup] missing python binary: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -e "${TARGET_CHECKPOINTS_LINK}" ]; then
  if [ -d "${SOURCE_CHECKPOINTS_DIR}" ]; then
    ln -s "${SOURCE_CHECKPOINTS_DIR}" "${TARGET_CHECKPOINTS_LINK}"
    echo "[setup] linked checkpoints -> ${SOURCE_CHECKPOINTS_DIR}"
  else
    echo "[setup] source checkpoints directory not found: ${SOURCE_CHECKPOINTS_DIR}" >&2
    exit 1
  fi
fi

if [ ! -f "${TARGET_PI3_DIR}/pyproject.toml" ]; then
  if [ -d "${SOURCE_PI3_DIR}" ] && [ -f "${SOURCE_PI3_DIR}/pyproject.toml" ]; then
    mkdir -p "${TARGET_PI3_DIR}"
    cp -a "${SOURCE_PI3_DIR}/." "${TARGET_PI3_DIR}/"
    echo "[setup] restored Pi3 from ${SOURCE_PI3_DIR}"
  else
    echo "[setup] usable Pi3 source not found: ${SOURCE_PI3_DIR}" >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"
export PIP_PROGRESS_BAR=off
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -e .
"${PYTHON_BIN}" -m pip install -e third_party/Pi3

echo "[setup] done"
