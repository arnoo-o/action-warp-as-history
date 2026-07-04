#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/root/autodl-tmp/warp_as_history_cs2_20260703}"
TRAIN_PID="${2:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"

LOG_DIR="${REPO_ROOT}/logs"
RUN_DIR="${REPO_ROOT}/runs/cs2_wah_lora"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_PATH="${LOG_DIR}/train_finish_report_${STAMP}.txt"

mkdir -p "${LOG_DIR}"

if [ -z "${TRAIN_PID}" ]; then
  TRAIN_PID="$(ps -eo pid,cmd | grep 'scripts/train_warp_as_history_lora.py' | grep -v grep | awk 'NR==1 {print $1}')"
fi

if [ -z "${TRAIN_PID}" ]; then
  echo "[monitor] no training pid found; refusing to arm auto-shutdown" > "${REPORT_PATH}"
  exit 1
fi

{
  echo "[monitor] started at $(date -Iseconds)"
  echo "[monitor] repo=${REPO_ROOT}"
  echo "[monitor] train_pid=${TRAIN_PID:-unknown}"
} > "${REPORT_PATH}"

if [ -n "${TRAIN_PID}" ]; then
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    echo "[monitor] waiting for pid ${TRAIN_PID} at $(date -Iseconds)" >> "${REPORT_PATH}"
    sleep "${POLL_SECONDS}"
  done
else
  echo "[monitor] no training pid detected, collecting current outputs immediately" >> "${REPORT_PATH}"
fi

LATEST_LOG="$(ls -1t "${LOG_DIR}"/cs2_wah_train_*.log 2>/dev/null | head -n 1 || true)"

{
  echo
  echo "[monitor] training finished at $(date -Iseconds)"
  echo "[monitor] latest_log=${LATEST_LOG:-none}"
  echo
  echo "[monitor] output files"
  ls -lah "${RUN_DIR}" 2>/dev/null || true
  echo
  echo "[monitor] latest checkpoints"
  find "${RUN_DIR}" -maxdepth 1 -type f \( -name 'visible_lora_state*.pt' -o -name 'train_*.json' \) | sort || true
  echo
  echo "[monitor] tail latest log"
  if [ -n "${LATEST_LOG}" ]; then
    tail -n 80 "${LATEST_LOG}" || true
  fi
} >> "${REPORT_PATH}"

python - <<PY >> "${REPORT_PATH}" 2>&1
from __future__ import annotations
import json
from pathlib import Path

run_dir = Path(r"${RUN_DIR}")
loss_path = run_dir / "train_loss.json"
config_path = run_dir / "train_config.json"

print()
print("[monitor] parsed summary")
if config_path.exists():
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        train_args = payload.get("train_args", {})
        print({
            "max_steps": train_args.get("max_steps"),
            "limit": train_args.get("limit"),
            "save_every": train_args.get("save_every"),
            "log_every": train_args.get("log_every"),
            "prompt_csv": train_args.get("prompt_csv"),
        })
    except Exception as exc:
        print(f"config_parse_error: {exc}")

if loss_path.exists():
    try:
        payload = json.loads(loss_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            print("loss_entries:", len(payload))
            print("last_record:", payload[-1])
        else:
            print("loss_entries: 0")
    except Exception as exc:
        print(f"loss_parse_error: {exc}")
else:
    print("loss file missing")
PY

echo "[monitor] powering off at $(date -Iseconds)" >> "${REPORT_PATH}"
/sbin/poweroff
