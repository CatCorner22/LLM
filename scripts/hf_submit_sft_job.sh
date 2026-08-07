#!/usr/bin/env bash
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.24"]
# ///
#
# Submit a TRL SFT training job on Hugging Face Jobs.
#
# Usage:
#   scripts/hf_submit_sft_job.sh --help
#   scripts/hf_submit_sft_job.sh --hub-model-id USER/model --flavor a10g-large
#
# Requires: HF Pro/Team/Enterprise + HF_TOKEN with jobs + write scope.
# Prefer: hf jobs uv run (flags BEFORE script URL).

set -euo pipefail

FLAVOR="a10g-large"
TIMEOUT="2h"
HUB_MODEL_ID=""
MODEL="Qwen/Qwen2.5-0.5B"
DATASET="trl-lib/Capybara"
EXPERIMENT="pioneer-sft"
MAX_STEPS="100"
SCRIPT_URL=""

usage() {
  cat <<'EOF'
Usage: hf_submit_sft_job.sh [options]

Submit a Pioneer TRL SFT job to Hugging Face Jobs infrastructure.

Options:
  --hub-model-id ID   Hub repo to push adapters to (required)
  --model ID          Base model (default: Qwen/Qwen2.5-0.5B)
  --dataset ID        Dataset name (default: trl-lib/Capybara)
  --experiment NAME   Experiment / Trackio run name
  --max-steps N       Cap training steps for demos (default: 100)
  --flavor NAME       GPU flavor (default: a10g-large)
  --timeout DURATION  Job timeout (default: 2h)
  --script-url URL    UV script URL (default: uses inline train_sft_trl.py via Hub upload)
  --help              Show this help

Environment:
  HF_TOKEN            Required. Write + jobs scopes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hub-model-id) HUB_MODEL_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --script-url) SCRIPT_URL="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set" >&2
  exit 1
fi

if [[ -z "$HUB_MODEL_ID" ]]; then
  echo "ERROR: --hub-model-id is required (Jobs are ephemeral — must push to Hub)" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found. Install: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_SCRIPT="$ROOT/scripts/train_sft_trl.py"

if [[ -z "$SCRIPT_URL" ]]; then
  if [[ ! -f "$LOCAL_SCRIPT" ]]; then
    echo "ERROR: missing $LOCAL_SCRIPT" >&2
    exit 1
  fi
  USERNAME="$(curl -sS -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/api/whoami-v2 | python3 -c 'import sys,json; print(json.load(sys.stdin)["name"])')"
  REPO="${USERNAME}/pioneer-training-scripts"
  echo "Ensuring script repo ${REPO}..."
  hf repos create "$REPO" --type model --exist-ok 2>/dev/null || true
  hf upload "$REPO" "$LOCAL_SCRIPT" train_sft_trl.py
  SCRIPT_URL="https://huggingface.co/${REPO}/resolve/main/train_sft_trl.py"
fi

echo "Submitting HF Job..."
echo "  flavor=${FLAVOR} timeout=${TIMEOUT}"
echo "  script=${SCRIPT_URL}"
echo "  hub_model_id=${HUB_MODEL_ID}"

# CRITICAL: flags BEFORE script URL; use --secrets (plural)
hf jobs uv run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env HUB_MODEL_ID="$HUB_MODEL_ID" \
  --env MODEL_NAME="$MODEL" \
  --env DATASET_NAME="$DATASET" \
  --env EXPERIMENT_NAME="$EXPERIMENT" \
  --env MAX_STEPS="$MAX_STEPS" \
  "$SCRIPT_URL"

echo "Job submitted. Check status with: hf jobs ps"
echo "Logs: hf jobs logs <job-id>"
