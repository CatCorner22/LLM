#!/usr/bin/env bash
# Read model IDs from stdin, emit enriched NDJSON metadata per line.
#
# Usage:
#   printf '%s\n' Qwen/Qwen2.5-0.5B | scripts/hf_enrich_models.sh
#   scripts/hf_api.sh models --limit 5 | jq -r '.[].id' | scripts/hf_enrich_models.sh

set -euo pipefail

API="https://huggingface.co/api"
AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: hf_enrich_models.sh

Reads Hugging Face model IDs from stdin (one per line) and prints one JSON
object per line with id, downloads, likes, pipeline_tag, and tags.

Uses HF_TOKEN when set.
EOF
  exit 0
fi

while IFS= read -r model_id; do
  [[ -z "$model_id" ]] && continue
  encoded="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe="/"))' "$model_id")"
  curl -sS "${AUTH_HEADER[@]}" "${API}/models/${encoded}" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print(json.dumps({"id": None, "error": "invalid_json"}))
    raise SystemExit(0)
out = {
    "id": data.get("id") or data.get("modelId"),
    "downloads": data.get("downloads"),
    "likes": data.get("likes"),
    "pipeline_tag": data.get("pipeline_tag"),
    "tags": data.get("tags", [])[:12],
    "lastModified": data.get("lastModified"),
}
if "error" in data:
    out["error"] = data["error"]
print(json.dumps(out))
'
done
