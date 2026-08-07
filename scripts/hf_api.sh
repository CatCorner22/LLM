#!/usr/bin/env bash
# Hugging Face API baseline — authenticated whoami + optional model search.
#
# Usage:
#   scripts/hf_api.sh --help
#   scripts/hf_api.sh whoami
#   scripts/hf_api.sh models --query "risk" --limit 5
#   scripts/hf_api.sh models --limit 10 | jq -r '.[].id' | scripts/hf_enrich_models.sh

set -euo pipefail

API="https://huggingface.co/api"
AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

usage() {
  cat <<'EOF'
Usage: hf_api.sh <command> [options]

Commands:
  whoami                 Print authenticated identity (JSON)
  models [--query Q] [--limit N] [--author ORG]
                         Search models (raw JSON array)
  datasets [--query Q] [--limit N]
                         Search datasets (raw JSON array)

Uses HF_TOKEN when set for higher rate limits and private access.
EOF
}

cmd="${1:-}"
shift || true

case "$cmd" in
  --help|-h|"") usage; [[ -n "$cmd" ]] || exit 1; exit 0 ;;
  whoami)
    curl -sS "${AUTH_HEADER[@]}" "${API}/whoami-v2"
    echo
    ;;
  models)
    QUERY=""; LIMIT=10; AUTHOR=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --query) QUERY="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --author) AUTHOR="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
      esac
    done
    URL="${API}/models?limit=${LIMIT}&full=true"
    [[ -n "$QUERY" ]] && URL+="&search=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$QUERY")"
    [[ -n "$AUTHOR" ]] && URL+="&author=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$AUTHOR")"
    curl -sS "${AUTH_HEADER[@]}" "$URL"
    echo
    ;;
  datasets)
    QUERY=""; LIMIT=10
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --query) QUERY="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
      esac
    done
    URL="${API}/datasets?limit=${LIMIT}"
    [[ -n "$QUERY" ]] && URL+="&search=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$QUERY")"
    curl -sS "${AUTH_HEADER[@]}" "$URL"
    echo
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
