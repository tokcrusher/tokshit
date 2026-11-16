#!/usr/bin/env bash
# Read-only guard for GGUF model files.
#
#   scripts/model_guard.sh [model.gguf] [--hash]
#
# Fails if the file is missing, empty, or writable by the current user.
# --hash also prints the sha256 (reads the whole file, can take a while).
# This script NEVER opens the model for writing.
set -euo pipefail

MODEL="${1:-${TOKSHIT_MODEL:-$HOME/models/flint-UD-Q3_K_XL.gguf}}"
DO_HASH=0
for arg in "$@"; do
  [[ "$arg" == "--hash" ]] && DO_HASH=1
done

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

size="$(stat -c '%s' "$MODEL")"
mode="$(stat -c '%A' "$MODEL")"
[[ "$size" =~ ^[0-9]+$ ]] && ((size > 0)) || { echo "empty model: $MODEL" >&2; exit 1; }

printf 'model: %s\n' "$MODEL"
printf 'size:  %s bytes (%.2f GiB)\n' "$size" "$(awk -v s="$size" 'BEGIN{print s/1073741824}')"
printf 'mode:  %s\n' "$mode"

if [[ -w "$MODEL" ]]; then
  cat >&2 <<EOF
[guard] FAIL: the model is writable by this user.
[guard] A GGUF must never be modified. Run: chmod 444 "$MODEL"
EOF
  exit 2
fi

echo "[guard] read-only: ok (no write permission)"
if ((DO_HASH)); then
  echo "[guard] sha256 (read-only):"
  sha256sum "$MODEL"
fi
