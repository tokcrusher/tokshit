#!/data/data/com.termux/files/usr/bin/bash
# Robust launcher for Inkling-2-Mini on the worktree/main build.
# Usage:
#   scripts/run_inkling2.sh                 # interactive chat
#   scripts/run_inkling2.sh "привет" 16     # one answer, 16 tokens
# Env overrides: TOKSHIT_TEMPERATURE, TOKSHIT_KV_CTX, TOKSHIT_POOL_WORKERS, ...
set -euo pipefail

ROOT=/data/data/com.termux/files/home/tokshit
MODEL=/data/data/com.termux/files/home/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf
BIN="$ROOT/build/tokshit"
GUARD="$ROOT/scripts/safe_run.sh"

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN (run: cmake --build $ROOT/build -j8)" >&2
  exit 1
fi

cd "$ROOT"
export TOKSHIT_MIN_AVAIL_MB="${TOKSHIT_MIN_AVAIL_MB:-1200}"
export TOKSHIT_HOT="${TOKSHIT_HOT:-0}"
export TOKSHIT_EXPERT_CACHE_MB="${TOKSHIT_EXPERT_CACHE_MB:-512}"
export TOKSHIT_PREFAULT="${TOKSHIT_PREFAULT:-1}"
export TOKSHIT_POOL_WORKERS="${TOKSHIT_POOL_WORKERS:-4}"
export TOKSHIT_POOL_PIN="${TOKSHIT_POOL_PIN:-1}"
export TOKSHIT_TEMPERATURE="${TOKSHIT_TEMPERATURE:-0}"

exec "$GUARD" "$BIN" --temperature "$TOKSHIT_TEMPERATURE" "$MODEL" "$@"
