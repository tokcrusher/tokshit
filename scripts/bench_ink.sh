#!/data/data/com.termux/files/usr/bin/bash
# Repeatable Inkling-2-Mini benchmark: lossless vs LOSSY fast profile.
# Usage: scripts/bench_ink.sh [runs] [tokens]
set -euo pipefail
ROOT=/data/data/com.termux/files/home/tokshit
MODEL=/data/data/com.termux/files/home/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf
RUNS="${1:-3}"
TOK="${2:-16}"
PROMPT="hi"

run_one() {
  local label="$1"; shift
  printf '%-10s ' "$label"
  env "$@" scripts/safe_run.sh build/tokshit --temperature 0 "$MODEL" "$PROMPT" "$TOK" 2>&1 \
    | grep -o '[0-9.]* tok/s' | tail -1
}

cd "$ROOT"
for i in $(seq 1 "$RUNS"); do
  printf 'lossless#%s ' "$i"
  env TOKSHIT_MIN_AVAIL_MB=1200 TOKSHIT_HOT=0 TOKSHIT_EXPERT_CACHE_MB=512 \
    TOKSHIT_PREFAULT=1 TOKSHIT_KV_CTX=512 TOKSHIT_POOL_WORKERS=4 \
    TOKSHIT_POOL_PIN=1 scripts/safe_run.sh build/tokshit --temperature 0 \
    "$MODEL" "$PROMPT" "$TOK" 2>&1 | grep -o '[0-9.]* tok/s' | tail -1
  printf 'lossy#%s    ' "$i"
  env TOKSHIT_MIN_AVAIL_MB=1200 TOKSHIT_HOT=0 TOKSHIT_EXPERT_CACHE_MB=512 \
    TOKSHIT_PREFAULT=1 TOKSHIT_KV_CTX=512 TOKSHIT_POOL_WORKERS=6 \
    TOKSHIT_POOL_PIN=1 TOKSHIT_DRAFT_TOPK=2 TOKSHIT_INK_PASSES=2 \
    TOKSHIT_HOT_Q4=1 scripts/safe_run.sh build/tokshit --temperature 0 \
    "$MODEL" "$PROMPT" "$TOK" 2>&1 | grep -o '[0-9.]* tok/s' | tail -1
done
