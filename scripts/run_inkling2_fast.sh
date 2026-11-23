#!/data/data/com.termux/files/usr/bin/bash
# LOSSY speed profile for Inkling-2-Mini.
# top-2 experts + 2 core passes + Q4_0 dense + 6 workers.
# Measured ~10.5 tok/s burst / 9.2-10.5 short runs; quality is degraded.
set -euo pipefail

ROOT=/data/data/com.termux/files/home/tokshit
MODEL=/data/data/com.termux/files/home/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf
BIN="$ROOT/build/tokshit"
GUARD="$ROOT/scripts/safe_run.sh"

echo "LOSSY: top-2 experts + 2 passes + Q4_0 dense + 6 workers" >&2

cd "$ROOT"
export TOKSHIT_MIN_AVAIL_MB="${TOKSHIT_MIN_AVAIL_MB:-1200}"
export TOKSHIT_HOT=0
export TOKSHIT_EXPERT_CACHE_MB="${TOKSHIT_EXPERT_CACHE_MB:-512}"
export TOKSHIT_PREFAULT=1
export TOKSHIT_KV_CTX="${TOKSHIT_KV_CTX:-512}"
export TOKSHIT_POOL_WORKERS=6
export TOKSHIT_POOL_PIN=1
export TOKSHIT_DRAFT_TOPK=2
export TOKSHIT_INK_PASSES=2
export TOKSHIT_HOT_Q4=1
export TOKSHIT_TEMPERATURE="${TOKSHIT_TEMPERATURE:-0}"

exec "$GUARD" "$BIN" --temperature "$TOKSHIT_TEMPERATURE" "$MODEL" "$@"
