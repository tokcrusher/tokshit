#!/data/data/com.termux/files/usr/bin/bash
# LOSSY but coherent profile for Inkling-2-Mini.
# top-4 experts + 3 core passes + original dense weights + 6 workers.
# 4/8 experts = quality loss, but the 3-pass schedule keeps text readable.
# Do NOT use top-2 or 2 passes: measured loops ("Capital capital capital...").
# Cold ~6.2-6.5 tok/s. Override TOKSHIT_POOL_WORKERS=4 for less heat.
set -euo pipefail

ROOT=/data/data/com.termux/files/home/tokshit
MODEL=/data/data/com.termux/files/home/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf
BIN="$ROOT/build/tokshit"
GUARD="$ROOT/scripts/safe_run.sh"

echo "LOSSY: top-4 experts + 3 passes + lossless dense + 6 workers" >&2

cd "$ROOT"
export TOKSHIT_MIN_AVAIL_MB="${TOKSHIT_MIN_AVAIL_MB:-1200}"
export TOKSHIT_HOT=0
export TOKSHIT_EXPERT_CACHE_MB="${TOKSHIT_EXPERT_CACHE_MB:-512}"
export TOKSHIT_PREFAULT=1
export TOKSHIT_KV_CTX="${TOKSHIT_KV_CTX:-512}"
export TOKSHIT_POOL_WORKERS="${TOKSHIT_POOL_WORKERS:-6}"
export TOKSHIT_POOL_PIN=1
export TOKSHIT_DRAFT_TOPK=4
export TOKSHIT_TEMPERATURE="${TOKSHIT_TEMPERATURE:-0}"

exec "$GUARD" "$BIN" --temperature "$TOKSHIT_TEMPERATURE" "$MODEL" "$@"
