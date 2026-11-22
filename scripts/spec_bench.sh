#!/data/data/com.termux/files/usr/bin/bash
# Honest spec/base benchmark for Inkling-2-Mini on a busy phone.
# Usage: scripts/spec_bench.sh [model] [draft] [prompt] [max_new]
set -u
MODEL="${1:-$HOME/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf}"
DRAFT="${2:-$HOME/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf}"
PROMPT="${3:-Write a Python function that reverses a string. Return only code.}"
MAXN="${4:-16}"
cd "$(dirname "$0")/.."

BASE=(TOKSHIT_MIN_AVAIL_MB=1200 TOKSHIT_HOT=0 TOKSHIT_EXPERT_CACHE_MB=512
      TOKSHIT_DROP_SIBLINGS=1 TOKSHIT_PREFAULT=0 TOKSHIT_KV_CTX=512
      TOKSHIT_PRINT_IDS=1)

echo "=== plain top8 ==="
env "${BASE[@]}" scripts/safe_run.sh build/tokshit --temperature 0 "$MODEL" \
  "$PROMPT" "$MAXN" 2>&1 | grep -E "\[prefill|generated|\[ids|guard" | head -4

echo "=== plain top2 ==="
env "${BASE[@]}" TOKSHIT_DRAFT_TOPK=2 scripts/safe_run.sh build/tokshit \
  --temperature 0 "$MODEL" "$PROMPT" "$MAXN" 2>&1 \
  | grep -E "\[prefill|generated|\[ids|guard" | head -4

echo "=== dspark sequential ==="
env "${BASE[@]}" TOKSHIT_DSPARK_BATCH=0 scripts/safe_run.sh build/tokshit \
  --temperature 0 --dspark "$DRAFT" "$MODEL" "$PROMPT" "$MAXN" 2>&1 \
  | grep -E "\[prefill|\[dspark|\[ids|guard" | head -4

echo "=== dspark batched ==="
env "${BASE[@]}" TOKSHIT_DSPARK_BATCH=1 scripts/safe_run.sh build/tokshit \
  --temperature 0 --dspark "$DRAFT" "$MODEL" "$PROMPT" "$MAXN" 2>&1 \
  | grep -E "\[prefill|\[dspark|\[ids|guard" | head -4
