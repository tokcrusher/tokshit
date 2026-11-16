#!/usr/bin/env bash
# Safe benchmark wrapper.
#
#   ./bench.sh "<flags>" [max_new] [runs]
#   ./bench.sh "--cache-mb 512" 8 2
#   ./bench.sh "--spec --spec-k 3 --spec-verify-topk 2 --cache-mb 512" 8 2
#
# Every run goes through scripts/safe_run.sh, which kills the process if
# MemAvailable falls below TOKSHIT_MIN_AVAIL_MB (default 1200). Do not run two
# model processes at once.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${TOKSHIT_BIN:-$ROOT/build/tokshit}"
MODEL="${TOKSHIT_MODEL:-$HOME/models/flint-UD-Q3_K_XL.gguf}"
FLAGS="${1:-}"
N="${2:-8}"
RUNS="${3:-2}"
MIN_AVAIL="${TOKSHIT_MIN_AVAIL_MB:-1200}"
RUN_TIMEOUT="${TOKSHIT_RUN_TIMEOUT:-900}"

[[ -x "$BIN" ]] || { echo "missing binary: $BIN (run ./install.sh or build first)" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL (set TOKSHIT_MODEL)" >&2; exit 1; }
[[ -x "$ROOT/scripts/safe_run.sh" ]] || { echo "missing scripts/safe_run.sh" >&2; exit 1; }

echo "== model: $MODEL"
echo "== flags: ${FLAGS:-<default>}"
echo "== tokens: $N, runs: $RUNS, min MemAvailable: ${MIN_AVAIL} MB"

for i in $(seq 1 "$RUNS"); do
  echo "--- run $i"
  out="$(TOKSHIT_MIN_AVAIL_MB="$MIN_AVAIL" timeout "$RUN_TIMEOUT" \
    "$ROOT/scripts/safe_run.sh" "$BIN" $FLAGS --temperature 0 \
    "$MODEL" "привет" "$N" 2>&1)" || echo "  (run exited non-zero)"

  rate="$(printf '%s\n' "$out" | grep -oE '= [0-9.]+ tok/s' | tail -1 | tr -dc '0-9.')"
  hit="$(printf '%s\n' "$out" | grep -oE 'rate=[0-9.]+%' | tail -1 || true)"
  printf '  gen: %s tok/s  %s\n' "${rate:-?}" "${hit:-}"
  printf '%s\n' "$out" | grep -E '^\[(время|matvec|guard|spec):' | sed 's/^/  /' || true
done
