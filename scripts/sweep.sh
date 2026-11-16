#!/usr/bin/env bash
# Quick config sweep on the device: speed, expert union, hit rate and
# temperature. Run from the repo root. Every run is guarded by safe_run.sh.
#
#   bash scripts/sweep.sh
#   TOKSHIT_MIN_AVAIL_MB=1400 bash scripts/sweep.sh
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M="${M:-${TOKSHIT_MODEL:-$HOME/models/flint-UD-Q3_K_XL.gguf}}"
N="${N:-8}"
CTX="${CTX:-512}"
BIN="${BIN:-$ROOT/build/tokshit}"
TEMP_ZONE="${TEMP_ZONE:-/sys/class/thermal/thermal_zone0/temp}"
MIN_AVAIL="${TOKSHIT_MIN_AVAIL_MB:-1200}"

[[ -x "$BIN" ]] || { echo "missing binary: $BIN" >&2; exit 1; }
[[ -f "$M" ]] || { echo "missing model: $M" >&2; exit 1; }
if pgrep -x tokshit >/dev/null 2>&1; then
  echo "another tokshit process is running; stop it first" >&2
  exit 1
fi

temp() { cat "$TEMP_ZONE" 2>/dev/null || echo 0; }

run() {
  local name=$1
  shift
  local t0 t1
  t0=$(temp)
  echo "=== $name | temp_before=$t0 ==="
  env TOKSHIT_MIN_AVAIL_MB="$MIN_AVAIL" timeout 600 \
    "$ROOT/scripts/safe_run.sh" "$BIN" "$@" --temperature 0 --ctx "$CTX" \
    --expert-stats "$M" "hi" "$N" 2>&1 |
    grep -E "tok/s|expert-stats|\[io:|prefill|guard" | tail -8 || true
  t1=$(temp)
  echo "--- $name temp_after=$t1"
}

# 1. plain top-6, the normal decode path
run "plain-top6" --cache-mb 512
# 2. lossy fast top-2 (page-cache experts)
run "fast-top2" --fast
# 3. speculative preset with reduced verify
run "spec-fast" --spec-fast
# 4. resident cache + pin frequent experts (full verify)
run "pin8-full" --cache-mb 1024 --pin-experts 8 --spec --spec-k 4 --spec-maxb 8
# 5. same, but verify top-2: the union does not explode
run "pin8-verify2" --cache-mb 1024 --pin-experts 8 --spec --spec-k 4 \
  --spec-maxb 8 --spec-verify-topk 2
# 6. buffered I/O + pin (page cache stays warm)
run "pin8-buffered" --cache-mb 1024 --o-direct=0 --pin-experts 8 --spec \
  --spec-k 4 --spec-maxb 8 --spec-verify-topk 2
