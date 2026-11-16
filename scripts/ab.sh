#!/usr/bin/env bash
# A/B run on the device: interleave configs, log temperature and memory.
# The phone throttles, so only interleaved comparisons in one session are fair.
#
#   bash scripts/ab.sh [tokens] [rounds]
#   TOKSHIT_AB_CFGS='4 6|8 6|3 3|2 3' bash scripts/ab.sh 12 2
#
# Every run goes through scripts/safe_run.sh. Do not run two models at once.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TOKSHIT_BIN:-$ROOT/build/tokshit}"
M="${TOKSHIT_AB_MODEL:-${TOKSHIT_MODEL:-$HOME/models/flint-UD-Q3_K_XL.gguf}}"
TOKENS="${1:-12}"
ROUNDS="${2:-2}"
CACHE_MB="${TOKSHIT_AB_CACHE:-512}"
CFGS="${TOKSHIT_AB_CFGS:-4 6|8 6|3 3|2 3}"
MIN_AVAIL="${TOKSHIT_MIN_AVAIL_MB:-1200}"
IFS='|' read -r -a CFG_LIST <<< "$CFGS"

[[ -x "$BIN" ]] || { echo "missing binary: $BIN" >&2; exit 1; }
[[ -f "$M" ]] || { echo "missing model: $M" >&2; exit 1; }
if pgrep -x tokshit >/dev/null 2>&1; then
  echo "another tokshit process is running; stop it first" >&2
  exit 1
fi
if ((CACHE_MB > 3072)); then
  echo "warning: cache ${CACHE_MB} MB is large; global AHB budget is 5 GiB incl. hot weights" >&2
fi

temp() {
  cat "${TEMP_ZONE:-/sys/class/thermal/thermal_zone0/temp}" 2>/dev/null || echo 0
}

mem_avail() {
  awk '/^MemAvailable:/ {print int($2 / 1024) " MB"}' /proc/meminfo
}

for round in $(seq 1 "$ROUNDS"); do
  for cfg in "${CFG_LIST[@]}"; do
    set -- $cfg
    io="${1:-4}"
    pool="${2:-6}"
    echo "=== round=$round io=$io pool=$pool temp=$(temp) mem=$(mem_avail) ==="
    env GGUF_TIME=1 TOKSHIT_MIN_AVAIL_MB="$MIN_AVAIL" timeout 600 \
      "$ROOT/scripts/safe_run.sh" "$BIN" \
      --io-threads "$io" --pool-workers "$pool" --cache-mb "$CACHE_MB" \
      "$M" "hi" "$TOKENS" 2>&1 |
      grep -E "tok/s|время|guard|read_bytes" || true
  done
done
