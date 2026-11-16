#!/usr/bin/env bash
# Compatibility wrapper around scripts/bench_matrix.sh.
#
#   ./bench.sh "<flags>" [max_new] [runs]
#   ./bench.sh "--cache-mb 512" 24 3
#
# The full harness adds JSON/CSV, temperature/memory logging, interleaved A/B,
# prefill mode and the regression gate. See docs/BENCH.ru.md. Every run goes
# through scripts/safe_run.sh and the device lock.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "$ROOT/scripts/bench_matrix.sh" --help
fi

FLAGS="${1:-}"
N="${2:-8}"
RUNS="${3:-2}"

exec "$ROOT/scripts/bench_matrix.sh" \
  --mode decode --flags "$FLAGS" --tokens "$N" --runs "$RUNS" \
  --cooldown "${TOKSHIT_BENCH_COOLDOWN:-5}"
