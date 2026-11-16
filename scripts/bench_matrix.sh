#!/usr/bin/env bash
# bench_matrix.sh — воспроизводимый decode/prefill/A-B харнесс для tokshit.
#
# Каждый прогон идёт через scripts/safe_run.sh под общим device-lock, с
# таймаутом, проверкой pgrep и MemAvailable. Конфиги в A/B чередуются
# round-robin со сдвигом на каждом раунде, поэтому троттлинг не перекашивает
# один режим. На выходе — человекочитаемая таблица, JSON и CSV.
#
#   scripts/bench_matrix.sh --mode decode --runs 3 --tokens 8
#   scripts/bench_matrix.sh --mode prefill --prompt-tokens 128,512
#   scripts/bench_matrix.sh --mode both \
#     --config 'base|--cache-mb 512' \
#     --config 'pin|--cache-mb 512 --pin-experts 8' --gate
#
# Первый --config — reference для A/B и цель regression gate.
# Полная документация: docs/BENCH.ru.md.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TOKSHIT_BIN:-$ROOT/build/tokshit}"
MODEL="${TOKSHIT_MODEL:-$HOME/models/flint-UD-Q3_K_XL.gguf}"
SAFE_RUN="$ROOT/scripts/safe_run.sh"
MODEL_GUARD="$ROOT/scripts/model_guard.sh"
TEMP_ZONE="${TOKSHIT_TEMP_ZONE:-/sys/class/thermal/thermal_zone0/temp}"

MODE="${TOKSHIT_BENCH_MODE:-decode}"
RUNS="${TOKSHIT_BENCH_RUNS:-3}"
TOKENS="${TOKSHIT_BENCH_TOKENS:-24}"
COOLDOWN="${TOKSHIT_BENCH_COOLDOWN:-15}"
CACHE_MB="${TOKSHIT_BENCH_CACHE_MB:-512}"
IO_THREADS="${TOKSHIT_BENCH_IO_THREADS:-4}"
POOL_WORKERS="${TOKSHIT_BENCH_POOL_WORKERS:-6}"
EXTRA_FLAGS="${TOKSHIT_BENCH_FLAGS:-}"
PROMPT="${TOKSHIT_BENCH_PROMPT:-hi}"
PROMPT_TOKENS="${TOKSHIT_BENCH_PROMPT_TOKENS:-128}"
MIN_AVAIL_MB="${TOKSHIT_MIN_AVAIL_MB:-1200}"
MIN_START_MB="${TOKSHIT_BENCH_MIN_START_MB:-4000}"
RUN_TIMEOUT="${TOKSHIT_RUN_TIMEOUT:-1200}"
LOCK_WAIT="${TOKSHIT_DEVICE_LOCK_WAIT:-3600}"
BUSY_WAIT="${TOKSHIT_BENCH_BUSY_WAIT:-120}"
MEM_WAIT="${TOKSHIT_BENCH_MEM_WAIT:-120}"
OUT_DIR="${TOKSHIT_BENCH_OUT:-${TMPDIR:-/tmp}/tokshit-bench}"
STAMP="${TOKSHIT_BENCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
JSON_OUT=""
CSV_OUT=""
SAVE_LOGS=1
GATE=0
GATE_PCT="${TOKSHIT_GATE_PCT:-10}"
BASE_DECODE="${TOKSHIT_BASELINE_DECODE:-1.1}"
BASE_DENSE="${TOKSHIT_BASELINE_DENSE:-12.55}"
BASE_HIT="${TOKSHIT_BASELINE_HIT:-27.5}"
CONFIGS=()

usage() {
  cat <<'EOF'
usage: scripts/bench_matrix.sh [options]

  --mode decode|prefill|both   что мерить (default: decode)
  --runs N                      повторов на конфиг (default: 3)
  --tokens N                    новых токенов в decode-прогоне (default: 24)
  --cooldown SEC                пауза между прогонами (default: 15)
  --cache-mb N                  кэш экспертов (default: 512, max 4096)
  --io-threads N                потоков чтения экспертов (default: 4)
  --pool-workers N              потоков пула (default: 6)
  --flags "..."                 общие дополнительные флаги
  --config 'name|flags'         конфиг для A/B; можно повторять.
                                первый конфиг — reference/gate.
  --prompt TEXT                 короткий промпт для decode (default: hi)
  --prompt-tokens N[,N...]      целевые длины prefill-промптов (default: 128)
  --model FILE                  .gguf (default: $TOKSHIT_MODEL)
  --bin FILE                    бинарь (default: build/tokshit)
  --timeout SEC                 timeout на один прогон (default: 1200)
  --min-start-mb N              требовать MemAvailable >= N перед прогоном
                                (default: 4000)
  --min-avail-mb N              порог safe_run watchdog (default: 1200)
  --out DIR                     каталог для JSON/CSV/логов
  --json FILE                   явный путь JSON
  --csv FILE                    явный путь CSV
  --no-logs                     не сохранять сырые логи прогонов
  --gate                        exit 3 при регрессе > GATE_PCT к baseline
  --gate-pct N                  порог регресса, % (default: 10)
  --baseline-decode N           baseline decode tok/s (default: 1.1)
  --baseline-dense N            baseline dense GB/s (default: 12.55)
  --baseline-hit N              baseline hit-rate % (default: 27.5)
  --lock-wait SEC               ожидание device-lock (default: 3600)
  --busy-wait SEC               ожидание чужого tokshit (default: 120)
  --mem-wait SEC                ожидание MemAvailable (default: 120)
  -h, --help                    эта справка

Exit codes: 0 ok, 2 usage, 3 gate failed, 4 unsafe environment,
            75 lock timeout, 124 model timeout (per-run, matrix continues).
EOF
}

die() { echo "bench_matrix: $*" >&2; exit "${BENCH_EXIT:-2}"; }

need_num() {
  [[ "${2:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$1 expects a number"
}
need_int() {
  [[ "${2:-}" =~ ^[0-9]+$ ]] || die "$1 expects an integer"
}

while (( $# )); do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --runs) RUNS="${2:-}"; shift 2 ;;
    --runs=*) RUNS="${1#*=}"; shift ;;
    --tokens) TOKENS="${2:-}"; shift 2 ;;
    --tokens=*) TOKENS="${1#*=}"; shift ;;
    --cooldown) COOLDOWN="${2:-}"; shift 2 ;;
    --cooldown=*) COOLDOWN="${1#*=}"; shift ;;
    --cache-mb) CACHE_MB="${2:-}"; shift 2 ;;
    --cache-mb=*) CACHE_MB="${1#*=}"; shift ;;
    --io-threads) IO_THREADS="${2:-}"; shift 2 ;;
    --io-threads=*) IO_THREADS="${1#*=}"; shift ;;
    --pool-workers) POOL_WORKERS="${2:-}"; shift 2 ;;
    --pool-workers=*) POOL_WORKERS="${1#*=}"; shift ;;
    --flags) EXTRA_FLAGS="${2:-}"; shift 2 ;;
    --flags=*) EXTRA_FLAGS="${1#*=}"; shift ;;
    --config) CONFIGS+=("${2:-}"); shift 2 ;;
    --config=*) CONFIGS+=("${1#*=}"); shift ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --prompt=*) PROMPT="${1#*=}"; shift ;;
    --prompt-tokens) PROMPT_TOKENS="${2:-}"; shift 2 ;;
    --prompt-tokens=*) PROMPT_TOKENS="${1#*=}"; shift ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --bin) BIN="${2:-}"; shift 2 ;;
    --bin=*) BIN="${1#*=}"; shift ;;
    --timeout) RUN_TIMEOUT="${2:-}"; shift 2 ;;
    --timeout=*) RUN_TIMEOUT="${1#*=}"; shift ;;
    --min-start-mb) MIN_START_MB="${2:-}"; shift 2 ;;
    --min-start-mb=*) MIN_START_MB="${1#*=}"; shift ;;
    --min-avail-mb) MIN_AVAIL_MB="${2:-}"; shift 2 ;;
    --min-avail-mb=*) MIN_AVAIL_MB="${1#*=}"; shift ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    --out=*) OUT_DIR="${1#*=}"; shift ;;
    --json) JSON_OUT="${2:-}"; shift 2 ;;
    --json=*) JSON_OUT="${1#*=}"; shift ;;
    --csv) CSV_OUT="${2:-}"; shift 2 ;;
    --csv=*) CSV_OUT="${1#*=}"; shift ;;
    --no-logs) SAVE_LOGS=0; shift ;;
    --gate) GATE=1; shift ;;
    --gate-pct) GATE_PCT="${2:-}"; shift 2 ;;
    --gate-pct=*) GATE_PCT="${1#*=}"; shift ;;
    --baseline-decode) BASE_DECODE="${2:-}"; shift 2 ;;
    --baseline-decode=*) BASE_DECODE="${1#*=}"; shift ;;
    --baseline-dense) BASE_DENSE="${2:-}"; shift 2 ;;
    --baseline-dense=*) BASE_DENSE="${1#*=}"; shift ;;
    --baseline-hit) BASE_HIT="${2:-}"; shift 2 ;;
    --baseline-hit=*) BASE_HIT="${1#*=}"; shift ;;
    --lock-wait) LOCK_WAIT="${2:-}"; shift 2 ;;
    --lock-wait=*) LOCK_WAIT="${1#*=}"; shift ;;
    --busy-wait) BUSY_WAIT="${2:-}"; shift 2 ;;
    --busy-wait=*) BUSY_WAIT="${1#*=}"; shift ;;
    --mem-wait) MEM_WAIT="${2:-}"; shift 2 ;;
    --mem-wait=*) MEM_WAIT="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "bench_matrix: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in decode|prefill|both) ;; *) die "--mode must be decode|prefill|both" ;; esac
need_int --runs "$RUNS"
need_int --tokens "$TOKENS"
need_int --cooldown "$COOLDOWN"
need_int --cache-mb "$CACHE_MB"
need_int --io-threads "$IO_THREADS"
need_int --pool-workers "$POOL_WORKERS"
need_int --timeout "$RUN_TIMEOUT"
need_int --min-start-mb "$MIN_START_MB"
need_int --min-avail-mb "$MIN_AVAIL_MB"
need_num --gate-pct "$GATE_PCT"
need_num --baseline-decode "$BASE_DECODE"
need_num --baseline-dense "$BASE_DENSE"
need_num --baseline-hit "$BASE_HIT"
need_int --lock-wait "$LOCK_WAIT"
need_int --busy-wait "$BUSY_WAIT"
need_int --mem-wait "$MEM_WAIT"
(( RUNS >= 1 )) || die "--runs must be >= 1"
(( TOKENS >= 1 )) || die "--tokens must be >= 1"
(( CACHE_MB <= 4096 )) || die "--cache-mb > 4096: AHB budget is 5 GiB incl. hot weights"
(( IO_THREADS >= 1 )) || die "--io-threads must be >= 1"
(( POOL_WORKERS >= 1 )) || die "--pool-workers must be >= 1"
(( MIN_START_MB >= MIN_AVAIL_MB )) || die "--min-start-mb < --min-avail-mb"

[[ -x "$BIN" ]] || die "missing binary: $BIN (build first)"
[[ -f "$MODEL" ]] || die "missing model: $MODEL (set TOKSHIT_MODEL)"
[[ -x "$SAFE_RUN" ]] || die "missing $SAFE_RUN"
command -v python3 >/dev/null 2>&1 || die "python3 is required for JSON/CSV reports"
command -v timeout >/dev/null 2>&1 || die "timeout is required"

# --- prompt lengths -------------------------------------------------------
TARGETS=()
IFS=',' read -r -a _raw_targets <<< "$PROMPT_TOKENS"
for t in "${_raw_targets[@]}"; do
  t="$(printf '%s' "$t" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -n "$t" ]] || continue
  [[ "$t" =~ ^[1-9][0-9]*$ ]] || die "--prompt-tokens: '$t' is not a positive integer"
  TARGETS+=("$t")
done
(( ${#TARGETS[@]} > 0 )) || die "--prompt-tokens is empty"

# --- configs --------------------------------------------------------------
CONFIG_NAMES=()
CONFIG_FLAGS=()
if (( ${#CONFIGS[@]} == 0 )); then
  CONFIG_NAMES=(base)
  CONFIG_FLAGS=("")
else
  for c in "${CONFIGS[@]}"; do
    [[ "$c" != *$'\t'* && "$c" != *$'\n'* ]] || die "config contains a tab/newline"
    if [[ "$c" == *"|"* ]]; then
      name="${c%%|*}"
      flags="${c#*|}"
    else
      name="$c"
      flags=""
    fi
    name="$(printf '%s' "$name" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    flags="$(printf '%s' "$flags" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -n "$name" ]] || die "empty config name in '$c'"
    for existing in "${CONFIG_NAMES[@]}"; do
      [[ "$existing" != "$name" ]] || die "duplicate config name: $name"
    done
    CONFIG_NAMES+=("$name")
    CONFIG_FLAGS+=("$flags")
  done
fi
REF_CONFIG="${CONFIG_NAMES[0]}"

# --- environment helpers --------------------------------------------------
mem_avail_mb() {
  awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

read_temp_c() {
  local raw
  raw="$(cat "$TEMP_ZONE" 2>/dev/null || true)"
  if [[ "$raw" =~ ^-?[0-9]+$ ]] && (( raw > 1000 )); then
    awk -v r="$raw" 'BEGIN {printf "%.1f", r / 1000.0}'
  elif [[ "$raw" =~ ^-?[0-9]+$ ]]; then
    printf '%s.0' "$raw"
  else
    printf ''
  fi
}

wait_no_tokshit() {
  local waited=0
  while pgrep -x tokshit >/dev/null 2>&1; do
    if (( waited >= BUSY_WAIT )); then
      echo "bench_matrix: another tokshit is still running after ${BUSY_WAIT}s" >&2
      return 1
    fi
    if (( waited % 10 == 0 )); then
      echo "[busy] waiting for another tokshit (${waited}s)" >&2
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

wait_mem() {
  local need="$1" waited=0 avail
  while :; do
    avail="$(mem_avail_mb)"
    if (( avail >= need )); then return 0; fi
    if (( waited >= MEM_WAIT )); then
      echo "bench_matrix: MemAvailable=${avail}MB < ${need}MB after ${MEM_WAIT}s" >&2
      return 1
    fi
    echo "[mem] MemAvailable=${avail}MB < ${need}MB; waiting (${waited}s)" >&2
    sleep 5
    waited=$((waited + 5))
  done
}

# --- safety checks --------------------------------------------------------
if ! "$MODEL_GUARD" "$MODEL" >/dev/null 2>&1; then
  "$MODEL_GUARD" "$MODEL" >&2 || true
  die "model guard failed: $MODEL must be read-only"
fi

if (( CACHE_MB > 2048 )); then
  echo "[warn] cache ${CACHE_MB} MB: global AHB budget is 5 GiB incl. hot weights" >&2
fi

device_lock_acquire() {
  local lock_dir lock_file
  lock_dir="${TMPDIR:-/data/data/com.termux/files/usr/tmp}"
  mkdir -p "$lock_dir" 2>/dev/null || lock_dir="/tmp"
  lock_file="$lock_dir/tokshit-device.lock"
  command -v flock >/dev/null 2>&1 || return 0
  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[device-lock] busy; waiting up to ${LOCK_WAIT}s..." >&2
    if ! flock -w "$LOCK_WAIT" 9; then
      echo "bench_matrix: device-lock wait timed out" >&2
      exit 75
    fi
  fi
  echo "[device-lock] acquired" >&2
}
device_lock_acquire

wait_no_tokshit || exit 4
wait_mem "$MIN_START_MB" || exit 4

mkdir -p "$OUT_DIR"
[[ -n "$JSON_OUT" ]] || JSON_OUT="$OUT_DIR/bench-$MODE-$STAMP.json"
[[ -n "$CSV_OUT" ]] || CSV_OUT="$OUT_DIR/bench-$MODE-$STAMP.csv"
mkdir -p "$(dirname -- "$JSON_OUT")" "$(dirname -- "$CSV_OUT")"
LOG_DIR="$OUT_DIR/logs-$MODE-$STAMP"
if (( SAVE_LOGS )); then mkdir -p "$LOG_DIR"; fi

META="$OUT_DIR/.meta-$MODE-$STAMP.tsv"
TSV="$OUT_DIR/.runs-$MODE-$STAMP.tsv"
: > "$META"
: > "$TSV"

meta_kv() {
  local v="$2"
  v="${v//$'\t'/ }"
  v="${v//$'\n'/ }"
  printf '%s\t%s\n' "$1" "$v" >> "$META"
}

# --- prefill prompts ------------------------------------------------------
prompt_dir="${TMPDIR:-/tmp}/tokshit-bench-prompts"
mkdir -p "$prompt_dir"
gen_prompt() {
  local target="$1" file="$prompt_dir/prompt-$1.txt"
  if [[ ! -s "$file" ]]; then
    awk -v n="$target" '
      BEGIN {
        split("the quick brown fox jumps over the lazy dog and then runs away from a big red cat with a small green hat near the old river where many people walk every day", w, " ")
        for (i = 0; i < n; ++i) {
          if (i > 0) printf " "
          printf "%s", w[(i % length(w)) + 1]
        }
        printf "\n"
      }' > "$file"
  fi
  printf '%s\n' "$file"
}

# --- parsing --------------------------------------------------------------
clean_num() {
  if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then printf '%s' "$1"; fi
}

parse_run() {
  local out="$1"
  P_DECODE=""; P_PREFILL=""; P_PTOK=""; P_PSEC=""; P_GTOK=""; P_GSEC=""
  P_HIT=""; P_DENSE=""; P_IO=""; P_RSS=""; P_MINAVAIL=""; P_LOAD=""

  local line
  line="$(grep -m1 '^\[prefill:' <<<"$out" || true)"
  if [[ -n "$line" ]]; then
    P_PTOK="$(clean_num "$(sed -E 's/^\[prefill: ([0-9]+) tok.*/\1/' <<<"$line")")"
    P_PSEC="$(clean_num "$(sed -E 's/^\[prefill: [0-9]+ tok за ([0-9]+([.][0-9]+)?)s.*/\1/' <<<"$line")")"
    P_GTOK="$(clean_num "$(sed -E 's/.*\| генерация: ([0-9]+) tok.*/\1/' <<<"$line")")"
    P_GSEC="$(clean_num "$(sed -E 's/.*генерация: [0-9]+ tok за ([0-9]+([.][0-9]+)?)s.*/\1/' <<<"$line")")"
    P_DECODE="$(clean_num "$(sed -E 's/.* = ([0-9]+([.][0-9]+)?) tok\/s\].*/\1/' <<<"$line")")"
  fi
  if [[ -n "$P_PTOK" && -n "$P_PSEC" ]]; then
    P_PREFILL="$(awk -v t="$P_PTOK" -v s="$P_PSEC" 'BEGIN {if (s > 0) printf "%.4f", t / s}')"
  fi

  P_HIT="$(clean_num "$(grep -m1 '^\[gen-only\]' <<<"$out" | grep -oE 'rate=[0-9]+([.][0-9]+)?%' | head -1 | tr -dc '0-9.' || true)")"
  P_DENSE="$(clean_num "$(grep -m1 '^\[matvec:' <<<"$out" | grep -oE '= [0-9]+([.][0-9]+)? GB/s' | head -1 | grep -oE '[0-9]+([.][0-9]+)?' || true)")"
  P_IO="$(clean_num "$(grep -m1 '^\[io:' <<<"$out" | grep -oE 'read_bytes=[0-9]+([.][0-9]+)? GB' | head -1 | grep -oE '[0-9]+([.][0-9]+)?' || true)")"
  P_LOAD="$(clean_num "$(grep -m1 '^loaded in ' <<<"$out" | grep -oE 'loaded in [0-9]+([.][0-9]+)?s' | grep -oE '[0-9]+([.][0-9]+)?' || true)")"

  line="$(grep -m1 '^\[guard\] peak RSS=' <<<"$out" || true)"
  if [[ -n "$line" ]]; then
    P_RSS="$(clean_num "$(grep -oE 'peak RSS=[0-9]+MB' <<<"$line" | grep -oE '[0-9]+' || true)")"
    P_MINAVAIL="$(clean_num "$(grep -oE 'min MemAvailable=[0-9]+MB' <<<"$line" | grep -oE '[0-9]+' || true)")"
  fi
}

# --- tsv ----------------------------------------------------------------
RUN_HEADER=(
  suite config round order prompt_target rc decode_tps prefill_tps
  prompt_tokens prefill_s gen_tokens gen_s hit_rate dense_gbps io_gb
  rss_mb guard_min_avail_mb temp_before_c temp_after_c mem_before_mb
  mem_after_mb duration_s load_s flags log timestamp
)
tsv_header() {
  local IFS=$'\t'
  printf '%s\n' "${RUN_HEADER[*]}" >> "$TSV"
}
tsv_row() {
  local IFS=$'\t' row=""
  local v
  for v in "$@"; do
    v="${v//$'\t'/ }"
    v="${v//$'\n'/ }"
    row+="${v}"$'\t'
  done
  printf '%s\n' "${row%$'\t'}" >> "$TSV"
}
tsv_header

TOTAL_RUNS=0
if [[ "$MODE" == "decode" || "$MODE" == "both" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + RUNS * ${#CONFIG_NAMES[@]}))
fi
if [[ "$MODE" == "prefill" || "$MODE" == "both" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + RUNS * ${#CONFIG_NAMES[@]} * ${#TARGETS[@]}))
fi

meta_kv mode "$MODE"
meta_kv runs "$RUNS"
meta_kv tokens "$TOKENS"
meta_kv cooldown_s "$COOLDOWN"
meta_kv cache_mb "$CACHE_MB"
meta_kv io_threads "$IO_THREADS"
meta_kv pool_workers "$POOL_WORKERS"
meta_kv extra_flags "$EXTRA_FLAGS"
meta_kv prompt "$PROMPT"
meta_kv prompt_targets "$(IFS=,; echo "${TARGETS[*]}")"
meta_kv model "$MODEL"
meta_kv model_bytes "$(stat -c '%s' "$MODEL" 2>/dev/null || echo 0)"
meta_kv binary "$BIN"
meta_kv binary_sha256 "$(sha256sum "$BIN" 2>/dev/null | awk '{print $1}')"
meta_kv host "$(uname -srm)"
meta_kv mem_total_mb "$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)"
meta_kv temp_zone "$TEMP_ZONE"
meta_kv started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
meta_kv min_start_mb "$MIN_START_MB"
meta_kv min_avail_mb "$MIN_AVAIL_MB"
meta_kv timeout_s "$RUN_TIMEOUT"
meta_kv ref_config "$REF_CONFIG"
meta_kv gate_enabled "$GATE"
meta_kv gate_pct "$GATE_PCT"
meta_kv baseline_decode "$BASE_DECODE"
meta_kv baseline_dense "$BASE_DENSE"
meta_kv baseline_hit "$BASE_HIT"
meta_kv config_order "$(IFS=,; echo "${CONFIG_NAMES[*]}")"
n=0
for cfg in "${CONFIG_NAMES[@]}"; do
  meta_kv "config_flags_$n" "${CONFIG_FLAGS[$n]}"
  meta_kv "config_name_$n" "$cfg"
  n=$((n + 1))
done

echo "== bench_matrix: mode=$MODE runs=$RUNS tokens=$TOKENS cooldown=${COOLDOWN}s"
echo "== model: $MODEL"
echo "== bin:   $BIN"
echo "== flags: cache=${CACHE_MB}MB io=${IO_THREADS} pool=${POOL_WORKERS} extra='${EXTRA_FLAGS}'"
echo "== configs:"
n=0
for cfg in "${CONFIG_NAMES[@]}"; do
  echo "   $cfg | ${CONFIG_FLAGS[$n]:-<none>}"
  n=$((n + 1))
done
echo "== total runs: $TOTAL_RUNS (timeout ${RUN_TIMEOUT}s each, min MemAvailable ${MIN_START_MB}MB)"
echo

COMMON_FLAGS=(--temperature 0 --cache-mb "$CACHE_MB" --io-threads "$IO_THREADS" --pool-workers "$POOL_WORKERS")

run_model() { # suite target label logfile extra-args...
  local suite="$1" target="$2" label="$3" log="$4"
  shift 4
  local t_before t_after m_before m_after start end rc out
  t_before="$(read_temp_c)"
  m_before="$(mem_avail_mb)"
  start="$(date +%s)"

  echo "--- [$suite] cfg=$label round=${ROUND:-?} target=${target:-} temp=${t_before:-?}C mem=${m_before}MB"
  if out="$(env GGUF_TIME=1 TOKSHIT_PROF=1 TOKSHIT_EXPERT_STATS=1 \
      TOKSHIT_MIN_AVAIL_MB="$MIN_AVAIL_MB" \
      timeout -k 15 "$RUN_TIMEOUT" "$SAFE_RUN" "$BIN" "$@" 2>&1 </dev/null)"; then
    rc=0
  else
    rc=$?
  fi
  end="$(date +%s)"
  t_after="$(read_temp_c)"
  m_after="$(mem_avail_mb)"

  if (( SAVE_LOGS )); then
    printf '%s\n' "$out" > "$log"
  else
    log=""
  fi

  parse_run "$out"
  local dur=$((end - start))
  if (( rc == 124 )); then
    echo "    timeout after ${dur}s" >&2
  elif (( rc != 0 )); then
    echo "    run exited rc=$rc after ${dur}s" >&2
  fi
  if [[ -n "$P_DECODE" ]]; then
    echo "    decode=${P_DECODE} tok/s hit=${P_HIT:-?}% dense=${P_DENSE:-?} GB/s"
  fi
  if [[ -n "$P_PREFILL" ]]; then
    echo "    prefill=${P_PREFILL} tok/s (${P_PTOK} tok / ${P_PSEC}s)"
  fi

  tsv_row \
    "$suite" "$label" "${ROUND:-0}" "${ORDER:-0}" "$target" "$rc" \
    "$P_DECODE" "$P_PREFILL" "$P_PTOK" "$P_PSEC" "$P_GTOK" "$P_GSEC" \
    "$P_HIT" "$P_DENSE" "$P_IO" "$P_RSS" "$P_MINAVAIL" \
    "$t_before" "$t_after" "$m_before" "$m_after" "$dur" "$P_LOAD" \
    "${CFG_FLAGS_RUN:-}" "$log" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_no=0
for round in $(seq 1 "$RUNS"); do
  ROUND="$round"
  offset=$(( (round - 1) % ${#CONFIG_NAMES[@]} ))
  echo "########## round $round/$RUNS (order offset $offset) ##########"
  for idx in $(seq 0 $((${#CONFIG_NAMES[@]} - 1))); do
    ci=$(( (idx + offset) % ${#CONFIG_NAMES[@]} ))
    cfg="${CONFIG_NAMES[$ci]}"
    cfg_flags="${CONFIG_FLAGS[$ci]}"
    CFG_FLAGS_RUN="$cfg_flags"
    ORDER=$((idx + 1))
    cfg_arr=()
    if [[ -n "$cfg_flags" ]]; then read -r -a cfg_arr <<< "$cfg_flags"; fi
    extra_arr=()
    if [[ -n "$EXTRA_FLAGS" ]]; then read -r -a extra_arr <<< "$EXTRA_FLAGS"; fi
    safe_cfg="$(printf '%s' "$cfg" | tr -c 'A-Za-z0-9_.-' '_')"

    if [[ "$MODE" == "decode" || "$MODE" == "both" ]]; then
      run_no=$((run_no + 1))
      wait_no_tokshit || exit 4
      wait_mem "$MIN_START_MB" || exit 4
      run_model decode "" "$cfg" "$LOG_DIR/decode-r${round}-${safe_cfg}.log" \
        "${COMMON_FLAGS[@]}" "${extra_arr[@]}" "${cfg_arr[@]}" \
        "$MODEL" "$PROMPT" "$TOKENS"
      if (( run_no < TOTAL_RUNS && COOLDOWN > 0 )); then
        echo "[cooldown] ${COOLDOWN}s"
        sleep "$COOLDOWN"
        ROUND="$round"
      fi
    fi

    if [[ "$MODE" == "prefill" || "$MODE" == "both" ]]; then
      for target in "${TARGETS[@]}"; do
        run_no=$((run_no + 1))
        pfile="$(gen_prompt "$target")"
        wait_no_tokshit || exit 4
        wait_mem "$MIN_START_MB" || exit 4
        run_model prefill "$target" "$cfg" \
          "$LOG_DIR/prefill-r${round}-${safe_cfg}-t${target}.log" \
          "${COMMON_FLAGS[@]}" "${extra_arr[@]}" "${cfg_arr[@]}" \
          --prompt-file "$pfile" "$MODEL" "_" 1
        if (( run_no < TOTAL_RUNS && COOLDOWN > 0 )); then
          echo "[cooldown] ${COOLDOWN}s"
          sleep "$COOLDOWN"
          ROUND="$round"
        fi
      done
    fi
  done
done

meta_kv finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
meta_kv runs_done "$run_no"

report_rc=0
python3 - "$TSV" "$META" "$JSON_OUT" "$CSV_OUT" <<'PY' || report_rc=$?
import csv
import json
import os
import sys

tsv_path, meta_path, json_path, csv_path = sys.argv[1:5]

def read_meta(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, _, v = line.partition("\t")
            d[k] = v
    return d

def read_runs(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows

meta = read_meta(meta_path)
runs = read_runs(tsv_path)
order = []
i = 0
while f"config_name_{i}" in meta:
    order.append(meta[f"config_name_{i}"])
    i += 1
if not order:
    order = list(dict.fromkeys(r["config"] for r in runs))

def num(v):
    if v is None:
        return None
    v = str(v).strip()
    if not v or v.upper() in ("NA", "NAN", "NULL", "-"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def stats(values):
    vals = [x for x in (num(v) for v in values) if x is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {
        "n": n,
        "median": round(med, 4),
        "min": round(vals[0], 4),
        "max": round(vals[-1], 4),
    }

summary = {}
for cfg in order:
    cfg_runs = [r for r in runs if r["config"] == cfg]
    dec = [r["decode_tps"] for r in cfg_runs if r["suite"] == "decode"]
    pre = [r["prefill_tps"] for r in cfg_runs if r["suite"] == "prefill"]
    hits = [r["hit_rate"] for r in cfg_runs if r["suite"] == "decode"]
    dense = [r["dense_gbps"] for r in cfg_runs if r["suite"] == "decode"]
    temps = [r["temp_after_c"] for r in cfg_runs]
    mems = [r["mem_after_mb"] for r in cfg_runs]
    ptok = [r["prompt_tokens"] for r in cfg_runs if r["suite"] == "prefill"]
    failed = sum(1 for r in cfg_runs if str(r.get("rc", "0")) not in ("0", ""))
    summary[cfg] = {
        "flags": next((r["flags"] for r in cfg_runs), ""),
        "runs": len(cfg_runs),
        "failed": failed,
        "decode_tps": stats(dec),
        "prefill_tps": stats(pre),
        "hit_rate": stats(hits),
        "dense_gbps": stats(dense),
        "temp_after_c": stats(temps),
        "mem_after_mb": stats(mems),
        "prompt_tokens": stats(ptok),
    }

def med(cfg, key):
    s = summary.get(cfg, {}).get(key)
    return s["median"] if s else None

ab = []
ref = meta.get("ref_config") or order[0]
for cfg in order:
    if cfg == ref:
        continue
    row = {"config": cfg, "reference": ref}
    for key, label in (("decode_tps", "decode_delta_pct"),
                       ("prefill_tps", "prefill_delta_pct"),
                       ("dense_gbps", "dense_delta_pct")):
        a, b = med(cfg, key), med(ref, key)
        row[label] = round((a / b - 1.0) * 100.0, 2) if a is not None and b else None
    a, b = med(cfg, "hit_rate"), med(ref, "hit_rate")
    row["hit_delta_pp"] = round(a - b, 2) if a is not None and b is not None else None
    ab.append(row)

gate_enabled = meta.get("gate_enabled", "0") == "1"
gate_pct = float(meta.get("gate_pct", "10") or 10)
baselines = {
    "decode_tps": float(meta.get("baseline_decode", "1.1") or 1.1),
    "dense_gbps": float(meta.get("baseline_dense", "12.55") or 12.55),
    "hit_rate": float(meta.get("baseline_hit", "27.5") or 27.5),
}
gate_cfg = ref
gate_checks = []
gate_failed = False
for key, label, unit in (("decode_tps", "decode", "tok/s"),
                         ("dense_gbps", "dense", "GB/s"),
                         ("hit_rate", "hit", "%")):
    actual = med(gate_cfg, key)
    base = baselines[key]
    threshold = base * (1.0 - gate_pct / 100.0)
    if actual is None:
        gate_checks.append({
            "metric": label, "status": "skip", "actual": None,
            "baseline": base, "threshold": round(threshold, 4), "unit": unit,
        })
        continue
    ok = actual >= threshold
    if not ok:
        gate_failed = True
    gate_checks.append({
        "metric": label, "status": "pass" if ok else "fail",
        "actual": round(actual, 4), "baseline": base,
        "threshold": round(threshold, 4), "unit": unit,
        "delta_pct": round((actual / base - 1.0) * 100.0, 2) if base else None,
    })

mode = meta.get("mode", "")
if gate_enabled and mode in ("decode", "both") and med(gate_cfg, "decode_tps") is None:
    gate_failed = True
    gate_checks.append({
        "metric": "decode", "status": "fail", "actual": None,
        "baseline": baselines["decode_tps"], "threshold": None,
        "unit": "tok/s", "note": "no decode data",
    })

gate = {
    "enabled": gate_enabled,
    "config": gate_cfg,
    "threshold_pct": gate_pct,
    "passed": (not gate_failed) if gate_enabled else None,
    "checks": gate_checks,
}

doc = {
    "schema": "tokshit.bench.v1",
    "mode": mode,
    "meta": meta,
    "summary": summary,
    "ab": ab,
    "gate": gate,
    "runs": runs,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
    f.write("\n")

fieldnames = list(runs[0].keys()) if runs else []
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(runs)

def fmt(st, digits=2, suffix=""):
    if not st:
        return "-"
    return f"{st['median']:.{digits}f} [{st['min']:.{digits}f}..{st['max']:.{digits}f}]{suffix}"

def fmt_pct(v):
    if v is None:
        return "-"
    return f"{v:+.2f}%"

print()
print("== runs ==")
print(f"{'suite':<9}{'config':<14}{'r':>2}{'ord':>4}{'target':>7}{'rc':>4}"
      f"{'decode':>9}{'prefill':>9}{'hit%':>7}{'dense':>8}{'temp':>13}{'memMB':>13}")
for r in runs:
    temp = f"{r.get('temp_before_c','') or '?'}->{r.get('temp_after_c','') or '?'}"
    mem = f"{r.get('mem_before_mb','') or '?'}->{r.get('mem_after_mb','') or '?'}"
    print(f"{r['suite']:<9}{r['config'][:13]:<14}{r['round']:>2}{r['order']:>4}"
          f"{r.get('prompt_target','') or '-':>7}{r['rc']:>4}"
          f"{(r.get('decode_tps') or '-'):>9}{(r.get('prefill_tps') or '-'):>9}"
          f"{(r.get('hit_rate') or '-'):>7}{(r.get('dense_gbps') or '-'):>8}"
          f"{temp:>13}{mem:>13}")

print()
print("== summary (median [min..max]) ==")
hdr = (f"{'config':<14}{'n':>3}  {'decode tok/s':<24}{'prefill tok/s':<24}"
       f"{'hit %':<18}{'dense GB/s':<22}{'temp C':<16}{'MemAvail MB':<16}")
print(hdr)
for cfg in order:
    s = summary[cfg]
    n = s["runs"]
    print(f"{cfg[:13]:<14}{n:>3}  {fmt(s['decode_tps']):<24}{fmt(s['prefill_tps']):<24}"
          f"{fmt(s['hit_rate'], 1):<18}{fmt(s['dense_gbps']):<22}"
          f"{fmt(s['temp_after_c'], 1):<16}{fmt(s['mem_after_mb'], 0):<16}")

print()
if ab:
    print(f"== A/B delta vs reference '{ref}' (median) ==")
    print(f"{'config':<14}{'decode':>10}{'prefill':>10}{'hit pp':>9}{'dense':>10}")
    print(f"{ref[:13]:<14}{'0.00%':>10}{'0.00%':>10}{'0.0':>9}{'0.00%':>10}")
    for row in ab:
        hit_delta = (f"{row['hit_delta_pp']:+.2f}"
                     if row['hit_delta_pp'] is not None else "-")
        print(f"{row['config'][:13]:<14}"
              f"{fmt_pct(row['decode_delta_pct']):>10}"
              f"{fmt_pct(row['prefill_delta_pct']):>10}"
              f"{hit_delta:>9}"
              f"{fmt_pct(row['dense_delta_pct']):>10}")

print()
print("== regression gate ==")
print(f"config='{gate_cfg}' threshold={gate_pct:g}% "
      f"({'enabled' if gate_enabled else 'report only, use --gate'})")
for c in gate_checks:
    if c["status"] == "skip":
        print(f"  [skip] {c['metric']}: no data (baseline {c['baseline']:g} {c['unit']})")
    elif c["status"] == "fail" and c.get("note"):
        print(f"  [FAIL] {c['metric']}: {c['note']}")
    else:
        delta = f", {c['delta_pct']:+.2f}%" if c.get("delta_pct") is not None else ""
        print(f"  [{'PASS' if c['status'] == 'pass' else 'FAIL'}] {c['metric']}: "
              f"{c['actual']:.2f} {c['unit']} vs >= {c['threshold']:.2f} "
              f"(baseline {c['baseline']:g}{delta})")
if gate_enabled:
    print(f"[gate] {'PASS' if not gate_failed else 'FAIL'}")
else:
    print("[gate] not enforced")

print()
print(f"json: {json_path}")
print(f"csv:  {csv_path}")
print(f"meta: {meta_path}")
print(f"tsv:  {tsv_path}")
if meta.get("runs_done"):
    print(f"runs: {meta['runs_done']}")

if gate_enabled and gate_failed:
    sys.exit(3)
PY

if (( report_rc == 3 )); then
  echo "[bench_matrix] regression gate failed (exit 3)" >&2
  exit 3
elif (( report_rc != 0 )); then
  echo "[bench_matrix] report generation failed (rc=$report_rc)" >&2
  exit "$report_rc"
fi

exit 0
