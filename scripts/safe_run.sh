#!/data/data/com.termux/files/usr/bin/bash
# Запуск tokshit с аварийным тормозом по памяти.
#
#   scripts/safe_run.sh build/tokshit --temperature 0 model.gguf "привет" 8
#
# Если MemAvailable падает ниже TOKSHIT_MIN_AVAIL_MB (по умолчанию 1200),
# процесс убивается до того, как сработает OOM-killer. В конце печатается
# пиковый RSS и минимум свободной памяти.
set -u

limit_mb=${TOKSHIT_MIN_AVAIL_MB:-1200}
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

# timeout обычно убивает сам safe_run.sh, а child остаётся сиротой и держит
# модель/AHB. Ловим TERM/INT и гасим child до выхода.
# timeout usually kills safe_run.sh itself and leaves the child orphaned with
# the model/AHB. Catch TERM/INT and kill the child before exiting.
child_pid=""
cleanup_child() {
  if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null
    kill -KILL "$child_pid" 2>/dev/null
  fi
}
trap 'cleanup_child; exit 143' TERM INT

# Only one model process at a time. The lock is inherited by the child and is
# released when the model exits, so parallel agents cannot fight over RAM or
# corrupt each other's benchmarks.
lock_dir="${TMPDIR:-/data/data/com.termux/files/usr/tmp}"
mkdir -p "$lock_dir" 2>/dev/null || lock_dir="/tmp"
lock_file="$lock_dir/tokshit-model.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[guard] another model run holds the lock; waiting..." >&2
    if ! flock -w "${TOKSHIT_MODEL_LOCK_WAIT:-600}" 9; then
      echo "[guard] lock wait timed out; aborting" >&2
      exit 75
    fi
  fi
fi

# Keep the caller's stdin. A plain `cmd &` in a non-interactive shell
# redirects stdin from /dev/null, which makes the interactive TUI exit
# immediately. Explicit <&0 preserves the terminal (and is harmless for
# non-interactive runs, where stdin is already a pipe or /dev/null).
"$@" <&0 &
pid=$!
child_pid=$pid
min_avail=9999999
peak_rss=0
killed=0

while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
  rss=$(awk '/^VmRSS:/ {print int($2/1024)}' "/proc/$pid/status" 2>/dev/null)
  [ -z "${avail:-}" ] && avail=0
  [ -z "${rss:-}" ] && rss=0
  [ "$avail" -lt "$min_avail" ] && min_avail=$avail
  [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
  if [ "$avail" -lt "$limit_mb" ]; then
    echo "[guard] MemAvailable=${avail}MB < ${limit_mb}MB: killing $pid" >&2
    kill -9 "$pid" 2>/dev/null
    killed=1
    break
  fi
  sleep 1
done

wait "$pid" 2>/dev/null
rc=$?
echo "[guard] peak RSS=${peak_rss}MB, min MemAvailable=${min_avail}MB" >&2
if [ "$killed" -eq 1 ]; then
  exit 137
fi
exit "$rc"
