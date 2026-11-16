#!/data/data/com.termux/files/usr/bin/bash
# Сериализация тяжёлых работ на устройстве (сборки и замеры производительности).
# Сборка, идущая параллельно с замером, ломает цифры; два замера — тоже.
# Использование:
#   TOKSHIT_DEVICE_LOCK_WAIT=3600 scripts/device_lock.sh cmake --build build --parallel 2
#   TOKSHIT_DEVICE_LOCK_WAIT=3600 scripts/device_lock.sh scripts/safe_run.sh build/tokshit ...
set -u

lock_dir="${TMPDIR:-/data/data/com.termux/files/usr/tmp}"
mkdir -p "$lock_dir" 2>/dev/null || lock_dir="/tmp"
lock_file="$lock_dir/tokshit-device.lock"
wait_s="${TOKSHIT_DEVICE_LOCK_WAIT:-3600}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

if command -v flock >/dev/null 2>&1; then
  exec 8>"$lock_file"
  if ! flock -n 8; then
    echo "[device-lock] another build/bench holds the device; waiting (up to ${wait_s}s)..." >&2
    if ! flock -w "$wait_s" 8; then
      echo "[device-lock] lock wait timed out; aborting" >&2
      exit 75
    fi
  fi
fi

exec "$@"
