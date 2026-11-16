#!/usr/bin/env bash
# tokshit installer / builder.
#
# Works on Termux (Android) and on a normal Linux box. It checks for the
# toolchain, installs whatever is missing with the system package manager,
# configures a Release build with Ninja, builds the runtime, runs the unit
# tests, and installs the `tokshit` binary.
#
# Usage:
#   ./install.sh                 # Termux/Linux, install into $PREFIX/bin or /usr/local/bin
#   ./install.sh --no-install    # build only
#   ./install.sh --clean         # wipe ./build first
#   ./install.sh -j 6            # build with 6 jobs
#   ./install.sh --prefix "$HOME/.local"
#
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
INSTALL_PREFIX=""
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
DO_TESTS=1
DO_TOOLS=1
DO_INSTALL=1
DO_CLEAN=0
INSTALL_TOOLS=0

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
tokshit installer

  ./install.sh [options]

Options:
  -j, --jobs N        parallel build jobs (default: all CPUs)
  --build-dir DIR     build directory (default: ./build)
  --prefix DIR        install prefix; binary goes to DIR/bin
                      (default: $PREFIX on Termux, /usr/local elsewhere)
  --no-install        build only, do not copy the binary
  --install-tools     also install qstat/kbench/bwbench/pbench/vcheck/ahbtest
  --no-tests          do not build/run the unit tests
  --no-tools          do not build the benchmark tools
  --clean             remove the build directory first
  --yes               accepted for compatibility; package managers are non-interactive
  -h, --help          show this help

The script installs missing packages automatically:
  Termux: git cmake ninja clang
  Linux:  git cmake ninja/ninja-build and clang (or another C++20 compiler)
EOF
}

while (($#)); do
  case "$1" in
    -j|--jobs)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      JOBS="$2"; shift 2;;
    --jobs=*) JOBS="${1#*=}"; shift;;
    --build-dir)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      BUILD_DIR="$2"; shift 2;;
    --build-dir=*) BUILD_DIR="${1#*=}"; shift;;
    --prefix)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      INSTALL_PREFIX="$2"; shift 2;;
    --prefix=*) INSTALL_PREFIX="${1#*=}"; shift;;
    --no-install) DO_INSTALL=0; shift;;
    --install-tools) INSTALL_TOOLS=1; shift;;
    --no-tests) DO_TESTS=0; shift;;
    --no-tools) DO_TOOLS=0; shift;;
    --clean) DO_CLEAN=1; shift;;
    -y|--yes) shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown option: $1 (try --help)";;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "invalid --jobs value: $JOBS"
if [[ "$BUILD_DIR" != /* ]]; then
  BUILD_DIR="$ROOT/$BUILD_DIR"
fi

# ---------------------------------------------------------------- environment
IS_TERMUX=0
if [[ -n "${TERMUX_VERSION:-}" ]] || [[ "${PREFIX:-}" == *com.termux* ]] ||
   [[ -d /data/data/com.termux/files/usr ]]; then
  IS_TERMUX=1
fi
TERMUX_PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- dependencies
need_pkgs=()
have git   || need_pkgs+=(git)
have cmake || need_pkgs+=(cmake)
have ninja || need_pkgs+=(ninja)

have_cxx=0
if have clang++ || have g++ || have c++; then
  have_cxx=1
else
  need_pkgs+=(clang)
fi

install_packages() {
  ((${#need_pkgs[@]})) || return 0
  log "Installing missing packages: ${need_pkgs[*]}"

  if ((IS_TERMUX)); then
    have pkg || die "Termux detected, but the 'pkg' command is missing"
    # Fresh Termux images sometimes have a stale package index. Try the cheap
    # path first, then update once and retry.
    if ! pkg install -y "${need_pkgs[@]}"; then
      warn "pkg install failed; updating the package index and retrying"
      pkg update -y
      pkg install -y "${need_pkgs[@]}"
    fi
    return 0
  fi

  local sudo_cmd=()
  if [[ "$(id -u)" -ne 0 ]]; then
    have sudo || die "missing packages and sudo is not available; install them manually: ${need_pkgs[*]}"
    sudo_cmd=(sudo)
  fi

  if have apt-get; then
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y git cmake ninja-build clang
  elif have dnf; then
    "${sudo_cmd[@]}" dnf install -y git cmake ninja-build clang
  elif have yum; then
    "${sudo_cmd[@]}" yum install -y git cmake ninja-build clang
  elif have pacman; then
    "${sudo_cmd[@]}" pacman -S --needed --noconfirm git cmake ninja clang
  elif have zypper; then
    "${sudo_cmd[@]}" zypper install -y git cmake ninja clang
  elif have apk; then
    "${sudo_cmd[@]}" apk add git cmake ninja clang
  else
    die "no supported package manager; install manually: ${need_pkgs[*]}"
  fi
}

install_packages

# Re-check after installation.
for c in git cmake ninja; do
  have "$c" || die "required command still missing: $c"
done
if ((!have_cxx)); then
  have clang++ || have g++ || have c++ ||
    die "no C++ compiler found (need clang++ or g++)"
fi

# CMake >= 3.20 is required by CMakeLists.txt.
CMAKE_VER="$(cmake --version | awk 'NR==1 {print $3}')"
if ! printf '%s\n%s\n' "3.20" "$CMAKE_VER" | sort -V -C; then
  die "CMake >= 3.20 required, found $CMAKE_VER"
fi

# ---------------------------------------------------------------- sanity checks
avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [[ -n "$avail_kb" ]] && ((avail_kb < 2 * 1024 * 1024)); then
  warn "less than 2 GiB free on $ROOT; the build may fail"
fi

avail_mb="$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || true)"
if [[ -n "$avail_mb" ]] && ((avail_mb < 1024)); then
  warn "only ${avail_mb} MiB MemAvailable; reduce -j or close other apps"
fi

# ---------------------------------------------------------------- build
if ((DO_CLEAN)); then
  log "Removing $BUILD_DIR"
  rm -rf -- "$BUILD_DIR"
fi

CMAKE_ARGS=(
  -S "$ROOT"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DTOKSHIT_BUILD_TESTS="$([[ $DO_TESTS -eq 1 ]] && echo ON || echo OFF)"
  -DTOKSHIT_BUILD_TOOLS="$([[ $DO_TOOLS -eq 1 ]] && echo ON || echo OFF)"
)

log "Configuring Release/Ninja in $BUILD_DIR"
cmake "${CMAKE_ARGS[@]}"

log "Building tokshit with $JOBS job(s)"
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if ((DO_TESTS)); then
  log "Running unit tests"
  if [[ -x "$BUILD_DIR/tokshit_tests" ]]; then
    "$BUILD_DIR/tokshit_tests"
  else
    (cd "$BUILD_DIR" && ctest --output-on-failure)
  fi
fi

# ---------------------------------------------------------------- install
if ((DO_INSTALL)); then
  if [[ -z "$INSTALL_PREFIX" ]]; then
    if ((IS_TERMUX)); then
      INSTALL_PREFIX="$TERMUX_PREFIX"
    else
      INSTALL_PREFIX="/usr/local"
    fi
  fi
  BINDIR="$INSTALL_PREFIX/bin"

  SUDO=()
  if [[ ! -d "$BINDIR" ]]; then
    parent="$(dirname "$BINDIR")"
    if [[ ! -w "$parent" ]]; then
      if [[ "$(id -u)" -ne 0 ]]; then
        have sudo || die "cannot write to $parent and sudo is unavailable"
        SUDO=(sudo)
      fi
    fi
    "${SUDO[@]}" mkdir -p "$BINDIR"
  elif [[ ! -w "$BINDIR" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then
      have sudo || die "cannot write to $BINDIR and sudo is unavailable"
      SUDO=(sudo)
    fi
  fi

  log "Installing tokshit to $BINDIR"
  "${SUDO[@]}" install -m 0755 "$BUILD_DIR/tokshit" "$BINDIR/tokshit"

  if ((INSTALL_TOOLS)); then
    for t in qstat kbench bwbench pbench vcheck ahbtest; do
      if [[ -x "$BUILD_DIR/$t" ]]; then
        "${SUDO[@]}" install -m 0755 "$BUILD_DIR/$t" "$BINDIR/$t"
      fi
    done
  fi
fi

# ---------------------------------------------------------------- safety hooks
if [[ -d "$ROOT/.githooks" ]] &&
   git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" config core.hooksPath .githooks
  log "Enabled local git safety hooks (.githooks)"
fi

# ---------------------------------------------------------------- next steps
RUN_BIN="$BUILD_DIR/tokshit"
INSTALLED_LINE=""
if ((DO_INSTALL)); then
  RUN_BIN="$BINDIR/tokshit"
  INSTALLED_LINE="  ->  $RUN_BIN"
fi
if ((DO_TESTS)); then TEST_LINE="passed"; else TEST_LINE="skipped"; fi

printf '\n\033[1;32m[ok]\033[0m tokshit is built.\n'
cat <<EOF
  binary:  $BUILD_DIR/tokshit$INSTALLED_LINE
  tests:   $TEST_LINE

Next:
  "$RUN_BIN" --help

The model is not included (it is many GiB). Put a .gguf file somewhere and run:
  TOKSHIT_EXPERT_CACHE_MB=512 "$ROOT/scripts/safe_run.sh" "$RUN_BIN" model.gguf "привет" 32

Read README.ru.md and docs/ for memory limits and performance notes.
EOF
