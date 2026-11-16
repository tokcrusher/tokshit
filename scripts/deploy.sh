#!/usr/bin/env bash
# Ship the source tree to a device over SSH and build it there.
#
# Environment:
#   TOKSHIT_DEPLOY_HOST   phone host, default 10.253.61.103
#   TOKSHIT_DEPLOY_PORT   sshd port, default 8022
#   TOKSHIT_DEPLOY_USER   user, default u0_a443
#   TOKSHIT_DEPLOY_PASS   password, passed to sshpass via SSHPASS (required)
#
# The remote build goes through ./install.sh, so it checks/installs the
# Termux toolchain and uses an out-of-source Ninja build.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${TOKSHIT_DEPLOY_HOST:-10.253.61.103}"
PORT="${TOKSHIT_DEPLOY_PORT:-8022}"
USER_NAME="${TOKSHIT_DEPLOY_USER:-u0_a443}"

: "${TOKSHIT_DEPLOY_PASS:?export TOKSHIT_DEPLOY_PASS with the device password}"
export SSHPASS="$TOKSHIT_DEPLOY_PASS"

if [[ -d /data/data/com.termux/files/usr/tmp ]]; then
  TMPBASE="/data/data/com.termux/files/usr/tmp"
else
  TMPBASE="${TMPDIR:-/tmp}"
fi
mkdir -p "$TMPBASE"
TARBALL="$(mktemp "$TMPBASE/tokshit-deploy.XXXXXX")"
trap 'rm -f "$TARBALL"' EXIT

tar czf "$TARBALL" \
  --exclude='.git' \
  --exclude='build' \
  --exclude='build-*' \
  --exclude='*.gguf' \
  --exclude='.code*' \
  --exclude='.fixbak*' \
  -C .. tokshit

sshpass -e scp -O -P "$PORT" "$TARBALL" "$USER_NAME@$HOST":~/tokshit.tgz
sshpass -e ssh -o ServerAliveInterval=10 -p "$PORT" "$USER_NAME@$HOST" '
  set -e
  cd ~
  if [ -d tokshit ]; then
    mv tokshit "tokshit.bak.$(date +%s)"
  fi
  tar xzf tokshit.tgz
  cd tokshit
  ./install.sh --no-install --no-tests -j 2
'
echo "deploy ok: ssh -p $PORT $USER_NAME@$HOST 'cd ~/tokshit && ./build/tokshit --help'"
