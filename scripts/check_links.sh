#!/usr/bin/env bash
#
# check_links.sh — documentation link checker for tokshit.
#
# Scans README.md, README.ru.md and docs/**/*.md, then verifies:
#   * relative file links and images (relative to the Markdown file);
#   * #anchors against GitHub-style heading slugs (duplicates get -1, -2, ...);
#   * forbidden references to PORT_CPP.md and the legacy project name.
#
# External links (http://, https://, mailto:) are ignored by default.
# Absolute paths (/...) are ignored by default as well.
#
# Usage:
#   scripts/check_links.sh                 # strict check, local links only
#   scripts/check_links.sh --warn-only     # report problems, always exit 0
#   scripts/check_links.sh --external      # also HEAD-check external links
#   scripts/check_links.sh --list-files    # print scanned files and exit
#   scripts/check_links.sh --help
#
# Options:
#   --strict        Exit non-zero on broken links (default).
#   --warn-only     Never fail because of broken links.
#   --external      HEAD-check external links (curl/wget, 5 s timeout).
#   --list-files    List the Markdown files that would be scanned.
#   -h, --help      Show this help.
#
# Requires bash, grep, sed, tr, find, sort. No Python.
#
# Exit status: 0 if everything is fine (or --warn-only), 1 if broken links
# or forbidden references were found in strict mode, 2 on usage errors.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2

STRICT=1
CHECK_EXTERNAL=0
LIST_FILES=0

usage() {
  sed -n '2,31p' "$0" | sed -e 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --strict)      STRICT=1 ;;
    --warn-only)   STRICT=0 ;;
    --external)    CHECK_EXTERNAL=1 ;;
    --list-files)  LIST_FILES=1 ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "check_links: unknown option: $arg" >&2
      echo "Try: $0 --help" >&2
      exit 2
      ;;
  esac
done

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

FILES=()
for f in README.md README.ru.md; do
  [ -f "$f" ] && FILES+=("$f")
done
if [ -d docs ]; then
  while IFS= read -r f; do
    FILES+=("$f")
  done < <(find docs -type f -name '*.md' | LC_ALL=C sort)
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "check_links: no Markdown files found under $ROOT" >&2
  exit 2
fi

if [ "$LIST_FILES" -eq 1 ]; then
  printf '%s\n' "${FILES[@]}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TMPDIR_LINKS=$(mktemp -d "${TMPDIR:-/tmp}/tokshit-links.XXXXXX") || exit 2
trap 'rm -rf "$TMPDIR_LINKS"' EXIT

# Legacy project name, assembled so the literal never lives in the source.
FORBIDDEN_NAME=$(printf '\157\156\171\170')
LINK_RE='!?\[[^]]*\]\([^)]*\)'

# GitHub-style heading slug. Keeps Unicode letters, lowercases ASCII,
# turns spaces into dashes and drops ASCII punctuation.
slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/<[^>]*>//g' \
          -e 's/`//g' \
          -e 's/!\[[^]]*\]([^)]*)//g' \
          -e 's/\[\([^]]*\)\]([^)]*)/\1/g' \
          -e 's/  */ /g' \
          -e 's/^ //' \
          -e 's/ $//' \
          -e 's/ /-/g' \
          -e 's/[][!"#$%&'"'"'()*+,.\/:;<=>?@\^_{|}~]//g'
}

# Print one anchor per line for an ATX-heading Markdown file.
# Fenced code blocks are skipped, as GitHub does.
build_anchor_cache() {
  local file=$1 cache=$2 line heading base candidate n
  local in_fence=0
  local -A seen=()
  : > "$cache"

  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '```'*|'~~~'*)
        if [ "$in_fence" -eq 0 ]; then in_fence=1; else in_fence=0; fi
        continue
        ;;
    esac
    [ "$in_fence" -eq 1 ] && continue

    case "$line" in
      \#*) ;;
      *) continue ;;
    esac

    if [[ "$line" =~ ^#{1,6}[[:space:]]+(.*)$ ]]; then
      heading=${BASH_REMATCH[1]}
      # Drop a closing #-sequence and trailing whitespace.
      heading=$(printf '%s' "$heading" \
        | sed -e 's/[[:space:]]*#\{1,\}[[:space:]]*$//' \
              -e 's/[[:space:]]*$//')
      base=$(slugify "$heading")
      [ -z "$base" ] && continue

      candidate=$base
      n=1
      while [ -n "${seen[$candidate]:-}" ]; do
        candidate="$base-$n"
        n=$((n + 1))
      done
      seen[$candidate]=1
      printf '%s\n' "$candidate" >> "$cache"
    fi
  done < "$file"
}

# Resolve a relative path against the directory of the source file.
resolve_path() {
  local src=$1 rel=$2
  local dir
  dir=$(dirname "$src")
  if [ "$dir" = "." ]; then
    printf '%s/%s' "$ROOT" "$rel"
  else
    printf '%s/%s/%s' "$ROOT" "$dir" "$rel"
  fi
}

# HEAD-check an external URL. Echo "OK <code>" or "FAIL <reason>".
check_external_url() {
  local url=$1 code
  if command -v curl >/dev/null 2>&1; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' -I -L \
      --max-time 5 --max-redirs 3 "$url" 2>/dev/null) || code=000
    case "$code" in
      2*|3*) printf 'OK %s' "$code" ;;
      000)   printf 'FAIL unreachable' ;;
      *)     printf 'FAIL HTTP %s' "$code" ;;
    esac
  elif command -v wget >/dev/null 2>&1; then
    if wget -q --spider --timeout=5 -T 5 "$url" >/dev/null 2>&1; then
      printf 'OK 200'
    else
      printf 'FAIL unreachable'
    fi
  else
    printf 'SKIP no curl/wget'
  fi
}

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

ERRORS=0
LOCAL_LINKS=0
EXTERNAL_LINKS=0
EXTERNAL_CHECKED=0
IGNORED_LINKS=0
ANCHOR_CACHE_SEQ=0
declare -A ANCHOR_CACHES=()

report_error() {
  local file=$1 line=$2 target=$3 reason=$4
  printf 'BROKEN %s:%s -> %s -- %s\n' "$file" "$line" "$target" "$reason"
  ERRORS=$((ERRORS + 1))
}

for file in "${FILES[@]}"; do
  in_fence=0
  lineno=0

  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))

    case "$line" in
      '```'*|'~~~'*)
        if [ "$in_fence" -eq 0 ]; then in_fence=1; else in_fence=0; fi
        continue
        ;;
    esac
    [ "$in_fence" -eq 1 ] && continue

    # Forbidden references: PORT_CPP.md and the legacy project name.
    # Do it with pure bash: one grep/fork per Markdown line is unusably slow
    # on Termux.
    if [[ "$line" == *PORT_CPP.md* ]]; then
      report_error "$file" "$lineno" 'PORT_CPP.md' 'forbidden reference'
    fi
    if [[ "${line,,}" == *"$FORBIDDEN_NAME"* ]]; then
      report_error "$file" "$lineno" "$FORBIDDEN_NAME" 'forbidden name'
    fi

    # Extract Markdown links and images: [text](target), ![alt](target).
    # Bash regex avoids a fork per line; only the target/anchor handling below
    # shells out, and those are few.
    rest=$line
    while [[ "$rest" =~ $LINK_RE ]]; do
      link=${BASH_REMATCH[0]}
      rest=${rest#*"$link"}
      [ -z "$link" ] && continue

      target=${link#*](}
      target=${target%)}

      # Trim whitespace.
      target=$(printf '%s' "$target" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

      # Angle-bracket destinations may contain spaces.
      case "$target" in
        '<'*'>'*)
          target=${target#<}
          target=${target%%>*}
          ;;
        *)
          # Drop an optional link title: [x](path "title").
          target=${target%%[[:space:]]*}
          ;;
      esac

      [ -z "$target" ] && continue

      case "$target" in
        http://*|https://*|mailto:*|tel:*|ftp:*|ftps:*|data:*|'//'*)
          EXTERNAL_LINKS=$((EXTERNAL_LINKS + 1))
          if [ "$CHECK_EXTERNAL" -eq 1 ]; then
            EXTERNAL_CHECKED=$((EXTERNAL_CHECKED + 1))
            result=$(check_external_url "$target")
            case "$result" in
              OK\ *) ;;
              SKIP\ *) printf 'SKIP   %s:%s -> %s -- %s\n' "$file" "$lineno" "$target" "${result#SKIP }" ;;
              *) report_error "$file" "$lineno" "$target" "${result#FAIL }" ;;
            esac
          fi
          continue
          ;;
        /*)
          IGNORED_LINKS=$((IGNORED_LINKS + 1))
          continue
          ;;
        '#')
          IGNORED_LINKS=$((IGNORED_LINKS + 1))
          continue
          ;;
      esac

      LOCAL_LINKS=$((LOCAL_LINKS + 1))

      path=${target%%#*}
      anchor=""
      case "$target" in
        *'#'*) anchor=${target#*#} ;;
      esac

      # Decode the only common escape in local paths.
      path=${path//%20/ }
      anchor=${anchor//%20/ }

      if [ -z "$path" ]; then
        target_file="$ROOT/$file"
        display_file=$file
      else
        target_file=$(resolve_path "$file" "$path")
        display_file=$path
      fi

      if [ ! -e "$target_file" ]; then
        report_error "$file" "$lineno" "$target" "missing file: $display_file"
        continue
      fi

      if [ -n "$anchor" ]; then
        if [ ! -f "$target_file" ]; then
          report_error "$file" "$lineno" "$target" "anchor on a non-file path"
          continue
        fi
        case "$target_file" in
          *.md|*.markdown) ;;
          *)
            report_error "$file" "$lineno" "$target" "anchor on a non-Markdown file"
            continue
            ;;
        esac

        cache=${ANCHOR_CACHES[$target_file]:-}
        if [ -z "$cache" ]; then
          ANCHOR_CACHE_SEQ=$((ANCHOR_CACHE_SEQ + 1))
          cache="$TMPDIR_LINKS/anchors.$ANCHOR_CACHE_SEQ"
          build_anchor_cache "$target_file" "$cache"
          ANCHOR_CACHES[$target_file]=$cache
        fi

        slug=$(slugify "$anchor")
        if grep -Fqx -- "$anchor" "$cache" || grep -Fqx -- "$slug" "$cache"; then
          :
        else
          report_error "$file" "$lineno" "$target" "missing anchor: #$anchor"
        fi
      fi
    done
  done < "$file"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

printf '\n'
printf 'Files scanned : %s\n' "${#FILES[@]}"
printf 'Local links   : %s\n' "$LOCAL_LINKS"
printf 'External links: %s' "$EXTERNAL_LINKS"
if [ "$CHECK_EXTERNAL" -eq 1 ]; then
  printf ' (checked: %s)' "$EXTERNAL_CHECKED"
fi
printf '\n'
printf 'Ignored       : %s (absolute/empty anchors)\n' "$IGNORED_LINKS"

if [ "$ERRORS" -eq 0 ]; then
  printf 'Result        : OK\n'
  exit 0
fi

printf 'Result        : %s problem(s)\n' "$ERRORS"
if [ "$STRICT" -eq 1 ]; then
  exit 1
fi
printf 'Warn-only mode: exiting 0\n'
exit 0
