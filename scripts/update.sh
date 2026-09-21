#!/usr/bin/env bash
# Update this checkout and its environment, then report what still needs a restart.
#
#   scripts/update.sh                 pull, sync, and report
#   scripts/update.sh /path/project   pull, sync, restart that project's UI, and report
#   scripts/update.sh --check         report only; change nothing
#   scripts/update.sh --offline       do not touch the network (use with --check)
#
# The MCP server cannot be reloaded in place: the MCP client owns that process, so the
# last step is always a client action — restart the server in the MCP panel, or reload
# the window. `status.version.restart_required` reports whether the running server
# predates the installed code.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
OFFLINE=0
PROJECT_ROOT=""

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^$/d'
}

for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --offline) OFFLINE=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) printf 'update.sh: unknown option: %s\n' "$arg" >&2; exit 2 ;;
    *) PROJECT_ROOT="$arg" ;;
  esac
done

PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || true)"
fi
if [ -z "$PYTHON" ]; then
  printf 'update.sh: no Python interpreter found\n' >&2
  exit 1
fi

declared_version() {
  "$PYTHON" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' \
    "$REPO_ROOT/pyproject.toml" 2>/dev/null || printf 'unknown'
}

installed_version() {
  "$PYTHON" -c 'from importlib.metadata import version; print(version("research-ultra-rag-mcp"))' \
    2>/dev/null || printf 'unknown'
}

upstream_ref() {
  git -C "$REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true
}

behind_count() {
  local upstream
  upstream="$(upstream_ref)"
  if [ -z "$upstream" ]; then
    printf 'unknown'
    return
  fi
  git -C "$REPO_ROOT" rev-list --count "HEAD..$upstream" 2>/dev/null || printf 'unknown'
}

report_versions() {
  printf '  declared in pyproject: %s\n' "$(declared_version)"
  printf '  installed in the venv: %s\n' "$(installed_version)"
}

if [ "$CHECK" = 1 ]; then
  printf 'Check only: nothing is changed.\n'
  printf '  checkout: %s\n' "$REPO_ROOT"
  printf '  branch:   %s\n' "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  if [ "$OFFLINE" = 1 ]; then
    printf '  remote:   not fetched (--offline)\n'
  elif git -C "$REPO_ROOT" fetch --quiet --prune --tags; then
    printf '  remote:   fetched\n'
  else
    printf 'update.sh: git fetch failed; reporting the local state only\n' >&2
  fi
  printf '  commits behind upstream: %s\n' "$(behind_count)"
  report_versions
  if [ "$OFFLINE" != 1 ]; then
    printf '  dependency changes: '
    if (cd "$REPO_ROOT" && uv sync --dry-run) 2>/dev/null | grep -q 'Would install\|Would uninstall'; then
      printf 'pending — run scripts/update.sh\n'
    else
      printf 'none\n'
    fi
  fi
  printf 'The MCP server keeps running the code it started with; restart it in your client.\n'
  exit 0
fi

printf 'Updating %s\n' "$REPO_ROOT"
if [ "$OFFLINE" = 1 ]; then
  printf 'update.sh: --offline cannot pull; drop --offline to update\n' >&2
  exit 2
fi
if ! git -C "$REPO_ROOT" pull --ff-only; then
  printf 'update.sh: git pull --ff-only failed; resolve the checkout by hand and rerun\n' >&2
  exit 1
fi
(cd "$REPO_ROOT" && uv sync)
printf 'Environment synced.\n'
report_versions

if [ -n "$PROJECT_ROOT" ]; then
  launcher="$PROJECT_ROOT/open-ui.sh"
  if [ -x "$launcher" ]; then
    printf 'Restarting the UI for %s\n' "$PROJECT_ROOT"
    "$launcher" --stop || true
    "$launcher" || printf 'update.sh: the UI did not start; read the launcher log\n' >&2
  else
    printf 'update.sh: no launcher at %s; start the UI yourself\n' "$launcher" >&2
  fi
fi

printf '\nThe MCP server itself still runs the version it started with. Restart it in your\n'
printf 'client (MCP panel: toggle or restart the server, or reload the window), then check\n'
printf 'status.version.restart_required — it should read false.\n'
