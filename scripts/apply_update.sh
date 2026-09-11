#!/usr/bin/env bash
# =============================================================================
# RIPPLE in-place updater. Spawned by the app after the user confirms Update.
# Waits for the Java process to exit, fast-forwards from official GitHub main,
# rebuilds with quickstart, then relaunches.
# =============================================================================
set -e

PID="${1:-}"
MODE="${2:-cpu}"
if [[ "$MODE" != "gpu" ]]; then
    MODE="cpu"
fi

REMOTE="${RIPPLE_UPDATE_REMOTE:-https://github.com/Le0nZim/ripple.git}"
BRANCH="${RIPPLE_UPDATE_BRANCH:-main}"
WAIT_SECONDS="${RIPPLE_UPDATE_WAIT_SECONDS:-120}"
SKIP_QUICKSTART="${RIPPLE_UPDATE_SKIP_QUICKSTART:-0}"
SKIP_RELAUNCH="${RIPPLE_UPDATE_SKIP_RELAUNCH:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "${REPO_ROOT}/tools"
LOG="${REPO_ROOT}/tools/ripple-update.log"

log() {
    echo "$@"
}

log "=== RIPPLE update $(date) ==="
log "Install: ${REPO_ROOT}"
log "Mode: ${MODE}"
log "Remote: ${REMOTE} (${BRANCH})"

if [[ -n "$PID" ]]; then
    log "Waiting for RIPPLE (PID ${PID}) to exit..."
    waited=0
    while kill -0 "$PID" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if [[ "$waited" -ge "$WAIT_SECONDS" ]]; then
            log "Timed out waiting for RIPPLE to exit."
            exit 1
        fi
    done
    log "RIPPLE exited."
fi

if ! command -v git >/dev/null 2>&1; then
    log "git was not found on PATH."
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "This folder is not a git checkout. Re-clone from https://github.com/Le0nZim/ripple.git"
    exit 1
fi

DIRTY="$(git status --porcelain --untracked-files=no || true)"
if [[ -n "$DIRTY" ]]; then
    log "Working tree has local source changes; aborting update."
    log "$DIRTY"
    exit 1
fi

log "Fetching ${REMOTE} ${BRANCH}..."
git fetch "${REMOTE}" "${BRANCH}"
log "Fast-forwarding..."
git merge --ff-only FETCH_HEAD

if [[ "$SKIP_QUICKSTART" != "1" ]]; then
    if [[ "$MODE" == "gpu" ]]; then
        bash "${REPO_ROOT}/quickstart.sh" --yes --gpu --no-launch
    else
        bash "${REPO_ROOT}/quickstart.sh" --yes --cpu --no-launch
    fi
else
    log "Skipping quickstart (RIPPLE_UPDATE_SKIP_QUICKSTART=1)."
fi

if [[ "$SKIP_RELAUNCH" == "1" ]]; then
    log "Skipping relaunch (RIPPLE_UPDATE_SKIP_RELAUNCH=1)."
    exit 0
fi

if [[ -x "${REPO_ROOT}/RIPPLE.sh" ]]; then
    log "Relaunching RIPPLE..."
    exec "${REPO_ROOT}/RIPPLE.sh"
fi

if [[ -f "${REPO_ROOT}/target/ripple.jar" ]]; then
    log "RIPPLE.sh missing; launching JAR directly."
    exec java -jar "${REPO_ROOT}/target/ripple.jar"
fi

log "Update finished but RIPPLE could not be relaunched. Start it with ./RIPPLE.sh"
exit 1
