#!/usr/bin/env bash
# ==============================================================================
# Persistent RAFT Tracking Service Management Interface (Linux/macOS)
# ==============================================================================
#
# Linux/macOS equivalent of run_persistent_tracking.bat
# Default transport (Windows parity): TCP (SOCKET_HOST/SOCKET_PORT)
# Optional Unix socket mode: set RIPPLE_TRANSPORT=unix and SOCKET_PATH
#
# Supported operations:
#   start              - Initialize persistent tracking server process
#   stop               - Terminate server with graceful shutdown sequence
#   status             - Query server operational state
#   restart            - Perform stop-start cycle with cleanup
#   compute_flow       - Compute optical flow for video
#   track_seed         - Track a single seed point
#   track_anchors      - Track with multiple anchor points
#   visualize_flow     - Generate flow visualization
#   load_flow          - Load cached flow into memory
#   compress_video     - Compress video for faster processing
#   clear_cache        - Clear server's flow cache
#   clear_memory       - Clear server's memory
#   preview_dog_detection     - Preview DoG detection
#   preview_trackpy_trajectories - Preview TrackPy trajectories
#   optimize_tracks    - Optimize track positions
#   ping               - Test server connection
#
# Environment variables (override as needed):
#   CONDA_ENV        (default: ripple-env)
#   MODEL_SIZE       (default: large)
#   SELECTED_GPU     (default: auto - use 0, 1, etc. for specific GPU)
#   SERVER_SCRIPT    (default: <repo>/src/main/python/tracking_server.py)
#   SEND_CMD_SCRIPT  (default: <repo>/src/main/python/send_command.py)
#   LOG_FILE         (default: /tmp/ripple-env.log)
#   PID_FILE         (default: /tmp/ripple-env.pid)
#   SOCKET_PATH      (default: /tmp/ripple-env.sock)   [unix transport]
#   SOCKET_HOST      (default: 127.0.0.1)              [tcp transport]
#   SOCKET_PORT      (default: 9876)                   [tcp transport]
# ==============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-ripple-env}"
MODEL_SIZE="${MODEL_SIZE:-large}"
SELECTED_GPU="${SELECTED_GPU:-auto}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${PROJECT_DIR}/src/main/python/tracking_server.py}"
SEND_CMD_SCRIPT="${SEND_CMD_SCRIPT:-${PROJECT_DIR}/src/main/python/send_command.py}"

LOG_FILE_DEFAULT="/tmp/ripple-env.log"
PID_FILE_DEFAULT="/tmp/ripple-env.pid"
SOCKET_PATH_DEFAULT="/tmp/ripple-env.sock"

if [[ -z "${LOG_FILE+x}" ]]; then LOG_FILE="${LOG_FILE_DEFAULT}"; fi
if [[ -z "${PID_FILE+x}" ]]; then PID_FILE="${PID_FILE_DEFAULT}"; fi
if [[ -z "${SOCKET_PATH+x}" ]]; then SOCKET_PATH="${SOCKET_PATH_DEFAULT}"; fi
SOCKET_HOST="${SOCKET_HOST:-127.0.0.1}"
SOCKET_PORT="${SOCKET_PORT:-9876}"

info()    { echo -e "\033[36m[INFO]\033[0m $*"; }
success() { echo -e "\033[32m[SUCCESS]\033[0m $*"; }
warning() { echo -e "\033[33m[WARNING]\033[0m $*"; }
error()   { echo -e "\033[31m[ERROR]\033[0m $*" >&2; }

pick_runtime_dir() {
  local d testfile
  for d in "${XDG_RUNTIME_DIR:-}" "/tmp" "/var/tmp" "$HOME/.cache/ripple"; do
    [[ -n "${d}" ]] || continue
    mkdir -p "${d}" 2>/dev/null || continue
    [[ -w "${d}" ]] || continue
    testfile="${d}/.ripple_write_test.$$"
    if ( : >"${testfile}" ) 2>/dev/null; then
      rm -f "${testfile}" 2>/dev/null || true
      echo "${d}"
      return 0
    fi
  done

  echo "/tmp"
  return 0
}

ensure_writable_append_file() {
  local path dir
  path="$1"
  dir="$(dirname "${path}")"
  mkdir -p "${dir}" 2>/dev/null || return 1
  [[ -w "${dir}" ]] || return 1
  if [[ -e "${path}" && ! -w "${path}" ]]; then
    return 1
  fi
  ( : >>"${path}" ) 2>/dev/null || return 1
  return 0
}

RUNTIME_DIR="$(pick_runtime_dir)"

if [[ "${LOG_FILE}" == "${LOG_FILE_DEFAULT}" ]]; then LOG_FILE="${RUNTIME_DIR}/ripple-env.log"; fi
if [[ "${PID_FILE}" == "${PID_FILE_DEFAULT}" ]]; then PID_FILE="${RUNTIME_DIR}/ripple-env.pid"; fi
if [[ "${SOCKET_PATH}" == "${SOCKET_PATH_DEFAULT}" ]]; then SOCKET_PATH="${RUNTIME_DIR}/ripple-env.sock"; fi

transport_is_tcp() {
  local t
  # Windows parity: default to TCP unless explicitly set otherwise.
  t="${RIPPLE_TRANSPORT:-tcp}"
  t_lower=$(echo "$t" | tr '[:upper:]' '[:lower:]')
  [[ "${t_lower}" == "tcp" || "${t_lower}" == "inet" ]]
}

activate_conda() {
  # Conda activation scripts can reference unset variables
  set +u

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" >/dev/null 2>&1 || {
      set -u
      error "Failed to activate conda env: ${CONDA_ENV}"
      exit 1
    }
    set -u
    return
  fi

  for d in "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
    if [[ -f "$d/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      . "$d/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV}" >/dev/null 2>&1 || {
        set -u
        error "Failed to activate conda env: ${CONDA_ENV}"
        exit 1
      }
      set -u
      return
    fi
  done

  set -u
  error "conda not found. Install Miniconda/Anaconda or add 'conda' to PATH."
  exit 1
}

python_exec() {
  # Prefer the activated env python if available
  if command -v python >/dev/null 2>&1; then
    echo "python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    error "Python not found in PATH (after conda activation)."
    exit 1
  fi
}

python_exec_client() {
  # send_command.py uses only Python stdlib; avoid conda activation cost.
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    # Fallback: use conda env python if system python is unavailable.
    activate_conda
    python_exec
  fi
}

tcp_can_connect() {
  local py
  py="$(python_exec)"
  "${py}" - <<PY >/dev/null 2>&1
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.25)
try:
    s.connect(("${SOCKET_HOST}", int("${SOCKET_PORT}")))
    print("ok")
finally:
    s.close()
PY
}

is_server_running() {
  if transport_is_tcp; then
    tcp_can_connect
    return $?
  fi

  # Unix socket mode
  if [[ -S "${SOCKET_PATH}" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
      local pid
      pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        return 0
      fi
    else
      # Socket exists even without PID file; treat as running
      return 0
    fi
  fi
  return 1
}

start_server() {
  if is_server_running; then
    info "Server is already running"
    return 0
  fi

  activate_conda

  if ! ensure_writable_append_file "${LOG_FILE}"; then
    local fallback
    fallback="${RUNTIME_DIR}/ripple-env-$(id -u).log"
    warning "Log file not writable: ${LOG_FILE}"
    warning "Falling back to: ${fallback}"
    LOG_FILE="${fallback}"
    ensure_writable_append_file "${LOG_FILE}" || {
      error "Fallback log file still not writable: ${LOG_FILE}"
      return 1
    }
  fi

  mkdir -p "$(dirname "${LOG_FILE}")" 2>/dev/null || true
  rm -f "${PID_FILE}" 2>/dev/null || true
  if ! transport_is_tcp; then
    rm -f "${SOCKET_PATH}" 2>/dev/null || true
  fi

  local py
  py="$(python_exec)"

  info "Starting persistent tracking server..."
  info "Using Python: ${py} (conda env: ${CONDA_ENV})"
  info "Script: ${SERVER_SCRIPT}"

  local args
  if transport_is_tcp; then
    args=("--tcp-host" "${SOCKET_HOST}" "--tcp-port" "${SOCKET_PORT}")
  else
    args=("--socket" "${SOCKET_PATH}")
  fi

  # Build GPU arguments
  local gpu_args=()
  if [[ "${SELECTED_GPU}" != "auto" && -n "${SELECTED_GPU}" ]]; then
    gpu_args=("--gpu" "${SELECTED_GPU}")
    info "Using GPU: ${SELECTED_GPU}"
  else
    info "Using GPU: auto (automatic selection)"
  fi

  # Start detached
  nohup "${py}" "${SERVER_SCRIPT}" \
    "${args[@]}" \
    --model "${MODEL_SIZE}" \
    --device auto \
    "${gpu_args[@]}" \
    >>"${LOG_FILE}" 2>&1 &

  echo "$!" >"${PID_FILE}" 2>/dev/null || true

  info "Waiting for server to start..."
  local i
  for i in {1..30}; do
    if is_server_running; then
      success "Server started"
      if transport_is_tcp; then
        echo "  Host: ${SOCKET_HOST}:${SOCKET_PORT}"
      else
        echo "  Socket: ${SOCKET_PATH}"
      fi
      echo "  Logs: ${LOG_FILE}"
      return 0
    fi
    sleep 1
  done

  error "Server failed to start within timeout"
  error "Check log file: ${LOG_FILE}"
  return 1
}

stop_server() {
  info "Stopping server..."

  # Try graceful shutdown first
  if is_server_running; then
    local py
    py="$(python_exec_client)"

    # Force transport selection for send_command.py
    if transport_is_tcp; then
      RIPPLE_TRANSPORT=tcp SOCKET_HOST="${SOCKET_HOST}" SOCKET_PORT="${SOCKET_PORT}" \
        "${py}" "${SEND_CMD_SCRIPT}" stop >/dev/null 2>&1 || true
    else
      RIPPLE_TRANSPORT=unix SOCKET_PATH="${SOCKET_PATH}" \
        "${py}" "${SEND_CMD_SCRIPT}" stop >/dev/null 2>&1 || true
    fi
    sleep 1
  fi

  # Kill lingering process if we have a PID
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi

  rm -f "${PID_FILE}" 2>/dev/null || true
  if ! transport_is_tcp; then
    rm -f "${SOCKET_PATH}" 2>/dev/null || true
  fi

  success "Server stopped"
}

check_status() {
  if is_server_running; then
    success "Server is RUNNING"
    if transport_is_tcp; then
      echo "  Host: ${SOCKET_HOST}:${SOCKET_PORT}"
    else
      echo "  Socket: ${SOCKET_PATH}"
    fi

    local py
    py="$(python_exec_client)"
    if transport_is_tcp; then
      if RIPPLE_TRANSPORT=tcp SOCKET_HOST="${SOCKET_HOST}" SOCKET_PORT="${SOCKET_PORT}" \
        "${py}" "${SEND_CMD_SCRIPT}" ping >/dev/null 2>&1; then
        echo "  Status: Responding to ping"
      else
        echo "  Status: Not responding to ping"
      fi
    else
      if RIPPLE_TRANSPORT=unix SOCKET_PATH="${SOCKET_PATH}" \
        "${py}" "${SEND_CMD_SCRIPT}" ping >/dev/null 2>&1; then
        echo "  Status: Responding to ping"
      else
        echo "  Status: Not responding to ping"
      fi
    fi
    return 0
  fi

  warning "Server is NOT RUNNING"
  echo "  Use '$0 start' to start the server"
  return 1
}

status_json() {
  if is_server_running; then
    local py
    py="$(python_exec_client)"
    if transport_is_tcp; then
      RIPPLE_TRANSPORT=tcp SOCKET_HOST="${SOCKET_HOST}" SOCKET_PORT="${SOCKET_PORT}" \
        "${py}" "${SEND_CMD_SCRIPT}" status
    else
      RIPPLE_TRANSPORT=unix SOCKET_PATH="${SOCKET_PATH}" \
        "${py}" "${SEND_CMD_SCRIPT}" status
    fi
    return $?
  fi

  echo '{"status":"error","message":"Server not running","busy":false}'
  return 1
}

restart_server() {
  stop_server || true
  start_server
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args...]

Lifecycle:
  start | stop | status | status_json | restart

All other commands are forwarded to send_command.py (same as Windows):
  compute_flow, track_seed, track_anchors, visualize_flow, load_flow,
  compress_video, clear_cache, clear_memory, preview_dog_detection,
  preview_trackpy_trajectories, optimize_tracks, ping, etc.

Transport:
  Default (Windows parity): TCP (SOCKET_HOST/SOCKET_PORT)
  Optional Unix socket: set RIPPLE_TRANSPORT=unix and SOCKET_PATH
EOF
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd
  cmd="$1"
  shift || true

  case "${cmd}" in
    start) start_server; exit $? ;;
    stop) stop_server; exit $? ;;
    status) check_status; exit $? ;;
    status_json) status_json; exit $? ;;
    restart) restart_server; exit $? ;;
    *)
      # Windows parity: ensure server is running then forward command.
      if ! is_server_running; then
        info "Server not running, starting it now..."
        start_server || { error "Failed to start server"; exit 1; }
      fi

      local py
      py="$(python_exec_client)"

      if transport_is_tcp; then
        RIPPLE_TRANSPORT=tcp SOCKET_HOST="${SOCKET_HOST}" SOCKET_PORT="${SOCKET_PORT}" \
          "${py}" "${SEND_CMD_SCRIPT}" "${cmd}" "$@"
      else
        RIPPLE_TRANSPORT=unix SOCKET_PATH="${SOCKET_PATH}" \
          "${py}" "${SEND_CMD_SCRIPT}" "${cmd}" "$@"
      fi
      exit $?
      ;;
  esac
}

main "$@"
