#!/usr/bin/env bash
# Validates conda environment completeness for the selected RIPPLE install mode.

validate_conda_environment() {
    local gpu_mode="$1"
    local env_name="$2"
    local script_dir="$3"
    local stamp_file="${script_dir}/.ripple-env.stamp"
    local req_file=""
    local req_hash=""
    local stamp_hash=""
    local missing=()

    if [[ "$gpu_mode" == "gpu" ]]; then
        req_file="${script_dir}/requirements/requirements-gpu.txt"
    else
        req_file="${script_dir}/requirements/requirements-cpu.txt"
    fi

    if [[ -f "$req_file" ]]; then
        req_hash=$(sha256sum "$req_file" | awk '{print $1}')
    fi
    if [[ -f "$stamp_file" ]]; then
        stamp_hash=$(awk -F= '/requirements_hash=/{print $2}' "$stamp_file" | tail -n1)
    fi

    local modules=(numpy scipy pandas tifffile cv2 trackpy torch)
    if [[ "$gpu_mode" == "gpu" ]]; then
        modules+=(torchvision)
    fi

    for module in "${modules[@]}"; do
        if ! python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
        then
            missing+=("$module")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        if [[ -n "$req_hash" && "$req_hash" == "$stamp_hash" ]]; then
            echo "  OK: Existing ${env_name} environment validated (fast path)"
            return 0
        fi
        if [[ -z "$stamp_hash" ]]; then
            write_conda_env_stamp "$gpu_mode" "$script_dir"
            echo "  OK: Existing ${env_name} environment validated; stamp written"
            return 0
        fi
        echo "  ! Requirements changed since last install; refreshing dependencies"
        return 2
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "  ! Incomplete ${env_name} environment; missing modules: ${missing[*]}"
        return 2
    fi

    echo "  ! Existing ${env_name} environment needs validation refresh"
    return 2
}

write_conda_env_stamp() {
    local gpu_mode="$1"
    local script_dir="$2"
    local stamp_file="${script_dir}/.ripple-env.stamp"
    local req_file=""
    if [[ "$gpu_mode" == "gpu" ]]; then
        req_file="${script_dir}/requirements/requirements-gpu.txt"
    else
        req_file="${script_dir}/requirements/requirements-cpu.txt"
    fi
    if [[ -f "$req_file" ]]; then
        {
            echo "requirements_hash=$(sha256sum "$req_file" | awk '{print $1}')"
            echo "gpu_mode=${gpu_mode}"
            date -u +generated_utc=%Y-%m-%dT%H:%M:%SZ
        } > "$stamp_file"
    fi
}

repair_conda_environment() {
    local gpu_mode="$1"
    local script_dir="$2"
    pip install --upgrade pip wheel setuptools -q
    if [[ "$gpu_mode" == "gpu" && -f "${script_dir}/requirements/requirements-gpu.txt" ]]; then
        echo "  Installing GPU packages (this may take a few minutes)..."
        pip install -r "${script_dir}/requirements/requirements-gpu.txt" -q
    else
        echo "  Installing CPU packages..."
        pip install -r "${script_dir}/requirements/requirements-cpu.txt" -q
    fi
    write_conda_env_stamp "$gpu_mode" "$script_dir"
}
