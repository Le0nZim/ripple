#!/usr/bin/env bash
# =============================================================================
# RIPPLE Quick Start Script
# =============================================================================
# One-command setup and launch for biology labs (macOS + Linux).
#
# This script:
#   1. Checks system requirements and downloads a portable JDK, Maven,
#      and Miniconda when they are missing (no admin/sudo)
#   2. Detects NVIDIA GPU availability
#   3. Asks the user to choose CPU or GPU mode
#   4. Creates/activates the ripple-env conda environment
#   5. Installs Python dependencies
#   6. Builds and launches RIPPLE
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_help() {
    cat <<'EOF'
Usage: bash quickstart.sh [options]

One-command RIPPLE setup for macOS and Linux. Missing JDK 17+, Maven, and
Miniconda are downloaded into tools/ or ~/miniconda3 after confirmation.

  --yes, -y       Install missing tools without prompting
  --cpu           Force CPU mode
  --gpu           Prefer GPU mode when an NVIDIA GPU is available
  --check         Doctor mode: report tools, do not install or launch
  --no-launch     Set up but do not start RIPPLE
  --help, -h      Show this help

Environment:
  RIPPLE_ASSUME_YES=1     Same as --yes
  RIPPLE_NO_LAUNCH=1      Same as --no-launch
  RIPPLE_TOOLS_DIR=...    Override portable JDK/Maven location
  RIPPLE_FORCE_REBUILD=1  Force mvn clean package
EOF
}

ASSUME_YES="${RIPPLE_ASSUME_YES:-0}"
FORCE_CPU=0
FORCE_GPU=0
CHECK_ONLY=0
NO_LAUNCH="${RIPPLE_NO_LAUNCH:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y) ASSUME_YES=1 ;;
        --cpu) FORCE_CPU=1 ;;
        --gpu) FORCE_GPU=1 ;;
        --check) CHECK_ONLY=1 ;;
        --no-launch) NO_LAUNCH=1 ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
    shift
done

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                     RIPPLE                               ║"
echo "║        Video Annotation Tool for Biology                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export RIPPLE_PROJECT_DIR="$SCRIPT_DIR"

CONDA_ENV_NAME="ripple-env"
RIPPLE_REQUIRED_JAVA_MAJOR=17
APPLE_SILICON=false

# shellcheck source=scripts/lib/java_env_check.sh
source "${SCRIPT_DIR}/scripts/lib/java_env_check.sh"
# shellcheck source=scripts/lib/conda_env_check.sh
source "${SCRIPT_DIR}/scripts/lib/conda_env_check.sh"
# shellcheck source=scripts/lib/bootstrap_tools.sh
source "${SCRIPT_DIR}/scripts/lib/bootstrap_tools.sh"

print_manual_tool_help() {
    echo ""
    echo -e "  ${YELLOW}Portable install skipped.${NC} You can also install tools yourself:"
    echo "    JDK 17+ (full JDK with javac): https://adoptium.net/"
    echo "    Maven 3.8+: https://maven.apache.org/download.cgi"
    echo "    Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "  Then re-run: bash quickstart.sh"
}

# =============================================================================
# STEP 1: Check System Requirements
# =============================================================================
echo -e "${BLUE}[1/6] Checking system requirements...${NC}"

OS=$(uname -s)
ARCH=$(uname -m)
if [[ "$OS" == "Linux" ]]; then
    echo -e "  ${GREEN}✓${NC} Linux detected ($ARCH)"
elif [[ "$OS" == "Darwin" ]]; then
    echo -e "  ${GREEN}✓${NC} macOS detected ($ARCH)"
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "  ${GREEN}✓${NC} Apple Silicon detected"
        APPLE_SILICON=true
    fi
    if xattr -l "$SCRIPT_DIR/quickstart.sh" 2>/dev/null | grep -q "com.apple.quarantine"; then
        echo -e "  ${YELLOW}!${NC} Gatekeeper quarantine detected"
        echo -e "    Run this command to fix: ${BOLD}xattr -cr \"$SCRIPT_DIR\"${NC}"
        echo ""
        if [[ "$ASSUME_YES" == "1" ]]; then
            xattr -cr "$SCRIPT_DIR"
            echo -e "  ${GREEN}✓${NC} Quarantine flags cleared"
        else
            read -r -p "  Would you like to clear quarantine now? [Y/n]: " CLEAR_QUARANTINE
            if [[ ! "$CLEAR_QUARANTINE" =~ ^[Nn]$ ]]; then
                xattr -cr "$SCRIPT_DIR"
                echo -e "  ${GREEN}✓${NC} Quarantine flags cleared"
            fi
        fi
    fi
elif [[ "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
    echo -e "  ${RED}✗${NC} Please run quickstart.bat on Windows (double-click or from cmd)."
    exit 1
else
    echo -e "  ${YELLOW}!${NC} Unknown OS: $OS"
fi

ripple_scan_tools
echo ""
echo -e "  ${BOLD}Preflight${NC}"
if [[ "$NEED_JDK" == "0" ]]; then
    echo -e "    ${GREEN}✓${NC} JDK 17+: $JDK_LOCATION"
else
    echo -e "    ${YELLOW}!${NC} JDK 17+ not found (will download Eclipse Temurin into tools/jdk)"
fi
if [[ "$NEED_MAVEN" == "0" ]]; then
    echo -e "    ${GREEN}✓${NC} Maven: $MAVEN_LOCATION"
else
    echo -e "    ${YELLOW}!${NC} Maven 3.8+ not found (will download Apache Maven into tools/maven)"
fi
if [[ "$NEED_CONDA" == "0" ]]; then
    echo -e "    ${GREEN}✓${NC} Conda: $CONDA_LOCATION"
else
    echo -e "    ${YELLOW}!${NC} Conda not found (will install Miniconda to ${RIPPLE_MINICONDA_HOME})"
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    echo ""
    if [[ "$NEED_JDK" == "0" && "$NEED_MAVEN" == "0" && "$NEED_CONDA" == "0" ]]; then
        ripple_apply_toolchain
        validate_java_toolchain 1
        echo -e "  ${GREEN}✓${NC} Doctor check passed"
        exit 0
    fi
    echo -e "  ${YELLOW}!${NC} Doctor check found missing tools. Re-run without --check to install."
    exit 1
fi

if [[ "$NEED_JDK" == "1" || "$NEED_MAVEN" == "1" || "$NEED_CONDA" == "1" ]]; then
    echo ""
    if [[ "$ASSUME_YES" != "1" ]]; then
        read -r -p "  Install missing tools now? [Y/n]: " INSTALL_MISSING
        if [[ "$INSTALL_MISSING" =~ ^[Nn]$ ]]; then
            print_manual_tool_help
            exit 1
        fi
    else
        echo "  Installing missing tools (--yes)..."
    fi
    if ! ripple_install_missing_tools; then
        echo -e "  ${RED}✗${NC} Portable tool install failed."
        echo "    Check your network, firewall, or proxy, then re-run:"
        echo "      bash quickstart.sh --yes"
        exit 1
    fi
else
    ripple_apply_toolchain
    ripple_init_conda_shell || true
fi

echo "  Checking Java toolchain (JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ required)..."
if ! validate_java_toolchain 1; then
    echo -e "  ${RED}✗${NC} Compatible JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ not found after setup"
    echo "    Re-run: bash quickstart.sh --yes"
    echo "    Or install a full JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ from https://adoptium.net/"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ toolchain validated"

if ! command -v conda >/dev/null 2>&1; then
    if ! ripple_init_conda_shell; then
        echo -e "  ${RED}✗${NC} Conda not found after setup"
        echo "    Re-run: bash quickstart.sh --yes"
        echo "    Or install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
fi
echo -e "  ${GREEN}✓${NC} Conda found"

if ! command -v mvn >/dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} Maven not found after setup"
    echo "    Re-run: bash quickstart.sh --yes"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Maven found"

# =============================================================================
# STEP 2: Detect GPU
# =============================================================================
echo -e "\n${BLUE}[2/6] Detecting GPU...${NC}"

GPU_AVAILABLE=false
if [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected: $GPU_NAME ($GPU_MEM)"
        GPU_AVAILABLE=true
    elif command -v nvidia-smi &> /dev/null; then
        echo -e "  ${YELLOW}!${NC} nvidia-smi found but GPU not accessible"
    else
        echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    echo -e "  ${YELLOW}!${NC} macOS detected - GPU mode not available"
else
    echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected"
fi

# =============================================================================
# STEP 3: User Selection
# =============================================================================
echo -e "\n${BLUE}[3/6] Installation mode selection...${NC}"

if [[ "$FORCE_CPU" == "1" ]]; then
    GPU_MODE="cpu"
    echo -e "  ${GREEN}✓${NC} CPU mode selected (--cpu)"
elif [[ "$FORCE_GPU" == "1" ]]; then
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        GPU_MODE="gpu"
        echo -e "  ${GREEN}✓${NC} GPU mode selected (--gpu)"
    else
        GPU_MODE="cpu"
        echo -e "  ${YELLOW}!${NC} --gpu requested but no NVIDIA GPU found; using CPU mode"
    fi
elif [[ "$GPU_AVAILABLE" == "true" ]]; then
    echo ""
    echo -e "  ${BOLD}Please select installation mode:${NC}"
    echo ""
    echo "    [1] GPU mode (recommended)"
    echo "        - Full functionality: RAFT, LocoTrack, TrackPy, DIS"
    echo "        - Requires NVIDIA GPU with CUDA support"
    echo ""
    echo "    [2] CPU mode"
    echo "        - Limited functionality: TrackPy, DIS optical flow"
    echo "        - Works on any system"
    echo ""
    while true; do
        read -r -p "  Enter your choice [1/2]: " choice
        case $choice in
            1)
                GPU_MODE="gpu"
                echo -e "\n  ${GREEN}✓${NC} GPU mode selected"
                break
                ;;
            2)
                GPU_MODE="cpu"
                echo -e "\n  ${GREEN}✓${NC} CPU mode selected"
                break
                ;;
            *)
                echo -e "  ${RED}Invalid choice. Please enter 1 or 2.${NC}"
                ;;
        esac
    done
else
    GPU_MODE="cpu"
    echo -e "  ${YELLOW}→${NC} CPU mode will be used (no NVIDIA GPU available)"
    echo "    Note: RAFT and LocoTrack require an NVIDIA GPU"
    echo "    TrackPy and DIS optical flow will be available"
fi

# =============================================================================
# STEP 4: Setup Conda Environment
# =============================================================================
echo -e "\n${BLUE}[4/6] Setting up conda environment...${NC}"

if ! ripple_init_conda_shell; then
    echo -e "  ${RED}✗${NC} Could not initialize conda for this shell"
    exit 1
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >/dev/null 2>&1 || true

CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo -e "  ${GREEN}✓${NC} Environment '${CONDA_ENV_NAME}' already exists"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "  Creating new environment '${CONDA_ENV_NAME}'..."
    if [[ "$GPU_MODE" == "gpu" ]]; then
        ENV_FILE="conda/environment.yml"
    else
        ENV_FILE="conda/environment-cpu.yml"
    fi

    if [[ "$OS" == "Darwin" && "$APPLE_SILICON" == "true" ]]; then
        echo -e "  ${YELLOW}Note:${NC} Apple Silicon detected. Some packages may be installed"
        echo -e "        via Rosetta 2 emulation if ARM64 wheels are unavailable."
    fi

    if [[ -f "$ENV_FILE" ]]; then
        conda env create -f "$ENV_FILE" -n "${CONDA_ENV_NAME}"
    else
        echo "  Environment file not found, creating minimal environment..."
        conda create -n "${CONDA_ENV_NAME}" python=3.11 pip -y
    fi

    conda activate "${CONDA_ENV_NAME}"
    echo -e "  ${GREEN}✓${NC} Environment created and activated"
fi

# =============================================================================
# STEP 5: Install Dependencies
# =============================================================================
echo -e "\n${BLUE}[5/6] Installing dependencies...${NC}"

set +e
validate_conda_environment "$GPU_MODE" "$CONDA_ENV_NAME" "$SCRIPT_DIR"
ENV_STATUS=$?
set -e

if [[ "$ENV_STATUS" -eq 0 ]]; then
    echo -e "  ${GREEN}✓${NC} Dependencies already installed"
elif [[ "$ENV_STATUS" -eq 2 ]]; then
    repair_conda_environment "$GPU_MODE" "$SCRIPT_DIR"
    echo -e "  ${GREEN}✓${NC} Dependencies installed"
else
    echo -e "  ${RED}✗${NC} Conda environment validation failed"
    exit 1
fi

# =============================================================================
# STEP 6: Build and Launch
# =============================================================================
echo -e "\n${BLUE}[6/6] Building and launching RIPPLE...${NC}"

FORCE_REBUILD="${RIPPLE_FORCE_REBUILD:-0}"

get_latest_source_mtime() {
    python - <<'PY'
import os

paths = [
    'pom.xml',
    'src/main/java',
    'src/main/resources',
    'src/main/python',
]

max_mtime = 0.0
for p in paths:
    if os.path.isfile(p):
        try:
            max_mtime = max(max_mtime, os.path.getmtime(p))
        except OSError:
            pass
    elif os.path.isdir(p):
        for root, _dirs, files in os.walk(p):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    max_mtime = max(max_mtime, os.path.getmtime(fp))
                except OSError:
                    pass

print(int(max_mtime))
PY
}

get_file_mtime() {
    local path="$1"
    python - <<PY
import os
p = r'''$path'''
print(int(os.path.getmtime(p)) if os.path.exists(p) else 0)
PY
}

JAR_PATH="target/ripple.jar"
NEED_BUILD=false

if [[ "$FORCE_REBUILD" == "1" ]]; then
    NEED_BUILD=true
elif [[ ! -f "$JAR_PATH" ]]; then
    NEED_BUILD=true
else
    LATEST_SRC_MTIME="$(get_latest_source_mtime)"
    JAR_MTIME="$(get_file_mtime "$JAR_PATH")"
    if [[ "$LATEST_SRC_MTIME" -gt "$JAR_MTIME" ]]; then
        NEED_BUILD=true
    fi
fi

if [[ "$NEED_BUILD" == "true" ]]; then
    echo "  Building Java application..."
    if [[ "$FORCE_REBUILD" == "1" ]]; then
        mvn clean package -DskipTests -q
    else
        mvn package -DskipTests -q
    fi
    echo -e "  ${GREEN}✓${NC} Build complete"
else
    echo -e "  ${GREEN}✓${NC} JAR already up to date"
fi

echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  RIPPLE Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# Create launch shortcuts
# =============================================================================
echo -e "${BLUE}Creating launch shortcuts...${NC}"

LAUNCHER_JAVA_HOME="${RIPPLE_RESOLVED_JAVA_HOME:-${JAVA_HOME:-}}"

cat > "${SCRIPT_DIR}/RIPPLE.sh" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# RIPPLE Launcher - Auto-generated by quickstart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x "${SCRIPT_DIR}/tools/jdk/bin/java" ]]; then
    export JAVA_HOME="${SCRIPT_DIR}/tools/jdk"
    export PATH="${JAVA_HOME}/bin:${PATH}"
elif [[ -n "${RIPPLE_LAUNCHER_JAVA_HOME}" && -x "${RIPPLE_LAUNCHER_JAVA_HOME}/bin/java" ]]; then
    export JAVA_HOME="${RIPPLE_LAUNCHER_JAVA_HOME}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
fi

if [[ -x "${SCRIPT_DIR}/tools/maven/bin/mvn" ]]; then
    export PATH="${SCRIPT_DIR}/tools/maven/bin:${PATH}"
fi

find_conda() {
    local CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "$HOME/mambaforge/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"
        "/opt/homebrew/Caskroom/mambaforge/base/bin/conda"
        "/usr/local/Caskroom/miniconda/base/bin/conda"
        "/usr/local/Caskroom/miniforge/base/bin/conda"
    )
    local cpath pattern
    for cpath in "${CONDA_PATHS[@]}"; do
        if [[ -x "$cpath" ]]; then
            echo "$cpath"
            return 0
        fi
    done
    for pattern in "/opt/homebrew/Caskroom/miniconda"/*/base/bin/conda \
                   "/opt/homebrew/Caskroom/miniforge"/*/base/bin/conda \
                   "/opt/homebrew/Caskroom/mambaforge"/*/base/bin/conda \
                   "/usr/local/Caskroom/miniconda"/*/base/bin/conda \
                   "/usr/local/Caskroom/miniforge"/*/base/bin/conda; do
        for cpath in $pattern; do
            if [[ -x "$cpath" ]]; then
                echo "$cpath"
                return 0
            fi
        done
    done
    return 1
}

if ! command -v conda &> /dev/null; then
    CONDA_EXE=$(find_conda)
    if [[ -n "$CONDA_EXE" ]]; then
        eval "$($CONDA_EXE shell.bash hook)"
    fi
fi

if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [[ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/mamba.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
else
    echo "ERROR: conda not found. Please ensure conda is installed and in your PATH."
    echo "       Or run: bash quickstart.sh to set up the environment."
    exit 1
fi
LAUNCHER_EOF

if [[ -n "$LAUNCHER_JAVA_HOME" ]]; then
    # Insert the captured system JAVA_HOME fallback after the shebang block.
    tmp_launcher="${SCRIPT_DIR}/RIPPLE.sh.tmp"
    awk -v home="$LAUNCHER_JAVA_HOME" '
        NR==6 { print "RIPPLE_LAUNCHER_JAVA_HOME=\"" home "\"" }
        { print }
    ' "${SCRIPT_DIR}/RIPPLE.sh" > "$tmp_launcher"
    mv "$tmp_launcher" "${SCRIPT_DIR}/RIPPLE.sh"
fi

{
    echo "conda activate ${CONDA_ENV_NAME}"
    echo "export RIPPLE_MODE=${GPU_MODE}"
    echo 'java -jar target/ripple.jar "$@"'
} >> "${SCRIPT_DIR}/RIPPLE.sh"

chmod +x "${SCRIPT_DIR}/RIPPLE.sh"
echo -e "  ${GREEN}✓${NC} Created RIPPLE.sh"

if [[ "$OS" == "Linux" ]]; then
    DESKTOP_FILE="${HOME}/Desktop/RIPPLE.desktop"
    APPLICATIONS_FILE="${HOME}/.local/share/applications/ripple.desktop"

    cat > "${SCRIPT_DIR}/RIPPLE.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RIPPLE
Comment=Video Annotation Tool for Biology
Exec=bash -c 'cd "${SCRIPT_DIR}" && ./RIPPLE.sh'
Icon=applications-science
Terminal=true
Categories=Science;Education;
StartupNotify=true
EOF

    if [[ -d "${HOME}/Desktop" ]]; then
        cp "${SCRIPT_DIR}/RIPPLE.desktop" "${DESKTOP_FILE}"
        chmod +x "${DESKTOP_FILE}"
        gio set "${DESKTOP_FILE}" metadata::trusted true 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Created desktop shortcut"
    fi

    mkdir -p "${HOME}/.local/share/applications"
    cp "${SCRIPT_DIR}/RIPPLE.desktop" "${APPLICATIONS_FILE}"
    echo -e "  ${GREEN}✓${NC} Added to applications menu"

elif [[ "$OS" == "Darwin" ]]; then
    cat > "${SCRIPT_DIR}/RIPPLE.command" << MACOS_EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
./RIPPLE.sh
MACOS_EOF
    chmod +x "${SCRIPT_DIR}/RIPPLE.command"
    echo -e "  ${GREEN}✓${NC} Created RIPPLE.command (double-click to launch)"

    if [[ -d "${HOME}/Desktop" ]]; then
        ln -sf "${SCRIPT_DIR}/RIPPLE.command" "${HOME}/Desktop/RIPPLE.command" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Created desktop alias"
    fi
fi

echo ""
echo -e "${BOLD}Launch options:${NC}"
echo "  1. Run ./RIPPLE.sh from this folder"
if [[ "$OS" == "Linux" ]]; then
    echo "  2. Use the desktop shortcut or find RIPPLE in applications menu"
elif [[ "$OS" == "Darwin" ]]; then
    echo "  2. Double-click RIPPLE.command"
fi
echo ""
GPU_MODE_UPPER=$(echo "$GPU_MODE" | tr '[:lower:]' '[:upper:]')
echo -e "  Mode: ${BOLD}${GPU_MODE_UPPER}${NC}"
echo ""

export RIPPLE_MODE="${GPU_MODE}"

if [[ "$NO_LAUNCH" == "1" ]]; then
    echo -e "${BOLD}Setup finished without launching (--no-launch).${NC}"
    echo ""
    exit 0
fi

echo -e "${BOLD}${GREEN}Launching RIPPLE now...${NC}"
echo ""
java -jar target/ripple.jar
