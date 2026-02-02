#!/usr/bin/env bash
# =============================================================================
# RIPPLE Quick Start Script
# =============================================================================
# One-command setup and launch for biology labs.
#
# This script:
#   1. Checks system requirements (Java, Conda, Maven) and auto-installs on macOS
#   2. Detects NVIDIA GPU availability
#   3. Asks user to choose CPU or GPU version
#   4. Creates/activates ripple-env conda environment
#   5. Installs all dependencies
#   6. Builds and launches RIPPLE
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                     RIPPLE                               ║"
echo "║        Video Annotation Tool for Biology                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
CONDA_ENV_NAME="ripple-env"

# =============================================================================
# STEP 1: Check System Requirements
# =============================================================================
echo -e "${BLUE}[1/6] Checking system requirements...${NC}"

# Check OS
OS=$(uname -s)
ARCH=$(uname -m)
if [[ "$OS" == "Linux" ]]; then
    echo -e "  ${GREEN}✓${NC} Linux detected ($ARCH)"
elif [[ "$OS" == "Darwin" ]]; then
    echo -e "  ${GREEN}✓${NC} macOS detected ($ARCH)"
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "  ${GREEN}✓${NC} Apple Silicon detected"
        APPLE_SILICON=true
    else
        APPLE_SILICON=false
    fi
    # Check for Gatekeeper quarantine flag and warn user
    if xattr -l "$SCRIPT_DIR/quickstart.sh" 2>/dev/null | grep -q "com.apple.quarantine"; then
        echo -e "  ${YELLOW}!${NC} Gatekeeper quarantine detected"
        echo -e "    Run this command to fix: ${BOLD}xattr -cr \"$SCRIPT_DIR\"${NC}"
        echo ""
        read -p "  Would you like to clear quarantine now? [Y/n]: " CLEAR_QUARANTINE
        if [[ ! "$CLEAR_QUARANTINE" =~ ^[Nn]$ ]]; then
            xattr -cr "$SCRIPT_DIR"
            echo -e "  ${GREEN}✓${NC} Quarantine flags cleared"
        fi
    fi
else
    echo -e "  ${YELLOW}!${NC} Unknown OS: $OS"
fi

# Check Java
if ! command -v java &> /dev/null && [[ "$OS" == "Darwin" ]]; then
    JAVA_HOME_CANDIDATE=$(/usr/libexec/java_home 2>/dev/null || true)
    if [[ -n "$JAVA_HOME_CANDIDATE" ]]; then
        export JAVA_HOME="$JAVA_HOME_CANDIDATE"
        export PATH="$JAVA_HOME/bin:$PATH"
        echo -e "  ${YELLOW}!${NC} Java found via java_home; using it for this session"
        
        # Persist JAVA_HOME and PATH in ~/.zshrc
        ZSHRC="$HOME/.zshrc"
        touch "$ZSHRC"
        if ! grep -q "# RIPPLE JAVA" "$ZSHRC"; then
            {
                echo ""
                echo "# RIPPLE JAVA"
                echo "export JAVA_HOME=\"\$(/usr/libexec/java_home)\""
                echo "export PATH=\"\$JAVA_HOME/bin:\$PATH\""
            } >> "$ZSHRC"
        else
            # Update existing block
            sed -i '' '/# RIPPLE JAVA/,+2c\
            # RIPPLE JAVA\
            export JAVA_HOME="$(/usr/libexec/java_home)"\
            export PATH="$JAVA_HOME/bin:$PATH"' "$ZSHRC"
        fi
        echo -e "  ${GREEN}✓${NC} Updated ~/.zshrc with JAVA_HOME and PATH"
    fi
fi

if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [[ "$JAVA_VERSION" -ge 11 ]]; then
        echo -e "  ${GREEN}✓${NC} Java $JAVA_VERSION"
    else
        echo -e "  ${YELLOW}!${NC} Java $JAVA_VERSION (11+ recommended)"
    fi
else
    if [[ "$OS" == "Darwin" ]]; then
        echo -e "  ${RED}✗${NC} Java not found"
        echo "    Download and install JDK 25:"
        echo "    https://download.oracle.com/java/25/latest/jdk-25_macos-aarch64_bin.dmg"
        echo "    After install, run: /usr/libexec/java_home -V"
        echo "    Add to ~/.zshrc:" 
        echo "      export JAVA_HOME=\"\$(/usr/libexec/java_home -v VERSION)\""
        echo "      export PATH=\"\$JAVA_HOME/bin:\$PATH\""
        echo "    Then: source ~/.zshrc && java -version"
        exit 1
    else
        echo -e "  ${RED}✗${NC} Java not found"
        echo "    Please install Java 11 or newer (OpenJDK recommended)"
        echo "    Ubuntu/Debian: sudo apt install openjdk-17-jdk"
        exit 1
    fi
fi

# Check Conda (handle Homebrew, Miniforge, Miniconda, Anaconda)
if ! command -v conda &> /dev/null; then
    # Try common conda installation paths
    CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "$HOME/mambaforge/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"
        "/usr/local/Caskroom/miniconda/base/bin/conda"
    )
    for cpath in "${CONDA_PATHS[@]}"; do
        if [[ -x "$cpath" ]]; then
            eval "$($cpath shell.bash hook)"
            break
        fi
    done
fi

if ! command -v conda &> /dev/null && [[ "$OS" == "Darwin" ]]; then
    echo -e "  ${YELLOW}!${NC} Conda not found - installing Miniconda"
    mkdir -p ~/miniconda3
    if [[ "$ARCH" == "arm64" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    curl "$MINICONDA_URL" -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
fi

if ! command -v conda &> /dev/null; then
    echo -e "  ${RED}✗${NC} Conda not found"
    echo "    Please install Miniconda or Anaconda:"
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda found"

# Check Maven
if ! command -v mvn &> /dev/null; then
    if [[ "$OS" == "Darwin" ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "  ${YELLOW}!${NC} Homebrew not found - installing"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            if [[ -x /opt/homebrew/bin/brew ]]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [[ -x /usr/local/bin/brew ]]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
        fi
        echo -e "  ${YELLOW}!${NC} Maven not found - installing via Homebrew"
        brew install maven
    else
        echo -e "  ${RED}✗${NC} Maven not found"
        echo "    Please install Maven 3.8+ from https://maven.apache.org/download.cgi"
        echo "    Ubuntu/Debian: sudo apt install maven"
        exit 1
    fi
fi

if ! command -v mvn &> /dev/null; then
    echo -e "  ${RED}✗${NC} Maven not found"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Maven found"

# =============================================================================
# STEP 2: Detect GPU
# =============================================================================
echo -e "\n${BLUE}[2/6] Detecting GPU...${NC}"

GPU_AVAILABLE=false

# Only check for NVIDIA GPU on Linux
if [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
            echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected: $GPU_NAME ($GPU_MEM)"
            GPU_AVAILABLE=true
        else
            echo -e "  ${YELLOW}!${NC} nvidia-smi found but GPU not accessible"
        fi
    else
        echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected"
    fi
else
    echo -e "  ${YELLOW}!${NC} macOS detected - GPU mode not available"
fi

# =============================================================================
# STEP 3: User Selection
# =============================================================================
echo -e "\n${BLUE}[3/6] Installation mode selection...${NC}"

if [[ "$GPU_AVAILABLE" == "true" ]]; then
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
        read -p "  Enter your choice [1/2]: " choice
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

# Accept conda Terms of Service for default channels (match Windows quickstart.bat)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >/dev/null 2>&1 || true

# Find conda base and source it (handle various conda installations)
CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]]; then
    # Miniforge/Mambaforge use mamba.sh
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
else
    # Fallback: try to initialize via conda command
    eval "$(conda shell.bash hook)"
fi

# Check if environment already exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo -e "  ${GREEN}✓${NC} Environment '${CONDA_ENV_NAME}' already exists"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "  Creating new environment '${CONDA_ENV_NAME}'..."
    
    # Select appropriate environment file
    if [[ "$GPU_MODE" == "gpu" ]]; then
        ENV_FILE="conda/environment.yml"
    else
        ENV_FILE="conda/environment-cpu.yml"
    fi
    
    # Warn about Apple Silicon compatibility
    if [[ "$OS" == "Darwin" ]] && [[ "$APPLE_SILICON" == "true" ]]; then
        echo -e "  ${YELLOW}Note:${NC} Apple Silicon detected. Some packages may be installed"
        echo -e "        via Rosetta 2 emulation if ARM64 wheels are unavailable."
    fi
    
    if [[ -f "$ENV_FILE" ]]; then
        # Create environment from file, but override the name
        conda env create -f "$ENV_FILE" -n "${CONDA_ENV_NAME}" -y
    else
        # Fallback: create minimal environment and install via pip
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

# Match Windows quickstart.bat behavior: only install if trackpy missing
if python -c "import trackpy" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Dependencies already installed"
else
    pip install --upgrade pip wheel setuptools -q
    if [[ "$GPU_MODE" == "gpu" ]]; then
        if [[ -f "requirements/requirements-gpu.txt" ]]; then
            echo "  Installing GPU packages (this may take a few minutes)..."
            pip install -r requirements/requirements-gpu.txt -q
        else
            pip install -r requirements/requirements-cpu.txt -q
        fi
    else
        echo "  Installing CPU packages..."
        pip install -r requirements/requirements-cpu.txt -q
    fi
    echo -e "  ${GREEN}✓${NC} Dependencies installed"
fi

# =============================================================================
# STEP 6: Build and Launch
# =============================================================================
echo -e "\n${BLUE}[6/6] Building and launching RIPPLE...${NC}"

# Build Java application
# Previous behavior only checked for existence of target/ripple.jar, which can be stale
# if sources changed after the last build. We now compare timestamps.
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

# Launch
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  RIPPLE Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# Create launch shortcuts
# =============================================================================
echo -e "${BLUE}Creating launch shortcuts...${NC}"

# Create the launcher shell script with improved conda detection
cat > "${SCRIPT_DIR}/RIPPLE.sh" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# RIPPLE Launcher - Auto-generated by quickstart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Initialize conda (handle Homebrew, Miniforge, and other installations)
if ! command -v conda &> /dev/null; then
    CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "$HOME/mambaforge/bin/conda"
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"
        "/usr/local/Caskroom/miniconda/base/bin/conda"
    )
    for cpath in "${CONDA_PATHS[@]}"; do
        if [[ -x "$cpath" ]]; then
            eval "$($cpath shell.bash hook)"
            break
        fi
    done
fi

CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
else
    eval "$(conda shell.bash hook)"
fi
LAUNCHER_EOF

# Add environment and mode dynamically
echo "conda activate ${CONDA_ENV_NAME}" >> "${SCRIPT_DIR}/RIPPLE.sh"
echo "export RIPPLE_MODE=${GPU_MODE}" >> "${SCRIPT_DIR}/RIPPLE.sh"
echo 'java -jar target/ripple.jar "$@"' >> "${SCRIPT_DIR}/RIPPLE.sh"

chmod +x "${SCRIPT_DIR}/RIPPLE.sh"
echo -e "  ${GREEN}✓${NC} Created RIPPLE.sh"

# Create desktop shortcut based on OS
if [[ "$OS" == "Linux" ]]; then
    # Create .desktop file for Linux
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

    # Copy to Desktop if it exists
    if [[ -d "${HOME}/Desktop" ]]; then
        cp "${SCRIPT_DIR}/RIPPLE.desktop" "${DESKTOP_FILE}"
        chmod +x "${DESKTOP_FILE}"
        # Make it trusted (GNOME)
        gio set "${DESKTOP_FILE}" metadata::trusted true 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Created desktop shortcut"
    fi
    
    # Copy to applications menu
    mkdir -p "${HOME}/.local/share/applications"
    cp "${SCRIPT_DIR}/RIPPLE.desktop" "${APPLICATIONS_FILE}"
    echo -e "  ${GREEN}✓${NC} Added to applications menu"

elif [[ "$OS" == "Darwin" ]]; then
    # Create macOS .command file (double-clickable)
    cat > "${SCRIPT_DIR}/RIPPLE.command" << MACOS_EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
./RIPPLE.sh
MACOS_EOF
    chmod +x "${SCRIPT_DIR}/RIPPLE.command"
    echo -e "  ${GREEN}✓${NC} Created RIPPLE.command (double-click to launch)"
    
    # Create alias on Desktop
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
echo -e "${BOLD}${GREEN}Launching RIPPLE now...${NC}"
echo ""

# Set execution mode and launch
export RIPPLE_MODE="${GPU_MODE}"

# Run the Java application directly
java -jar target/ripple.jar
