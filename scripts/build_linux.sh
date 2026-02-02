#!/usr/bin/env bash
# =============================================================================
# RIPPLE Build Script for Linux
# =============================================================================
# This script creates a distributable package for Linux including:
# - Java application (JAR or native binary)
# - Bundled Python environment
# - Pre-downloaded models
# - AppImage or .deb package
#
# Prerequisites:
#   - JDK 17+ with jpackage
#   - Python 3.10+
#   - Maven 3.8+
#   - appimagetool (for AppImage)
#   - dpkg (for .deb)
#
# Usage:
#   ./build/build_linux.sh [--appimage|--deb|--tar]
# =============================================================================

set -Eeuo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist/linux"
APP_NAME="RIPPLE"
APP_VERSION="1.0.0"
MAIN_CLASS="com.ripple.VideoAnnotationTool"
MAIN_JAR="ripple.jar"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# =============================================================================
# Cleanup
# =============================================================================
cleanup() {
    info "Cleaning up..."
    rm -rf "$BUILD_DIR/temp" 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# Check Prerequisites
# =============================================================================
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Java
    if ! command -v java &> /dev/null; then
        error "Java not found. Please install JDK 17+."
        exit 1
    fi
    
    JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [[ "$JAVA_VERSION" -lt 17 ]]; then
        error "Java 17+ required. Found version $JAVA_VERSION."
        exit 1
    fi
    info "Java version: $JAVA_VERSION"
    
    # Check jpackage
    if ! command -v jpackage &> /dev/null; then
        warn "jpackage not found. Will create JAR-based distribution."
        HAS_JPACKAGE=false
    else
        HAS_JPACKAGE=true
    fi
    
    # Check Maven
    if ! command -v mvn &> /dev/null; then
        error "Maven not found. Please install Maven 3.8+."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.10+."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    info "Python version: $PYTHON_VERSION"
}

# =============================================================================
# Build Java Application
# =============================================================================
build_java() {
    info "Building Java application..."
    
    cd "$PROJECT_DIR"
    
    # Check if pom.xml exists
    if [[ -f "pom.xml" ]]; then
        mvn clean package -DskipTests -q
        
        # Find the built JAR
        JAR_PATH=$(find target -name "*.jar" -not -name "*-sources.jar" -not -name "*-javadoc.jar" | head -1)
        if [[ -z "$JAR_PATH" ]]; then
            error "JAR file not found after build"
            exit 1
        fi
        
        info "JAR built: $JAR_PATH"
    else
        warn "pom.xml not found, skipping Java build"
        JAR_PATH=""
    fi
}

# =============================================================================
# Create Python Bundle
# =============================================================================
create_python_bundle() {
    info "Creating Python bundle..."
    
    local PYTHON_DIR="$BUILD_DIR/app/python"
    mkdir -p "$PYTHON_DIR"
    
    # Create virtual environment
    python3 -m venv "$PYTHON_DIR"
    
    # Activate and install packages
    source "$PYTHON_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools -q

    # Windows parity: build bundles CPU requirements for maximum compatibility.
    if [[ -f "$PROJECT_DIR/requirements/requirements-cpu.txt" ]]; then
        info "Installing CPU Python packages..."
        pip install -r "$PROJECT_DIR/requirements/requirements-cpu.txt" -q
    else
        warn "requirements/requirements-cpu.txt not found; skipping Python package install"
    fi
    
    deactivate
    
    success "Python bundle created"
}

# =============================================================================
# Copy Runtime Files
# =============================================================================
copy_runtime_files() {
    info "Copying runtime files..."
    
    local APP_DIR="$BUILD_DIR/app"
    mkdir -p "$APP_DIR/runtime"
    
    # Copy Python scripts from repo layout (src/main/python)
    cp "$PROJECT_DIR/src/main/python/tracking_server.py" "$APP_DIR/runtime/" 2>/dev/null || true
    cp "$PROJECT_DIR/src/main/python/locotrack_flow.py" "$APP_DIR/runtime/" 2>/dev/null || true
    cp "$PROJECT_DIR/src/main/python/trackmate_dog.py" "$APP_DIR/runtime/" 2>/dev/null || true
    cp "$PROJECT_DIR/src/main/python/trackpy_flow.py" "$APP_DIR/runtime/" 2>/dev/null || true
    cp "$PROJECT_DIR/src/main/python/runtime_config.py" "$APP_DIR/runtime/" 2>/dev/null || true
    cp "$PROJECT_DIR/src/main/python/send_command.py" "$APP_DIR/runtime/" 2>/dev/null || true
    
    # Copy launcher
    cp "$PROJECT_DIR/src/main/python/ripple_launcher.py" "$APP_DIR/"
    
    # Copy JAR if built (Maven shade plugin produces target/ripple.jar)
    if [[ -f "$PROJECT_DIR/target/ripple.jar" ]]; then
        cp "$PROJECT_DIR/target/ripple.jar" "$APP_DIR/$MAIN_JAR"
    elif [[ -n "${JAR_PATH:-}" && -f "$JAR_PATH" ]]; then
        cp "$JAR_PATH" "$APP_DIR/$MAIN_JAR"
    fi
    
    # Create shell launcher script
    cat > "$APP_DIR/$APP_NAME" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use bundled Python if available
if [[ -f "$SCRIPT_DIR/python/bin/python3" ]]; then
    exec "$SCRIPT_DIR/python/bin/python3" "$SCRIPT_DIR/ripple_launcher.py" "$@"
else
    exec python3 "$SCRIPT_DIR/ripple_launcher.py" "$@"
fi
EOF
    chmod +x "$APP_DIR/$APP_NAME"
    
    success "Runtime files copied"
}

# =============================================================================
# Create Desktop Entry
# =============================================================================
create_desktop_entry() {
    info "Creating desktop entry..."
    
    local APP_DIR="$BUILD_DIR/app"
    
    cat > "$APP_DIR/$APP_NAME.desktop" << EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Video Annotation Tool for Biology
Exec=$APP_NAME
Icon=applications-science
Terminal=false
Categories=Science;Biology;Graphics;
Keywords=tracking;annotation;microscopy;
EOF
}

# =============================================================================
# Create Tarball Distribution
# =============================================================================
create_tarball() {
    info "Creating tarball distribution..."
    
    local DIST_NAME="${APP_NAME}-${APP_VERSION}-linux-x86_64"
    local TARBALL="$BUILD_DIR/${DIST_NAME}.tar.gz"
    
    cd "$BUILD_DIR"
    mv app "$DIST_NAME"
    tar -czf "$TARBALL" "$DIST_NAME"
    mv "$DIST_NAME" app
    
    success "Created: $TARBALL"
}

# =============================================================================
# Create AppImage
# =============================================================================
create_appimage() {
    info "Creating AppImage..."
    
    # Check for appimagetool
    if ! command -v appimagetool &> /dev/null; then
        # Download appimagetool
        local APPIMAGETOOL="$BUILD_DIR/temp/appimagetool"
        mkdir -p "$BUILD_DIR/temp"
        
        info "Downloading appimagetool..."
        wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
            -O "$APPIMAGETOOL"
        chmod +x "$APPIMAGETOOL"
    else
        APPIMAGETOOL="appimagetool"
    fi
    
    # Create AppDir structure
    local APPDIR="$BUILD_DIR/temp/${APP_NAME}.AppDir"
    mkdir -p "$APPDIR/usr/bin"
    mkdir -p "$APPDIR/usr/share/applications"
    mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
    
    # Copy application files
    cp -r "$BUILD_DIR/app"/* "$APPDIR/usr/bin/"
    cp "$BUILD_DIR/app/$APP_NAME.desktop" "$APPDIR/"
    cp "$BUILD_DIR/app/$APP_NAME.desktop" "$APPDIR/usr/share/applications/"
    
    # Create AppRun
    cat > "$APPDIR/AppRun" << 'EOF'
#!/usr/bin/env bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/RIPPLE" "$@"
EOF
    chmod +x "$APPDIR/AppRun"
    
    # Build AppImage
    ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$BUILD_DIR/${APP_NAME}-${APP_VERSION}-x86_64.AppImage"
    
    success "Created: $BUILD_DIR/${APP_NAME}-${APP_VERSION}-x86_64.AppImage"
}

# =============================================================================
# Create .deb Package
# =============================================================================
create_deb() {
    info "Creating .deb package..."
    
    local DEB_DIR="$BUILD_DIR/temp/deb"
    local INSTALL_DIR="$DEB_DIR/opt/$APP_NAME"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/usr/bin"
    mkdir -p "$DEB_DIR/usr/share/applications"
    
    # Copy application files
    cp -r "$BUILD_DIR/app"/* "$INSTALL_DIR/"
    
    # Create symlink in /usr/bin
    ln -sf "/opt/$APP_NAME/$APP_NAME" "$DEB_DIR/usr/bin/$APP_NAME"
    
    # Copy desktop file
    cp "$BUILD_DIR/app/$APP_NAME.desktop" "$DEB_DIR/usr/share/applications/"
    # Fix Exec path
    sed -i "s|Exec=$APP_NAME|Exec=/opt/$APP_NAME/$APP_NAME|" \
        "$DEB_DIR/usr/share/applications/$APP_NAME.desktop"
    
    # Create control file
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: ripple
Version: $APP_VERSION
Section: science
Priority: optional
Architecture: amd64
Depends: default-jre (>= 17) | openjdk-17-jre | openjdk-21-jre, python3 (>= 3.10)
Maintainer: Leonidas Zimi <lzimi001@odu.edu>
Description: $APP_NAME - Video Annotation Tool for Biology
 RIPPLE is a comprehensive video annotation and particle tracking tool
 designed for biological research. It supports GPU-accelerated tracking
 using RAFT and LocoTrack models.
EOF
    
    # Create postinst script
    cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
chmod +x /opt/RIPPLE/RIPPLE
chmod +x /opt/RIPPLE/python/bin/python3 2>/dev/null || true
EOF
    chmod +x "$DEB_DIR/DEBIAN/postinst"
    
    # Build .deb
    dpkg-deb --build "$DEB_DIR" "$BUILD_DIR/${APP_NAME}_${APP_VERSION}_amd64.deb"
    
    success "Created: $BUILD_DIR/${APP_NAME}_${APP_VERSION}_amd64.deb"
}

# =============================================================================
# Main
# =============================================================================
main() {
    local PACKAGE_TYPE="${1:---tar}"
    
    echo ""
    echo "=========================================="
    echo " $APP_NAME Build Script for Linux"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    
    # Create build directory
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/app"
    
    build_java
    create_python_bundle
    copy_runtime_files
    create_desktop_entry
    
    case "$PACKAGE_TYPE" in
        --appimage)
            create_appimage
            ;;
        --deb)
            create_deb
            ;;
        --tar|*)
            create_tarball
            ;;
    esac
    
    echo ""
    success "Build complete! Output in: $BUILD_DIR"
    echo ""
}

main "$@"
