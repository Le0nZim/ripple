#!/usr/bin/env bash
# Portable JDK / Maven / Miniconda bootstrap for RIPPLE (macOS + Linux).
# Source from quickstart.sh. Running directly with "urls" prints download URLs.

RIPPLE_REQUIRED_JAVA_MAJOR="${RIPPLE_REQUIRED_JAVA_MAJOR:-17}"
RIPPLE_MAVEN_VERSION="${RIPPLE_MAVEN_VERSION:-3.9.9}"
RIPPLE_MINICONDA_HOME="${RIPPLE_MINICONDA_HOME:-$HOME/miniconda3}"

ripple_tools_dir() {
    if [[ -n "${RIPPLE_TOOLS_DIR:-}" ]]; then
        echo "${RIPPLE_TOOLS_DIR}"
        return 0
    fi
    if [[ -n "${RIPPLE_PROJECT_DIR:-}" ]]; then
        echo "${RIPPLE_PROJECT_DIR}/tools"
        return 0
    fi
    echo "$(pwd)/tools"
}

ripple_adoptium_os() {
    local os="${1:-}"
    if [[ -z "$os" ]]; then
        os="$(uname -s)"
    fi
    case "$os" in
        Darwin|darwin|mac|macos) echo "mac" ;;
        Linux|linux) echo "linux" ;;
        Windows|windows|win) echo "windows" ;;
        *) echo "linux" ;;
    esac
}

ripple_adoptium_arch() {
    local arch="${1:-}"
    if [[ -z "$arch" ]]; then
        arch="$(uname -m)"
    fi
    case "$arch" in
        arm64|aarch64) echo "aarch64" ;;
        x86_64|amd64|x64) echo "x64" ;;
        *) echo "x64" ;;
    esac
}

ripple_jdk_download_url() {
    local os arch
    os="$(ripple_adoptium_os "${1:-}")"
    arch="$(ripple_adoptium_arch "${2:-}")"
    echo "https://api.adoptium.net/v3/binary/latest/${RIPPLE_REQUIRED_JAVA_MAJOR}/ga/${os}/${arch}/jdk/hotspot/normal/eclipse?project=jdk"
}

ripple_maven_archive_name() {
    local os
    os="$(ripple_adoptium_os "${1:-}")"
    if [[ "$os" == "windows" ]]; then
        echo "apache-maven-${RIPPLE_MAVEN_VERSION}-bin.zip"
    else
        echo "apache-maven-${RIPPLE_MAVEN_VERSION}-bin.tar.gz"
    fi
}

ripple_maven_download_url() {
    local name
    name="$(ripple_maven_archive_name "${1:-}")"
    echo "https://archive.apache.org/dist/maven/maven-3/${RIPPLE_MAVEN_VERSION}/binaries/${name}"
}

ripple_miniconda_download_url() {
    local os_key arch_key
    os_key="$(ripple_adoptium_os "${1:-}")"
    arch_key="${2:-$(uname -m)}"
    case "$os_key" in
        mac)
            if [[ "$arch_key" == "arm64" || "$arch_key" == "aarch64" ]]; then
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
            ;;
        windows)
            if [[ "$arch_key" == "arm64" || "$arch_key" == "aarch64" ]]; then
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-arm64.exe"
            else
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
            fi
            ;;
        *)
            if [[ "$arch_key" == "arm64" || "$arch_key" == "aarch64" ]]; then
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            else
                echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            fi
            ;;
    esac
}

ripple_print_download_urls() {
    local os="${1:-}"
    local arch="${2:-}"
    echo "jdk=$(ripple_jdk_download_url "$os" "$arch")"
    echo "maven=$(ripple_maven_download_url "$os")"
    echo "miniconda=$(ripple_miniconda_download_url "$os" "$arch")"
}

ripple_download() {
    local url="$1"
    local dest="$2"
    local attempt
    mkdir -p "$(dirname "$dest")"
    for attempt in 1 2; do
        if command -v curl >/dev/null 2>&1; then
            if curl -fL --retry 2 --connect-timeout 30 -o "$dest" "$url"; then
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -O "$dest" "$url"; then
                return 0
            fi
        else
            echo "  ERROR: neither curl nor wget is available. Install curl and re-run."
            return 1
        fi
        echo "  Download failed (attempt ${attempt}/2)"
    done
    echo "  ERROR: download failed. Check your network, firewall, or proxy."
    echo "         URL: $url"
    return 1
}

ripple_runtime_java_home() {
    local java_bin="${1:-java}"
    if ! command -v "$java_bin" >/dev/null 2>&1 && [[ ! -x "$java_bin" ]]; then
        return 1
    fi
    "$java_bin" -XshowSettings:properties -version 2>&1 | awk -F'= ' '/^[[:space:]]*java\.home[[:space:]]*=/{print $2; exit}'
}

ripple_jdk_is_usable() {
    local home="$1"
    local major
    if [[ -z "$home" || ! -x "$home/bin/java" || ! -x "$home/bin/javac" ]]; then
        return 1
    fi
    if declare -f parse_java_major >/dev/null 2>&1; then
        major=$(parse_java_major "$("$home/bin/java" -version 2>&1)")
    else
        major=$("$home/bin/java" -version 2>&1 | awk -F '"' '/version/{print $2; exit}' | cut -d. -f1)
    fi
    [[ "$major" =~ ^[0-9]+$ ]] && [[ "$major" -ge "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]
}

ripple_resolve_jdk() {
    local home tools
    if command -v java >/dev/null 2>&1 && command -v javac >/dev/null 2>&1; then
        home="$(ripple_runtime_java_home "$(command -v java)" || true)"
        if ripple_jdk_is_usable "$home"; then
            echo "$home"
            return 0
        fi
        home="$(dirname "$(dirname "$(command -v javac)")")"
        if ripple_jdk_is_usable "$home"; then
            echo "$home"
            return 0
        fi
    fi

    tools="$(ripple_tools_dir)"
    for home in "${tools}/jdk" "${tools}/jdk/Contents/Home"; do
        if ripple_jdk_is_usable "$home"; then
            echo "$home"
            return 0
        fi
    done

    if [[ -n "${JAVA_HOME:-}" ]] && ripple_jdk_is_usable "$JAVA_HOME"; then
        echo "$JAVA_HOME"
        return 0
    fi

    if [[ "$(uname -s)" == "Darwin" ]] && [[ -x /usr/libexec/java_home ]]; then
        home=$(/usr/libexec/java_home -v "${RIPPLE_REQUIRED_JAVA_MAJOR}" 2>/dev/null || true)
        if ripple_jdk_is_usable "$home"; then
            echo "$home"
            return 0
        fi
    fi

    for home in \
        /usr/lib/jvm/java-21-openjdk-amd64 \
        /usr/lib/jvm/java-21-openjdk \
        /usr/lib/jvm/java-17-openjdk-amd64 \
        /usr/lib/jvm/java-17-openjdk \
        /usr/lib/jvm/java-17-oracle \
        /usr/lib/jvm/default-java; do
        if ripple_jdk_is_usable "$home"; then
            echo "$home"
            return 0
        fi
    done
    return 1
}

ripple_maven_version_ok() {
    local ver="$1"
    local major minor
    major="$(echo "$ver" | cut -d. -f1)"
    minor="$(echo "$ver" | cut -d. -f2)"
    [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 1
    if [[ "$major" -gt 3 ]]; then
        return 0
    fi
    [[ "$major" -eq 3 && "$minor" -ge 8 ]]
}

ripple_maven_version_from_home() {
    local home="$1"
    local mvn_bin="${home}/bin/mvn"
    if [[ ! -x "$mvn_bin" ]]; then
        return 1
    fi
    "$mvn_bin" -version 2>/dev/null | awk '/Apache Maven/{print $3; exit}'
}

ripple_resolve_maven() {
    local tools mvn_bin home ver
    if command -v mvn >/dev/null 2>&1; then
        ver="$(mvn -version 2>/dev/null | awk '/Apache Maven/{print $3; exit}' || true)"
        if ripple_maven_version_ok "$ver"; then
            home="$(mvn -version 2>/dev/null | awk -F': ' '/Maven home/{print $2; exit}' || true)"
            if [[ -n "$home" ]]; then
                echo "$home"
                return 0
            fi
            mvn_bin="$(command -v mvn)"
            echo "$(dirname "$(dirname "$mvn_bin")")"
            return 0
        fi
    fi
    tools="$(ripple_tools_dir)"
    if [[ -x "${tools}/maven/bin/mvn" ]]; then
        ver="$(ripple_maven_version_from_home "${tools}/maven" || true)"
        if [[ -z "$ver" ]] || ripple_maven_version_ok "$ver"; then
            echo "${tools}/maven"
            return 0
        fi
    fi
    return 1
}

ripple_resolve_conda() {
    local cpath pattern
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        echo "${CONDA_EXE}"
        return 0
    fi
    for cpath in \
        "${RIPPLE_MINICONDA_HOME}/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/opt/homebrew/Caskroom/mambaforge/base/bin/conda" \
        "/usr/local/Caskroom/miniconda/base/bin/conda" \
        "/usr/local/Caskroom/miniforge/base/bin/conda"; do
        if [[ -x "$cpath" ]]; then
            echo "$cpath"
            return 0
        fi
    done
    for pattern in \
        "/opt/homebrew/Caskroom/miniconda"/*/base/bin/conda \
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

ripple_scan_tools() {
    NEED_JDK=0
    NEED_MAVEN=0
    NEED_CONDA=0
    JDK_LOCATION=""
    MAVEN_LOCATION=""
    CONDA_LOCATION=""

    if JDK_LOCATION="$(ripple_resolve_jdk)"; then
        :
    else
        JDK_LOCATION=""
        NEED_JDK=1
    fi
    if MAVEN_LOCATION="$(ripple_resolve_maven)"; then
        :
    else
        MAVEN_LOCATION=""
        NEED_MAVEN=1
    fi
    if CONDA_LOCATION="$(ripple_resolve_conda)"; then
        :
    else
        CONDA_LOCATION=""
        NEED_CONDA=1
    fi
}

ripple_apply_toolchain() {
    local jdk maven
    jdk="$(ripple_resolve_jdk || true)"
    maven="$(ripple_resolve_maven || true)"
    if [[ -n "$jdk" ]]; then
        export JAVA_HOME="$jdk"
        export PATH="${JAVA_HOME}/bin:${PATH}"
        export RIPPLE_RESOLVED_JAVA_HOME="$jdk"
    fi
    if [[ -n "$maven" && -x "${maven}/bin/mvn" ]]; then
        export PATH="${maven}/bin:${PATH}"
        export RIPPLE_RESOLVED_MAVEN_HOME="$maven"
    elif command -v mvn >/dev/null 2>&1; then
        export RIPPLE_RESOLVED_MAVEN_HOME="$(dirname "$(dirname "$(command -v mvn)")")"
    fi
}

ripple_install_jdk() {
    local tools url archive tmp javac_path jdk_home
    tools="$(ripple_tools_dir)"
    mkdir -p "$tools"
    url="$(ripple_jdk_download_url)"
    archive="${tools}/temurin-jdk.tar.gz"
    echo "  Downloading Eclipse Temurin JDK ${RIPPLE_REQUIRED_JAVA_MAJOR} (portable, no admin)..."
    if ! ripple_download "$url" "$archive"; then
        return 1
    fi
    tmp="$(mktemp -d "${tools}/jdk-extract.XXXXXX")"
    if ! tar -xzf "$archive" -C "$tmp"; then
        echo "  ERROR: failed to extract JDK archive"
        rm -rf "$tmp" "$archive"
        return 1
    fi
    javac_path="$(find "$tmp" -type f -name javac | head -n1)"
    if [[ -z "$javac_path" ]]; then
        echo "  ERROR: extracted JDK is missing javac"
        rm -rf "$tmp" "$archive"
        return 1
    fi
    jdk_home="$(cd "$(dirname "$javac_path")/.." && pwd)"
    rm -rf "${tools}/jdk"
    mkdir -p "${tools}/jdk"
    cp -a "${jdk_home}/." "${tools}/jdk/"
    rm -rf "$tmp" "$archive"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        xattr -cr "${tools}/jdk" 2>/dev/null || true
    fi
    if ! ripple_jdk_is_usable "${tools}/jdk"; then
        echo "  ERROR: portable JDK did not validate after extract"
        return 1
    fi
    echo "  Installed portable JDK to ${tools}/jdk"
    return 0
}

ripple_install_maven() {
    local tools url archive tmp inner
    tools="$(ripple_tools_dir)"
    mkdir -p "$tools"
    url="$(ripple_maven_download_url)"
    archive="${tools}/$(ripple_maven_archive_name)"
    echo "  Downloading Apache Maven ${RIPPLE_MAVEN_VERSION} (portable, no admin)..."
    if ! ripple_download "$url" "$archive"; then
        return 1
    fi
    tmp="$(mktemp -d "${tools}/maven-extract.XXXXXX")"
    if ! tar -xzf "$archive" -C "$tmp"; then
        echo "  ERROR: failed to extract Maven archive"
        rm -rf "$tmp" "$archive"
        return 1
    fi
    inner="$(find "$tmp" -maxdepth 1 -type d -name 'apache-maven-*' | head -n1)"
    if [[ -z "$inner" ]]; then
        echo "  ERROR: unexpected Maven archive layout"
        rm -rf "$tmp" "$archive"
        return 1
    fi
    rm -rf "${tools}/maven"
    mv "$inner" "${tools}/maven"
    rm -rf "$tmp" "$archive"
    if [[ ! -x "${tools}/maven/bin/mvn" ]]; then
        echo "  ERROR: portable Maven did not validate after extract"
        return 1
    fi
    echo "  Installed portable Maven to ${tools}/maven"
    return 0
}

ripple_sanitize_conda_state() {
    # conda activate crashes if CONDA_SHLVL is set but CONDA_PREFIX is missing
    # or points at a directory that is no longer visible.
    local prefix="${CONDA_PREFIX:-}"
    if [[ -n "${CONDA_SHLVL:-}" ]] && [[ -z "$prefix" || ! -d "$prefix" ]]; then
        unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
        unset CONDA_PYTHON_EXE CONDA_ROOT
    fi
}

ripple_install_miniconda() {
    local url installer
    ripple_sanitize_conda_state
    url="$(ripple_miniconda_download_url)"
    mkdir -p "${RIPPLE_MINICONDA_HOME}"
    installer="${RIPPLE_MINICONDA_HOME}/miniconda-installer.sh"
    echo "  Downloading Miniconda (user-local, no admin)..."
    if ! ripple_download "$url" "$installer"; then
        return 1
    fi
    bash "$installer" -b -u -p "${RIPPLE_MINICONDA_HOME}"
    rm -f "$installer"
    if [[ ! -x "${RIPPLE_MINICONDA_HOME}/bin/conda" ]]; then
        echo "  ERROR: Miniconda install did not produce a conda executable"
        return 1
    fi
    # Hook this session; initialize common login shells so later terminals work.
    # shellcheck disable=SC1091
    source "${RIPPLE_MINICONDA_HOME}/bin/activate" 2>/dev/null || true
    "${RIPPLE_MINICONDA_HOME}/bin/conda" init bash >/dev/null 2>&1 || true
    if [[ "$(uname -s)" == "Darwin" ]]; then
        "${RIPPLE_MINICONDA_HOME}/bin/conda" init zsh >/dev/null 2>&1 || true
    fi
    echo "  Installed Miniconda to ${RIPPLE_MINICONDA_HOME}"
    return 0
}

ripple_init_conda_shell() {
    local conda_bin base
    ripple_sanitize_conda_state
    conda_bin="$(ripple_resolve_conda || true)"
    if [[ -z "$conda_bin" ]]; then
        return 1
    fi
    if eval "$("${conda_bin}" shell.bash hook 2>/dev/null)"; then
        return 0
    fi
    base="$("${conda_bin}" info --base 2>/dev/null || true)"
    if [[ -n "$base" && -f "${base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${base}/etc/profile.d/conda.sh"
        return 0
    fi
    if [[ -n "$base" && -f "${base}/etc/profile.d/mamba.sh" ]]; then
        # shellcheck disable=SC1091
        source "${base}/etc/profile.d/mamba.sh"
        return 0
    fi
    return 1
}

ripple_install_missing_tools() {
    if [[ "${NEED_JDK:-0}" == "1" ]]; then
        ripple_install_jdk || return 1
    fi
    if [[ "${NEED_MAVEN:-0}" == "1" ]]; then
        ripple_install_maven || return 1
    fi
    if [[ "${NEED_CONDA:-0}" == "1" ]]; then
        ripple_install_miniconda || return 1
    fi
    ripple_apply_toolchain
    ripple_init_conda_shell || true
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        urls)
            ripple_print_download_urls "${2:-}" "${3:-}"
            ;;
        *)
            echo "Usage: $0 urls [linux|mac|windows] [x64|aarch64|arm64|x86_64]"
            exit 1
            ;;
    esac
fi
