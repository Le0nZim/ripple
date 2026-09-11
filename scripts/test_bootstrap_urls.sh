#!/usr/bin/env bash
# Dry-run checks that bootstrap URL helpers pick the right download endpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bootstrap_tools.sh
source "${SCRIPT_DIR}/scripts/lib/bootstrap_tools.sh"

pass=0
fail=0

assert_eq() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "[PASS] $name"
        pass=$((pass + 1))
    else
        echo "[FAIL] $name"
        echo "       expected: $expected"
        echo "       actual:   $actual"
        fail=$((fail + 1))
    fi
}

echo "Running bootstrap URL dry-run tests..."

JDK_PREFIX="https://api.adoptium.net/v3/binary/latest/17/ga"
JDK_SUFFIX="jdk/hotspot/normal/eclipse?project=jdk"
MAVEN_TGZ="https://archive.apache.org/dist/maven/maven-3/${RIPPLE_MAVEN_VERSION}/binaries/apache-maven-${RIPPLE_MAVEN_VERSION}-bin.tar.gz"
MAVEN_ZIP="https://archive.apache.org/dist/maven/maven-3/${RIPPLE_MAVEN_VERSION}/binaries/apache-maven-${RIPPLE_MAVEN_VERSION}-bin.zip"

assert_eq "JDK linux/x64" "$(ripple_jdk_download_url linux x64)" "${JDK_PREFIX}/linux/x64/${JDK_SUFFIX}"
assert_eq "JDK linux/aarch64" "$(ripple_jdk_download_url linux aarch64)" "${JDK_PREFIX}/linux/aarch64/${JDK_SUFFIX}"
assert_eq "JDK darwin/arm64" "$(ripple_jdk_download_url darwin arm64)" "${JDK_PREFIX}/mac/aarch64/${JDK_SUFFIX}"
assert_eq "JDK darwin/x86_64" "$(ripple_jdk_download_url darwin x86_64)" "${JDK_PREFIX}/mac/x64/${JDK_SUFFIX}"
assert_eq "JDK windows/x64" "$(ripple_jdk_download_url windows x64)" "${JDK_PREFIX}/windows/x64/${JDK_SUFFIX}"

assert_eq "Maven linux tarball" "$(ripple_maven_download_url linux)" "$MAVEN_TGZ"
assert_eq "Maven mac tarball" "$(ripple_maven_download_url mac)" "$MAVEN_TGZ"
assert_eq "Maven windows zip" "$(ripple_maven_download_url windows)" "$MAVEN_ZIP"

assert_eq "Miniconda linux/x86_64" "$(ripple_miniconda_download_url linux x86_64)" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
assert_eq "Miniconda linux/aarch64" "$(ripple_miniconda_download_url linux aarch64)" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
assert_eq "Miniconda darwin/arm64" "$(ripple_miniconda_download_url darwin arm64)" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
assert_eq "Miniconda darwin/x86_64" "$(ripple_miniconda_download_url darwin x86_64)" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
assert_eq "Miniconda windows/x64" "$(ripple_miniconda_download_url windows x64)" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

echo
echo "Results: ${pass} passed, ${fail} failed"
if [[ "$fail" -gt 0 ]]; then
    exit 1
fi
