#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/java_env_check.sh
source "${SCRIPT_DIR}/scripts/lib/java_env_check.sh"

pass=0
fail=0

assert_ok() {
    local name="$1"
    shift
    if "$@"; then
        echo "[PASS] $name"
        pass=$((pass + 1))
    else
        echo "[FAIL] $name"
        fail=$((fail + 1))
    fi
}

assert_fail() {
    local name="$1"
    shift
    if "$@"; then
        echo "[FAIL] $name (expected rejection)"
        fail=$((fail + 1))
    else
        echo "[PASS] $name"
        pass=$((pass + 1))
    fi
}

run_case_java17_runtime_and_maven() {
    (
        export PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:$PATH"
        export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
        validate_java_toolchain 1
    )
}

run_case_java11_only() {
    (
        export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH"
        export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
        validate_java_toolchain 1
    )
}

find_system_jdk_for_test() {
    local candidate
    for candidate in \
        /usr/lib/jvm/java-21-openjdk-amd64 \
        /usr/lib/jvm/java-17-openjdk-amd64 \
        /usr/lib/jvm/java-21-openjdk \
        /usr/lib/jvm/java-17-openjdk \
        "${JAVA_HOME:-}"; do
        if [[ -n "$candidate" && -x "$candidate/bin/java" && -x "$candidate/bin/javac" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    if command -v javac >/dev/null 2>&1; then
        echo "$(dirname "$(dirname "$(command -v javac)")")"
        return 0
    fi
    return 1
}

run_case_tools_jdk_without_java_home() {
    local sys tmp found
    sys="$(find_system_jdk_for_test)" || return 1
    tmp="$(mktemp -d)"
    ln -s "$sys" "${tmp}/jdk"
    (
        unset JAVA_HOME
        export RIPPLE_TOOLS_DIR="$tmp"
        export RIPPLE_PROJECT_DIR="$SCRIPT_DIR"
        export PATH="${tmp}/jdk/bin:/usr/bin:/bin"
        found="$(find_jdk_home)"
        [[ "$found" == "${tmp}/jdk" ]]
    )
    local rc=$?
    rm -rf "$tmp"
    return $rc
}

run_case_missing_javac() {
    (
        export PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:$PATH"
        export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
        local backup
        backup=$(command -v javac)
        mv "$backup" "${backup}.bak" 2>/dev/null || return 1
        validate_java_toolchain 1
        local rc=$?
        mv "${backup}.bak" "$backup" 2>/dev/null || true
        return $rc
    )
}

echo "Running Java environment validation tests..."
echo "(Cases skip automatically when the referenced JDK paths are unavailable.)"

if [[ -x /usr/lib/jvm/java-17-openjdk-amd64/bin/java && -x /usr/lib/jvm/java-17-openjdk-amd64/bin/javac ]]; then
    assert_ok "Java 17 runtime and Maven using Java 17" run_case_java17_runtime_and_maven
else
    echo "[SKIP] Java 17 runtime and Maven using Java 17"
fi

if [[ -x /usr/lib/jvm/java-11-openjdk-amd64/bin/java ]]; then
    assert_fail "Java 11 only rejected" run_case_java11_only
else
    echo "[SKIP] Java 11 only rejected"
fi

if [[ -x /usr/lib/jvm/java-17-openjdk-amd64/bin/java ]]; then
    assert_fail "Java 17 without javac rejected" run_case_missing_javac
else
    echo "[SKIP] Java 17 without javac rejected"
fi

if find_system_jdk_for_test >/dev/null; then
    assert_ok "tools/jdk accepted when JAVA_HOME is unset" run_case_tools_jdk_without_java_home
else
    echo "[SKIP] tools/jdk accepted when JAVA_HOME is unset"
fi

echo
echo "Results: ${pass} passed, ${fail} failed"
if [[ "$fail" -gt 0 ]]; then
    exit 1
fi
