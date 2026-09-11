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

echo
echo "Results: ${pass} passed, ${fail} failed"
if [[ "$fail" -gt 0 ]]; then
    exit 1
fi
