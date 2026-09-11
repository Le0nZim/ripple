#!/usr/bin/env bash
# Shared Java/JDK validation helpers for RIPPLE installer scripts.

RIPPLE_REQUIRED_JAVA_MAJOR="${RIPPLE_REQUIRED_JAVA_MAJOR:-17}"

parse_java_major() {
    local version_output="$1"
    local version_string
    version_string=$(echo "$version_output" | head -n1 | sed -n 's/.*"\([^"]*\)".*/\1/p')
    if [[ -z "$version_string" ]]; then
        version_string=$(echo "$version_output" | head -n1 | sed -n 's/.*Java version: \([0-9][^, ]*\).*/\1/p')
    fi
    if [[ -z "$version_string" ]]; then
        version_string=$(echo "$version_output" | head -n1 | awk '/javac/{print $2; exit}')
    fi
    if [[ -z "$version_string" ]]; then
        version_string=$(echo "$version_output" | head -n1 | awk '{print $NF}')
    fi
    if [[ -z "$version_string" ]]; then
        echo "0"
        return 0
    fi
    if [[ "$version_string" =~ ^1\.([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    echo "$version_string" | cut -d'.' -f1
}

find_jdk_home() {
    local candidate
    local tools_root="${RIPPLE_TOOLS_DIR:-}"
    if [[ -z "$tools_root" && -n "${RIPPLE_PROJECT_DIR:-}" ]]; then
        tools_root="${RIPPLE_PROJECT_DIR}/tools"
    fi
    for candidate in \
        "${tools_root:+$tools_root/jdk}" \
        "${tools_root:+$tools_root/jdk/Contents/Home}" \
        "${JAVA_HOME:-}" \
        "$(dirname "$(dirname "$(command -v javac 2>/dev/null || true)")")" \
        "$(dirname "$(dirname "$(command -v java 2>/dev/null || true)")")" \
        /usr/lib/jvm/java-21-openjdk-amd64 \
        /usr/lib/jvm/java-21-openjdk \
        /usr/lib/jvm/java-17-openjdk-amd64 \
        /usr/lib/jvm/java-17-openjdk \
        /usr/lib/jvm/java-17-oracle \
        /usr/lib/jvm/default-java; do
        if [[ -n "$candidate" && -x "$candidate/bin/javac" && -x "$candidate/bin/java" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

validate_java_toolchain() {
    local strict="${1:-1}"
    local java_major=0
    local javac_major=0
    local maven_major=0
    local java_home_major=0
    local jdk_home=""

    echo "  Runtime diagnostics:"
    if command -v java >/dev/null 2>&1; then
        java -version 2>&1 | sed 's/^/    /'
        java_major=$(parse_java_major "$(java -version 2>&1)")
    else
        echo "    java: not found"
    fi

    if command -v javac >/dev/null 2>&1; then
        javac -version 2>&1 | sed 's/^/    /'
        javac_major=$(parse_java_major "$(javac -version 2>&1)")
    else
        echo "    javac: not found"
    fi

    if command -v mvn >/dev/null 2>&1; then
        mvn -version 2>&1 | sed 's/^/    /'
        maven_major=$(parse_java_major "$(mvn -version 2>&1 | awk '/Java version/{print; exit}')")
    else
        echo "    mvn: not found"
    fi

    echo "    JAVA_HOME=${JAVA_HOME:-<unset>}"

    if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
        java_home_major=$(parse_java_major "$("${JAVA_HOME}/bin/java" -version 2>&1)")
    fi

    if [[ ! "$java_major" =~ ^[0-9]+$ ]] || [[ "$java_major" -lt "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]; then
        echo "  ERROR: Java runtime ${java_major:-unknown} found; JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ required."
        return 1
    fi

    if [[ ! "$javac_major" =~ ^[0-9]+$ ]] || [[ "$javac_major" -lt "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]; then
        echo "  ERROR: javac ${javac_major:-missing} found; a full JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ is required."
        return 1
    fi

    if [[ "$strict" == "1" && "$maven_major" =~ ^[0-9]+$ && "$maven_major" -lt "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]; then
        jdk_home=$(find_jdk_home || true)
        if [[ -n "$jdk_home" ]]; then
            export JAVA_HOME="$jdk_home"
            export PATH="$JAVA_HOME/bin:$PATH"
            maven_major=$(parse_java_major "$(mvn -version 2>&1 | awk '/Java version/{print; exit}')")
            echo "  ! Adjusted JAVA_HOME for Maven: $JAVA_HOME"
            mvn -version 2>&1 | sed 's/^/    /'
        fi
    fi

    if [[ "$strict" == "1" && "$maven_major" =~ ^[0-9]+$ && "$maven_major" -lt "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]; then
        echo "  ERROR: Maven is using Java ${maven_major}; JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ is required."
        echo "  Fix on Ubuntu/WSL:"
        echo "    sudo apt update"
        echo "    sudo apt install -y openjdk-17-jdk"
        echo "    sudo update-alternatives --config java"
        echo "    sudo update-alternatives --config javac"
        echo "    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
        echo "    export PATH=\"\$JAVA_HOME/bin:\$PATH\""
        return 1
    fi

    if [[ -n "${JAVA_HOME:-}" && "$java_home_major" =~ ^[0-9]+$ && "$java_home_major" -lt "$RIPPLE_REQUIRED_JAVA_MAJOR" ]]; then
        echo "  ERROR: JAVA_HOME points to Java ${java_home_major}, but JDK ${RIPPLE_REQUIRED_JAVA_MAJOR}+ is required."
        return 1
    fi

    echo "  OK: Java runtime ${java_major}, javac ${javac_major}, Maven Java ${maven_major:-unknown}"
    return 0
}
