#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== RIPPLE installation validation ==="

echo
echo "[1/3] Java unit tests"
mvn -q test

echo
echo "[2/3] Python parameter tests"
python -m unittest src/test/python/test_tracking_parameters.py

echo
echo "[3/3] Java toolchain checks"
bash scripts/test_java_env.sh

echo
echo "All automated installation validation checks passed."
