#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== RIPPLE installation validation ==="

echo
echo "[1/4] Java unit tests"
mvn -q test

echo
echo "[2/4] Python parameter tests"
python -m unittest src/test/python/test_tracking_parameters.py

echo
echo "[3/4] Java toolchain checks"
bash scripts/test_java_env.sh

echo
echo "[4/4] Bootstrap URL dry-run"
bash scripts/test_bootstrap_urls.sh

echo
echo "All automated installation validation checks passed."
