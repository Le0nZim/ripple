#!/usr/bin/env bash
# Create a minimal source archive (~2–3 MB) containing only git-tracked files.
# Excludes local bloat: .venv (~460MB), target/, downloaded weights, video/, etc.
#
# Usage:
#   bash scripts/make_source_archive.sh
#   bash scripts/make_source_archive.sh /tmp/ripple-source.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

OUTPUT="${1:-${SCRIPT_DIR}/ripple-source.zip}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: run this script from the RIPPLE git repository root." >&2
    exit 1
fi

REF="${RIPPLE_ARCHIVE_REF:-HEAD}"
echo "Creating source archive from ${REF} → ${OUTPUT}"
git archive --format=zip --output="$OUTPUT" "$REF"

SIZE=$(du -h "$OUTPUT" | awk '{print $1}')
echo "Done (${SIZE}). Extract and run: bash quickstart.sh"
