#!/usr/bin/env bash
set -euo pipefail

DATASET_SIZE="${1:-1000}"
MAX_CHUNK_SIZE="${2:-500}"
FIXED_CHUNK_OVERLAP="${3:-50}"
SEMANTIC_THRESHOLD="${4:-0.75}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/scripts/build_rag.py" \
    --dataset-size "$DATASET_SIZE" \
    --max-chunk-size "$MAX_CHUNK_SIZE" \
    --fixed-chunk-overlap "$FIXED_CHUNK_OVERLAP" \
    --semantic-threshold "$SEMANTIC_THRESHOLD"
