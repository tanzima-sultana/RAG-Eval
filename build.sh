#!/usr/bin/env bash
set -euo pipefail

DATASET_SIZE="${1:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/scripts/build_rag.py" --dataset-size "$DATASET_SIZE"
