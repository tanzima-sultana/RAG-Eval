# Builds the indexes: ingestion, chunking, embedding, indexing and vector DB. Saved parts are loaded, not rebuilt.
#
# Usage:
#   bash scripts/build.sh                         # the settings in config/build_config.yaml
#   bash scripts/build.sh --dataset_size 20       # any setting can be overridden for one run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/build_log.txt"

python3 src/run_build.py --config config/build_config.yaml --input_path "$INPUT_PATH" "$@" 2>&1 | tee "$LOG"
