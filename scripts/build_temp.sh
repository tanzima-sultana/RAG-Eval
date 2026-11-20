# Builds the indexes: ingestion, chunking, embedding, indexing and vector DB. Saved parts are loaded, not rebuilt.
#
# Usage:
#   bash scripts/build_temp.sh                         # the settings in config_temp.yaml
#   bash scripts/build_temp.sh --dataset_size 20       # any setting can be overridden for one run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/log.txt"

python3 src/run_build.py --config config_temp.yaml "$@" 2>&1 | tee "$LOG"
