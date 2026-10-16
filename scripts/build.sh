# Builds the RAG indexes with the settings in the build part of configs/input_args.yaml.
#
# Usage:
#   bash scripts/build.sh                      # the build settings in configs/input_args.yaml
#   bash scripts/build.sh --dataset_size 20    # any setting can be overridden for one run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/log.txt"

python3 src/build.py --config configs/input_args.yaml "$@" 2>&1 | tee "$LOG"
