# Benchmarks retrieval: loads the saved eval dataset and build, transforms the queries, retrieves the val queries and scores them.
#
# Usage:
#   bash scripts/benchmark_temp.sh                         # the settings in config_temp.yaml
#   bash scripts/benchmark_temp.sh --dataset_size 20       # any setting can be overridden for one run
#
# Needs the eval dataset from scripts/eval_dataset.sh and the Qdrant server on localhost:6333.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/log.txt"

python3 src/run_benchmark.py --config config_temp.yaml "$@" 2>&1 | tee "$LOG"
