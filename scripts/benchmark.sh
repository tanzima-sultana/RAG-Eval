# Benchmarks retrieval: reads one build manifest and one eval dataset metadata, transforms the queries, retrieves the
# val queries on every index of the build, scores them and saves the best setups.
#
# Usage:
#   bash scripts/benchmark.sh                                   # the settings in config/benchmark_config.yaml
#   bash scripts/benchmark.sh --retrieval.top_k 10              # any setting can be overridden for one run, as --{section}.{key}
#
# Needs a build from scripts/build.sh, an eval dataset from scripts/eval_dataset.sh and the Qdrant server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/benchmark_log.txt"

python3 src/run_benchmark.py --config config/benchmark_config.yaml "$@" 2>&1 | tee "$LOG"
