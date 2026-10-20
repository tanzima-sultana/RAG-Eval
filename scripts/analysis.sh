# Compares the newest n_compare_runs runs that eval saved, with the settings in the build and eval parts of config.yaml.
#
# Usage:
#   bash scripts/analysis.sh                    # the build and eval settings in config.yaml
#   bash scripts/analysis.sh --n_compare_runs 3 # any setting can be overridden for one run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/analysis_log.txt"

python3 src/analysis.py --config config.yaml "$@" 2>&1 | tee "$LOG"
