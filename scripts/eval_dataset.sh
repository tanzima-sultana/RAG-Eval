# Loads the saved eval dataset, or generates and saves one if there is none.
#
# Usage:
#   bash scripts/eval_dataset.sh                                   # the settings in eval_dataset_config.yaml
#   bash scripts/eval_dataset.sh --documents.dataset_size 20       # any setting can be overridden for one run, as --{section}.{key}
#
# Generating calls a paid LLM, so it runs only with ALLOW_PAID_API=1:
#   ALLOW_PAID_API=1 bash scripts/eval_dataset.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/log.txt"

python3 src/run_eval_dataset.py --config eval_dataset_config.yaml "$@" 2>&1 | tee "$LOG"
