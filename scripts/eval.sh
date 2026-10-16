# Evaluates the RAG build with the settings in the build and eval parts of configs/input_args.yaml.
#
# Usage:
#   bash scripts/eval.sh                  # the build and eval settings in configs/input_args.yaml
#   bash scripts/eval.sh --top_k 10       # any setting can be overridden for one run
#   ALLOW_PAID_API=1 bash scripts/eval.sh # when a layer uses a paid LLM and its cache does not exist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/eval_log.txt"

python3 src/eval.py --config configs/input_args.yaml "$@" 2>&1 | tee "$LOG"
