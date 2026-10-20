# Loads the documents, then loads or generates the eval QnA and the queries, with the settings in the build and eval parts of config.yaml.
#
# Usage:
#   bash scripts/question.sh                  # the build and eval settings in config.yaml
#   bash scripts/question.sh --n_multi_qus 3  # any setting can be overridden for one run
#   ALLOW_PAID_API=1 bash scripts/question.sh # when a layer uses a paid LLM and its cache does not exist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/question_log.txt"

python3 src/question.py --config config.yaml "$@" 2>&1 | tee "$LOG"
