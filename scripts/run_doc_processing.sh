
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate

if [ -f .env ]; then
  source .env
fi

export PYTHONPATH="$(pwd)/src"

# 1. Document processing
DOC_TYPE="docx"
DATASET_NAME="docx-corpus"
DATASET_SIZE=5000
DATASET_URL="superdoc-dev/docx-corpus"
DATASET_PATH=""

mkdir -p log/l1_doc_processing
LOG="log/l1_doc_processing/log_${DATASET_NAME}_${DATASET_SIZE}.txt"

# run

python3 -m l1_doc_processing.main \
  --doc_type $DOC_TYPE \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --dataset_url $DATASET_URL \
  2>&1 | tee "$LOG"