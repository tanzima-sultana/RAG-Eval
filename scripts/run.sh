
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

mkdir -p log
LOG="log/log.txt"

# 1. Document processing
DOC_TYPE="pdf"
DATASET_NAME="CUAD"
DATASET_SIZE=20
DATASET_URL="dvgodoy/CUAD_v1_Contract_Understanding_PDF"
DATASET_PATH=""

# run

python3 src/rag.py \
  --doc_type $DOC_TYPE \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --dataset_url $DATASET_URL \
  --dataset_path $DATASET_PATH \
  2>&1 | tee "$LOG"