
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src"

mkdir -p log/l1_doc_processing
LOG="log/l1_doc_processing/log.txt"

# 1. Document processing
DOC_TYPE="pdf"
DATASET_NAME="CUAD"
DATASET_SIZE=20
DATASET_URL="dvgodoy/CUAD_v1_Contract_Understanding_PDF"
DATASET_PATH=""

# run

python3 -m l1_doc_processing.main \
  --doc_type $DOC_TYPE \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --dataset_url $DATASET_URL \
  2>&1 | tee "$LOG"