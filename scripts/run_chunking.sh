
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

# 2. Chunking
DATASET_NAME="CUAD"
# Must not exceed the number of documents run_doc_processing.sh
# actually produced under CUAD/data/processed -- l1's own
# DATASET_SIZE (20) is how many raw rows it requests, but one of
# those 20 CUAD rows is an exact duplicate of another, so only 19
# unique documents end up processed. If run_doc_processing.sh's
# DATASET_SIZE changes, re-check CUAD/data/processed's actual count
# before changing this.
DATASET_SIZE=19

CHUNK_TYPE="fixed" # fixed, sentence, semantic
MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.3
MODEL_NAME="all-MiniLM-L6-v2"

mkdir -p log/l2_chunking
LOG="log/l2_chunking/log_${CHUNK_TYPE}.txt"

# run

python3 -m l2_chunking.main \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --chunk_type $CHUNK_TYPE \
  --min_chunk_size $MIN_CHUNK_SIZE \
  --max_chunk_size $MAX_CHUNK_SIZE \
  --fixed_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THRESHOLD \
  --model_name $MODEL_NAME \
  2>&1 | tee "$LOG"