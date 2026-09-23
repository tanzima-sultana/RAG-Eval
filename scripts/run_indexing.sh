
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

# 4. Indexing
DATASET_NAME="CUAD"
DATASET_SIZE=100

INDEX_TYPE="flatip" #flatip, ivf, hnsw, bm25, qdrant

CHUNK_TYPE="fixed" # fixed, sentence, semantic
MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.3

mkdir -p log/l4_indexing
LOG="log/l4_indexing/log_${INDEX_TYPE}.txt"

# run

python3 -m l4_indexing.main \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --index_type $INDEX_TYPE \
  --chunk_type $CHUNK_TYPE \
  --min_chunk_size $MIN_CHUNK_SIZE \
  --max_chunk_size $MAX_CHUNK_SIZE \
  --fixed_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THRESHOLD \
  2>&1 | tee "$LOG"