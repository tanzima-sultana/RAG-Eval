
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src"

# 3. Embedding
DATASET_NAME="CUAD"
DATASET_SIZE=200

CHUNK_TYPE="fixed" # fixed, sentence, semantic
MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.3
MODEL_NAME="all-MiniLM-L6-v2"

mkdir -p log/l3_embedding
LOG="log/l3_embedding/log_${CHUNK_TYPE}.txt"

# run

python3 -m l3_embedding.main \
  --dataset_name $DATASET_NAME \
  --dataset_size $DATASET_SIZE \
  --chunk_type $CHUNK_TYPE \
  --min_chunk_size $MIN_CHUNK_SIZE \
  --max_chunk_size $MAX_CHUNK_SIZE \
  --fixed_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THRESHOLD \
  --model_name $MODEL_NAME \
  2>&1 | tee "$LOG"