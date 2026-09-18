
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/src"

mkdir -p log/l2_chunking
LOG="log/l2_chunking/log.txt"

# 2. Chunking
CHUNK_TYPE="fixed" # fixed, sentence, semantic
MAX_CHUNK_SIZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.3
MODEL_NAME="all-MiniLM-L6-v2"

# run

python3 -m l2_chunking.main \
  --chunk_type $CHUNK_TYPE \
  --max_chunk_size $MAX_CHUNK_SIZE \
  --fixed_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THRESHOLD \
  --model_name $MODEL_NAME \
  2>&1 | tee "$LOG"