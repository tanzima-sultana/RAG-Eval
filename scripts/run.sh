SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/log.txt"

# 1. Ingestion
DATASET_SIZE=100
INPUT_PATH="pixparse/pdfa-eng-wds"
OUTPUT_PATH="outputs"
DOWNLOAD_WORKERS=1
MAX_BATCH_ROWS=1
HUGGING_FACE_SPLIT="train"
PARSE_WORKERS=16
PARSE_TIMEOUT_SECONDS=0
DEDUP_THRESHOLD=0.0
PDF_PARSER="pypdfium2"

# 2. Chunking
FRAMEWORK="" # langchain or ""
MODEL_NAME="all-MiniLM-L6-v2"
CHUNKING_TYPE="fixed" # fixed, sentence, semantic
MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIXED_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.2

# run

python3 src/rag.py \
  --dataset_size "$DATASET_SIZE" \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --download_workers "$DOWNLOAD_WORKERS" \
  --max_batch_rows "$MAX_BATCH_ROWS" \
  --hugging_face_split "$HUGGING_FACE_SPLIT" \
  --parse_workers "$PARSE_WORKERS" \
  --parse_timeout_seconds "$PARSE_TIMEOUT_SECONDS" \
  --dedup_threshold "$DEDUP_THRESHOLD" \
  --pdf_parser "$PDF_PARSER" \
  --framework "$FRAMEWORK" \
  --model_name "$MODEL_NAME" \
  --chunk_type "$CHUNKING_TYPE" \
  --min_chunk_size "$MIN_CHUNK_SIZE" \
  --max_chunk_size "$MAX_CHUNK_SIZE" \
  --fixed_chunk_overlap "$FIXED_CHUNK_OVERLAP" \
  --semantic_threshold "$SEMANTIC_THRESHOLD" \
  2>&1 | tee "$LOG"
