SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/eval_log.txt"

# Which build to evaluate. These must match the values scripts/run.sh was run with.

# 1. Dataset
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

# 4. Indexing
FAISS_INDEX_TYPE="faiss_flatip" # faiss_flatip, faiss_ivf, faiss_hnsw
IVF_NLIST=256
HNSW_M=32
HNSW_EF_CONSTRUCTION=200
BM25_K1=1.5
BM25_B=0.75
BM25_TOKENIZER="whitespace"

# 5. Eval QnA
EVAL_QNA_LLM="claude" # LLM provider RAGAS uses to generate the eval QnA
EVAL_QNA_MODEL="all-MiniLM-L6-v2" # Local embedding model RAGAS uses to build its knowledge graph
N_SAMPLE_DOCUMENTS=200 # Documents randomly picked for RAGAS to generate the eval QnA from
N_SAMPLE_EVALS=50 # Eval QnA items to generate

# run

python3 src/eval.py \
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
  --faiss_index_type "$FAISS_INDEX_TYPE" \
  --ivf_nlist "$IVF_NLIST" \
  --hnsw_m "$HNSW_M" \
  --hnsw_ef_construction "$HNSW_EF_CONSTRUCTION" \
  --bm25_k1 "$BM25_K1" \
  --bm25_b "$BM25_B" \
  --bm25_tokenizer "$BM25_TOKENIZER" \
  --eval_qna_llm "$EVAL_QNA_LLM" \
  --eval_qna_model "$EVAL_QNA_MODEL" \
  --n_sample_documents "$N_SAMPLE_DOCUMENTS" \
  --n_sample_evals "$N_SAMPLE_EVALS" \
  2>&1 | tee "$LOG"
