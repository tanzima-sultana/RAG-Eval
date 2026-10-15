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
DATASET_SIZE=5000
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
CHUNKING_TYPES="fixed sentence semantic" # one or more of fixed, sentence, semantic, separated by spaces
MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIXED_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.2

# 4. Indexing
INDEX_TYPES="faiss_flatip faiss_ivf faiss_hnsw bm25 qdrant" # any of faiss_flatip, faiss_ivf, faiss_hnsw, bm25, qdrant, separated by spaces
IVF_NLIST=256
HNSW_M=32
HNSW_EF_CONSTRUCTION=200
BM25_K1=1.5
BM25_B=0.75
BM25_TOKENIZER="whitespace"

# 5. Eval QnA
EVAL_QNA_LLM="claude" # LLM provider RAGAS uses to generate the eval QnA
EVAL_QNA_MODEL="all-MiniLM-L6-v2" # Local embedding model RAGAS uses to build its knowledge graph
N_SAMPLE_DOCUMENTS=100 # Documents randomly picked for RAGAS to generate the eval QnA from
N_SAMPLE_EVALS=200 # Eval QnA items to generate

# 6. Query
QUERY_LLM="ollama" # Local LLM provider that transforms the eval questions
QUERY_LLM_MODEL="llama3.1:8b" # Local model the query LLM provider runs
N_MULTI_QUS=3 # Phrasings the query LLM writes for each eval question

# 7. Retrieval
RETRIEVAL_TYPES="dense bm25 hybrid" # any of dense, bm25, hybrid, separated by spaces
TOP_K=20 # Chunks each retriever returns for a query
RERANKING=true # true to also rerank the TOP_K chunks with the cross-encoder, false not to
RERANK_TOP_K=5 # Chunks kept after the cross-encoder reranks the TOP_K chunks
TUNING_QUERIES_SIZE=150 # Eval questions used for tuning; the rest are held out for one final check
N_RETRIEVAL_QUALITY=1 # Best plain setups and best reranked setups whose retrieved outputs go on to the next layers

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
  --chunk_types $CHUNKING_TYPES \
  --min_chunk_size "$MIN_CHUNK_SIZE" \
  --max_chunk_size "$MAX_CHUNK_SIZE" \
  --fixed_chunk_overlap "$FIXED_CHUNK_OVERLAP" \
  --semantic_threshold "$SEMANTIC_THRESHOLD" \
  --index_types $INDEX_TYPES \
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
  --query_llm "$QUERY_LLM" \
  --query_llm_model "$QUERY_LLM_MODEL" \
  --n_multi_qus "$N_MULTI_QUS" \
  --retrieval_types $RETRIEVAL_TYPES \
  --top_k "$TOP_K" \
  --reranking "$RERANKING" \
  --rerank_top_k "$RERANK_TOP_K" \
  --tuning_queries_size "$TUNING_QUERIES_SIZE" \
  --n_retrieval_quality "$N_RETRIEVAL_QUALITY" \
  2>&1 | tee "$LOG"
