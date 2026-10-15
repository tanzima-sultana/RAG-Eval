SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

mkdir -p log
LOG="log/eval_log.txt"

# Models: LLM provider, model name and price in US dollars per million tokens. Each layer below picks one
CLAUDE="claude"
SONNET="claude-sonnet-5-5"
SONNET_INPUT_PRICE_PER_MIL=2.00
SONNET_OUTPUT_PRICE_PER_MIL=10.00
HAIKU="claude-haiku-5-5"
HAIKU_INPUT_PRICE_PER_MIL=0.10
HAIKU_OUTPUT_PRICE_PER_MIL=0.50

OLLAMA="ollama"
LLAMA="llama3.1:8b"
LLAMA_INPUT_PRICE_PER_MIL=0
LLAMA_OUTPUT_PRICE_PER_MIL=0

GEMINI="gemini"
FLASH_LITE_35="gemini-3.5-flash-lite"
FLASH_LITE_35_INPUT_PRICE_PER_MIL=0.30
FLASH_LITE_35_OUTPUT_PRICE_PER_MIL=2.50
FLASH_LITE_31="gemini-3.1-flash-lite"
FLASH_LITE_31_INPUT_PRICE_PER_MIL=0.25
FLASH_LITE_31_OUTPUT_PRICE_PER_MIL=1.50

CHATGPT="chatgpt"
GPT_MINI="gpt-4.1-mini"
GPT_MINI_INPUT_PRICE_PER_MIL=0.40
GPT_MINI_OUTPUT_PRICE_PER_MIL=1.60
GPT_NANO="gpt-4.1-nano"
GPT_NANO_INPUT_PRICE_PER_MIL=0.10
GPT_NANO_OUTPUT_PRICE_PER_MIL=0.40

# Settings of every LLM call, the same for every layer
LLM_MAX_TOKENS=16000 # Maximum output tokens per call
LLM_TIMEOUT_SECONDS=180 # Seconds before one call times out

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
QNA_LLM="$CLAUDE" # LLM RAGAS uses to write the eval questions and answers
QNA_LLM_MODEL="$SONNET"
QNA_INPUT_PRICE="$SONNET_INPUT_PRICE_PER_MIL"
QNA_OUTPUT_PRICE="$SONNET_OUTPUT_PRICE_PER_MIL"
QNA_KG_LLM="$CLAUDE" # LLM RAGAS uses to build its knowledge graph
QNA_KG_LLM_MODEL="$HAIKU"
QNA_KG_INPUT_PRICE="$HAIKU_INPUT_PRICE_PER_MIL"
QNA_KG_OUTPUT_PRICE="$HAIKU_OUTPUT_PRICE_PER_MIL"
EVAL_QNA_MODEL="all-MiniLM-L6-v2" # Local embedding model RAGAS uses to build its knowledge graph
N_SAMPLE_DOCUMENTS=100 # Documents randomly picked for RAGAS to generate the eval QnA from
N_SAMPLE_EVALS=200 # Eval QnA items to generate

# 6. Query
QUERY_LLM="$OLLAMA" # LLM that transforms the eval questions
QUERY_LLM_MODEL="$LLAMA"
QUERY_INPUT_PRICE="$LLAMA_INPUT_PRICE_PER_MIL"
QUERY_OUTPUT_PRICE="$LLAMA_OUTPUT_PRICE_PER_MIL"
N_MULTI_QUS=3 # Phrasings the query LLM writes for each eval question

# 7. Retrieval
RETRIEVAL_TYPES="dense bm25 hybrid" # any of dense, bm25, hybrid, separated by spaces
TOP_K=20 # Chunks each retriever returns for a query
RERANKING=true # true to also rerank the TOP_K chunks with the cross-encoder, false not to
RERANK_TOP_K=5 # Chunks kept after the cross-encoder reranks the TOP_K chunks
TUNING_QUERIES_SIZE=150 # Eval questions used for tuning; the rest are held out for one final check
N_RETRIEVAL_QUALITY=1 # Best plain setups and best reranked setups whose retrieved outputs go on to the next layers

# 8. Generation
GENERATION_LLM="$OLLAMA" # LLM that answers the prompts
GENERATION_LLM_MODEL="$LLAMA"
GENERATION_INPUT_PRICE="$LLAMA_INPUT_PRICE_PER_MIL"
GENERATION_OUTPUT_PRICE="$LLAMA_OUTPUT_PRICE_PER_MIL"

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
  --qna_llm "$QNA_LLM" \
  --qna_llm_model "$QNA_LLM_MODEL" \
  --qna_input_price "$QNA_INPUT_PRICE" \
  --qna_output_price "$QNA_OUTPUT_PRICE" \
  --qna_kg_llm "$QNA_KG_LLM" \
  --qna_kg_llm_model "$QNA_KG_LLM_MODEL" \
  --qna_kg_input_price "$QNA_KG_INPUT_PRICE" \
  --qna_kg_output_price "$QNA_KG_OUTPUT_PRICE" \
  --eval_qna_model "$EVAL_QNA_MODEL" \
  --n_sample_documents "$N_SAMPLE_DOCUMENTS" \
  --n_sample_evals "$N_SAMPLE_EVALS" \
  --query_llm "$QUERY_LLM" \
  --query_llm_model "$QUERY_LLM_MODEL" \
  --query_input_price "$QUERY_INPUT_PRICE" \
  --query_output_price "$QUERY_OUTPUT_PRICE" \
  --n_multi_qus "$N_MULTI_QUS" \
  --retrieval_types $RETRIEVAL_TYPES \
  --top_k "$TOP_K" \
  --reranking "$RERANKING" \
  --rerank_top_k "$RERANK_TOP_K" \
  --tuning_queries_size "$TUNING_QUERIES_SIZE" \
  --n_retrieval_quality "$N_RETRIEVAL_QUALITY" \
  --generation_llm "$GENERATION_LLM" \
  --generation_llm_model "$GENERATION_LLM_MODEL" \
  --generation_input_price "$GENERATION_INPUT_PRICE" \
  --generation_output_price "$GENERATION_OUTPUT_PRICE" \
  --llm_max_tokens "$LLM_MAX_TOKENS" \
  --llm_timeout_seconds "$LLM_TIMEOUT_SECONDS" \
  2>&1 | tee "$LOG"
