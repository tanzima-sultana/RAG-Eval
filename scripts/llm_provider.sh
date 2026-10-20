# Checks the QnA and query layers end to end with one LLM provider, on a few documents.
# Run it again for every provider each time a provider is added or the LLM code changes.
#
# Usage:
#   bash scripts/llm_provider.sh ollama                     # free, local
#   ALLOW_PAID_API=1 bash scripts/llm_provider.sh claude    # paid, a few cents
#   ALLOW_PAID_API=1 bash scripts/llm_provider.sh gemini    # paid, a few cents
#   ALLOW_PAID_API=1 bash scripts/llm_provider.sh chatgpt   # paid, a few cents
#
# Writes only under outputs/llm_check/{provider}/, never under the real outputs/{dataset}/ caches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate
export PYTHONPATH="$(pwd)/src"

if [ -f .env ]; then
      source .env
fi

PROVIDER="$1"

mkdir -p log
LOG="log/llm_check_${PROVIDER}.txt"

# Models: LLM provider, model name and price in US dollars per million tokens. The same as config.yaml
CLAUDE="claude"
SONNET="claude-sonnet-5-5"
SONNET_INPUT_PRICE_PER_MIL=2.00
SONNET_OUTPUT_PRICE_PER_MIL=10.00
SONNET_CACHE_WRITE_PRICE_PER_MIL=2.50
SONNET_CACHE_READ_PRICE_PER_MIL=0.10
HAIKU="claude-haiku-5-5"
HAIKU_INPUT_PRICE_PER_MIL=0.10
HAIKU_OUTPUT_PRICE_PER_MIL=0.50
HAIKU_CACHE_WRITE_PRICE_PER_MIL=0.125
HAIKU_CACHE_READ_PRICE_PER_MIL=0.01

OLLAMA="ollama"
LLAMA="llama3.1:8b"
LLAMA_INPUT_PRICE_PER_MIL=0
LLAMA_OUTPUT_PRICE_PER_MIL=0
LLAMA_CACHE_WRITE_PRICE_PER_MIL=0
LLAMA_CACHE_READ_PRICE_PER_MIL=0

GEMINI="gemini"
# The 2.5 models are listed but no longer available to new users, so the 3.x models are used
FLASH_LITE_35="gemini-3.5-flash-lite"
FLASH_LITE_35_INPUT_PRICE_PER_MIL=0.30
FLASH_LITE_35_OUTPUT_PRICE_PER_MIL=2.50
FLASH_LITE_35_CACHE_WRITE_PRICE_PER_MIL=0.30
FLASH_LITE_35_CACHE_READ_PRICE_PER_MIL=0.03
FLASH_LITE_31="gemini-3.1-flash-lite"
FLASH_LITE_31_INPUT_PRICE_PER_MIL=0.25
FLASH_LITE_31_OUTPUT_PRICE_PER_MIL=1.50
FLASH_LITE_31_CACHE_WRITE_PRICE_PER_MIL=0.25
FLASH_LITE_31_CACHE_READ_PRICE_PER_MIL=0.025

CHATGPT="chatgpt"
GPT_MINI="gpt-4.1-mini"
GPT_MINI_INPUT_PRICE_PER_MIL=0.40
GPT_MINI_OUTPUT_PRICE_PER_MIL=1.60
GPT_MINI_CACHE_WRITE_PRICE_PER_MIL=0.40
GPT_MINI_CACHE_READ_PRICE_PER_MIL=0.10
GPT_NANO="gpt-4.1-nano"
GPT_NANO_INPUT_PRICE_PER_MIL=0.10
GPT_NANO_OUTPUT_PRICE_PER_MIL=0.40
GPT_NANO_CACHE_WRITE_PRICE_PER_MIL=0.10
GPT_NANO_CACHE_READ_PRICE_PER_MIL=0.025

# Settings of every LLM call
LLM_MAX_TOKENS=16000
LLM_TIMEOUT_SECONDS=180
LLM_CACHE_TTL=5m

# The models each layer uses for this provider
if [ "$PROVIDER" == "$CLAUDE" ]; then
      QNA_LLM="$CLAUDE"
      QNA_LLM_MODEL="$SONNET"
      QNA_INPUT_PRICE="$SONNET_INPUT_PRICE_PER_MIL"
      QNA_OUTPUT_PRICE="$SONNET_OUTPUT_PRICE_PER_MIL"
      QNA_CACHE_WRITE_PRICE="$SONNET_CACHE_WRITE_PRICE_PER_MIL"
      QNA_CACHE_READ_PRICE="$SONNET_CACHE_READ_PRICE_PER_MIL"
      QNA_KG_LLM="$CLAUDE"
      QNA_KG_LLM_MODEL="$HAIKU"
      QNA_KG_INPUT_PRICE="$HAIKU_INPUT_PRICE_PER_MIL"
      QNA_KG_OUTPUT_PRICE="$HAIKU_OUTPUT_PRICE_PER_MIL"
      QNA_KG_CACHE_WRITE_PRICE="$HAIKU_CACHE_WRITE_PRICE_PER_MIL"
      QNA_KG_CACHE_READ_PRICE="$HAIKU_CACHE_READ_PRICE_PER_MIL"
      QUERY_LLM="$CLAUDE"
      QUERY_LLM_MODEL="$HAIKU"
      QUERY_INPUT_PRICE="$HAIKU_INPUT_PRICE_PER_MIL"
      QUERY_OUTPUT_PRICE="$HAIKU_OUTPUT_PRICE_PER_MIL"
      QUERY_CACHE_WRITE_PRICE="$HAIKU_CACHE_WRITE_PRICE_PER_MIL"
      QUERY_CACHE_READ_PRICE="$HAIKU_CACHE_READ_PRICE_PER_MIL"
elif [ "$PROVIDER" == "$OLLAMA" ]; then
      QNA_LLM="$OLLAMA"
      QNA_LLM_MODEL="$LLAMA"
      QNA_INPUT_PRICE="$LLAMA_INPUT_PRICE_PER_MIL"
      QNA_OUTPUT_PRICE="$LLAMA_OUTPUT_PRICE_PER_MIL"
      QNA_CACHE_WRITE_PRICE="$LLAMA_CACHE_WRITE_PRICE_PER_MIL"
      QNA_CACHE_READ_PRICE="$LLAMA_CACHE_READ_PRICE_PER_MIL"
      QNA_KG_LLM="$OLLAMA"
      QNA_KG_LLM_MODEL="$LLAMA"
      QNA_KG_INPUT_PRICE="$LLAMA_INPUT_PRICE_PER_MIL"
      QNA_KG_OUTPUT_PRICE="$LLAMA_OUTPUT_PRICE_PER_MIL"
      QNA_KG_CACHE_WRITE_PRICE="$LLAMA_CACHE_WRITE_PRICE_PER_MIL"
      QNA_KG_CACHE_READ_PRICE="$LLAMA_CACHE_READ_PRICE_PER_MIL"
      QUERY_LLM="$OLLAMA"
      QUERY_LLM_MODEL="$LLAMA"
      QUERY_INPUT_PRICE="$LLAMA_INPUT_PRICE_PER_MIL"
      QUERY_OUTPUT_PRICE="$LLAMA_OUTPUT_PRICE_PER_MIL"
      QUERY_CACHE_WRITE_PRICE="$LLAMA_CACHE_WRITE_PRICE_PER_MIL"
      QUERY_CACHE_READ_PRICE="$LLAMA_CACHE_READ_PRICE_PER_MIL"
elif [ "$PROVIDER" == "$GEMINI" ]; then
      QNA_LLM="$GEMINI"
      QNA_LLM_MODEL="$FLASH_LITE_35"
      QNA_INPUT_PRICE="$FLASH_LITE_35_INPUT_PRICE_PER_MIL"
      QNA_OUTPUT_PRICE="$FLASH_LITE_35_OUTPUT_PRICE_PER_MIL"
      QNA_CACHE_WRITE_PRICE="$FLASH_LITE_35_CACHE_WRITE_PRICE_PER_MIL"
      QNA_CACHE_READ_PRICE="$FLASH_LITE_35_CACHE_READ_PRICE_PER_MIL"
      QNA_KG_LLM="$GEMINI"
      QNA_KG_LLM_MODEL="$FLASH_LITE_31"
      QNA_KG_INPUT_PRICE="$FLASH_LITE_31_INPUT_PRICE_PER_MIL"
      QNA_KG_OUTPUT_PRICE="$FLASH_LITE_31_OUTPUT_PRICE_PER_MIL"
      QNA_KG_CACHE_WRITE_PRICE="$FLASH_LITE_31_CACHE_WRITE_PRICE_PER_MIL"
      QNA_KG_CACHE_READ_PRICE="$FLASH_LITE_31_CACHE_READ_PRICE_PER_MIL"
      QUERY_LLM="$GEMINI"
      QUERY_LLM_MODEL="$FLASH_LITE_31"
      QUERY_INPUT_PRICE="$FLASH_LITE_31_INPUT_PRICE_PER_MIL"
      QUERY_OUTPUT_PRICE="$FLASH_LITE_31_OUTPUT_PRICE_PER_MIL"
      QUERY_CACHE_WRITE_PRICE="$FLASH_LITE_31_CACHE_WRITE_PRICE_PER_MIL"
      QUERY_CACHE_READ_PRICE="$FLASH_LITE_31_CACHE_READ_PRICE_PER_MIL"
elif [ "$PROVIDER" == "$CHATGPT" ]; then
      QNA_LLM="$CHATGPT"
      QNA_LLM_MODEL="$GPT_MINI"
      QNA_INPUT_PRICE="$GPT_MINI_INPUT_PRICE_PER_MIL"
      QNA_OUTPUT_PRICE="$GPT_MINI_OUTPUT_PRICE_PER_MIL"
      QNA_CACHE_WRITE_PRICE="$GPT_MINI_CACHE_WRITE_PRICE_PER_MIL"
      QNA_CACHE_READ_PRICE="$GPT_MINI_CACHE_READ_PRICE_PER_MIL"
      QNA_KG_LLM="$CHATGPT"
      QNA_KG_LLM_MODEL="$GPT_NANO"
      QNA_KG_INPUT_PRICE="$GPT_NANO_INPUT_PRICE_PER_MIL"
      QNA_KG_OUTPUT_PRICE="$GPT_NANO_OUTPUT_PRICE_PER_MIL"
      QNA_KG_CACHE_WRITE_PRICE="$GPT_NANO_CACHE_WRITE_PRICE_PER_MIL"
      QNA_KG_CACHE_READ_PRICE="$GPT_NANO_CACHE_READ_PRICE_PER_MIL"
      QUERY_LLM="$CHATGPT"
      QUERY_LLM_MODEL="$GPT_NANO"
      QUERY_INPUT_PRICE="$GPT_NANO_INPUT_PRICE_PER_MIL"
      QUERY_OUTPUT_PRICE="$GPT_NANO_OUTPUT_PRICE_PER_MIL"
      QUERY_CACHE_WRITE_PRICE="$GPT_NANO_CACHE_WRITE_PRICE_PER_MIL"
      QUERY_CACHE_READ_PRICE="$GPT_NANO_CACHE_READ_PRICE_PER_MIL"
else
      echo "Usage: bash scripts/llm_provider.sh claude|ollama|gemini|chatgpt"
      exit 1
fi

# 1. Dataset: a few documents, in a folder of its own
DATASET_SIZE=20
INPUT_PATH="pixparse/pdfa-eng-wds"
OUTPUT_PATH="outputs/llm_check/$PROVIDER"
DOWNLOAD_WORKERS=1
MAX_BATCH_ROWS=1
HUGGING_FACE_SPLIT="train"
PARSE_WORKERS=16
PARSE_TIMEOUT_SECONDS=0
DEDUP_THRESHOLD=0.0
PDF_PARSER="pypdfium2"

# 2. Eval QnA: a few questions from a few short documents, to keep the calls small
# Only documents up to MAX_DOCUMENT_TOKENS are used, so every provider gets the same documents and they fit in
# Ollama's context window (32768 tokens, see src/llm/ollama.py). The check tests the code, not long documents
MAX_DOCUMENT_TOKENS=8000
EVAL_QNA_MODEL="all-MiniLM-L6-v2"
N_SAMPLE_DOCUMENTS=3
N_SAMPLE_EVALS=2

# 3. Query
MODEL_NAME="all-MiniLM-L6-v2"
N_MULTI_QUS=3

# run

python3 src/llm_provider.py \
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
  --max_document_tokens "$MAX_DOCUMENT_TOKENS" \
  --eval_qna_model "$EVAL_QNA_MODEL" \
  --n_sample_documents "$N_SAMPLE_DOCUMENTS" \
  --n_sample_evals "$N_SAMPLE_EVALS" \
  --qna_llm "$QNA_LLM" \
  --qna_llm_model "$QNA_LLM_MODEL" \
  --qna_input_price "$QNA_INPUT_PRICE" \
  --qna_output_price "$QNA_OUTPUT_PRICE" \
  --qna_cache_write_price "$QNA_CACHE_WRITE_PRICE" \
  --qna_cache_read_price "$QNA_CACHE_READ_PRICE" \
  --qna_kg_llm "$QNA_KG_LLM" \
  --qna_kg_llm_model "$QNA_KG_LLM_MODEL" \
  --qna_kg_input_price "$QNA_KG_INPUT_PRICE" \
  --qna_kg_output_price "$QNA_KG_OUTPUT_PRICE" \
  --qna_kg_cache_write_price "$QNA_KG_CACHE_WRITE_PRICE" \
  --qna_kg_cache_read_price "$QNA_KG_CACHE_READ_PRICE" \
  --model_name "$MODEL_NAME" \
  --n_multi_qus "$N_MULTI_QUS" \
  --query_llm "$QUERY_LLM" \
  --query_llm_model "$QUERY_LLM_MODEL" \
  --query_input_price "$QUERY_INPUT_PRICE" \
  --query_output_price "$QUERY_OUTPUT_PRICE" \
  --query_cache_write_price "$QUERY_CACHE_WRITE_PRICE" \
  --query_cache_read_price "$QUERY_CACHE_READ_PRICE" \
  --llm_max_tokens "$LLM_MAX_TOKENS" \
  --llm_timeout_seconds "$LLM_TIMEOUT_SECONDS" \
  --llm_cache_ttl "$LLM_CACHE_TTL" \
  2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
