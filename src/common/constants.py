# 1. Doc ingestion

# Number of documents downloaded/decoded at the same time
DOWNLOAD_WORKERS = 64
MAX_BATCH_ROWS=256
HUGGING_FACE_SPLIT="train"

# Number of documents parsed at the same time, each in its own process
PARSE_WORKERS = 16
PARSE_TIMEOUT_SECONDS=120
DEDUP_THRESHOLD=0.90

# Number of documents sent to a parse worker process in one go
PDF_PARSER="pypdf"

# 2. Chunking

# Chunking frameworks: "" uses the built-in chunkers
NO_FRAMEWORK=""
LANGCHAIN_FRAMEWORK="langchain"

FIXED_CHUNKING="fixed"
SENTENCE_CHUNKING="sentence"
SEMANTIC_CHUNKING="semantic"

MIN_CHUNK_SIZE=128
MAX_CHUNK_SIZE=256
FIXED_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.2

# 4. Indexing

FAISS_FLATIP_INDEX="faiss_flatip"
FAISS_IVF_INDEX="faiss_ivf"
FAISS_HNSW_INDEX="faiss_hnsw"

IVF_NLIST=256
HNSW_M=32
HNSW_EF_CONSTRUCTION=200
BM25_K1=1.5
BM25_B=0.75
BM25_TOKENIZER="whitespace"
QDRANT_BATCH_SIZE=256

# Qdrant server started with the qdrant-docker skill
QDRANT_URL="http://localhost:6333"

# Eval

# Number of eval QnA items, saved as {output_path}/eval_qna_{dataset_size}_{SAMPLE_DOCUMENTS}_{N_EVAL_SAMPLES}_{EVAL_LLM}_{EVAL_MODEL}.json
N_EVAL_SAMPLES=50

# Claude model RAGAS uses to write the eval questions and answers
EVAL_LLM="claude-sonnet-5-5"
# Maximum output tokens per eval LLM call. RAGAS prompts return short JSON, so 16000 leaves room for thinking
EVAL_LLM_MAX_TOKENS=16000
# Seconds before one eval LLM call times out, the same as RAGAS's own per-job timeout (RunConfig.timeout)
EVAL_LLM_TIMEOUT_SECONDS=180
# Claude API price in US dollars per million tokens (Anthropic API rates, 2026-10-06)
CLAUDE_PRICES_PER_MILLION_TOKENS={
	"claude-opus-5-5": {"input": 4.00, "output": 20.00},
	"claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
	"claude-haiku-5-5": {"input": 0.10, "output": 0.50},
}
# Local HuggingFace SentenceTransformer model RAGAS uses to build its knowledge graph
EVAL_MODEL="all-MiniLM-L6-v2"
# Number of documents randomly picked from the dataset for RAGAS to generate the eval QnA from
SAMPLE_DOCUMENTS=200
# Seed for that pick, so the same documents are picked every run
SAMPLE_DOCUMENTS_SEED=42
# Only documents with more than this many tokens (tiktoken cl100k_base, as RAGAS counts) are sampled,
# because RAGAS only extracts headlines and summaries from documents over 500 tokens
EVAL_MIN_DOCUMENT_TOKENS=500
