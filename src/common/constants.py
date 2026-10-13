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

# Number of eval QnA items, saved as {output_path}/eval_qna_{SAMPLE_DOCUMENTS}_{N_EVAL_SAMPLES}_{EVAL_QNA_LLM}_{EVAL_QNA_MODEL}.json
N_EVAL_SAMPLES=50

# LLM provider RAGAS uses to generate the eval QnA. The models are picked in the code
CLAUDE_LLM_GENERATOR="claude"
EVAL_QNA_LLM=CLAUDE_LLM_GENERATOR
# Claude model for the RAGAS knowledge graph steps (headlines, summaries, filter, themes, entities)
CLAUDE_TRANSFORMS_MODEL="claude-haiku-5-5"
# Claude model for writing the personas, scenarios, questions and answers
CLAUDE_QUESTION_MODEL="claude-sonnet-5-5"
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
EVAL_QNA_MODEL="all-MiniLM-L6-v2"
# Number of documents randomly picked from the dataset for RAGAS to generate the eval QnA from
SAMPLE_DOCUMENTS=200
# Seed for that pick, so the same documents are picked every run
SAMPLE_DOCUMENTS_SEED=42
# Only documents with more than this many tokens (tiktoken cl100k_base, as RAGAS counts) are sampled,
# because RAGAS only extracts headlines and summaries from documents over 500 tokens
EVAL_MIN_DOCUMENT_TOKENS=500
# An eval passage is linked to the document that contains its first this many characters (whitespace collapsed),
# because RAGAS sometimes stitches non-adjacent sections of a document into one passage
EVAL_PASSAGE_MATCH_CHARS=80

# 5. Query

# Local LLM provider that rewrites the eval questions, writes the multi-query phrasings and the HyDE answers
OLLAMA_LLM="ollama"
# Ollama server, started by the Ollama install
OLLAMA_URL="http://localhost:11434"
# Seconds before one Ollama call times out. The first call after 5 idle minutes also reloads the model onto the GPU
OLLAMA_TIMEOUT_SECONDS=120
# Maximum output tokens per Ollama call. At temperature 0 a model can repeat a phrase forever, so the reply is cut here
OLLAMA_MAX_TOKENS=400
QUERY_LLM=OLLAMA_LLM
# Local model the query LLM provider runs
QUERY_LLM_MODEL="llama3.1:8b"
# Number of phrasings the query LLM writes for each eval question
N_MULTI_QUS=3
# Stored as the HyDE answer when the query LLM still refuses to write one, so l6 skips it
I_DONT_KNOW="I don't know"

# 6. Retrieval

# Query types searched for each eval question, one per l5 Query field
ORIGINAL_QUERY="original"
REWRITTEN_QUERY="rewritten"
MULTI_QUERY="multi"
HYDE_QUERY="hyde"
# Retrieval types
DENSE_RETRIEVAL="dense"
# Indexer name of the Qdrant collection, stored in Retrieved.indexer
QDRANT_INDEX="qdrant"

# Seed for the tuning / held-out split, so every run gets the same split
TUNING_SPLIT_SEED=42

# Number of chunks each retriever returns for a query
TOP_K=20
# Number of chunks kept after the cross-encoder reranks the TOP_K chunks
RERANK_TOP_K=5
