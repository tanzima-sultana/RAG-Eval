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
BM25_INDEX="bm25"
QDRANT_INDEX="qdrant"

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

# Number of eval QnA items, saved as {output_path}/eval_qna_{SAMPLE_DOCUMENTS}_{N_EVAL_SAMPLES}_{qna_llm_model}_{EVAL_QNA_MODEL}.json
N_EVAL_SAMPLES=50

# Claude LLM provider name
CLAUDE_LLM="claude"
# ChatGPT (OpenAI) LLM provider name
CHATGPT_LLM="chatgpt"
# Gemini (Google) LLM provider name
GEMINI_LLM="gemini"
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

# Ollama (local) LLM provider name
OLLAMA_LLM="ollama"
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
BM25_RETRIEVAL="bm25"
HYBRID_RETRIEVAL="hybrid"
# Reciprocal rank fusion constant for hybrid: a chunk at rank r (from 1) gets 1 / (RRF_K + r) from each list
RRF_K=60

# Seed for the tuning / held-out split, so every run gets the same split
TUNING_SPLIT_SEED=42

# Number of chunks each retriever returns for a query
TOP_K=20
# Number of chunks kept after the cross-encoder reranks the TOP_K chunks
RERANK_TOP_K=5
# Cross-encoder that reranks the TOP_K chunks against the eval question
CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

# Share of a chunk's words that must be in a context passage for the chunk to count as relevant at chunk level
CHUNK_CONTEXT_OVERLAP_THRESHOLD=0.5

# Number of best plain setups and of best reranked setups whose retrieved outputs go on to the next layers
N_RETRIEVAL_QUALITY=1
