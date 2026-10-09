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
