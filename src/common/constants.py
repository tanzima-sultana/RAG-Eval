# 1. Doc processing

# Number of documents downloaded/decoded at the same time
DOWNLOAD_WORKERS = 64

# Number of documents parsed at the same time, each in its own process
PARSE_WORKERS = 16

# Number of documents sent to a parse worker process in one go
PARSE_DOC_COUNT = 16

# 2. Chunking

FIXED_CHUNKING="fixed"
SENTENCE_CHUNKING="sentence"
SEMANTIC_CHUNKING="semantic"


