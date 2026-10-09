import dataclasses

import faiss
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# One record per index build, not per chunk: many chunks go into one
# FAISS index, one BM25 index and one Qdrant collection.

@dataclasses.dataclass
class FaissIndex:
	# faiss_flatip, faiss_ivf or faiss_hnsw
	faiss_index_type: str
	faiss_index: faiss.Index
	# FAISS position i holds the vector of chunk_ids[i]
	chunk_ids: list[str]
	index_file_path: str
	dimension: int

@dataclasses.dataclass
class Bm25Index:
	bm25_model: BM25Okapi
	# BM25 position i holds the text of chunk_ids[i]
	chunk_ids: list[str]
	index_file_path: str

@dataclasses.dataclass
class QdrantIndex:
	# No chunk_ids list: each Qdrant point stores its chunk_id itself
	qdrant_client: QdrantClient
	collection_name: str
	dimension: int
