import dataclasses

import faiss
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

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

# LangChain indexes, built with --framework langchain. Each LangChain
# Document stores the chunk text as page_content and {"chunk_id": chunk_id}
# as metadata, so a search returns the chunk_id directly and no chunk_ids
# list is needed.

@dataclasses.dataclass
class LangchainFaissIndex:
	# faiss_flatip, faiss_ivf or faiss_hnsw
	faiss_index_type: str
	vector_store: FAISS
	# Folder holding index.faiss and index.pkl, written by FAISS.save_local
	index_folder_path: str
	dimension: int

@dataclasses.dataclass
class LangchainBm25Index:
	retriever: BM25Retriever
	index_file_path: str

@dataclasses.dataclass
class LangchainQdrantIndex:
	vector_store: QdrantVectorStore
	collection_name: str
	dimension: int
