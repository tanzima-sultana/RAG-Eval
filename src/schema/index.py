import dataclasses

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# One record per index, not per chunk: all chunks of one chunking run go into one index


@dataclasses.dataclass
class FaissIndex:
	# faiss_flatip, faiss_ivf or faiss_hnsw
	index_type: str
	vector_store: FAISS
	index_path: str
	dimension: int


@dataclasses.dataclass
class Bm25Index:
	retriever: BM25Retriever
	index_path: str
