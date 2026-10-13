from common.args import RetrievalArgs
from l2_chunking.schema import Chunk
from l4_indexing.schema import FaissIndex
from l4_indexing.schema import Bm25Index
from l4_indexing.schema import QdrantIndex
from l4_indexing.schema import LangchainFaissIndex
from l4_indexing.schema import LangchainBm25Index
from l4_indexing.schema import LangchainQdrantIndex
from l5_query.schema import Query
from l6_retrieval.schema import Retrieved

def get_retrieved_outputs(
	queries: list[Query],
	chunks: dict[str, Chunk],
	faiss_index: FaissIndex | LangchainFaissIndex,
	bm25_index: Bm25Index | LangchainBm25Index,
	qdrant_index: QdrantIndex | LangchainQdrantIndex,
	retrieval_args: RetrievalArgs
) -> list[Retrieved]:
	"""
	Input: queries, the l5 queries; chunks, chunk_id to Chunk, for the chunk texts and doc_ids;
	faiss_index, bm25_index and qdrant_index, the l4 indexes, either all built-in or all LangChain;
	retrieval_args, top_k and rerank_top_k.
	Output: one Retrieved per query, query type and retrieval setup, with reranking off and on. Nothing is saved.
	What it does: checks once whether the indexes are LangChain or built-in, since each kind is searched differently.
	The rest is not written yet.
	"""
	# 1. Decide once which kind of indexes were loaded. load_index returns all three of the same kind
	if isinstance(faiss_index, LangchainFaissIndex):
		# LangChain indexes: not written yet
		pass
	else:
		# Built-in indexes: not written yet
		pass
