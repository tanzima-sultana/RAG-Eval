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
from l6_retrieval import validate
from l6_retrieval.retrievars import dense

def get_retrieved_outputs(
	queries: list[Query],
	chunks: dict[str, Chunk],
	faiss_index: FaissIndex | LangchainFaissIndex,
	bm25_index: Bm25Index | LangchainBm25Index,
	qdrant_index: QdrantIndex | LangchainQdrantIndex,
	retrieval_args: RetrievalArgs
) -> list[Retrieved]:
	"""
	Input: l5 queries, chunks, the three l4 indexes and retrieval_args.
	Output: one Retrieved per query, query type and retrieval setup.
	What it does: retrieves chunks for every query with every retrieval setup.
	"""
	# 1. Validate retrieval args
	validate.validate_retrieval_args(retrieval_args)

	retrieved_outputs = []

	# 2. Decide once which kind of indexes were loaded. load_index returns all three of the same kind
	if isinstance(faiss_index, LangchainFaissIndex):
		# LangChain indexes: not written yet
		pass
	else:
		# 3. Dense retrieval
		dense_outputs = dense.get_dense_outputs(queries, faiss_index, qdrant_index, retrieval_args.top_k)
		retrieved_outputs.extend(dense_outputs)
		print(f"Dense retrieved : {len(dense_outputs)}")

	return retrieved_outputs
