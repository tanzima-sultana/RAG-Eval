import dataclasses


@dataclasses.dataclass
class Retrieved:
	# The chunks found for one eval question with one query type and one retrieval setup
	# Same as the EvalSample and Query id, linking the result to its eval question
	id: str
	# original, rewritten, multi or hyde
	query_type: str
	# fixed, sentence or semantic
	chunk_type: str
	# dense, bm25 or hybrid
	retrieval_type: str
	# faiss_flatip, faiss_ivf, faiss_hnsw or qdrant for dense, bm25 for bm25, and both joined with "+" for hybrid,
	# e.g. faiss_flatip+bm25
	index_type: str
	top_k: int
	reranking: bool
	# Chunks kept after reranking. 0 when not reranked
	rerank_top_k: int
	# The chunks, best first
	chunk_ids: list[str]
	# The score of each chunk. Each retrieval type and reranking has its own scale
	chunk_scores: list[float]
	# The document of each chunk: the part of the chunk id before the last "_"
	doc_ids: list[str]
