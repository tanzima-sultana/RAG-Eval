import dataclasses


@dataclasses.dataclass(frozen=True)
class Setup:
	# One retrieval setup the benchmark scores and the pipeline runs. Frozen, so the evals can be grouped by it
	# fixed, sentence or semantic
	chunk_type: str
	# faiss_flatip, faiss_ivf, faiss_hnsw or qdrant for dense, bm25 for bm25, and both joined with "+" for hybrid,
	# e.g. faiss_flatip+bm25
	index_type: str
	# original, rewritten, multi or hyde. BM25 and hybrid have no hyde
	query_type: str
	# dense, bm25 or hybrid
	retrieval_type: str
	# True to rerank the top_k chunks with the cross-encoder and keep rerank_top_k
	reranking: bool
