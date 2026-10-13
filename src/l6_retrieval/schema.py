import dataclasses

@dataclasses.dataclass
class Retrieved:
	"""
	One retrieval result: the chunks found for one eval question, with one query type, one chunk type and one
	retrieval setup.

	Fields:
		id:             Same as the Query and EvalQnA id, linking the result to its eval question.
		query_type:     original, rewritten, multi or hyde.
		chunk_type:     fixed, sentence or semantic: the chunks and indexes that were searched.
		retrieval_type: dense, bm25 or hybrid.
		index_type:     faiss_flatip, faiss_ivf, faiss_hnsw or qdrant for dense, bm25 for bm25,
		                and both joined with "+" for hybrid, e.g. faiss_flatip+bm25.
		top_k:          Number of chunks the retriever returns.
		reranking:      True if the fixed cross-encoder reranked the chunks.
		rerank_top_k:   Number of chunks kept after reranking. 0 when not reranked.
		chunk_ids:      The final chunks, best first.
		chunk_scores:   The score of each chunk, in the same order as chunk_ids. Each retrieval type and reranking
						has its own scale, so scores are only comparable between results with the same ones.
		doc_ids:        The document of each chunk, in the same order as chunk_ids.
	"""
	id: str
	query_type: str
	chunk_type: str
	retrieval_type: str
	index_type: str
	top_k: int
	reranking: bool
	rerank_top_k: int
	chunk_ids: list[str]
	chunk_scores: list[float]
	doc_ids: list[str]
	
