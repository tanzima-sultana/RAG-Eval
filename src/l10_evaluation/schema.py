import dataclasses

@dataclasses.dataclass
class Quality:
	"""
	The retrieval quality of one Retrieved at one level, chunk or doc. Which chunks are relevant and what recall
	counts depend on the level, see RetrievalQuality.

	Fields:
		hit:                 1 if at least one chunk is relevant, else 0.
		precision:           Share of the k chunks that are relevant.
		recall:              Share of the relevant items found in the k chunks.
		mrr:                 1 / rank of the first relevant chunk. 0 for a miss.
		ndcg:                nDCG of the k chunks, rewarding relevant chunks near the top.
		first_relevant_rank: Rank of the first relevant chunk, starting at 1. 0 for a miss.
	"""
	hit: int
	precision: float
	recall: float
	mrr: float
	ndcg: float
	first_relevant_rank: int

@dataclasses.dataclass
class RetrievalQuality:
	"""
	The retrieval quality of one Retrieved against its eval question.

	Fields:
		k:             Number of chunks scored: top_k, or rerank_top_k for a reranked Retrieved.
		chunk_quality: The quality at chunk level. A chunk is relevant if it comes from one of the EvalQnA doc_ids and
		               its text overlaps one of the EvalQnA context passages. Recall is the share of context passages found.
		doc_quality:   The quality at doc level. A chunk is relevant if it comes from one of the EvalQnA doc_ids.
		               Recall is the share of EvalQnA doc_ids found.
	"""
	k: int
	chunk_quality: Quality
	doc_quality: Quality
