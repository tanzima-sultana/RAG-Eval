"""
hybrid.py

Hybrid retrieval: fuses the dense and BM25 outputs of the same queries with reciprocal rank fusion (RRF).
"""

from common.constants import BM25_INDEX
from common.constants import HYBRID_RETRIEVAL
from common.constants import RRF_K
from stages.evaluation.retrieval.schema import Retrieved

def get_hybrid_outputs(chunk_type: str, dense_index_type: str, dense_outputs: list[Retrieved], bm25_outputs: list[Retrieved], top_k: int) -> list[Retrieved]:
	"""
	Input: the chunk type, the dense index type, that index's dense outputs, the BM25 outputs and top_k.
	Output: one Retrieved per query and query type found in both, not reranked.
	What it does: for each dense output, finds the BM25 output of the same query and query type and fuses their
	chunks. HyDE has no BM25 output, so it has no hybrid output.
	"""
	# 1. Find each BM25 output by its query id and query type
	bm25_outputs_by_key = {}
	for bm25_retrieved in bm25_outputs:
		key = (bm25_retrieved.id, bm25_retrieved.query_type)
		bm25_outputs_by_key[key] = bm25_retrieved

	# 2. Fuse each dense output with its BM25 output
	hybrid_outputs = []
	for dense_retrieved in dense_outputs:
		key = (dense_retrieved.id, dense_retrieved.query_type)
		if key not in bm25_outputs_by_key:
			continue
		bm25_retrieved = bm25_outputs_by_key[key]

		fused_scores = fuse_with_rrf(dense_retrieved.chunk_ids, bm25_retrieved.chunk_ids)
		hybrid_retrieved = make_hybrid_retrieved(dense_retrieved.id, dense_retrieved.query_type, chunk_type, dense_index_type, top_k, fused_scores)
		hybrid_outputs.append(hybrid_retrieved)

	return hybrid_outputs

def fuse_with_rrf(dense_chunk_ids: list[str], bm25_chunk_ids: list[str]) -> dict[str, float]:
	"""
	Input: the dense chunk ids and the BM25 chunk ids, each best first.
	Output: a dict of chunk_id to fused score.
	What it does: gives each chunk 1 / (RRF_K + rank) from every list it is in, with rank starting at 1, and adds
	them up. Ranks are used instead of scores, because dense and BM25 scores are on different scales.
	"""
	fused_scores = {}

	for position in range(len(dense_chunk_ids)):
		chunk_id = dense_chunk_ids[position]
		if chunk_id not in fused_scores:
			fused_scores[chunk_id] = 0.0
		fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (RRF_K + position + 1)

	for position in range(len(bm25_chunk_ids)):
		chunk_id = bm25_chunk_ids[position]
		if chunk_id not in fused_scores:
			fused_scores[chunk_id] = 0.0
		fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (RRF_K + position + 1)

	return fused_scores

def make_hybrid_retrieved(query_id: str, query_type: str, chunk_type: str, dense_index_type: str, top_k: int, fused_scores: dict[str, float]) -> Retrieved:
	"""
	Input: the query id and type, the chunk type, the dense index type, top_k, and the fused score of each chunk.
	Output: a Retrieved holding the top_k chunks, best first.
	What it does: sorts the chunks by fused score and keeps top_k.
	"""
	# 1. Sort best first and keep top_k
	sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:top_k]

	# 2. Get each chunk's score and document. A chunk id is "{doc_id}_{chunk_idx}"
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(fused_scores[chunk_id])
		doc_id = chunk_id.rsplit("_", 1)[0]
		doc_ids.append(doc_id)

	return Retrieved(
		id=query_id,
		query_type=query_type,
		chunk_type=chunk_type,
		retrieval_type=HYBRID_RETRIEVAL,
		index_type=f"{dense_index_type}+{BM25_INDEX}",
		top_k=top_k,
		reranking=False,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)
