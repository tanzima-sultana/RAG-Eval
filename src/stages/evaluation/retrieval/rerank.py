"""
rerank.py

Reranking: scores each retrieved chunk against its eval question with a cross-encoder and keeps the best
rerank_top_k chunks.
"""

from sentence_transformers import CrossEncoder

from stages.build.chunking.schema import Chunk
from stages.question.query.schema import Query
from stages.evaluation.retrieval.schema import Retrieved

def get_reranked_outputs(queries: list[Query], chunks: dict[str, Chunk], retrieved_outputs: list[Retrieved], cross_encoder: CrossEncoder, rerank_top_k: int) -> list[Retrieved]:
	"""
	Input: the queries, the chunks, the not reranked outputs, the cross-encoder and rerank_top_k.
	Output: one reranked Retrieved per output, holding the best rerank_top_k chunks.
	What it does: scores every (question, chunk text) pair with the cross-encoder and sorts each output's chunks
	by that score.
	"""
	# 1. Find each eval question by its query id
	# Every query type is reranked against the original eval question, since that is what the answer must match
	question_by_query_id = {}
	for query in queries:
		question_by_query_id[query.id] = query.original_qus.qus

	# 2. Collect every (question, chunk text) pair, all outputs one after another
	pairs = []
	for retrieved in retrieved_outputs:
		question = question_by_query_id[retrieved.id]
		for chunk_id in retrieved.chunk_ids:
			pairs.append((question, chunks[chunk_id].text))

	# 3. Score all pairs in one call, so the cross-encoder runs in large batches
	pair_scores = []
	if len(pairs) > 0:
		pair_scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)

	# 4. Sort each output's chunks by their score and keep rerank_top_k
	reranked_outputs = []
	# Position of the output's first pair in pair_scores
	pair_position = 0
	for retrieved in retrieved_outputs:
		rerank_scores = {}
		for chunk_id in retrieved.chunk_ids:
			rerank_scores[chunk_id] = float(pair_scores[pair_position])
			pair_position = pair_position + 1

		sorted_chunk_ids = sorted(rerank_scores, key=rerank_scores.get, reverse=True)
		chunk_ids = sorted_chunk_ids[:rerank_top_k]

		# A chunk id is "{doc_id}_{chunk_idx}"
		chunk_scores = []
		doc_ids = []
		for chunk_id in chunk_ids:
			chunk_scores.append(rerank_scores[chunk_id])
			doc_ids.append(chunk_id.rsplit("_", 1)[0])

		reranked_outputs.append(Retrieved(
			id=retrieved.id,
			query_type=retrieved.query_type,
			chunk_type=retrieved.chunk_type,
			retrieval_type=retrieved.retrieval_type,
			index_type=retrieved.index_type,
			top_k=retrieved.top_k,
			reranking=True,
			rerank_top_k=rerank_top_k,
			chunk_ids=chunk_ids,
			chunk_scores=chunk_scores,
			doc_ids=doc_ids
		))

	return reranked_outputs
