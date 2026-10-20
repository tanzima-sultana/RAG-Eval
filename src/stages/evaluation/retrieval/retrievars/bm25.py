"""
bm25.py

BM25 retrieval: searches the built-in BM25 index with the query tokens.
"""

import numpy as np

from common.constants import BM25_RETRIEVAL
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from stages.build.indexing.schema import Bm25Index
from stages.question.query.schema import Query
from stages.evaluation.retrieval.schema import Retrieved

def get_bm25_outputs(queries: list[Query], chunk_type: str, index_type: str, bm25_index: Bm25Index, top_k: int, query_types: list[str]) -> list[Retrieved]:
	"""
	Input: the queries, the chunk type, the built-in BM25 index with its index type, top_k, and query_types, the query
	types to search with.
	Output: one Retrieved per query and query type in query_types (original, rewritten, multi), not reranked.
	What it does: searches the index with each query type's tokens and keeps the top_k chunks.
	HyDE has no tokens, because it is searched by embedding only, so it is not searched here.
	"""
	bm25_outputs = []

	for query in queries:
		# 1. Collect the tokens of each query type
		multi_tokens = []
		for multi_qus in query.multi_qus:
			multi_tokens.append(multi_qus.tokens)

		query_types_and_tokens = []
		query_types_and_tokens.append((ORIGINAL_QUERY, [query.original_qus.tokens]))
		query_types_and_tokens.append((REWRITTEN_QUERY, [query.rewritten_qus.tokens]))
		query_types_and_tokens.append((MULTI_QUERY, multi_tokens))

		# 2. Search the index with each query type
		# A query type with no tokens gets a Retrieved with no chunks, so it still counts as a miss
		for query_type, query_tokens_list in query_types_and_tokens:
			if query_type not in query_types:
				continue
			hits = search_bm25(bm25_index, query_tokens_list, top_k)
			retrieved = make_bm25_retrieved(query.id, query_type, chunk_type, index_type, top_k, hits)
			bm25_outputs.append(retrieved)

	return bm25_outputs

def search_bm25(bm25_index: Bm25Index, query_tokens_list: list[list[str]], top_k: int) -> list[tuple[str, float]]:
	"""
	Input: the built-in BM25 index, the token lists to search with and top_k.
	Output: (chunk_id, score) pairs, up to top_k for each token list.
	What it does: scores every chunk with BM25 and keeps the top_k. A chunk with score 0 shares no word with the
	query, so it is not kept.
	"""
	hits = []

	for query_tokens in query_tokens_list:
		scores = bm25_index.bm25_model.get_scores(query_tokens)
		# Positions of the top_k scores, best first
		top_positions = np.argsort(scores)[::-1][:top_k]
		for position in top_positions:
			score = float(scores[position])
			if score <= 0.0:
				continue
			chunk_id = bm25_index.chunk_ids[int(position)]
			hits.append((chunk_id, score))

	return hits

def make_bm25_retrieved(query_id: str, query_type: str, chunk_type: str, index_type: str, top_k: int, hits: list[tuple[str, float]]) -> Retrieved:
	"""
	Input: the query id and type, the chunk type, the index type, top_k, and the (chunk_id, score) pairs found.
	Output: a Retrieved holding the top_k chunks, best first.
	What it does: keeps each chunk once with its best score, so the multi query phrasings are merged.
	"""
	# 1. Keep each chunk's best score
	best_scores = {}
	for chunk_id, score in hits:
		if chunk_id not in best_scores or score > best_scores[chunk_id]:
			best_scores[chunk_id] = score

	# 2. Sort best first and keep top_k
	sorted_chunk_ids = sorted(best_scores, key=best_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:top_k]

	# 3. Get each chunk's score and document. A chunk id is "{doc_id}_{chunk_idx}"
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(best_scores[chunk_id])
		doc_id = chunk_id.rsplit("_", 1)[0]
		doc_ids.append(doc_id)

	return Retrieved(
		id=query_id,
		query_type=query_type,
		chunk_type=chunk_type,
		retrieval_type=BM25_RETRIEVAL,
		index_type=index_type,
		top_k=top_k,
		reranking=False,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)
