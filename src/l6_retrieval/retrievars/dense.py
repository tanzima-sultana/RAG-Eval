"""
dense.py

Dense retrieval: searches the built-in FAISS index and the Qdrant collection with the query embeddings.
"""

import numpy as np

from common.constants import DENSE_RETRIEVAL
from common.constants import HYDE_QUERY
from common.constants import I_DONT_KNOW
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import QDRANT_INDEX
from common.constants import REWRITTEN_QUERY
from l4_indexing.schema import FaissIndex
from l4_indexing.schema import QdrantIndex
from l5_query.schema import Query
from l6_retrieval.schema import Retrieved

def get_dense_outputs(queries: list[Query], chunk_type: str, index_type: str, index: FaissIndex | QdrantIndex, top_k: int) -> list[Retrieved]:
	"""
	Input: l5 queries, the chunk type, one built-in FAISS index or Qdrant collection with its index type, and top_k.
	Output: one Retrieved per query and query type, not reranked.
	What it does: searches the index with each query type's embeddings and keeps the top_k chunks.
	"""
	dense_outputs = []

	for query in queries:
		# 1. Collect the embeddings of each query type
		multi_vectors = []
		for multi_qus in query.multi_qus:
			multi_vectors.append(multi_qus.embeddings)

		# A HyDE answer the LLM refused to write is not searched
		hyde_vectors = []
		if query.hyde_ans.ans != I_DONT_KNOW:
			hyde_vectors.append(query.hyde_ans.embeddings)

		query_types_and_vectors = []
		query_types_and_vectors.append((ORIGINAL_QUERY, [query.original_qus.embeddings]))
		query_types_and_vectors.append((REWRITTEN_QUERY, [query.rewritten_qus.embeddings]))
		query_types_and_vectors.append((MULTI_QUERY, multi_vectors))
		query_types_and_vectors.append((HYDE_QUERY, hyde_vectors))

		# 2. Search the index with each query type
		# A query type with no embeddings gets a Retrieved with no chunks, so it still counts as a miss
		for query_type, query_vectors in query_types_and_vectors:
			if index_type == QDRANT_INDEX:
				hits = search_qdrant(index, query_vectors, top_k)
			else:
				hits = search_faiss(index, query_vectors, top_k)
			retrieved = make_dense_retrieved(query.id, query_type, chunk_type, index_type, top_k, hits)
			dense_outputs.append(retrieved)

	return dense_outputs

def search_faiss(faiss_index: FaissIndex, query_vectors: list[np.ndarray], top_k: int) -> list[tuple[str, float]]:
	"""
	Input: the built-in FAISS index, the query embeddings to search with and top_k.
	Output: (chunk_id, score) pairs, top_k for each query embedding.
	What it does: searches FAISS once with all the query embeddings. The score is the cosine similarity.
	"""
	hits = []
	if len(query_vectors) == 0:
		return hits

	query_matrix = np.array(query_vectors, dtype=np.float32)
	scores, positions = faiss_index.faiss_index.search(query_matrix, top_k)

	for row in range(len(query_vectors)):
		for column in range(top_k):
			position = int(positions[row][column])
			# FAISS returns -1 when it finds fewer than top_k chunks
			if position == -1:
				continue
			chunk_id = faiss_index.chunk_ids[position]
			hits.append((chunk_id, float(scores[row][column])))

	return hits

def search_qdrant(qdrant_index: QdrantIndex, query_vectors: list[np.ndarray], top_k: int) -> list[tuple[str, float]]:
	"""
	Input: the Qdrant collection, the query embeddings to search with and top_k.
	Output: (chunk_id, score) pairs, top_k for each query embedding.
	What it does: searches Qdrant once per query embedding. The score is the cosine similarity.
	"""
	hits = []

	for query_vector in query_vectors:
		response = qdrant_index.qdrant_client.query_points(
			collection_name=qdrant_index.collection_name,
			query=query_vector.tolist(),
			limit=top_k,
			with_payload=["chunk_id"]
		)
		for point in response.points:
			hits.append((point.payload["chunk_id"], float(point.score)))

	return hits

def make_dense_retrieved(query_id: str, query_type: str, chunk_type: str, index_type: str, top_k: int, hits: list[tuple[str, float]]) -> Retrieved:
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
		retrieval_type=DENSE_RETRIEVAL,
		index_type=index_type,
		top_k=top_k,
		reranking=False,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)
