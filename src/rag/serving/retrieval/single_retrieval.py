import dataclasses

from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from common import constants
from common.args import RetrievalArgs
from schema.chunk import Document
from schema.index import Bm25Index
from schema.query import Query
from schema.retrieved import Retrieved
from schema.setup import Setup


def dense_search(retrieval_args: RetrievalArgs, vector_store: FAISS | QdrantVectorStore, index_type: str, query: Query, query_type: str) -> Retrieved:
	"""
	Input: retrieval_args, the retrieval settings of one chunk type; vector_store, a LangChain FAISS or Qdrant vector
	store; index_type, its index type; query, one Query; query_type, the one query type of the query to search with.
	Output: the Retrieved of that query type, not reranked.
	What it does: searches the vector store with the query type's embeddings and keeps the top_k chunks. Multi has one
	embedding per phrasing, and each chunk is kept once with its best score. The score is the cosine similarity. A
	HyDE answer the LLM refused to write is not searched, so it gets no chunks and still counts as a miss. Nothing is
	saved.
	"""
	top_k = retrieval_args.top_k

	# 1. The embeddings of the query type
	query_vectors = []
	if query_type == constants.ORIGINAL_QUERY:
		query_vectors.append(query.original_qus.embeddings)
	elif query_type == constants.REWRITTEN_QUERY:
		query_vectors.append(query.rewritten_qus.embeddings)
	elif query_type == constants.MULTI_QUERY:
		for multi_qus in query.multi_qus:
			query_vectors.append(multi_qus.embeddings)
	else:
		if query.hyde_ans.ans != constants.I_DONT_KNOW:
			query_vectors.append(query.hyde_ans.embeddings)

	# 2. Keep each chunk once with its best score, so the multi-query phrasings are merged
	best_scores = {}
	for query_vector in query_vectors:
		documents_and_scores = vector_store.similarity_search_with_score_by_vector(query_vector.tolist(), k=top_k)
		for document, score in documents_and_scores:
			chunk_id = document.metadata["chunk_id"]
			if chunk_id not in best_scores or float(score) > best_scores[chunk_id]:
				best_scores[chunk_id] = float(score)

	# 3. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
	sorted_chunk_ids = sorted(best_scores, key=best_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:top_k]
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(best_scores[chunk_id])
		doc_ids.append(chunk_id.rsplit("_", 1)[0])

	return Retrieved(
		id=query.id,
		setup=Setup(
			chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
			index_type=index_type,
			query_type=query_type,
			retrieval_type=constants.DENSE_RETRIEVAL,
			reranking=False
		),
		top_k=top_k,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)


def bm25_search(retrieval_args: RetrievalArgs, bm25_index: Bm25Index, query: Query, query_type: str) -> Retrieved:
	"""
	Input: retrieval_args, the retrieval settings of one chunk type; bm25_index, the BM25 index of this chunk type;
	query, one Query; query_type, the one query type of the query to search with. HyDE has no tokens, so it is never
	given here.
	Output: the Retrieved of that query type, not reranked.
	What it does: scores every chunk with the query type's tokens and keeps the top_k chunks with a score above 0.
	Multi has tokens per phrasing, and each chunk is kept once with its best score. Nothing is saved.
	"""
	top_k = retrieval_args.top_k

	# 1. BM25 position i holds the LangChain Document of chunk i
	chunk_ids_by_position = []
	for document in bm25_index.retriever.docs:
		chunk_ids_by_position.append(document.metadata["chunk_id"])

	# 2. The tokens of the query type
	query_tokens_list = []
	if query_type == constants.ORIGINAL_QUERY:
		query_tokens_list.append(query.original_qus.tokens)
	elif query_type == constants.REWRITTEN_QUERY:
		query_tokens_list.append(query.rewritten_qus.tokens)
	else:
		for multi_qus in query.multi_qus:
			query_tokens_list.append(multi_qus.tokens)

	# 3. Keep each chunk once with its best score. BM25Retriever.invoke returns no scores, so its rank_bm25 model is
	# scored directly
	best_scores = {}
	for query_tokens in query_tokens_list:
		scores = bm25_index.retriever.vectorizer.get_scores(query_tokens)
		top_positions = scores.argsort()[::-1][:top_k]
		for position in top_positions:
			score = float(scores[position])
			# A chunk with score 0 shares no word with the query
			if score <= 0.0:
				continue
			chunk_id = chunk_ids_by_position[int(position)]
			if chunk_id not in best_scores or score > best_scores[chunk_id]:
				best_scores[chunk_id] = score

	# 4. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
	sorted_chunk_ids = sorted(best_scores, key=best_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:top_k]
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(best_scores[chunk_id])
		doc_ids.append(chunk_id.rsplit("_", 1)[0])

	return Retrieved(
		id=query.id,
		setup=Setup(
			chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
			index_type=constants.BM25_INDEX,
			query_type=query_type,
			retrieval_type=constants.BM25_RETRIEVAL,
			reranking=False
		),
		top_k=top_k,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)


def hybrid_search(retrieval_args: RetrievalArgs, vector_store: FAISS | QdrantVectorStore, index_type: str, bm25_index: Bm25Index, query: Query, query_type: str) -> Retrieved:
	"""
	Input: retrieval_args, the retrieval settings of one chunk type; vector_store, a LangChain FAISS or Qdrant vector
	store; index_type, its index type; bm25_index, the BM25 index of this chunk type; query, one Query; query_type, the
	one query type to search with. HyDE has no tokens, so it is never given here.
	Output: the hybrid Retrieved of that query type, with index type "{index_type}+bm25", not reranked.
	What it does: runs dense_search and bm25_search for the query type, then fuses the two by reciprocal rank fusion
	and keeps the top_k chunks. Ranks are used, not scores, because dense and BM25 scores are on different scales.
	Nothing is saved.
	"""
	top_k = retrieval_args.top_k

	# 1. The dense and the BM25 result of the query type
	dense_retrieved = dense_search(retrieval_args, vector_store, index_type, query, query_type)
	bm25_retrieved = bm25_search(retrieval_args, bm25_index, query, query_type)

	# 2. A chunk at rank r (from 1) gets 1 / (RRF_K + r) from each list it is in
	fused_scores = {}
	for position in range(len(dense_retrieved.chunk_ids)):
		chunk_id = dense_retrieved.chunk_ids[position]
		if chunk_id not in fused_scores:
			fused_scores[chunk_id] = 0.0
		fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (constants.RRF_K + position + 1)
	for position in range(len(bm25_retrieved.chunk_ids)):
		chunk_id = bm25_retrieved.chunk_ids[position]
		if chunk_id not in fused_scores:
			fused_scores[chunk_id] = 0.0
		fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (constants.RRF_K + position + 1)

	# 3. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
	sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:top_k]
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(fused_scores[chunk_id])
		doc_ids.append(chunk_id.rsplit("_", 1)[0])

	return Retrieved(
		id=query.id,
		setup=Setup(
			chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
			index_type=f"{index_type}+{constants.BM25_INDEX}",
			query_type=query_type,
			retrieval_type=constants.HYBRID_RETRIEVAL,
			reranking=False
		),
		top_k=top_k,
		rerank_top_k=0,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)


def rerank_result(retrieval_args: RetrievalArgs, cross_encoder: CrossEncoder, retrieved: Retrieved, original_qus: str, chunks: dict[str, Document]) -> Retrieved:
	"""
	Input: retrieval_args, the retrieval settings of one chunk type; cross_encoder, the loaded cross-encoder;
	retrieved, one plain result; original_qus, its original question text; chunks, a dict of chunk_id to Document.
	Output: the reranked Retrieved, holding its best rerank_top_k chunks.
	What it does: scores each chunk against the original question with the cross-encoder and keeps the best
	rerank_top_k. Nothing is saved.
	"""
	rerank_top_k = retrieval_args.rerank_top_k

	# 1. Score every (question, chunk text) pair in one call
	pairs = []
	for chunk_id in retrieved.chunk_ids:
		pairs.append((original_qus, chunks[chunk_id].text))
	pair_scores = []
	if len(pairs) > 0:
		pair_scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)

	# 2. Sort the chunks by their score and keep rerank_top_k. A chunk id is "{doc_id}_{chunk_index}"
	rerank_scores = {}
	for position in range(len(retrieved.chunk_ids)):
		rerank_scores[retrieved.chunk_ids[position]] = float(pair_scores[position])

	sorted_chunk_ids = sorted(rerank_scores, key=rerank_scores.get, reverse=True)
	chunk_ids = sorted_chunk_ids[:rerank_top_k]
	chunk_scores = []
	doc_ids = []
	for chunk_id in chunk_ids:
		chunk_scores.append(rerank_scores[chunk_id])
		doc_ids.append(chunk_id.rsplit("_", 1)[0])

	return Retrieved(
		id=retrieved.id,
		setup=dataclasses.replace(retrieved.setup, reranking=True),
		top_k=retrieval_args.top_k,
		rerank_top_k=rerank_top_k,
		chunk_ids=chunk_ids,
		chunk_scores=chunk_scores,
		doc_ids=doc_ids
	)
