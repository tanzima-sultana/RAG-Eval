import dataclasses

import torch
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from common import constants
from common.args_temp import RetrievalArgs
from schema.chunk import Chunk
from schema.index import Bm25Index
from schema.query import Query
from schema.retrieved import Retrieved
from schema.setup import Setup


class SingleRetrieval:
	def __init__(self, retrieval_args: RetrievalArgs):
		"""
		Input: retrieval_args, the retrieval settings of one chunk type.
		Output: nothing.
		What it does: stores them, validates them and, only with reranking, loads the cross-encoder.
		"""
		self.retrieval_args = retrieval_args
		self.validate()

		self.cross_encoder = None
		if self.retrieval_args.reranking:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.cross_encoder = CrossEncoder(constants.CROSS_ENCODER_MODEL, device=device)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a retrieval setting is not valid.
		"""
		retrieval_args = self.retrieval_args

		supported_retrieval_types = [
			constants.DENSE_RETRIEVAL,
			constants.BM25_RETRIEVAL,
			constants.HYBRID_RETRIEVAL
		]
		for retrieval_type in retrieval_args.retrieval_types:
			if retrieval_type not in supported_retrieval_types:
				raise ValueError(
					f"retrieval_type ({retrieval_type}) is not supported. Use \"{constants.DENSE_RETRIEVAL}\", "
					f"\"{constants.BM25_RETRIEVAL}\" or \"{constants.HYBRID_RETRIEVAL}\"."
				)

		if retrieval_args.top_k < 1:
			raise ValueError(f"top_k ({retrieval_args.top_k}) must be at least 1.")

		if retrieval_args.rerank_top_k < 1:
			raise ValueError(f"rerank_top_k ({retrieval_args.rerank_top_k}) must be at least 1.")

		if retrieval_args.rerank_top_k > retrieval_args.top_k:
			raise ValueError(
				f"rerank_top_k ({retrieval_args.rerank_top_k}) must not be more than top_k ({retrieval_args.top_k}), "
				"since the reranker only reorders the top_k chunks the retriever returns."
			)

	def dense_search(self, vector_store: FAISS | QdrantVectorStore, index_type: str, query: Query) -> list[Retrieved]:
		"""
		Input: vector_store, a LangChain FAISS or Qdrant vector store; index_type, its index type; query, one Query.
		Output: one Retrieved per query type the query holds, not reranked.
		What it does: searches the vector store with each query type's embeddings and keeps the top_k chunks. The score
		is the cosine similarity. Nothing is saved.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k

		# 1. Collect the embeddings of each query type the query holds. A query type not in this run is None
		query_types_and_vectors = []
		if query.original_qus is not None:
			query_types_and_vectors.append((constants.ORIGINAL_QUERY, [query.original_qus.embeddings]))
		if query.rewritten_qus is not None:
			query_types_and_vectors.append((constants.REWRITTEN_QUERY, [query.rewritten_qus.embeddings]))
		if query.multi_qus is not None:
			multi_vectors = []
			for multi_qus in query.multi_qus:
				multi_vectors.append(multi_qus.embeddings)
			query_types_and_vectors.append((constants.MULTI_QUERY, multi_vectors))
		if query.hyde_ans is not None:
			# A HyDE answer the LLM refused to write is not searched
			hyde_vectors = []
			if query.hyde_ans.ans != constants.I_DONT_KNOW:
				hyde_vectors.append(query.hyde_ans.embeddings)
			query_types_and_vectors.append((constants.HYDE_QUERY, hyde_vectors))

		# A query type with no embeddings gets a Retrieved with no chunks, so it still counts as a miss
		retrieved_items = []
		for query_type, query_vectors in query_types_and_vectors:

			# 2. Keep each chunk once with its best score, so the multi-query phrasings are merged
			best_scores = {}
			for query_vector in query_vectors:
				documents_and_scores = vector_store.similarity_search_with_score_by_vector(
					query_vector.tolist(), k=top_k
				)
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

			retrieved_items.append(Retrieved(
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
			))

		return retrieved_items

	def bm25_search(self, index: Bm25Index, query: Query) -> list[Retrieved]:
		"""
		Input: index, the BM25 index of this chunk type; query, one Query.
		Output: one Retrieved per query type the query holds except hyde, not reranked.
		What it does: scores every chunk with each query type's tokens and keeps the top_k chunks with a score above 0.
		HyDE has no tokens, so it has no BM25 result. Nothing is saved.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k

		# 1. BM25 position i holds the Document of chunk i
		chunk_ids_by_position = []
		for document in index.retriever.docs:
			chunk_ids_by_position.append(document.metadata["chunk_id"])

		# 2. Collect the tokens of each query type the query holds. A query type not in this run is None
		query_types_and_tokens = []
		if query.original_qus is not None:
			query_types_and_tokens.append((constants.ORIGINAL_QUERY, [query.original_qus.tokens]))
		if query.rewritten_qus is not None:
			query_types_and_tokens.append((constants.REWRITTEN_QUERY, [query.rewritten_qus.tokens]))
		if query.multi_qus is not None:
			multi_tokens = []
			for multi_qus in query.multi_qus:
				multi_tokens.append(multi_qus.tokens)
			query_types_and_tokens.append((constants.MULTI_QUERY, multi_tokens))

		# A query type with no tokens gets a Retrieved with no chunks, so it still counts as a miss
		retrieved_items = []
		for query_type, query_tokens_list in query_types_and_tokens:

			# 3. Keep each chunk once with its best score. BM25Retriever.invoke returns no scores, so its
			# rank_bm25 model is scored directly
			best_scores = {}
			for query_tokens in query_tokens_list:
				scores = index.retriever.vectorizer.get_scores(query_tokens)
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

			retrieved_items.append(Retrieved(
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
			))

		return retrieved_items

	def hybrid_fuse(self, dense_retrieved: Retrieved, bm25_retrieved: Retrieved) -> Retrieved:
		"""
		Input: dense_retrieved and bm25_retrieved, the plain dense and BM25 results of one query and query type.
		Output: their hybrid Retrieved, not reranked.
		What it does: fuses the two by reciprocal rank fusion and keeps the top_k chunks. Ranks are used, not scores,
		because dense and BM25 scores are on different scales. Nothing is saved.
		"""
		top_k = self.retrieval_args.top_k

		# 1. A chunk at rank r (from 1) gets 1 / (RRF_K + r) from each list it is in
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

		# 2. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
		sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
		chunk_ids = sorted_chunk_ids[:top_k]
		chunk_scores = []
		doc_ids = []
		for chunk_id in chunk_ids:
			chunk_scores.append(fused_scores[chunk_id])
			doc_ids.append(chunk_id.rsplit("_", 1)[0])

		return Retrieved(
			id=dense_retrieved.id,
			setup=Setup(
				chunk_type=dense_retrieved.setup.chunk_type,
				index_type=f"{dense_retrieved.setup.index_type}+{constants.BM25_INDEX}",
				query_type=dense_retrieved.setup.query_type,
				retrieval_type=constants.HYBRID_RETRIEVAL,
				reranking=False
			),
			top_k=top_k,
			rerank_top_k=0,
			chunk_ids=chunk_ids,
			chunk_scores=chunk_scores,
			doc_ids=doc_ids
		)

	def rerank_result(self, retrieved: Retrieved, original_qus: str, chunks: dict[str, Chunk]) -> Retrieved:
		"""
		Input: retrieved, one plain result; original_qus, its original question; chunks, a dict of chunk_id to Chunk.
		Output: the reranked Retrieved, holding its best rerank_top_k chunks.
		What it does: scores each chunk against the original question with the cross-encoder and keeps the best
		rerank_top_k. Nothing is saved.
		"""
		rerank_top_k = self.retrieval_args.rerank_top_k

		# 1. Score every (question, chunk text) pair in one call
		pairs = []
		for chunk_id in retrieved.chunk_ids:
			pairs.append((original_qus, chunks[chunk_id].text))
		pair_scores = []
		if len(pairs) > 0:
			pair_scores = self.cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)

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
			top_k=self.retrieval_args.top_k,
			rerank_top_k=rerank_top_k,
			chunk_ids=chunk_ids,
			chunk_scores=chunk_scores,
			doc_ids=doc_ids
		)
