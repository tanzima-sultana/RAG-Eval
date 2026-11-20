import dataclasses
import json
import os

import torch
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from common import constants
from common import paths_temp
from common.args_temp import QueryArgs
from common.args_temp import RetrievalArgs
from schema.chunk import Chunk
from schema.eval_sample import EvalSample
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.query import Query
from schema.retrieved import Retrieved
from schema.vectordb import VectorDb


def save_retrieved(retrieved_file: str, retrieved_items: list[Retrieved]) -> None:
	"""
	Input: retrieved_file, the cache file of one retrieval setup; retrieved_items, its Retrieved list.
	Output: nothing.
	What it does: writes the list as JSON to a temporary file, then renames it to retrieved_file.
	"""
	os.makedirs(os.path.dirname(retrieved_file), exist_ok=True)

	retrieved_dicts = []
	for retrieved in retrieved_items:
		retrieved_dicts.append(dataclasses.asdict(retrieved))

	with open(retrieved_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(retrieved_dicts, f)
	os.replace(retrieved_file + ".tmp", retrieved_file)


def load_retrieved(retrieved_file: str) -> list[Retrieved] | None:
	"""
	Input: retrieved_file, the cache file of one retrieval setup.
	Output: the saved Retrieved list, or None if the file does not exist.
	What it does: reads the JSON file and builds one Retrieved per entry.
	"""
	if not os.path.exists(retrieved_file):
		return None

	with open(retrieved_file, encoding="utf-8") as f:
		retrieved_dicts = json.load(f)

	retrieved_items = []
	for retrieved_dict in retrieved_dicts:
		retrieved_items.append(Retrieved(**retrieved_dict))
	return retrieved_items


class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs, query_args: QueryArgs, val_queries_count: int):
		"""
		Input: retrieval_args, the retrieval settings of one chunk type; query_args and val_queries_count, the number
		of val queries searched, used only to name the retrieved folders.
		Output: nothing.
		What it does: stores them, validates retrieval_args and, only with reranking, loads the cross-encoder.
		"""
		self.retrieval_args = retrieval_args
		self.query_args = query_args
		self.val_queries_count = val_queries_count
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

	def compute_retrieval(self, index: FaissIndex | Bm25Index, index_type: str, queries: list[Query], output_path: str) -> list[Retrieved]:
		"""
		Input: index, one FAISS or BM25 index of this chunk type; index_type, its index type; queries; output_path,
		the dataset's output folder.
		Output: one Retrieved per query and query type, or [] if no requested retrieval type uses this index.
		What it does: calls BM25 retrieval for the BM25 index and dense retrieval for a FAISS index. Hybrid fuses
		both, so each also runs when only hybrid is requested.
		"""
		retrieval_types = self.retrieval_args.retrieval_types
		hybrid_requested = constants.HYBRID_RETRIEVAL in retrieval_types

		if index_type == constants.BM25_INDEX:
			if constants.BM25_RETRIEVAL in retrieval_types or hybrid_requested:
				return self.bm25_retrieval(index, queries, output_path)
			return []

		if constants.DENSE_RETRIEVAL in retrieval_types or hybrid_requested:
			return self.dense_retrieval(index.vector_store, index_type, queries, output_path)
		return []

	def compute_vectordb_retrieval(self, vectordb: VectorDb, queries: list[Query], output_path: str) -> list[Retrieved]:
		"""
		Input: vectordb, one vector DB collection of this chunk type; queries; output_path, the dataset's output folder.
		Output: one Retrieved per query and query type, or [] if neither dense nor hybrid is requested.
		What it does: calls dense retrieval on the collection's vector store.
		"""
		retrieval_types = self.retrieval_args.retrieval_types
		if constants.DENSE_RETRIEVAL in retrieval_types or constants.HYBRID_RETRIEVAL in retrieval_types:
			return self.dense_retrieval(vectordb.vector_store, vectordb.vectordb_type, queries, output_path)
		return []

	def dense_retrieval(self, vector_store: FAISS | QdrantVectorStore, index_type: str, queries: list[Query], output_path: str) -> list[Retrieved]:
		"""
		Input: vector_store, a LangChain FAISS or Qdrant vector store; index_type, its index type; queries;
		output_path, the dataset's output folder.
		Output: one Retrieved per query and query type the query holds, not reranked.
		What it does: loads the saved results, or searches the vector store with each query type's saved embeddings,
		keeps the top_k chunks and saves them. The score is the cosine similarity.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k

		# 1. Load the saved results
		retrieved_path = paths_temp.get_retrieved_path(output_path, retrieval_args, self.query_args, self.val_queries_count, index_type)
		retrieved_file = os.path.join(retrieved_path, f"{constants.DENSE_RETRIEVAL}_top{top_k}.json")
		retrieved_items = load_retrieved(retrieved_file)
		if retrieved_items is not None:
			print(f"Dense {index_type} results loaded from cache : {retrieved_file}")
			return retrieved_items

		# 2. Collect the embeddings of each query type the query holds. A query type not in this run is None
		retrieved_items = []
		for query in queries:
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
			for query_type, query_vectors in query_types_and_vectors:

				# 3. Keep each chunk once with its best score, so the multi-query phrasings are merged
				best_scores = {}
				for query_vector in query_vectors:
					documents_and_scores = vector_store.similarity_search_with_score_by_vector(
						query_vector.tolist(), k=top_k
					)
					for document, score in documents_and_scores:
						chunk_id = document.metadata["chunk_id"]
						if chunk_id not in best_scores or float(score) > best_scores[chunk_id]:
							best_scores[chunk_id] = float(score)

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
					query_type=query_type,
					chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
					retrieval_type=constants.DENSE_RETRIEVAL,
					index_type=index_type,
					top_k=top_k,
					reranking=False,
					rerank_top_k=0,
					chunk_ids=chunk_ids,
					chunk_scores=chunk_scores,
					doc_ids=doc_ids
				))

		# 5. Save the results
		save_retrieved(retrieved_file, retrieved_items)
		print(f"Dense {index_type} results : {len(retrieved_items)}")
		return retrieved_items

	def bm25_retrieval(self, index: Bm25Index, queries: list[Query], output_path: str) -> list[Retrieved]:
		"""
		Input: index, the BM25 index of this chunk type; queries; output_path, the dataset's output folder.
		Output: one Retrieved per query and query type the query holds except hyde, not reranked.
		What it does: loads the saved results, or scores every chunk with each query type's saved tokens, keeps the
		top_k chunks with a score above 0 and saves them. HyDE has no tokens, so it has no BM25 result.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k

		# 1. Load the saved results
		retrieved_path = paths_temp.get_retrieved_path(output_path, retrieval_args, self.query_args, self.val_queries_count, constants.BM25_INDEX)
		retrieved_file = os.path.join(retrieved_path, f"{constants.BM25_RETRIEVAL}_top{top_k}.json")
		retrieved_items = load_retrieved(retrieved_file)
		if retrieved_items is not None:
			print(f"BM25 results loaded from cache : {retrieved_file}")
			return retrieved_items

		# 2. BM25 position i holds the Document of chunk i
		chunk_ids_by_position = []
		for document in index.retriever.docs:
			chunk_ids_by_position.append(document.metadata["chunk_id"])

		# 3. Collect the tokens of each query type the query holds. A query type not in this run is None
		retrieved_items = []
		for query in queries:
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
			for query_type, query_tokens_list in query_types_and_tokens:

				# 4. Keep each chunk once with its best score. BM25Retriever.invoke returns no scores, so its
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

				# 5. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
				sorted_chunk_ids = sorted(best_scores, key=best_scores.get, reverse=True)
				chunk_ids = sorted_chunk_ids[:top_k]
				chunk_scores = []
				doc_ids = []
				for chunk_id in chunk_ids:
					chunk_scores.append(best_scores[chunk_id])
					doc_ids.append(chunk_id.rsplit("_", 1)[0])

				retrieved_items.append(Retrieved(
					id=query.id,
					query_type=query_type,
					chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
					retrieval_type=constants.BM25_RETRIEVAL,
					index_type=constants.BM25_INDEX,
					top_k=top_k,
					reranking=False,
					rerank_top_k=0,
					chunk_ids=chunk_ids,
					chunk_scores=chunk_scores,
					doc_ids=doc_ids
				))

		# 6. Save the results
		save_retrieved(retrieved_file, retrieved_items)
		print(f"BM25 results : {len(retrieved_items)}")
		return retrieved_items

	def hybrid_retrieval(self, retrieved_items: list[Retrieved], output_path: str) -> list[Retrieved]:
		"""
		Input: retrieved_items, the plain dense and BM25 results of this chunk type; output_path, the dataset's output
		folder.
		Output: one hybrid Retrieved per dense result that has a BM25 result of the same query and query type.
		What it does: for each dense index type, loads the saved results, or fuses its results with the BM25 results by
		reciprocal rank fusion, keeps the top_k chunks and saves them. Ranks are used, not scores, because dense and
		BM25 scores are on different scales.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k

		# 1. Find each BM25 result by its query id and query type, and list the dense index types in order
		bm25_items_by_key = {}
		dense_index_types = []
		for retrieved in retrieved_items:
			if retrieved.retrieval_type == constants.BM25_RETRIEVAL:
				bm25_items_by_key[(retrieved.id, retrieved.query_type)] = retrieved
			elif retrieved.index_type not in dense_index_types:
				dense_index_types.append(retrieved.index_type)

		hybrid_items = []
		for dense_index_type in dense_index_types:
			hybrid_index_type = f"{dense_index_type}+{constants.BM25_INDEX}"

			# 2. Load the saved results
			retrieved_path = paths_temp.get_retrieved_path(output_path, retrieval_args, self.query_args, self.val_queries_count, hybrid_index_type)
			retrieved_file = os.path.join(retrieved_path, f"{constants.HYBRID_RETRIEVAL}_top{top_k}.json")
			index_hybrid_items = load_retrieved(retrieved_file)
			if index_hybrid_items is not None:
				print(f"Hybrid {hybrid_index_type} results loaded from cache : {retrieved_file}")
				hybrid_items.extend(index_hybrid_items)
				continue

			# 3. Fuse each dense result of this index with its BM25 result. HyDE has no BM25 result, so no hybrid one
			index_hybrid_items = []
			for dense_retrieved in retrieved_items:
				if dense_retrieved.index_type != dense_index_type:
					continue
				key = (dense_retrieved.id, dense_retrieved.query_type)
				if key not in bm25_items_by_key:
					continue
				bm25_retrieved = bm25_items_by_key[key]

				# A chunk at rank r (from 1) gets 1 / (RRF_K + r) from each list it is in
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

				# 4. Sort best first and keep top_k. A chunk id is "{doc_id}_{chunk_index}"
				sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
				chunk_ids = sorted_chunk_ids[:top_k]
				chunk_scores = []
				doc_ids = []
				for chunk_id in chunk_ids:
					chunk_scores.append(fused_scores[chunk_id])
					doc_ids.append(chunk_id.rsplit("_", 1)[0])

				index_hybrid_items.append(Retrieved(
					id=dense_retrieved.id,
					query_type=dense_retrieved.query_type,
					chunk_type=dense_retrieved.chunk_type,
					retrieval_type=constants.HYBRID_RETRIEVAL,
					index_type=hybrid_index_type,
					top_k=top_k,
					reranking=False,
					rerank_top_k=0,
					chunk_ids=chunk_ids,
					chunk_scores=chunk_scores,
					doc_ids=doc_ids
				))

			# 5. Save the results
			save_retrieved(retrieved_file, index_hybrid_items)
			print(f"Hybrid {hybrid_index_type} results : {len(index_hybrid_items)}")
			hybrid_items.extend(index_hybrid_items)

		return hybrid_items

	def rerank_retrieval(self, retrieved_items: list[Retrieved], chunks: dict[str, Chunk], eval_samples: list[EvalSample], output_path: str) -> list[Retrieved]:
		"""
		Input: retrieved_items, the plain results of this chunk type; chunks, a dict of chunk_id to Chunk; eval_samples,
		the eval questions; output_path, the dataset's output folder.
		Output: one reranked Retrieved per plain result, holding its best rerank_top_k chunks.
		What it does: for each retrieval setup (retrieval type and index type), loads the saved reranked results, or
		scores each chunk against the original eval question with the cross-encoder, keeps the best rerank_top_k and
		saves them. Every query type is reranked against the original question, since that is what the answer must match.
		"""
		retrieval_args = self.retrieval_args
		top_k = retrieval_args.top_k
		rerank_top_k = retrieval_args.rerank_top_k
		cross_encoder_name = constants.CROSS_ENCODER_MODEL.replace("/", "-")

		# 1. Find each eval question by its id. A Query has the same id as its EvalSample
		original_qus_by_id = {}
		for eval_sample in eval_samples:
			original_qus_by_id[eval_sample.id] = eval_sample.qus

		# 2. List the retrieval setups in order
		setups = []
		for retrieved in retrieved_items:
			setup = (retrieved.retrieval_type, retrieved.index_type)
			if setup not in setups:
				setups.append(setup)

		reranked_items = []
		for retrieval_type, index_type in setups:
			# 3. Load the saved reranked results
			retrieved_path = paths_temp.get_retrieved_path(output_path, retrieval_args, self.query_args, self.val_queries_count, index_type)
			retrieved_file = os.path.join(
				retrieved_path, f"{retrieval_type}_top{top_k}_rerank{rerank_top_k}_{cross_encoder_name}.json"
			)
			setup_reranked_items = load_retrieved(retrieved_file)
			if setup_reranked_items is not None:
				print(f"Reranked {retrieval_type} {index_type} results loaded from cache : {retrieved_file}")
				reranked_items.extend(setup_reranked_items)
				continue

			# 4. Collect this setup's results and every (question, chunk text) pair, one result after another
			setup_items = []
			pairs = []
			for retrieved in retrieved_items:
				if retrieved.retrieval_type != retrieval_type or retrieved.index_type != index_type:
					continue
				setup_items.append(retrieved)
				for chunk_id in retrieved.chunk_ids:
					pairs.append((original_qus_by_id[retrieved.id], chunks[chunk_id].text))

			# 5. Score all pairs in one call, so the cross-encoder runs in large batches
			pair_scores = []
			if len(pairs) > 0:
				pair_scores = self.cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)

			# 6. Sort each result's chunks by their score and keep rerank_top_k
			setup_reranked_items = []
			# Position of the result's first pair in pair_scores
			pair_position = 0
			for retrieved in setup_items:
				rerank_scores = {}
				for chunk_id in retrieved.chunk_ids:
					rerank_scores[chunk_id] = float(pair_scores[pair_position])
					pair_position = pair_position + 1

				sorted_chunk_ids = sorted(rerank_scores, key=rerank_scores.get, reverse=True)
				chunk_ids = sorted_chunk_ids[:rerank_top_k]
				chunk_scores = []
				doc_ids = []
				for chunk_id in chunk_ids:
					chunk_scores.append(rerank_scores[chunk_id])
					doc_ids.append(chunk_id.rsplit("_", 1)[0])

				setup_reranked_items.append(Retrieved(
					id=retrieved.id,
					query_type=retrieved.query_type,
					chunk_type=retrieved.chunk_type,
					retrieval_type=retrieved.retrieval_type,
					index_type=retrieved.index_type,
					top_k=top_k,
					reranking=True,
					rerank_top_k=rerank_top_k,
					chunk_ids=chunk_ids,
					chunk_scores=chunk_scores,
					doc_ids=doc_ids
				))

			# 7. Save the results
			save_retrieved(retrieved_file, setup_reranked_items)
			print(f"Reranked {retrieval_type} {index_type} results : {len(setup_reranked_items)}")
			reranked_items.extend(setup_reranked_items)

		return reranked_items
