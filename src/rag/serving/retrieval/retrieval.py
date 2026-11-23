import torch
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from common import constants
from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from rag.serving.retrieval.cache import load_retrieved_entry
from rag.serving.retrieval.cache import save_retrieved_entry
from rag.serving.retrieval.single_retrieval import bm25_search
from rag.serving.retrieval.single_retrieval import dense_search
from rag.serving.retrieval.single_retrieval import hybrid_search
from rag.serving.retrieval.single_retrieval import rerank_result
from schema.chunk import Document
from schema.index import Bm25Index
from schema.query import Query
from schema.retrieved import Retrieved


class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs, query_args: QueryArgs, query_type: str, retrieval_type: str, reranking: bool, index_type: str | None, vector_store: FAISS | QdrantVectorStore | None, bm25_index: Bm25Index | None, chunks: dict[str, Document], retrieved_path: str):
		"""
		Input: retrieval_args, the retrieval settings of one chunk type; query_args, the query settings, used to name
		the cache files; query_type, the one query type this setup searches with; retrieval_type, dense, bm25 or
		hybrid; reranking, true to rerank the result with the cross-encoder; index_type, the dense index type for dense
		and hybrid, None for bm25; vector_store, the dense index's LangChain vector store for dense and hybrid, None for
		bm25; bm25_index, the BM25 index for bm25 and hybrid, None for dense; chunks, a dict of chunk_id to Document,
		used only with reranking; retrieved_path, the folder the cache files are saved in, used as given.
		Output: nothing.
		What it does: stores them, so one Retrieval stands for one setup, validates them and, only with reranking,
		loads the cross-encoder once, so every query reuses it.
		"""
		self.retrieval_args = retrieval_args
		self.query_args = query_args
		self.query_type = query_type
		self.retrieval_type = retrieval_type
		self.reranking = reranking
		self.index_type = index_type
		self.vector_store = vector_store
		self.bm25_index = bm25_index
		self.chunks = chunks
		self.retrieved_path = retrieved_path

		# 1. Validate the settings before anything is loaded
		self.validate()

		# 2. The cross-encoder, only with reranking
		self.cross_encoder = None
		if self.reranking:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.cross_encoder = CrossEncoder(constants.CROSS_ENCODER_MODEL, device=device)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a retrieval setting or this setup is not valid: the retrieval types, top_k
		and rerank_top_k; the query type and retrieval type of the setup; HyDE with BM25 or hybrid, since HyDE has no
		tokens; and the index objects and chunks the setup needs.
		"""
		retrieval_args = self.retrieval_args

		# 1. The retrieval settings of the chunk type
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

		# 2. The query type and retrieval type of this setup
		supported_query_types = [
			constants.ORIGINAL_QUERY,
			constants.REWRITTEN_QUERY,
			constants.MULTI_QUERY,
			constants.HYDE_QUERY
		]
		if self.query_type not in supported_query_types:
			raise ValueError(
				f"query_type ({self.query_type}) is not supported. Use \"{constants.ORIGINAL_QUERY}\", "
				f"\"{constants.REWRITTEN_QUERY}\", \"{constants.MULTI_QUERY}\" or \"{constants.HYDE_QUERY}\"."
			)

		if self.retrieval_type not in supported_retrieval_types:
			raise ValueError(
				f"retrieval_type ({self.retrieval_type}) is not supported. Use \"{constants.DENSE_RETRIEVAL}\", "
				f"\"{constants.BM25_RETRIEVAL}\" or \"{constants.HYBRID_RETRIEVAL}\"."
			)

		# HyDE is a passage searched by its embedding only, so it has no BM25 tokens
		if self.query_type == constants.HYDE_QUERY and self.retrieval_type != constants.DENSE_RETRIEVAL:
			raise ValueError(
				f"query_type \"{constants.HYDE_QUERY}\" has no BM25 tokens, so it can only be searched with "
				f"\"{constants.DENSE_RETRIEVAL}\", not \"{self.retrieval_type}\"."
			)

		# 3. The index objects this setup searches
		if self.retrieval_type == constants.DENSE_RETRIEVAL:
			if self.index_type is None or self.vector_store is None:
				raise ValueError("Dense retrieval needs index_type and vector_store.")
		elif self.retrieval_type == constants.BM25_RETRIEVAL:
			if self.bm25_index is None:
				raise ValueError("BM25 retrieval needs bm25_index.")
		else:
			if self.index_type is None or self.vector_store is None or self.bm25_index is None:
				raise ValueError("Hybrid retrieval needs index_type, vector_store and bm25_index.")

		# 4. The chunk texts the cross-encoder reads
		if self.reranking and self.chunks is None:
			raise ValueError("Reranking needs chunks, to read each chunk's text.")

	def retrieve(self, query: Query) -> Retrieved:
		"""
		Input: query, one Query.
		Output: the one Retrieved of this setup for the query.
		What it does: looks the result up in its own saved file, named by the hash of what decides it, keyed by the
		original question text, and gives it the query's id. If it is not saved, searches with this setup's retrieval
		type and query type, reranks it against the original question if reranking is on, and saves it. Prints this
		query's line.
		"""
		query_type = self.query_type

		# 1. Validate the input: the query holds this setup's query type. Original is always made
		if query_type == constants.REWRITTEN_QUERY and query.rewritten_qus is None:
			raise ValueError(f"Query {query.id} has no rewritten query. Add \"{constants.REWRITTEN_QUERY}\" to query_types.")
		if query_type == constants.MULTI_QUERY and query.multi_qus is None:
			raise ValueError(f"Query {query.id} has no multi query. Add \"{constants.MULTI_QUERY}\" to query_types.")
		if query_type == constants.HYDE_QUERY and query.hyde_ans is None:
			raise ValueError(f"Query {query.id} has no HyDE answer. Add \"{constants.HYDE_QUERY}\" to query_types.")

		original_qus = query.original_qus.qus

		# 2. Load the saved result of this setup and question
		retrieved_file = paths.get_retrieved_file_path(
			self.retrieved_path, self.retrieval_args, self.query_args, query_type, original_qus,
			self.index_type, self.retrieval_type, self.reranking
		)
		retrieved = load_retrieved_entry(retrieved_file)
		if retrieved is not None:
			retrieved.id = query.id
			setup = retrieved.setup
			print(
				f"Retrieved {query.id} : {setup.index_type} {setup.query_type} {setup.retrieval_type} "
				f"reranking={setup.reranking} : loaded from cache, {len(retrieved.chunk_ids)} chunks"
			)
			return retrieved

		# 3. Search with this setup's retrieval type
		if self.retrieval_type == constants.DENSE_RETRIEVAL:
			retrieved = dense_search(self.retrieval_args, self.vector_store, self.index_type, query, query_type)
		elif self.retrieval_type == constants.BM25_RETRIEVAL:
			retrieved = bm25_search(self.retrieval_args, self.bm25_index, query, query_type)
		else:
			retrieved = hybrid_search(self.retrieval_args, self.vector_store, self.index_type, self.bm25_index, query, query_type)

		# 4. Rerank against the original question
		if self.reranking:
			retrieved = rerank_result(self.retrieval_args, self.cross_encoder, retrieved, original_qus, self.chunks)

		# 5. Save the result and report it
		save_retrieved_entry(retrieved_file, original_qus, retrieved)
		setup = retrieved.setup
		print(
			f"Retrieved {query.id} : {setup.index_type} {setup.query_type} {setup.retrieval_type} "
			f"reranking={setup.reranking} : made, {len(retrieved.chunk_ids)} chunks"
		)
		return retrieved
