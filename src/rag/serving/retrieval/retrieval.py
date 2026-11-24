from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from common import constants
from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from rag.serving.retrieval.cache import load_retrieved_entry
from rag.serving.retrieval.cache import save_retrieved_entry
from rag.serving.retrieval.single_retrieval import invoke_retriever
from rag.serving.retrieval.single_retrieval import make_retrieved
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.query import Query
from schema.retrieved import Retrieved
from schema.setup import Setup
from schema.vectordb import VectorDb


class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs, query_args: QueryArgs, faiss_index: FaissIndex | None, vectordb_index: VectorDb | None, bm25_index: Bm25Index | None, retrieved_path: str):
		"""
		Input: retrieval_args, the retrieval settings of one setup, with its one retrieval type in retrieval_types;
		query_args, the query settings of one setup, with no query type in query_types for original or its one extra
		query type; faiss_index, a FAISS index record for dense and hybrid, or None; vectordb_index, a Qdrant record for
		dense and hybrid, or None; bm25_index, the BM25 index record for bm25 and hybrid, or None; retrieved_path, the
		folder the retrieved files are saved in, used as given.
		Output: nothing.
		What it does: stores them, so one Retrieval stands for one setup, validates them, then builds the setup's
		LangChain retriever once and, only with reranking, its LangChain cross-encoder reranker, so every query reuses
		them. The setup's index type is read from the index records.
		"""
		self.retrieval_args = retrieval_args
		self.query_args = query_args
		self.faiss_index = faiss_index
		self.vectordb_index = vectordb_index
		self.bm25_index = bm25_index
		self.retrieved_path = retrieved_path

		# 1. Validate the settings before anything is built
		self.validate()

		# 2. The one retrieval type and query type of this setup. No extra query type means original
		self.retrieval_type = retrieval_args.retrieval_types[0]
		self.query_type = constants.ORIGINAL_QUERY
		if len(query_args.query_types) == 1:
			self.query_type = query_args.query_types[0]

		# 3. The dense index record given, FAISS or Qdrant, and its index type
		dense_vector_store = None
		dense_index_type = None
		if faiss_index is not None:
			dense_vector_store = faiss_index.vector_store
			dense_index_type = faiss_index.index_type
		if vectordb_index is not None:
			dense_vector_store = vectordb_index.vector_store
			dense_index_type = vectordb_index.vectordb_type

		# 4. The LangChain retriever of this retrieval type, each returning top_k chunks. The BM25 retriever is a copy
		# with its own k, so the shared index is not changed
		top_k = retrieval_args.top_k
		if self.retrieval_type == constants.DENSE_RETRIEVAL:
			self.retriever = dense_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
			self.index_type = dense_index_type
		elif self.retrieval_type == constants.BM25_RETRIEVAL:
			self.retriever = bm25_index.retriever.model_copy(update={"k": top_k})
			self.index_type = constants.BM25_INDEX
		else:
			dense_retriever = dense_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
			bm25_retriever = bm25_index.retriever.model_copy(update={"k": top_k})
			# Reciprocal rank fusion with LangChain's default equal weights and constant, one chunk kept once
			self.retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], id_key="chunk_id")
			self.index_type = f"{dense_index_type}+{constants.BM25_INDEX}"

		# 5. The cross-encoder reranker, only with reranking
		self.reranker = None
		if retrieval_args.reranking:
			cross_encoder = HuggingFaceCrossEncoder(model_name=constants.CROSS_ENCODER_MODEL)
			self.reranker = CrossEncoderReranker(model=cross_encoder, top_n=retrieval_args.rerank_top_k)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a setting of this setup is not valid: exactly one supported retrieval type,
		top_k and rerank_top_k, at most one supported extra query type, HyDE only with dense, since HyDE is searched by
		its meaning only, and the index records the retrieval type needs.
		"""
		retrieval_args = self.retrieval_args
		query_args = self.query_args

		# 1. Exactly one retrieval type, since one Retrieval stands for one setup
		supported_retrieval_types = [
			constants.DENSE_RETRIEVAL,
			constants.BM25_RETRIEVAL,
			constants.HYBRID_RETRIEVAL
		]
		if len(retrieval_args.retrieval_types) != 1:
			raise ValueError(
				f"retrieval_types ({retrieval_args.retrieval_types}) must hold exactly one retrieval type, since one "
				"Retrieval stands for one setup."
			)
		retrieval_type = retrieval_args.retrieval_types[0]
		if retrieval_type not in supported_retrieval_types:
			raise ValueError(
				f"retrieval_type ({retrieval_type}) is not supported. Use \"{constants.DENSE_RETRIEVAL}\", "
				f"\"{constants.BM25_RETRIEVAL}\" or \"{constants.HYBRID_RETRIEVAL}\"."
			)

		# 2. top_k and rerank_top_k
		if retrieval_args.top_k < 1:
			raise ValueError(f"top_k ({retrieval_args.top_k}) must be at least 1.")

		if retrieval_args.rerank_top_k < 1:
			raise ValueError(f"rerank_top_k ({retrieval_args.rerank_top_k}) must be at least 1.")

		if retrieval_args.rerank_top_k > retrieval_args.top_k:
			raise ValueError(
				f"rerank_top_k ({retrieval_args.rerank_top_k}) must not be more than top_k ({retrieval_args.top_k}), "
				"since the reranker only reorders the top_k chunks the retriever returns."
			)

		# 3. At most one extra query type. None means original
		supported_query_types = [
			constants.REWRITTEN_QUERY,
			constants.MULTI_QUERY,
			constants.HYDE_QUERY
		]
		if len(query_args.query_types) > 1:
			raise ValueError(
				f"query_types ({query_args.query_types}) must hold at most one query type, none for original, since "
				"one Retrieval stands for one setup."
			)
		if len(query_args.query_types) == 1 and query_args.query_types[0] not in supported_query_types:
			raise ValueError(
				f"query_type ({query_args.query_types[0]}) is not supported. Use \"{constants.REWRITTEN_QUERY}\", "
				f"\"{constants.MULTI_QUERY}\" or \"{constants.HYDE_QUERY}\", or none for original."
			)

		# HyDE is a passage searched by its meaning, so it is searched with dense only
		if query_args.query_types == [constants.HYDE_QUERY] and retrieval_type != constants.DENSE_RETRIEVAL:
			raise ValueError(
				f"query_type \"{constants.HYDE_QUERY}\" can only be searched with \"{constants.DENSE_RETRIEVAL}\", "
				f"not \"{retrieval_type}\"."
			)

		# 4. The index records this retrieval type searches
		dense_index_count = 0
		if self.faiss_index is not None:
			dense_index_count = dense_index_count + 1
		if self.vectordb_index is not None:
			dense_index_count = dense_index_count + 1

		if retrieval_type == constants.DENSE_RETRIEVAL:
			if dense_index_count != 1 or self.bm25_index is not None:
				raise ValueError("Dense retrieval needs exactly one of faiss_index and vectordb_index, and no bm25_index.")
		elif retrieval_type == constants.BM25_RETRIEVAL:
			if dense_index_count != 0 or self.bm25_index is None:
				raise ValueError("BM25 retrieval needs bm25_index only, and no faiss_index or vectordb_index.")
		else:
			if dense_index_count != 1 or self.bm25_index is None:
				raise ValueError("Hybrid retrieval needs exactly one of faiss_index and vectordb_index, and bm25_index.")

	def retrieve(self, query: Query) -> Retrieved:
		"""
		Input: query, one Query.
		Output: the one Retrieved of this setup for the query.
		What it does: takes the texts of this setup's query type and looks the result up in its own saved file, named by
		the hash of the content it is made from, and gives it the query's id. If it is not saved, invokes the LangChain
		retriever with the texts, reranks the result against the original question if reranking is on, and saves it.
		Prints this query's line.
		"""
		retrieval_args = self.retrieval_args
		query_type = self.query_type

		# 1. Validate the input: the query holds this setup's query type. Original is always made
		if query_type == constants.REWRITTEN_QUERY and query.rewritten_qus is None:
			raise ValueError(f"Query {query.id} has no rewritten query. Add \"{constants.REWRITTEN_QUERY}\" to query_types.")
		if query_type == constants.MULTI_QUERY and query.multi_qus is None:
			raise ValueError(f"Query {query.id} has no multi query. Add \"{constants.MULTI_QUERY}\" to query_types.")
		if query_type == constants.HYDE_QUERY and query.hyde_ans is None:
			raise ValueError(f"Query {query.id} has no HyDE answer. Add \"{constants.HYDE_QUERY}\" to query_types.")

		# 2. The texts of this query type. A HyDE answer the LLM refused to write is not searched
		query_texts = []
		if query_type == constants.ORIGINAL_QUERY:
			query_texts.append(query.original_qus.qus)
		elif query_type == constants.REWRITTEN_QUERY:
			query_texts.append(query.rewritten_qus.qus)
		elif query_type == constants.MULTI_QUERY:
			for multi_qus in query.multi_qus:
				query_texts.append(multi_qus.qus)
		else:
			if query.hyde_ans.ans != constants.I_DONT_KNOW:
				query_texts.append(query.hyde_ans.ans)
		original_qus = query.original_qus.qus

		# 3. Load the saved result of this content
		retrieved_file = paths.get_retrieved_file_path(
			self.retrieved_path, retrieval_args, query_type, self.index_type, query_texts, original_qus
		)
		retrieved = load_retrieved_entry(retrieved_file)
		if retrieved is not None:
			retrieved.id = query.id
			print(
				f"Retrieved {query.id} : {self.index_type} {query_type} {self.retrieval_type} "
				f"reranking={retrieval_args.reranking} : loaded from cache, {len(retrieved.chunk_ids)} chunks"
			)
			return retrieved

		# 4. Search with the LangChain retriever
		documents = invoke_retriever(self.retriever, query_texts, retrieval_args.top_k)

		# 5. Rerank against the original question with the LangChain cross-encoder reranker
		rerank_top_k = 0
		if self.reranker is not None:
			rerank_top_k = retrieval_args.rerank_top_k
			if len(documents) > 0:
				documents = list(self.reranker.compress_documents(documents, original_qus))

		# 6. The result of this setup, saved and reported
		setup = Setup(
			chunk_type=retrieval_args.index_args.chunking_args.chunk_type,
			index_type=self.index_type,
			query_type=query_type,
			retrieval_type=self.retrieval_type,
			reranking=retrieval_args.reranking
		)
		retrieved = make_retrieved(query.id, setup, retrieval_args.top_k, rerank_top_k, documents)
		save_retrieved_entry(retrieved_file, original_qus, retrieved)
		print(
			f"Retrieved {query.id} : {self.index_type} {query_type} {self.retrieval_type} "
			f"reranking={retrieval_args.reranking} : made, {len(retrieved.chunk_ids)} chunks"
		)
		return retrieved
