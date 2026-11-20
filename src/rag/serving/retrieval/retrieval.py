import os

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from common import constants
from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from rag.serving.retrieval.cache import load_retrieved
from rag.serving.retrieval.cache import save_retrieved
from rag.serving.retrieval.single_retrieval import SingleRetrieval
from schema.chunk import Chunk
from schema.eval_sample import EvalSample
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.query import Query
from schema.retrieved import Retrieved
from schema.vectordb import VectorDb


class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs, query_args: QueryArgs, val_queries_count: int):
		"""
		Input: retrieval_args, the retrieval settings of one chunk type; query_args and val_queries_count, the number
		of val queries searched, used only to name the retrieved folders.
		Output: nothing.
		What it does: stores them and builds the SingleRetrieval, which validates retrieval_args and loads the cross-encoder.
		"""
		self.retrieval_args = retrieval_args
		self.query_args = query_args
		self.val_queries_count = val_queries_count
		self.search = SingleRetrieval(retrieval_args)

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
		What it does: loads the saved results, or searches each query with dense_search and saves them.
		"""
		top_k = self.retrieval_args.top_k

		# 1. Load the saved results
		retrieved_path = paths.get_retrieved_path(output_path, self.retrieval_args, self.query_args, self.val_queries_count, index_type)
		retrieved_file = os.path.join(retrieved_path, f"{constants.DENSE_RETRIEVAL}_top{top_k}.json")
		retrieved_items = load_retrieved(retrieved_file)
		if retrieved_items is not None:
			print(f"Dense {index_type} results loaded from cache : {retrieved_file}")
			return retrieved_items

		# 2. SingleRetrieval each query
		retrieved_items = []
		for query in queries:
			retrieved_items.extend(self.search.dense_search(vector_store, index_type, query))

		# 3. Save the results
		save_retrieved(retrieved_file, retrieved_items)
		print(f"Dense {index_type} results : {len(retrieved_items)}")
		return retrieved_items

	def bm25_retrieval(self, index: Bm25Index, queries: list[Query], output_path: str) -> list[Retrieved]:
		"""
		Input: index, the BM25 index of this chunk type; queries; output_path, the dataset's output folder.
		Output: one Retrieved per query and query type the query holds except hyde, not reranked.
		What it does: loads the saved results, or searches each query with bm25_search and saves them.
		"""
		top_k = self.retrieval_args.top_k

		# 1. Load the saved results
		retrieved_path = paths.get_retrieved_path(output_path, self.retrieval_args, self.query_args, self.val_queries_count, constants.BM25_INDEX)
		retrieved_file = os.path.join(retrieved_path, f"{constants.BM25_RETRIEVAL}_top{top_k}.json")
		retrieved_items = load_retrieved(retrieved_file)
		if retrieved_items is not None:
			print(f"BM25 results loaded from cache : {retrieved_file}")
			return retrieved_items

		# 2. SingleRetrieval each query
		retrieved_items = []
		for query in queries:
			retrieved_items.extend(self.search.bm25_search(index, query))

		# 3. Save the results
		save_retrieved(retrieved_file, retrieved_items)
		print(f"BM25 results : {len(retrieved_items)}")
		return retrieved_items

	def hybrid_retrieval(self, retrieved_items: list[Retrieved], output_path: str) -> list[Retrieved]:
		"""
		Input: retrieved_items, the plain dense and BM25 results of this chunk type; output_path, the dataset's output
		folder.
		Output: one hybrid Retrieved per dense result that has a BM25 result of the same query and query type.
		What it does: for each dense index type, loads the saved results, or fuses each dense result with its BM25
		result with hybrid_fuse and saves them.
		"""
		top_k = self.retrieval_args.top_k

		# 1. Find each BM25 result by its query id and query type, and list the dense index types in order
		bm25_items_by_key = {}
		dense_index_types = []
		for retrieved in retrieved_items:
			if retrieved.setup.retrieval_type == constants.BM25_RETRIEVAL:
				bm25_items_by_key[(retrieved.id, retrieved.setup.query_type)] = retrieved
			elif retrieved.setup.index_type not in dense_index_types:
				dense_index_types.append(retrieved.setup.index_type)

		hybrid_items = []
		for dense_index_type in dense_index_types:
			hybrid_index_type = f"{dense_index_type}+{constants.BM25_INDEX}"

			# 2. Load the saved results
			retrieved_path = paths.get_retrieved_path(output_path, self.retrieval_args, self.query_args, self.val_queries_count, hybrid_index_type)
			retrieved_file = os.path.join(retrieved_path, f"{constants.HYBRID_RETRIEVAL}_top{top_k}.json")
			index_hybrid_items = load_retrieved(retrieved_file)
			if index_hybrid_items is not None:
				print(f"Hybrid {hybrid_index_type} results loaded from cache : {retrieved_file}")
				hybrid_items.extend(index_hybrid_items)
				continue

			# 3. Fuse each dense result of this index with its BM25 result. HyDE has no BM25 result, so no hybrid one
			index_hybrid_items = []
			for dense_retrieved in retrieved_items:
				if dense_retrieved.setup.index_type != dense_index_type:
					continue
				key = (dense_retrieved.id, dense_retrieved.setup.query_type)
				if key not in bm25_items_by_key:
					continue
				index_hybrid_items.append(self.search.hybrid_fuse(dense_retrieved, bm25_items_by_key[key]))

			# 4. Save the results
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
		reranks each result with rerank_result against the original eval question and saves them.
		"""
		top_k = self.retrieval_args.top_k
		rerank_top_k = self.retrieval_args.rerank_top_k
		cross_encoder_name = constants.CROSS_ENCODER_MODEL.replace("/", "-")

		# 1. Find each eval question by its id. A Query has the same id as its EvalSample
		original_qus_by_id = {}
		for eval_sample in eval_samples:
			original_qus_by_id[eval_sample.id] = eval_sample.qus

		# 2. List the retrieval setups in order
		setups = []
		for retrieved in retrieved_items:
			setup = (retrieved.setup.retrieval_type, retrieved.setup.index_type)
			if setup not in setups:
				setups.append(setup)

		reranked_items = []
		for retrieval_type, index_type in setups:
			# 3. Load the saved reranked results
			retrieved_path = paths.get_retrieved_path(output_path, self.retrieval_args, self.query_args, self.val_queries_count, index_type)
			retrieved_file = os.path.join(
				retrieved_path, f"{retrieval_type}_top{top_k}_rerank{rerank_top_k}_{cross_encoder_name}.json"
			)
			setup_reranked_items = load_retrieved(retrieved_file)
			if setup_reranked_items is not None:
				print(f"Reranked {retrieval_type} {index_type} results loaded from cache : {retrieved_file}")
				reranked_items.extend(setup_reranked_items)
				continue

			# 4. Rerank each result of this setup
			setup_reranked_items = []
			for retrieved in retrieved_items:
				if retrieved.setup.retrieval_type != retrieval_type or retrieved.setup.index_type != index_type:
					continue
				setup_reranked_items.append(self.search.rerank_result(retrieved, original_qus_by_id[retrieved.id], chunks))

			# 5. Save the results
			save_retrieved(retrieved_file, setup_reranked_items)
			print(f"Reranked {retrieval_type} {index_type} results : {len(setup_reranked_items)}")
			reranked_items.extend(setup_reranked_items)

		return reranked_items
