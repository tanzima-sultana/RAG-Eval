from common.args import RetrievalArgs
from common.constants import LANGCHAIN_FRAMEWORK
from common.constants import DENSE_RETRIEVAL, BM25_RETRIEVAL, HYBRID_RETRIEVAL
from common.constants import FAISS_FLATIP_INDEX, FAISS_HNSW_INDEX, FAISS_IVF_INDEX, BM25_INDEX, QDRANT_INDEX
from l4_indexing.schema import FaissIndex
from l4_indexing.schema import Bm25Index
from l4_indexing.schema import QdrantIndex
from l4_indexing.schema import LangchainFaissIndex
from l4_indexing.schema import LangchainBm25Index
from l4_indexing.schema import LangchainQdrantIndex
from l5_query.schema import Query
from l6_retrieval.schema import Retrieved
from l6_retrieval import validate
from l6_retrieval.langchain import Langchain
from l6_retrieval.retrievars import dense
from l6_retrieval.retrievars import bm25
from l6_retrieval.retrievars import hybrid

class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs):
		"""
		Input: retrieval_args, the framework, retrieval_types, top_k and rerank_top_k.
		Output: none.
		What it does: validates retrieval_args and stores it. For framework "langchain", also creates the Langchain
		class once, which the dense and BM25 methods hand the work to.
		"""
		validate.validate_retrieval_args(retrieval_args)
		self.retrieval_args = retrieval_args

		self.langchain = None
		if self.retrieval_args.framework == LANGCHAIN_FRAMEWORK:
			self.langchain = Langchain(self.retrieval_args.top_k)

	def compute_dense_retrieval(self, queries: list[Query], chunk_type: str, index_type: str, index: FaissIndex | QdrantIndex | LangchainFaissIndex | LangchainQdrantIndex) -> list[Retrieved]:
		"""
		Input: l5 queries, the chunk type, and one l4 index with its index type.
		Output: one Retrieved per query and query type. Empty if dense is not in retrieval_types or the index is not
		FAISS or Qdrant.
		What it does: searches the FAISS or Qdrant index with the query embeddings.
		"""
		if DENSE_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		# Dense retrieval only searches FAISS and Qdrant
		if index_type not in (FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX, QDRANT_INDEX):
			return []

		if self.langchain is not None:
			return self.langchain.get_dense_outputs(queries, chunk_type, index_type, index)

		dense_outputs = dense.get_dense_outputs(queries, chunk_type, index_type, index, self.retrieval_args.top_k)
		return dense_outputs
	
	def compute_bm25_retrieval(self, queries: list[Query], chunk_type: str, index_type: str, index: Bm25Index | LangchainBm25Index) -> list[Retrieved]:
		"""
		Input: l5 queries, the chunk type, and one l4 index with its index type.
		Output: one Retrieved per query and query type. Empty if bm25 is not in retrieval_types or the index is not BM25.
		What it does: searches the BM25 index with the query tokens.
		"""
		if BM25_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		# BM25 retrieval only searches the BM25 index
		if index_type != BM25_INDEX:
			return []

		if self.langchain is not None:
			return self.langchain.get_bm25_outputs(queries, chunk_type, index_type, index)

		bm25_outputs = bm25.get_bm25_outputs(queries, chunk_type, index_type, index, self.retrieval_args.top_k)
		return bm25_outputs
	
	def compute_hybrid_retrieval(self, chunk_type: str, dense_index_type: str, dense_outputs: list[Retrieved], bm25_outputs: list[Retrieved]) -> list[Retrieved]:
		"""
		Input: the chunk type, the dense index type, that index's dense outputs and the BM25 outputs.
		Output: one Retrieved per query and query type found in both. Empty if hybrid is not in retrieval_types.
		What it does: fuses the dense and BM25 chunks of each query and query type with reciprocal rank fusion.
		"""
		if HYBRID_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		hybrid_outputs = hybrid.get_hybrid_outputs(chunk_type, dense_index_type, dense_outputs, bm25_outputs, self.retrieval_args.top_k)
		return hybrid_outputs
