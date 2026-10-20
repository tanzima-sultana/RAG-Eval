import dataclasses
import os

import torch
from sentence_transformers import CrossEncoder

from common import paths
from common.args import ChunkingArgs
from common.args import IndexArgs
from common.args import RetrievalArgs
from common.constants import CROSS_ENCODER_MODEL
from common.constants import LANGCHAIN_FRAMEWORK
from common.constants import DENSE_RETRIEVAL, BM25_RETRIEVAL, HYBRID_RETRIEVAL
from common.constants import FAISS_FLATIP_INDEX, FAISS_HNSW_INDEX, FAISS_IVF_INDEX, BM25_INDEX, QDRANT_INDEX
from stages.build.chunking.schema import Chunk
from stages.build.indexing.schema import FaissIndex
from stages.build.indexing.schema import Bm25Index
from stages.build.indexing.schema import QdrantIndex
from stages.build.indexing.schema import LangchainFaissIndex
from stages.build.indexing.schema import LangchainBm25Index
from stages.build.indexing.schema import LangchainQdrantIndex
from stages.question.query.schema import Query
from l6_retrieval.schema import Retrieved
from l6_retrieval import validate
from l6_retrieval.langchain import Langchain
from l6_retrieval.retrievars import dense
from l6_retrieval.retrievars import bm25
from l6_retrieval.retrievars import hybrid
from l6_retrieval import rerank
from l6_retrieval import cache

class Retrieval:
	def __init__(self, retrieval_args: RetrievalArgs, chunking_args: ChunkingArgs, index_args: IndexArgs, query_split: str, tuning_queries_size: int, output_path: str):
		"""
		Input: retrieval_args, the retrieval settings; chunking_args, the chunk type's settings; index_args, the
		run's index settings (each method sets its own index_type); query_split, tuning or held_out, the queries this
		run uses; tuning_queries_size; output_path, the dataset's output folder.
		Output: none.
		What it does: validates retrieval_args and stores everything the methods need to build their cache folders.
		For framework "langchain", also creates the Langchain class, which the dense and BM25 methods hand the work
		to. With reranking, also loads the cross-encoder.
		"""
		validate.validate_retrieval_args(retrieval_args)
		self.retrieval_args = retrieval_args
		self.chunking_args = chunking_args
		self.index_args = index_args
		self.query_split = query_split
		self.tuning_queries_size = tuning_queries_size
		self.output_path = output_path

		self.langchain = None
		if self.retrieval_args.framework == LANGCHAIN_FRAMEWORK:
			self.langchain = Langchain(self.retrieval_args.top_k, self.retrieval_args.query_types)

		self.cross_encoder = None
		if self.retrieval_args.reranking:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

	def compute_dense_retrieval(self, queries: list[Query], chunk_type: str, index_type: str, index: FaissIndex | QdrantIndex | LangchainFaissIndex | LangchainQdrantIndex) -> list[Retrieved]:
		"""
		Input: the queries, the chunk type, and one index with its index type.
		Output: one Retrieved per query and query type. Empty if neither dense nor hybrid is in retrieval_types, or the
		index is not FAISS or Qdrant. Hybrid fuses the dense outputs, so they are also computed for hybrid alone.
		What it does: loads the outputs from cache, or searches the FAISS or Qdrant index with the query embeddings
		and saves them.
		"""
		if DENSE_RETRIEVAL not in self.retrieval_args.retrieval_types and HYBRID_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		# Dense retrieval only searches FAISS and Qdrant
		if index_type not in (FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX, QDRANT_INDEX):
			return []

		dense_index_args = dataclasses.replace(self.index_args, index_type=index_type)
		retrieved_path = paths.get_retrieved_path(self.output_path, self.chunking_args, dense_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		dense_file = os.path.join(retrieved_path, f"{DENSE_RETRIEVAL}_top{self.retrieval_args.top_k}.json")
		cached_dense_outputs = cache.load_cached_retrieved(dense_file)
		if cached_dense_outputs is not None:
			print(f"Dense outputs loaded from cache : {dense_file}")
			return cached_dense_outputs

		if self.langchain is not None:
			dense_outputs = self.langchain.get_dense_outputs(queries, chunk_type, index_type, index)
		else:
			dense_outputs = dense.get_dense_outputs(queries, chunk_type, index_type, index, self.retrieval_args.top_k, self.retrieval_args.query_types)

		cache.save_retrieved(dense_file, dense_outputs)
		return dense_outputs
	
	def compute_bm25_retrieval(self, queries: list[Query], chunk_type: str, index_type: str, index: Bm25Index | LangchainBm25Index) -> list[Retrieved]:
		"""
		Input: the queries, the chunk type, and one index with its index type.
		Output: one Retrieved per query and query type. Empty if neither bm25 nor hybrid is in retrieval_types, or the
		index is not BM25. Hybrid fuses the BM25 outputs, so they are also computed for hybrid alone.
		What it does: loads the outputs from cache, or searches the BM25 index with the query tokens and saves them.
		"""
		if BM25_RETRIEVAL not in self.retrieval_args.retrieval_types and HYBRID_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		# BM25 retrieval only searches the BM25 index
		if index_type != BM25_INDEX:
			return []

		bm25_index_args = dataclasses.replace(self.index_args, index_type=BM25_INDEX)
		retrieved_path = paths.get_retrieved_path(self.output_path, self.chunking_args, bm25_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		bm25_file = os.path.join(retrieved_path, f"{BM25_RETRIEVAL}_top{self.retrieval_args.top_k}.json")
		cached_bm25_outputs = cache.load_cached_retrieved(bm25_file)
		if cached_bm25_outputs is not None:
			print(f"BM25 outputs loaded from cache : {bm25_file}")
			return cached_bm25_outputs

		if self.langchain is not None:
			bm25_outputs = self.langchain.get_bm25_outputs(queries, chunk_type, index_type, index)
		else:
			bm25_outputs = bm25.get_bm25_outputs(queries, chunk_type, index_type, index, self.retrieval_args.top_k, self.retrieval_args.query_types)

		cache.save_retrieved(bm25_file, bm25_outputs)
		return bm25_outputs
	
	def compute_hybrid_retrieval(self, chunk_type: str, dense_index_type: str, dense_outputs: list[Retrieved], bm25_outputs: list[Retrieved]) -> list[Retrieved]:
		"""
		Input: the chunk type, the dense index type, that index's dense outputs and the BM25 outputs.
		Output: one Retrieved per query and query type found in both. Empty if hybrid is not in retrieval_types.
		What it does: loads the outputs from cache, or fuses the dense and BM25 chunks of each query and query type
		with reciprocal rank fusion and saves them.
		"""
		if HYBRID_RETRIEVAL not in self.retrieval_args.retrieval_types:
			return []

		dense_index_args = dataclasses.replace(self.index_args, index_type=dense_index_type)
		bm25_index_args = dataclasses.replace(self.index_args, index_type=BM25_INDEX)
		retrieved_path = paths.get_hybrid_retrieved_path(self.output_path, self.chunking_args, dense_index_args, bm25_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		hybrid_file = os.path.join(retrieved_path, f"{HYBRID_RETRIEVAL}_top{self.retrieval_args.top_k}.json")
		cached_hybrid_outputs = cache.load_cached_retrieved(hybrid_file)
		if cached_hybrid_outputs is not None:
			print(f"Hybrid outputs loaded from cache : {hybrid_file}")
			return cached_hybrid_outputs

		hybrid_outputs = hybrid.get_hybrid_outputs(chunk_type, dense_index_type, dense_outputs, bm25_outputs, self.retrieval_args.top_k)

		cache.save_retrieved(hybrid_file, hybrid_outputs)
		return hybrid_outputs

	def get_reranked_dense_retrieval(self, queries: list[Query], chunks: dict[str, Chunk], index_type: str, dense_outputs: list[Retrieved]) -> list[Retrieved]:
		"""
		Input: the queries, the chunks, the dense index type, and that index's dense outputs.
		Output: one reranked Retrieved per dense output, holding the best rerank_top_k chunks.
		What it does: loads the reranked outputs from cache, or reranks the chunks of each dense output with the
		cross-encoder and saves them.
		"""
		dense_index_args = dataclasses.replace(self.index_args, index_type=index_type)
		retrieved_path = paths.get_retrieved_path(self.output_path, self.chunking_args, dense_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		cross_encoder_name = CROSS_ENCODER_MODEL.replace("/", "-")
		reranked_dense_file = os.path.join(
			retrieved_path,
			f"{DENSE_RETRIEVAL}_top{self.retrieval_args.top_k}_rerank{self.retrieval_args.rerank_top_k}_{cross_encoder_name}.json"
		)
		cached_reranked_dense_outputs = cache.load_cached_retrieved(reranked_dense_file)
		if cached_reranked_dense_outputs is not None:
			print(f"Reranked dense outputs loaded from cache : {reranked_dense_file}")
			return cached_reranked_dense_outputs

		reranked_dense_outputs = rerank.get_reranked_outputs(queries, chunks, dense_outputs, self.cross_encoder, self.retrieval_args.rerank_top_k)

		cache.save_retrieved(reranked_dense_file, reranked_dense_outputs)
		return reranked_dense_outputs

	def get_reranked_bm25_retrieval(self, queries: list[Query], chunks: dict[str, Chunk], bm25_outputs: list[Retrieved]) -> list[Retrieved]:
		"""
		Input: the queries, the chunks, and the BM25 outputs.
		Output: one reranked Retrieved per BM25 output, holding the best rerank_top_k chunks.
		What it does: loads the reranked outputs from cache, or reranks the chunks of each BM25 output with the
		cross-encoder and saves them.
		"""
		bm25_index_args = dataclasses.replace(self.index_args, index_type=BM25_INDEX)
		retrieved_path = paths.get_retrieved_path(self.output_path, self.chunking_args, bm25_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		cross_encoder_name = CROSS_ENCODER_MODEL.replace("/", "-")
		reranked_bm25_file = os.path.join(
			retrieved_path,
			f"{BM25_RETRIEVAL}_top{self.retrieval_args.top_k}_rerank{self.retrieval_args.rerank_top_k}_{cross_encoder_name}.json"
		)
		cached_reranked_bm25_outputs = cache.load_cached_retrieved(reranked_bm25_file)
		if cached_reranked_bm25_outputs is not None:
			print(f"Reranked BM25 outputs loaded from cache : {reranked_bm25_file}")
			return cached_reranked_bm25_outputs

		reranked_bm25_outputs = rerank.get_reranked_outputs(queries, chunks, bm25_outputs, self.cross_encoder, self.retrieval_args.rerank_top_k)

		cache.save_retrieved(reranked_bm25_file, reranked_bm25_outputs)
		return reranked_bm25_outputs

	def get_reranked_hybrid_retrieval(self, queries: list[Query], chunks: dict[str, Chunk], dense_index_type: str, hybrid_outputs: list[Retrieved]) -> list[Retrieved]:
		"""
		Input: the queries, the chunks, the dense index type of the hybrid pair, and its hybrid outputs.
		Output: one reranked Retrieved per hybrid output, holding the best rerank_top_k chunks.
		What it does: loads the reranked outputs from cache, or reranks the chunks of each hybrid output with the
		cross-encoder and saves them.
		"""
		dense_index_args = dataclasses.replace(self.index_args, index_type=dense_index_type)
		bm25_index_args = dataclasses.replace(self.index_args, index_type=BM25_INDEX)
		retrieved_path = paths.get_hybrid_retrieved_path(self.output_path, self.chunking_args, dense_index_args, bm25_index_args, self.query_split, self.tuning_queries_size, self.retrieval_args.query_types)
		cross_encoder_name = CROSS_ENCODER_MODEL.replace("/", "-")
		reranked_hybrid_file = os.path.join(
			retrieved_path,
			f"{HYBRID_RETRIEVAL}_top{self.retrieval_args.top_k}_rerank{self.retrieval_args.rerank_top_k}_{cross_encoder_name}.json"
		)
		cached_reranked_hybrid_outputs = cache.load_cached_retrieved(reranked_hybrid_file)
		if cached_reranked_hybrid_outputs is not None:
			print(f"Reranked hybrid outputs loaded from cache : {reranked_hybrid_file}")
			return cached_reranked_hybrid_outputs

		reranked_hybrid_outputs = rerank.get_reranked_outputs(queries, chunks, hybrid_outputs, self.cross_encoder, self.retrieval_args.rerank_top_k)

		cache.save_retrieved(reranked_hybrid_file, reranked_hybrid_outputs)
		return reranked_hybrid_outputs
