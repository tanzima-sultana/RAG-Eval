import dataclasses

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths_temp
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from common.args_temp import make_ingestion_args
from common.args_temp import make_ollama_args
from common.args_temp import make_query_args
from common.args_temp import make_retrieval_args
from common.args_temp import make_vectordb_args
from llm_provider.llm_provider import LLMProvider
from rag.build.chunking import Chunking
from rag.build.embedding import Embedding
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from rag.build.ingestion import ingestion
from rag.serving.query_transformation.single_query import transform_question
from rag.serving.retrieval.single_retrieval import SingleRetrieval
from schema.query import Query
from schema.retrieved import Retrieved
from schema.setup import Setup


class RagPipeline:
	def __init__(self, setup: Setup, args):
		"""
		Input: setup, the one setup this pipeline runs; args, the parsed args.
		Output: nothing.
		What it does: loads once what the setup needs: the chunks, its index (dense, BM25, or both for hybrid), the
		embedding model, the query LLM if the query type is not original, and the SingleRetrieval with its reranking.
		"""
		self.setup = setup
		self.validate()
		output_path = paths_temp.get_output_path(args.output_path, args.dataset_name)

		# 1. Chunks and embeddings of the setup's chunk type, which the index and the reranker need
		documents = ingestion(make_ingestion_args(args))
		chunking_args = make_chunking_args(args, setup.chunk_type)
		self.chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
		embeddings = Embedding(make_embedding_args(chunking_args)).compute_embeddings(self.chunks, output_path)

		# 2. The setup's index. Hybrid's index_type is the dense and BM25 index types joined with "+"
		self.vector_store = None
		self.bm25_index = None
		for index_type in setup.index_type.split("+"):
			if index_type == constants.BM25_INDEX:
				self.bm25_index = Indexing(make_index_args(args, chunking_args, index_type)).compute_index(self.chunks, embeddings, output_path)
			elif index_type == constants.QDRANT_INDEX:
				vectordb = IndexingVectorDb(make_vectordb_args(args, chunking_args, index_type)).compute_vectordb(self.chunks, embeddings, output_path)
				self.vector_store = vectordb.vector_store
			else:
				faiss_index = Indexing(make_index_args(args, chunking_args, index_type)).compute_index(self.chunks, embeddings, output_path)
				self.vector_store = faiss_index.vector_store

		# 3. The embedding model, normalized like the chunk embeddings, and the query LLM only if the query type needs one
		self.query_args = make_query_args(args)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = HuggingFaceEmbeddings(
			model_name=self.query_args.embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"normalize_embeddings": True}
		)
		self.query_llm_provider = None
		if setup.query_type != constants.ORIGINAL_QUERY:
			self.query_llm_provider = LLMProvider(self.query_args.query_llm_args, make_ollama_args(args))

		# 4. The SingleRetrieval with the setup's reranking, so the cross-encoder is loaded only if the setup reranks
		retrieval_args = dataclasses.replace(make_retrieval_args(args, chunking_args), reranking=setup.reranking)
		self.single_retrieval = SingleRetrieval(retrieval_args)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if the setup asks for something retrieval cannot do.
		"""
		setup = self.setup
		if setup.retrieval_type == constants.BM25_RETRIEVAL or setup.retrieval_type == constants.HYBRID_RETRIEVAL:
			if setup.query_type == constants.HYDE_QUERY:
				raise ValueError(f"{setup.retrieval_type} retrieval has no hyde, since HyDE has no BM25 tokens.")
		if setup.retrieval_type == constants.HYBRID_RETRIEVAL and len(setup.index_type.split("+")) != 2:
			raise ValueError(f"hybrid index_type ({setup.index_type}) must be a dense and the BM25 index type joined with \"+\".")

	def transform(self, question_id: str, question: str) -> Query:
		"""
		Input: question_id; question, one question.
		Output: its Query, holding only the setup's query type.
		What it does: transforms the question with the setup's query type.
		"""
		return transform_question(
			self.query_args, question_id, question, [self.setup.query_type], self.query_llm_provider, self.embedding_model
		)

	def retrieve(self, query: Query, question: str) -> Retrieved:
		"""
		Input: query, the Query of the question; question, the question, which the cross-encoder scores chunks against.
		Output: the Retrieved of the setup.
		What it does: searches by the setup's retrieval type, then reranks if the setup reranks.
		"""
		setup = self.setup

		# 1. Search by the setup's retrieval type
		if setup.retrieval_type == constants.DENSE_RETRIEVAL:
			retrieved = self.single_retrieval.dense_search(self.vector_store, setup.index_type, query)[0]
		elif setup.retrieval_type == constants.BM25_RETRIEVAL:
			retrieved = self.single_retrieval.bm25_search(self.bm25_index, query)[0]
		else:
			dense_index_type = setup.index_type.split("+")[0]
			dense_retrieved = self.single_retrieval.dense_search(self.vector_store, dense_index_type, query)[0]
			bm25_retrieved = self.single_retrieval.bm25_search(self.bm25_index, query)[0]
			retrieved = self.single_retrieval.hybrid_fuse(dense_retrieved, bm25_retrieved)

		# 2. Rerank against the question
		if setup.reranking:
			retrieved = self.single_retrieval.rerank_result(retrieved, question, self.chunks)

		return retrieved

	def run(self, question_id: str, question: str) -> Retrieved:
		"""
		Input: question_id; question, one question.
		Output: its Retrieved.
		What it does: runs the steps in order: transform, then retrieve.
		"""
		query = self.transform(question_id, question)
		return self.retrieve(query, question)
