import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common.args import PipelineArgs
from rag.build.indexing import load_index_from_path
from rag.build.indexing_vectordb import load_vectordb_from_path
from rag.serving.query.query import QueryTransformation
from rag.serving.retrieval.retrieval import Retrieval
from schema.retrieved import Retrieved


class Pipeline:
	def __init__(self, pipeline_args: PipelineArgs):
		"""
		Input: pipeline_args, every setting of one best setup's pipeline, one sub-args per layer.
		Output: nothing.
		What it does: builds everything once, so every question reuses it: the embedding model the indexes were built
		with, the setup's indexes, each loaded straight from its path, then the query layer and the retrieval layer. The
		context, prompt and answer layers are None until they are built.
		"""
		self.pipeline_args = pipeline_args

		# 1. The embedding model the indexes were built with, normalized like the chunk embeddings, so a query vector
		# matches the chunk vectors
		device = "cuda" if torch.cuda.is_available() else "cpu"
		embedding_model = HuggingFaceEmbeddings(
			model_name=pipeline_args.embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)

		# 2. The setup's indexes, each loaded straight from its path. Hybrid's index_type is the dense and BM25 index
		# types joined with "+", in the order of index_paths
		faiss_index = None
		vectordb_index = None
		bm25_index = None
		index_types = pipeline_args.index_type.split("+")
		for index_position in range(len(index_types)):
			index_type = index_types[index_position]
			index_path = pipeline_args.index_paths[index_position]
			if index_type == constants.QDRANT_INDEX:
				vectordb_index = load_vectordb_from_path(index_path, embedding_model, pipeline_args.qdrant_url)
			elif index_type == constants.BM25_INDEX:
				bm25_index = load_index_from_path(index_type, index_path, embedding_model)
			else:
				faiss_index = load_index_from_path(index_type, index_path, embedding_model)

		# 3. The query layer
		self.query_transformation = QueryTransformation(pipeline_args.query_args, pipeline_args.query_path)

		# 4. The retrieval layer
		self.retrieval = Retrieval(
			pipeline_args.retrieval_args, pipeline_args.query_args, faiss_index, vectordb_index, bm25_index,
			pipeline_args.retrieved_path
		)

		# 5. The context, prompt and answer layers, not built yet
		self.context = None
		self.prompt = None
		self.answer = None

	def invoke(self, qus_id: str, qus: str) -> tuple[Retrieved, None]:
		"""
		Input: qus_id, the id of one question; qus, its text.
		Output: (retrieved, answer). The answer is None until the context, prompt and answer layers are built.
		What it does: makes the question's query with the query layer, then retrieves its chunks with the retrieval
		layer.
		"""
		query = self.query_transformation.query(qus_id, qus)
		retrieved = self.retrieval.retrieve(query)
		return retrieved, None
