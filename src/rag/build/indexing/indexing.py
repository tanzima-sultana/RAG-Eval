from common import constants
from common.args import Bm25IndexingArgs
from common.args import EmbedderArgs
from common.args import QdrantArgs
from rag.build.embedding.embedding import make_embedding_model
from rag.build.indexing.bm25_indexing import Bm25Indexing
from rag.build.indexing.faiss_indexing import FaissIndexing
from rag.build.indexing.vectordb.qdrant import IndexingVectorDb


class Indexing:
	def __init__(
		self,
		embedder_args: EmbedderArgs,
		bm25_args: Bm25IndexingArgs,
		faiss_types: list[str],
		vectordb_types: list[str],
		qdrant_args: QdrantArgs | None,
		chunking_hash: str,
		embedding_hash: str
	):
		"""
		Input: embedder_args, the shared embedder settings; bm25_args, the BM25 settings; faiss_types, the FAISS types to
		build, which may be empty; vectordb_types, the vector DB types to build, which may be empty; qdrant_args, the
		Qdrant settings, None when vectordb_types is empty; chunking_hash, from chunking; embedding_hash, from embedding.
		Output: nothing.
		What it does: stores the inputs and validates them. BM25 is always set up. The FAISS indexing and the vector DB
		indexing are set up only when faiss_types or vectordb_types is not empty. The embedding model, which the FAISS
		and vector DB stores use to embed queries, is loaded only then, once, and shared.
		"""
		self.embedder_args = embedder_args
		self.bm25_args = bm25_args
		self.faiss_types = faiss_types
		self.vectordb_types = vectordb_types
		self.qdrant_args = qdrant_args
		self.chunking_hash = chunking_hash
		self.embedding_hash = embedding_hash
		self.validate()

		self.bm25_indexing = Bm25Indexing(bm25_args, chunking_hash)

		self.embedding_model = None
		self.faiss_indexing = None
		self.vectordb_indexing = None
		if len(faiss_types) > 0 or len(vectordb_types) > 0:
			# The same embedder as the embedding layer, so a query vector matches the chunk vectors
			self.embedding_model = make_embedding_model(embedder_args)
		if len(faiss_types) > 0:
			self.faiss_indexing = FaissIndexing(embedding_hash, self.embedding_model)
		if len(vectordb_types) > 0:
			self.vectordb_indexing = IndexingVectorDb(qdrant_args, embedding_hash, self.embedding_model)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if the embedding provider, a FAISS type or a vector DB type is not supported, if
		faiss_types and vectordb_types are both empty, or if vectordb_types is not empty and qdrant_args is None. The
		settings of each kind are checked by its own class.
		"""
		if self.embedder_args.embedding_provider != constants.HUGGINGFACE_EMBEDDING:
			raise ValueError(
				f"embedding_provider ({self.embedder_args.embedding_provider}) is not supported. Use "
				f"\"{constants.HUGGINGFACE_EMBEDDING}\"."
			)

		supported_faiss_types = [constants.FAISS_FLATIP_INDEX, constants.FAISS_IVF_INDEX, constants.FAISS_HNSW_INDEX]
		for faiss_type in self.faiss_types:
			if faiss_type not in supported_faiss_types:
				raise ValueError(
					f"faiss_type ({faiss_type}) is not supported. Use \"{constants.FAISS_FLATIP_INDEX}\", "
					f"\"{constants.FAISS_IVF_INDEX}\" or \"{constants.FAISS_HNSW_INDEX}\"."
				)

		for vectordb_type in self.vectordb_types:
			if vectordb_type != constants.QDRANT_INDEX:
				raise ValueError(
					f"vectordb_type ({vectordb_type}) is not supported. Use \"{constants.QDRANT_INDEX}\"."
				)

		if len(self.faiss_types) == 0 and len(self.vectordb_types) == 0:
			raise ValueError(
				"faiss_types and vectordb_types are both empty. Give at least one FAISS type in faiss_indexing "
				"or one vector DB type in vectordb_indexing."
			)

		if len(self.vectordb_types) > 0 and self.qdrant_args is None:
			raise ValueError("qdrant_args is missing. Give qdrant_url and qdrant_batch_size in the vectordb_indexing config.")
