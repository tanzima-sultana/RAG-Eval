from common import constants
from common.args import ChunkingArgs
from common.args import IndexArgs
from common import paths

from stages.build.chunking.schema import Chunk
from stages.build.embedding.schema import Embedded
from stages.build.indexing.schema import FaissIndex
from stages.build.indexing.schema import Bm25Index
from stages.build.indexing.schema import QdrantIndex
from stages.build.indexing.schema import LangchainFaissIndex
from stages.build.indexing.schema import LangchainBm25Index
from stages.build.indexing.schema import LangchainQdrantIndex
from stages.build.indexing.langchain import Langchain
from stages.build.indexing.indexers import faiss, bm25, qdrant
from stages.build.indexing import validate

class Indexing:
	def __init__(self, index_args: IndexArgs, chunking_args: ChunkingArgs):
		"""
		Input: index_args, the indexing settings with one index_type; chunking_args, the chunking
		settings the chunks and embeddings were made with.
		Output: none.
		What it does: validates index_args and stores both settings. For framework "langchain", also creates
		the Langchain class once, which the three compute methods hand the work to.
		"""
		validate.validate_index_args(index_args)
		self.index_args = index_args
		self.chunking_args = chunking_args

		self.langchain = None
		if self.chunking_args.framework == constants.LANGCHAIN_FRAMEWORK:
			self.langchain = Langchain(self.index_args, self.chunking_args)

	def compute_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex | Bm25Index | QdrantIndex | LangchainFaissIndex | LangchainBm25Index | LangchainQdrantIndex:
		"""
		Input: chunks and embeddings from chunking and embedding; output_path, the dataset's output folder.
		Output: the index of index_type, built-in or LangChain.
		What it does: builds the index of index_type, or loads it from cache.
		"""
		if self.index_args.index_type == constants.BM25_INDEX:
			return self.compute_bm25_index(chunks, output_path)

		if self.index_args.index_type == constants.QDRANT_INDEX:
			return self.compute_qdrant_index(chunks, embeddings, output_path)

		return self.compute_faiss_index(chunks, embeddings, output_path)

	def compute_faiss_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex | LangchainFaissIndex:
		"""
		Input: chunks and embeddings from chunking and embedding; output_path, the dataset's output folder.
		Output: the FAISS index of type index_type, built-in or LangChain.
		What it does: builds the FAISS index, or loads it from cache.
		"""
		faiss_index_path = paths.get_faiss_index_path(output_path, self.chunking_args, self.index_args)

		if self.langchain is not None:
			return self.langchain.get_faiss_index(faiss_index_path, chunks, embeddings)

		return faiss.get_faiss_index(faiss_index_path, self.index_args, embeddings)

	def compute_bm25_index(self, chunks: dict[str, Chunk], output_path: str) -> Bm25Index | LangchainBm25Index:
		"""
		Input: chunks from chunking; output_path, the dataset's output folder.
		Output: the BM25 index, built-in or LangChain.
		What it does: builds the BM25 index, or loads it from cache.
		"""
		bm25_index_path = paths.get_bm25_index_path(output_path, self.chunking_args, self.index_args)

		if self.langchain is not None:
			return self.langchain.get_bm25_index(bm25_index_path, chunks)

		return bm25.get_bm25_index(bm25_index_path, self.index_args, chunks)

	def compute_qdrant_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> QdrantIndex | LangchainQdrantIndex:
		"""
		Input: chunks and embeddings from chunking and embedding; output_path, the dataset's output folder.
		Output: the Qdrant collection, built-in or LangChain.
		What it does: builds the Qdrant collection, or uses the existing one if it holds the same chunks.
		"""
		qdrant_collection_name = paths.get_qdrant_collection_name(output_path, self.chunking_args)

		if self.langchain is not None:
			return self.langchain.get_qdrant_index(qdrant_collection_name, chunks, embeddings)

		return qdrant.get_qdrant_index(qdrant_collection_name, self.index_args, chunks, embeddings)
