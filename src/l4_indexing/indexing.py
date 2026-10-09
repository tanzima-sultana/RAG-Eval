from common import constants
from common.args import ChunkingArgs
from common.args import IndexArgs
from common import paths

from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded
from l4_indexing.schema import FaissIndex
from l4_indexing.schema import Bm25Index
from l4_indexing.schema import QdrantIndex
from l4_indexing.schema import LangchainFaissIndex
from l4_indexing.schema import LangchainBm25Index
from l4_indexing.schema import LangchainQdrantIndex
from l4_indexing.langchain import Langchain
from l4_indexing.indexers import faiss, bm25, qdrant
from l4_indexing import validate

class Indexing:
	def __init__(self, index_args: IndexArgs, chunking_args: ChunkingArgs):
		"""
		Input: index_args, the indexing settings; chunking_args, the chunking
		settings the chunks and embeddings were made with.
		Output: none.
		What it does: stores both settings for use by compute_index.
		"""
		self.index_args = index_args
		self.chunking_args = chunking_args

	def compute_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> tuple[FaissIndex, Bm25Index, QdrantIndex] | tuple[LangchainFaissIndex, LangchainBm25Index, LangchainQdrantIndex]:
		"""
		Input: chunks, a dict of chunk_id to Chunk from l2_chunking;
		embeddings, a dict of chunk_id to Embedded from l3_embedding;
		output_path, the dataset's output folder the index folders are built under.
		Output: a tuple of (FaissIndex, Bm25Index, QdrantIndex), or for
		framework "langchain" a tuple of (LangchainFaissIndex,
		LangchainBm25Index, LangchainQdrantIndex).
		What it does: validates index_args. For framework "langchain", hands
		the work to the Langchain class. Otherwise builds all three indexes
		from the same chunks: a FAISS index of the type in faiss_index_type,
		a BM25 index over the chunk texts, and a Qdrant collection of the
		embeddings.
		"""

		# 1. validate index args
		validate.validate_index_args(self.index_args)

		if self.chunking_args.framework == constants.LANGCHAIN_FRAMEWORK:
			return Langchain(self.index_args, self.chunking_args).compute_index(chunks, embeddings, output_path)

		# 2. Get the folder of the FAISS and BM25 indexes and the Qdrant collection name
		faiss_index_path = paths.get_faiss_index_path(output_path, self.chunking_args, self.index_args)
		bm25_index_path = paths.get_bm25_index_path(output_path, self.chunking_args, self.index_args)
		qdrant_collection_name = paths.get_qdrant_collection_name(output_path, self.chunking_args)

		faiss_index = faiss.get_faiss_index(faiss_index_path, self.index_args, embeddings)
		bm25_index = bm25.get_bm25_index(bm25_index_path, self.index_args, chunks)
		qdrant_index = qdrant.get_qdrant_index(qdrant_collection_name, self.index_args, chunks, embeddings)

		return faiss_index, bm25_index, qdrant_index
