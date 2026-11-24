from common import constants
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from common.args import IndexArgs
from rag.build.embedding.embedding import make_embedding_model
from rag.build.indexing.indexers.bm25_indexer import Bm25Indexer
from rag.build.indexing.indexers.faiss_indexer import FaissIndexer
from schema.chunk import Chunk
from schema.embed import Embedded
from schema.index import FaissIndex
from schema.index import Bm25Index


class Indexing:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, index_args: IndexArgs):
		"""
		Input: embedder_args, the shared embedder settings; chunking_args, the chunking settings of the chunks the index
		holds; index_args, the indexing settings for one index type.
		Output: nothing.
		What it does: stores the three args, validates them, loads the embedding model of the embedder's provider,
		which the FAISS vector store uses to embed queries, and creates the FAISS and BM25 indexers.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.index_args = index_args
		self.validate()

		# The same embedder as the embedding layer, so a query vector matches the chunk vectors
		self.embedding_model = make_embedding_model(self.embedder_args)

		# The indexers that hold the indexing code of each index type
		self.faiss_indexer = FaissIndexer(embedder_args, chunking_args, index_args, self.embedding_model)
		self.bm25_indexer = Bm25Indexer(embedder_args, chunking_args, index_args)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if the embedding provider or the index type is not supported, or if a setting of
		this index type is missing or not valid. Only the settings of index_args.index_type are checked: faiss_flatip
		has none.
		"""
		index_args = self.index_args

		if self.embedder_args.embedding_provider != constants.HUGGINGFACE_EMBEDDING:
			raise ValueError(
				f"embedding_provider ({self.embedder_args.embedding_provider}) is not supported. Use "
				f"\"{constants.HUGGINGFACE_EMBEDDING}\"."
			)

		supported_index_types = [
			constants.FAISS_FLATIP_INDEX,
			constants.FAISS_IVF_INDEX,
			constants.FAISS_HNSW_INDEX,
			constants.BM25_INDEX
		]
		if index_args.index_type not in supported_index_types:
			raise ValueError(
				f"index_type ({index_args.index_type}) is not supported. Use "
				f"\"{constants.FAISS_FLATIP_INDEX}\", \"{constants.FAISS_IVF_INDEX}\", "
				f"\"{constants.FAISS_HNSW_INDEX}\" or \"{constants.BM25_INDEX}\"."
			)

		if index_args.index_type == constants.FAISS_IVF_INDEX:
			if index_args.ivf_nlist_multiplier is None:
				raise ValueError("ivf_nlist_multiplier is missing. Give it in the faiss_ivf block of the indexing config.")
			if index_args.ivf_nlist_multiplier <= 0:
				raise ValueError(f"ivf_nlist_multiplier ({index_args.ivf_nlist_multiplier}) must be greater than 0.")

		if index_args.index_type == constants.FAISS_HNSW_INDEX:
			if index_args.hnsw_m is None:
				raise ValueError("hnsw_m is missing. Give it in the faiss_hnsw block of the indexing config.")
			if index_args.hnsw_m <= 0:
				raise ValueError(f"hnsw_m ({index_args.hnsw_m}) must be greater than 0.")
			if index_args.hnsw_ef_construction is None:
				raise ValueError("hnsw_ef_construction is missing. Give it in the faiss_hnsw block of the indexing config.")
			if index_args.hnsw_ef_construction <= 0:
				raise ValueError(f"hnsw_ef_construction ({index_args.hnsw_ef_construction}) must be greater than 0.")

		if index_args.index_type == constants.BM25_INDEX:
			if index_args.bm25_k1 is None:
				raise ValueError("bm25_k1 is missing. Give it in the bm25 block of the indexing config.")
			if index_args.bm25_b is None:
				raise ValueError("bm25_b is missing. Give it in the bm25 block of the indexing config.")
			if index_args.bm25_tokenizer != "whitespace":
				raise ValueError(f"bm25_tokenizer ({index_args.bm25_tokenizer}) is not supported. Use \"whitespace\".")

	def compute_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex | Bm25Index:
		"""
		Input: chunks, a dict of chunk_id to Chunk; embeddings, a dict of chunk_id to Embedded; output_path, the
		dataset's output folder.
		Output: a FaissIndex or a Bm25Index.
		What it does: calls the indexer method of the index type.
		"""
		index_type = self.index_args.index_type
		if index_type == constants.FAISS_FLATIP_INDEX:
			index = self.faiss_indexer.faiss_flatip_indexing(chunks, embeddings, output_path)
		elif index_type == constants.FAISS_IVF_INDEX:
			index = self.faiss_indexer.faiss_ivf_indexing(chunks, embeddings, output_path)
		elif index_type == constants.FAISS_HNSW_INDEX:
			index = self.faiss_indexer.faiss_hnsw_indexing(chunks, embeddings, output_path)
		else:
			index = self.bm25_indexer.bm25_indexing(chunks, output_path)

		return index
