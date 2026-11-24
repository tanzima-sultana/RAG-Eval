from common import constants
from common import paths
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from rag.build.chunking.cache import load_chunks
from rag.build.chunking.cache import save_chunks
from rag.build.chunking.chunkers import Chunker
from schema.file import File
from schema.chunk import Chunk


class Chunking:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs):
		"""
		Input: embedder_args, the shared embedder settings, whose model_name chunking uses; chunking_args, the chunking
		settings for one chunk type.
		Output: nothing.
		What it does: stores embedder_args and chunking_args, and creates the Chunker that holds the chunking code of
		every chunk type.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.chunker = Chunker(embedder_args, chunking_args)
		# The folder compute_chunks saved to, for the build manifest
		self.chunk_path = None

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a chunking setting is not valid.
		"""
		chunking_args = self.chunking_args

		if self.embedder_args.model_name == "":
			raise ValueError("model_name is empty.")

		if chunking_args.chunk_type not in [constants.FIXED_CHUNKING, constants.SENTENCE_CHUNKING, constants.SEMANTIC_CHUNKING]:
			raise ValueError(
				f"chunk_type ({chunking_args.chunk_type}) is not supported. Use "
				f"\"{constants.FIXED_CHUNKING}\", \"{constants.SENTENCE_CHUNKING}\" or \"{constants.SEMANTIC_CHUNKING}\"."
			)

		if chunking_args.min_chunk_size <= 0:
			raise ValueError(f"min_chunk_size ({chunking_args.min_chunk_size}) must be greater than 0.")

		if chunking_args.max_chunk_size < chunking_args.min_chunk_size:
			raise ValueError(
				f"max_chunk_size ({chunking_args.max_chunk_size}) must not be smaller than "
				f"min_chunk_size ({chunking_args.min_chunk_size})."
			)

		if chunking_args.fixed_chunk_overlap < 0 or chunking_args.fixed_chunk_overlap >= chunking_args.max_chunk_size:
			raise ValueError(
				f"fixed_chunk_overlap ({chunking_args.fixed_chunk_overlap}) must be at least 0 and smaller than "
				f"max_chunk_size ({chunking_args.max_chunk_size})."
			)

	def compute_chunks(self, documents: dict[str, File], output_path: str) -> dict[str, Chunk]:
		"""
		Input: documents, a dict of doc_id to File; output_path, the dataset's output folder.
		Output: a dict of chunk_id to Chunk for every document.
		What it does: loads saved chunks, chunks the remaining documents with LangChain and saves them.
		"""
		# 1. Validate the settings
		self.validate()

		# 2. Load saved chunks, and collect the documents that still need chunking
		chunk_path = paths.get_chunk_path(output_path, self.embedder_args, self.chunking_args)
		print(f"Chunk folder : {chunk_path}")
		self.chunk_path = chunk_path
		chunks = {}
		cached_chunk_count = 0
		uncached_documents = {}
		for doc_id in documents:
			cached_chunks = load_chunks(chunk_path, doc_id)
			if cached_chunks is not None:
				chunks.update(cached_chunks)
				cached_chunk_count += len(cached_chunks)
			else:
				uncached_documents[doc_id] = documents[doc_id]

		# 3. Chunk the documents that still need chunking, with the Chunker method of the chunk type
		chunks_by_document = {}
		if len(uncached_documents) > 0:
			chunk_type = self.chunking_args.chunk_type
			if chunk_type == constants.FIXED_CHUNKING:
				chunks_by_document = self.chunker.fixed_chunking(uncached_documents)
			elif chunk_type == constants.SENTENCE_CHUNKING:
				chunks_by_document = self.chunker.sentence_chunking(uncached_documents)
			elif chunk_type == constants.SEMANTIC_CHUNKING:
				chunks_by_document = self.chunker.semantic_chunking(uncached_documents)

		# 4. Save the new chunks, one file per document
		new_chunk_count = 0
		for doc_id in chunks_by_document:
			new_chunks = chunks_by_document[doc_id]
			save_chunks(chunk_path, doc_id, new_chunks)
			chunks.update(new_chunks)
			new_chunk_count += len(new_chunks)

		cached_document_count = len(documents) - len(uncached_documents)
		print(f"Documents : {len(documents)} ({cached_document_count} from cache, {len(uncached_documents)} newly chunked)")
		print(f"Chunks : {len(chunks)} ({cached_chunk_count} from cache, {new_chunk_count} new)")

		return chunks
