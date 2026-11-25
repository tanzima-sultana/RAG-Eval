from common import constants
from common import paths
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from rag.build.chunking.cache import load_document_chunks
from rag.build.chunking.cache import save_chunks
from rag.build.chunking.chunkers import Chunker
from schema.chunk import Chunk
from schema.file import File


class Chunking:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, ingestion_hash: str):
		"""
		Input: embedder_args, the shared embedder settings, whose model_name chunking uses; chunking_args, the chunking
		settings for one chunk type; ingestion_hash, the hash of the input files, from ingestion.
		Output: nothing.
		What it does: stores the three, and creates the Chunker that holds the chunking code of every chunk type.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.ingestion_hash = ingestion_hash
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

	def compute_chunks(self, files: list[File], output_path: str) -> tuple[dict[str, Chunk], str, str]:
		"""
		Input: files, a list of File, from ingestion; output_path, the dataset's output folder.
		Output: (chunks, chunking_hash, chunk_path): a dict of chunk_id to Chunk for every file, in file order, the hash of
		this chunking run, and the folder the chunks are saved in.
		What it does: loads the saved chunks of every file that has some, chunks only the other files with the chunker of
		the chunk type, and saves them. If one file changes, only that file is chunked.
		"""
		# 1. Validate the settings, then name the folder of these settings and this chunking run
		self.validate()
		chunk_path = paths.get_chunk_path(output_path, self.embedder_args, self.chunking_args)
		chunking_hash = paths.get_chunking_hash(self.embedder_args, self.chunking_args, self.ingestion_hash)
		print(f"Chunk folder : {chunk_path}")
		self.chunk_path = chunk_path

		# 2. Load the saved chunks of each file by its document id, and collect the files still to chunk
		cached_chunks_by_document_id = {}
		files_to_chunk = []
		for file in files:
			document_chunks = load_document_chunks(chunk_path, file.document_id)
			if document_chunks is not None:
				cached_chunks_by_document_id[file.document_id] = document_chunks
			else:
				files_to_chunk.append(file)

		# 3. Chunk the other files with the Chunker method of the chunk type
		new_chunks = {}
		if len(files_to_chunk) > 0:
			chunk_type = self.chunking_args.chunk_type
			if chunk_type == constants.FIXED_CHUNKING:
				new_chunks = self.chunker.fixed_chunking(files_to_chunk)
			elif chunk_type == constants.SENTENCE_CHUNKING:
				new_chunks = self.chunker.sentence_chunking(files_to_chunk)
			elif chunk_type == constants.SEMANTIC_CHUNKING:
				new_chunks = self.chunker.semantic_chunking(files_to_chunk)

		# 4. Put the saved and the new chunks together in file order
		new_chunks_by_document_id = {}
		for chunk_id in new_chunks:
			document_id = chunk_id.split("_", 1)[0]
			if document_id not in new_chunks_by_document_id:
				new_chunks_by_document_id[document_id] = {}
			new_chunks_by_document_id[document_id][chunk_id] = new_chunks[chunk_id]

		chunks = {}
		document_ids = []
		for file in files:
			document_ids.append(file.document_id)
			if file.document_id in cached_chunks_by_document_id:
				chunks.update(cached_chunks_by_document_id[file.document_id])
			elif file.document_id in new_chunks_by_document_id:
				chunks.update(new_chunks_by_document_id[file.document_id])

		# 5. Save the new chunks and the list of this run, so the next run loads them from the cache
		save_chunks(chunk_path, new_chunks, document_ids, chunking_hash)

		cached_count = len(cached_chunks_by_document_id)
		print(f"Files : {len(files)} ({cached_count} from cache, {len(files_to_chunk)} newly chunked)")
		print(f"Chunks : {len(chunks)}")

		return chunks, chunking_hash, chunk_path
