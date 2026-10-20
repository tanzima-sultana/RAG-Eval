import time

import torch

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


from common import constants
from common import paths
from common.args import ChunkingArgs

from stages.build.ingestion.schema import Document
from l2_chunking.schema import Chunk
from l2_chunking.langchain import Langchain
from l2_chunking import validate, cache

class Chunking:
	def __init__(self, chunking_args: ChunkingArgs):
		"""
		Input: chunking_args, a ChunkingArgs holding the chunking parameters
		and the SentenceTransformer model name.
		Output: nothing.
		What it does: stores chunking_args. For the built-in chunkers
		(framework ""), loads the SentenceTransformer model (on GPU if
		available) and keeps its tokenizer for counting tokens. For the
		langchain framework, the model is not loaded here, because LangChain
		loads its own copy; model and tokenizer stay None. The semantic
		embedding fields start empty and are filled by compute_chunks only
		for semantic chunking.
		"""

		self.chunking_args = chunking_args

		if self.chunking_args.framework == constants.NO_FRAMEWORK:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.model = SentenceTransformer(self.chunking_args.model_name, device=device)
			self.tokenizer = self.model.tokenizer
		else:
			self.model = None
			self.tokenizer = None

		# Only for semantic
		self.all_embeddings = None
		self.embeddings_offset = None

	def compute_chunks(self, documents: dict[str, Document], output_path:str)-> dict[str, Chunk]:
		"""
		Input: documents, a dict of doc_id to Document from stages.build.ingestion;
		output_path, the dataset's output folder the chunk folder is built under.
		Output: a dict of chunk_id to Chunk for every document.
		What it does: validates chunking_args, loads already saved chunks
		for each document from the chunk folder, and chunks the remaining
		documents with the strategy named by chunking_args.chunk_type,
		saving their new chunks. For semantic chunking, only the documents
		that still need chunking have their sentences embedded. Prints how
		many chunks were loaded from cache and how many were newly computed.
		"""

		# 1. validate chunking args
		validate.validate_chunking_args(self.chunking_args)

		if self.chunking_args.framework == constants.LANGCHAIN_FRAMEWORK:
			return Langchain(self.chunking_args).compute_chunks(documents, output_path)


		# imported here, not at module level, to avoid a circular
		# import: chunker.py imports Chunking (this class) at module
		# level for its type hint, so this module cannot also
		# import chunker.py at module level.
		from l2_chunking.chunker import get_chunker

		# Resolved before any chunking work, so an unregistered
		# chunk_type raises ValueError immediately.
		chunker_cls = get_chunker(self.chunking_args.chunk_type)

		chunk_path = paths.get_chunk_path(output_path, self.chunking_args)
		chunks = {}
		cached_chunk_count = 0
		new_chunk_count = 0

		# 2. Load cached chunks, and collect the documents that still need chunking
		uncached_doc_ids = []
		for doc_id in documents:
			cached_chunks = cache._load_cached_chunks(chunk_path, doc_id)
			if cached_chunks is not None:
				chunks.update(cached_chunks)
				cached_chunk_count += len(cached_chunks)
			else:
				uncached_doc_ids.append(doc_id)

		# 3. All embeddings calculation, only for the documents that still need chunking
		if self.chunking_args.chunk_type == constants.SEMANTIC_CHUNKING:
			all_sentences = []
			self.embeddings_offset = {}

			for doc_id in uncached_doc_ids:
				doc = documents[doc_id]
				text = doc.text
				sentences = sent_tokenize(text)

				start = len(all_sentences)
				all_sentences.extend(sentences)
				self.embeddings_offset[doc_id] = (start, len(all_sentences))

			embedding_start = time.perf_counter()
			if len(all_sentences) > 0:
				self.all_embeddings = np.asarray(
					self.model.encode(all_sentences, batch_size=256, show_progress_bar=False)
				)
			else:
				self.all_embeddings = np.empty((0, 0))
			embedding_seconds = time.perf_counter() - embedding_start
			print(f"Embedded {len(all_sentences)} sentences in {embedding_seconds:.3f}s")

		# 4. Chunk the documents that still need chunking
		chunker = chunker_cls(self)

		for doc_id in uncached_doc_ids:
			doc = documents[doc_id]
			new_chunks = chunker.chunk(doc)
			cache._save_chunks(chunk_path, doc_id, new_chunks)
			chunks.update(new_chunks)
			new_chunk_count += len(new_chunks)

		cached_document_count = len(documents) - len(uncached_doc_ids)
		print(f"Chunk folder : {chunk_path}")
		print(f"Documents : {len(documents)} ({cached_document_count} from cache, {len(uncached_doc_ids)} newly chunked)")
		print(f"Chunks : {len(chunks)} ({cached_chunk_count} from cache, {new_chunk_count} new)")

		return chunks
