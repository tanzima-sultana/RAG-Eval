import time

import torch

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from common.chunk_config import ChunkConfig
from common.constants import SEMANTIC_CHUNKING
from common import paths


from l1_doc_processing.cache import load_documents

from l2_chunking.validate import validate_chunk_config, validate_processed_dataset
from l2_chunking.cache import _load_cached_chunks, _save_chunks
from l2_chunking.schema import Chunk
from l2_chunking.summary import print_chunk_time_summary, print_run_summary, print_sentence_embedding_summary

class Chunking:
	def __init__(self, chunk_config: ChunkConfig, dataset_size:int, model_name: str):

		self.chunk_config = chunk_config
		self.dataset_size = dataset_size
		self.model_name = model_name

		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(self.model_name, device=device)
		self.tokenizer = self.model.tokenizer
		
		# Only for semantic
		self.all_embeddings = None
		self.embeddings_offset = None

	def compute_chunks(self)-> dict[str, Chunk]:
		"""
		Validates the chunk config and processed dataset, loads the
		processed documents, and chunks each one with the strategy named by
		chunk_config.chunk_type. Chunks already saved for a document are
		loaded instead of recomputed; new chunks are saved.

		Returns a dict of chunk_id to Chunk.
		"""

		# 1. validate chunk config
		validate_chunk_config(self.chunk_config)

		# 2. validate processed dataset
		validate_processed_dataset(self.chunk_config.dataset_name, self.dataset_size)

		# imported here, not at module level, to avoid a circular
		# import: chunker.py imports Chunking (this class) at module
		# level to use as a base class, so this module cannot also
		# import chunker.py at module level.
		from l2_chunking.chunker import get_chunker

		# Resolved before any document loading, so an unregistered
		# chunk_type raises ValueError immediately instead of only
		# after documents have already been loaded.
		chunker_cls = get_chunker(self.chunk_config.chunk_type)

		# Load documents - dict[doc_id, Document]
		documents = load_documents(self.chunk_config.dataset_name, self.dataset_size)

		# All embeddings calculation
		if self.chunk_config.chunk_type == SEMANTIC_CHUNKING:
			all_sentences = []
			self.embeddings_offset = {}

			for doc_id in documents:
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

		chunker = chunker_cls(self)
		chunks = {}

		output_path = paths.get_chunk_path(self.chunk_config)
		cached_document_count = 0
		cached_chunk_count = 0
		new_chunk_count = 0
		# (doc_id, seconds) for every document chunked in this run
		chunk_times = []

		# Compute chunks
		for doc_id in documents:
			# if chunks are already computed from documents
			cached_chunks = _load_cached_chunks(output_path, doc_id)
			if cached_chunks:
				chunks.update(cached_chunks)
				cached_document_count += 1
				cached_chunk_count += len(cached_chunks)
			else:
				document = documents[doc_id]
				chunk_start = time.perf_counter()
				new_chunks = chunker.chunk(document)
				chunk_times.append((doc_id, time.perf_counter() - chunk_start))
				_save_chunks(output_path, new_chunks)
				chunks.update(new_chunks)
				new_chunk_count += len(new_chunks)

		print_run_summary(len(documents), cached_document_count, cached_chunk_count, new_chunk_count)
		if self.chunk_config.chunk_type == SEMANTIC_CHUNKING:
			print_sentence_embedding_summary(len(all_sentences), embedding_seconds)
		print_chunk_time_summary(chunk_times)

		return chunks
