

import dataclasses
import datetime
import glob
import json
import os
import torch

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from common.constants import SEMANTIC_CHUNKING
from common import global_values
from common import paths
from l1_doc_processing.schema import Document
from l2_chunking.schema import Chunk

class Chunking:
	def __init__(self, min_chunk_size:int, max_chunk_size: int, fixed_chunk_overlap: int, semantic_threshold: float, model_name: str):

		self.min_chunk_size = min_chunk_size
		self.max_chunk_size = max_chunk_size
		self.fixed_chunk_overlap = fixed_chunk_overlap
		self.semantic_threshold = semantic_threshold
		self.model_name = model_name

		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(self.model_name, device=device)
		self.tokenizer = self.model.tokenizer
		
		# Only for semantic
		self.all_embeddings = None
		self.embeddings_offset = None


	def validate(self):
		if self.fixed_chunk_overlap >= self.max_chunk_size:
			raise ValueError(
				f"fixed_chunk_overlap ({self.fixed_chunk_overlap}) must be smaller than "
				f"max_chunk_size ({self.max_chunk_size}); otherwise the sliding window "
				"never advances."
			)
		
	def json_dict_to_document(self, data: dict) -> Document:
		"""
		data: a dict previously produced by l1_doc_processing's
			DocProcessing.document_to_json_dict, with timestamp stored
			as an ISO-8601 string.

		Mirrors DocProcessing.json_dict_to_document: reverses that
		conversion by parsing the timestamp string back into a
		datetime.datetime, so chunking reconstructs exactly the same
		Document shape doc_processing wrote.

		Returns the reconstructed Document.
		"""
		data = dict(data)
		data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
		return Document(**data)

	def load_documents(self, data_path: str, dataset_size:int) -> dict[str, Document]:
		"""
		data_path: directory holding processed Document JSON files
			already written by l1_doc_processing (one {doc_id}.json
			file per document), the same layout
			DocProcessing.load_processed_document reads from.
		dataset_size: how many of the processed documents to load,
			counting in sorted file name order so the same
			dataset_size always selects the same documents.

		Raises ValueError if dataset_size is larger than the number of
		documents already processed and sitting in data_path --
		load_documents only reads what l1_doc_processing has already
		written, it cannot process new raw documents itself.

		Reads the first dataset_size JSON files in data_path (sorted
		by file name) and reconstructs each one back into a Document
		via json_dict_to_document.

		Returns a dict of doc_id to Document.
		"""
		sorted_file_names = sorted(os.listdir(data_path))

		if dataset_size > len(sorted_file_names):
			raise ValueError(
				f"Requested dataset_size {dataset_size} exceeds the "
				f"{len(sorted_file_names)} document(s) already processed "
				f"under '{data_path}'. Run l1_doc_processing with a "
				f"dataset_size of at least {dataset_size} first."
			)

		selected_file_names = sorted_file_names[:dataset_size]

		documents = {}
		for file_name in selected_file_names:
			doc_id, _ = os.path.splitext(file_name)
			file_path = os.path.join(data_path, file_name)
			with open(file_path, encoding="utf-8") as f:
				documents[doc_id] = self.json_dict_to_document(json.load(f))
		return documents

	def chunk_to_json_dict(self, chunk: Chunk) -> dict:
		"""
		chunk: a fully-populated Chunk.

		Converts chunk to a plain dict of JSON-serializable values.
		Chunk has no non-JSON-native fields (no timestamp to convert),
		so this is a direct dataclass-to-dict conversion.

		Returns the JSON-serializable dict.
		"""
		return dataclasses.asdict(chunk)

	def json_dict_to_chunk(self, data: dict) -> Chunk:
		"""
		data: a dict previously produced by chunk_to_json_dict. Chunk
			has no non-JSON-native fields (no timestamp to reparse),
			so this only reconstructs the dataclass instance.

		Returns the reconstructed Chunk.
		"""
		return Chunk(**data)

	def _load_cached_chunks(self, chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
		"""
		chunk_path: directory previously returned by
			paths.get_chunk_path, holding one {chunk_id}.json file per
			already-computed chunk.
		doc_id: the document to look up cached chunks for. A doc_id's
			chunks are every file named "{doc_id}_*.json" in
			chunk_path, matching the chunk_id format _finalize_chunk
			builds.

		Looks for doc_id's chunk files already written under
		chunk_path. If chunk_path does not exist yet, or no files
		match doc_id, there is nothing cached.

		Returns a dict of chunk_id to Chunk for every cached file
		found, or None if doc_id has no cached chunks.
		"""
		if not os.path.exists(chunk_path):
			return None

		files = glob.glob(os.path.join(chunk_path, f"{doc_id}_*.json"))

		if not files:
			return None

		chunks = {}
		for file_path in sorted(files):
			chunk_id = os.path.splitext(os.path.basename(file_path))[0]
			with open(file_path, encoding="utf-8") as f:
				chunks[chunk_id] = self.json_dict_to_chunk(json.load(f))

		return chunks

	def _save_chunks(self, chunk_path: str, chunks: dict[str, Chunk]) -> None:
		"""
		chunk_path: directory to write the chunk JSON files into, the
			same directory _load_cached_chunks reads back from.
		chunks: dict of chunk_id to the newly computed Chunk to
			persist for one document.

		Writes each chunk to chunk_path as {chunk_id}.json, creating
		chunk_path first if it does not exist yet.

		Returns nothing.
		"""
		os.makedirs(chunk_path, exist_ok=True)
		for chunk_id in chunks:
			file_path = os.path.join(chunk_path, f"{chunk_id}.json")
			with open(file_path, "w", encoding="utf-8") as f:
				json.dump(self.chunk_to_json_dict(chunks[chunk_id]), f)

	def compute_chunks(self, chunk_type: str, dataset_size:int)-> dict[str, Chunk]:
		"""
		chunk_type: name of the chunking strategy to run (e.g.
			"fixed", "sentence", "semantic" from common.constants).

		Looks up chunk_type in the chunking strategy registry first,
		raising ValueError immediately if it is not a registered,
		implemented strategy, before any document loading happens.

		Returns a dict of chunk_id to Chunk.
		"""

		# Save global values
		global_values.set("max_chunk_size", self.max_chunk_size)
		global_values.set("fixed_chunk_overlap", self.fixed_chunk_overlap)
		global_values.set("semantic_threshold", self.semantic_threshold)
		global_values.set("model_name", self.model_name)

		self.validate()

		# Load documents - dict[doc_id, Document]
		data_path = paths.dataset_processed_path()
		documents = self.load_documents(data_path, dataset_size)

		# All embeddings calculation
		if chunk_type == SEMANTIC_CHUNKING:
			all_sentences = []
			self.embeddings_offset = {}

			for doc_id in documents:
				doc = documents[doc_id]
				text = doc.text
				sentences = sent_tokenize(text)

				start = len(all_sentences)
				all_sentences.extend(sentences)
				self.embeddings_offset[doc_id] = (start, len(all_sentences))

			if len(all_sentences) > 0:
				self.all_embeddings = np.asarray(
					self.model.encode(all_sentences, batch_size=256, show_progress_bar=False)
				)
			else:
				self.all_embeddings = np.empty((0, 0))

		# imported here, not at module level, to avoid a circular
		# import: chunker.py imports Chunking (this class) at module
		# level to use as a base class, so this module cannot also
		# import chunker.py at module level.
		from l2_chunking.chunker import get_chunker

		chunker_cls = get_chunker(chunk_type)
		chunker = chunker_cls(self)
		chunks = {}

		output_path = paths.get_chunk_path(chunk_type)
		# Compute chunks
		for doc_id in documents:
			# if chunks are already computed from documents
			cached_chunks = self._load_cached_chunks(output_path, doc_id)
			if cached_chunks:
				chunks.update(cached_chunks)
			else:	
				document = documents[doc_id]
				document_chunks = chunker.chunk(document)
				self._save_chunks(output_path, document_chunks)
				chunks.update(document_chunks)
			
		return chunks
