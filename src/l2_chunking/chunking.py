

import dataclasses
import datetime
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

	def load_documents(self, data_path: str) -> dict[str, Document]:
		"""
		data_path: directory holding processed Document JSON files
			already written by l1_doc_processing (one {doc_id}.json
			file per document), the same layout
			DocProcessing.load_processed_document reads from.

		Reads every JSON file in data_path and reconstructs each one
		back into a Document via json_dict_to_document.

		Returns a dict of doc_id to Document.
		"""
		documents = {}
		for file_name in sorted(os.listdir(data_path)):
			doc_id, _ = os.path.splitext(file_name)
			file_path = os.path.join(data_path, file_name)
			with open(file_path, encoding="utf-8") as f:
				documents[doc_id] = self.json_dict_to_document(json.load(f))
		return documents

	def json_dict_to_chunk(self, data: dict) -> Chunk:
		"""
		data: a dict previously produced by writing a Chunk out as
			JSON. Chunk has no non-JSON-native fields (no timestamp to
			reparse), so this only reconstructs the dataclass
			instance.

		Returns the reconstructed Chunk.
		"""
		return Chunk(**data)

	def load_chunks(self, chunk_path: str) -> dict[str, Chunk]:
		"""
		chunk_path: directory holding previously computed Chunk JSON
			files (one {chunk_id}.json per chunk), as returned by
			paths.get_chunk_path.

		Reads every JSON file in chunk_path and reconstructs each one
		back into a Chunk via json_dict_to_chunk.

		Returns a dict of chunk_id to Chunk.
		"""
		chunks = {}
		for file_name in sorted(os.listdir(chunk_path)):
			chunk_id, _ = os.path.splitext(file_name)
			file_path = os.path.join(chunk_path, file_name)
			with open(file_path, encoding="utf-8") as f:
				chunks[chunk_id] = self.json_dict_to_chunk(json.load(f))
		return chunks

	def chunk_to_json_dict(self, chunk: Chunk) -> dict:
		"""
		chunk: a fully-populated Chunk.

		Converts chunk to a plain dict of JSON-serializable values.
		Chunk has no non-JSON-native fields (no timestamp to convert),
		so this is a direct dataclass-to-dict conversion.

		Returns the JSON-serializable dict.
		"""
		return dataclasses.asdict(chunk)

	def save_chunk(self, chunk_path: str, chunk_id: str, chunk: Chunk) -> None:
		"""
		chunk_path: directory to write the chunk JSON file into, the
			same directory load_chunks reads back from.
		chunk_id: identifies the file to write.
		chunk: the Chunk to persist.

		Writes chunk to chunk_path as {chunk_id}.json.

		Returns nothing.
		"""
		file_path = os.path.join(chunk_path, f"{chunk_id}.json")
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(self.chunk_to_json_dict(chunk), f)


	def compute_chunks(self, chunk_type)-> dict[str, Chunk]:
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

		output_path = paths.get_chunk_path(chunk_type)
		# If already exists, return that
		if os.path.exists(output_path):
			print("Read chunks from disk")
			return self.load_chunks(output_path)

		# Load documents - dict[doc_id, Document]
		data_path = paths.dataset_processed_path()
		documents = self.load_documents(data_path)

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

		# Compute chunks
		for doc_id in documents:
			document = documents[doc_id]
			document_chunks = chunker.chunk(document)
			chunks.update(document_chunks)

		# Save chunks
		os.makedirs(output_path, exist_ok=True)
		for chunk_id in chunks:
			self.save_chunk(output_path, chunk_id, chunks[chunk_id])

		return chunks
