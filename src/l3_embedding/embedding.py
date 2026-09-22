import dataclasses
import os
import glob
import json
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from common.chunk_config import ChunkConfig
from common import paths

from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded

class Embedding:
	def __init__(self, chunk_config: ChunkConfig, dataset_size:int, model_name: str):
		self.chunk_config = chunk_config
		self.dataset_size = dataset_size
		self.model_name = model_name

		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(self.model_name, device=device)
	
	def load_document_ids(self, data_path: str, dataset_size:int) -> list[str]:
		
		if not os.path.exists(data_path):
			raise FileNotFoundError(
				f"'{data_path}' does not exist. Run l1_doc_processing "
				f"for this dataset before chunking and embedding it."
			)

		sorted_file_names = sorted(os.listdir(data_path))

		if len(sorted_file_names) == 0:
			raise ValueError(
				f"'{data_path}' exists but holds no processed "
				f"documents. Run l1_doc_processing for this dataset "
				f"before chunking and embedding it."
			)

		if dataset_size > len(sorted_file_names):
			raise ValueError(
				f"Requested dataset_size {dataset_size} exceeds the "
				f"{len(sorted_file_names)} document(s) already processed "
				f"under '{data_path}'. Run l1_doc_processing with a "
				f"dataset_size of at least {dataset_size} first."
			)

		selected_file_names = sorted_file_names[:dataset_size]

		doc_ids = []
		for file_name in selected_file_names:
			doc_id, _ = os.path.splitext(file_name)
			doc_ids.append(doc_id)
		return doc_ids
	
	def json_dict_to_chunk(self, data: dict) -> Chunk:
		return Chunk(**data)

	def _load_cached_chunks(self, chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
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


	def json_dict_to_embedding(self, data: dict) -> Embedded:
		data = dict(data)
		data["embeddings"] = np.array(data["embeddings"])
		return Embedded(**data)


	def _load_cached_embeddings(self, embedding_path, chunk_id: str) -> Embedded | None:
		
		file_path = paths.get_file_path(embedding_path, chunk_id, "json")
		if not os.path.exists(file_path):
			return None
		with open(file_path, encoding="utf-8") as f:
			return self.json_dict_to_embedding(json.load(f))
	
	def embedding_to_json_dict(self, embedding: Embedded) -> dict:
		data = dataclasses.asdict(embedding)
		data["embeddings"] = embedding.embeddings.tolist()
		return data
	
	def _save_embeddings(self, embedding_path: str, chunk_id: str, embeddings: Embedded) -> None:

		os.makedirs(embedding_path, exist_ok=True)
		file_path = os.path.join(embedding_path, f"{chunk_id}.json")
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(self.embedding_to_json_dict(embeddings), f)
			

	def compute_embeddings(self) -> dict[str, Embedded]:
		embeddings = {}

		# imported here, not at module level, to avoid importing the
		# whole l2_chunking package (and its heavier dependencies)
		# unless compute_embeddings is actually called.
		from l2_chunking.chunker import get_chunker

		# Raises ValueError for an unregistered chunk_type up front,
		# so a typo'd --chunk_type fails loudly instead of silently
		# matching no cached chunks and returning an empty result.
		get_chunker(self.chunk_config.chunk_type)

		# Load document_ids
		data_path = paths.get_processed_dataset_path(self.chunk_config.dataset_name)
		document_ids = self.load_document_ids(data_path, self.dataset_size)

		# Load chunks and cached embeddings
		chunk_path = paths.get_chunk_path(self.chunk_config)
		output_path = paths.get_embeddings_path(self.chunk_config)

		# First pass: resolve every already-cached embedding straight
		# from disk, and collect the text of every chunk that still
		# needs embedding into pending_chunk_ids/pending_chunk_texts
		# (kept in lockstep -- the same index in both lists always
		# refers to the same chunk).
		pending_chunk_ids = []
		pending_chunk_texts = []
		for doc_id in document_ids:
			chunks = self._load_cached_chunks(chunk_path, doc_id)
			if not chunks:
				continue
			for chunk_id in chunks:
				# if embeddings are already computed from chunks
				cached_embeddings = self._load_cached_embeddings(output_path, chunk_id)
				if cached_embeddings:
					embeddings[chunk_id] = cached_embeddings
				else:
					text = chunks[chunk_id].text
					pending_chunk_ids.append(chunk_id)
					pending_chunk_texts.append(text)

		# Second pass: embed every not-yet-cached chunk in one batched
		# model.encode call instead of one call per chunk
		if len(pending_chunk_texts) > 0:
			pending_embeddings = self.model.encode(
				pending_chunk_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
			)

			for pending_index in range(len(pending_chunk_ids)):
				chunk_id = pending_chunk_ids[pending_index]
				new_embeddings = Embedded(chunk_id, pending_embeddings[pending_index], True)
				embeddings[chunk_id] = new_embeddings

				self._save_embeddings(output_path, chunk_id, new_embeddings)

		return embeddings