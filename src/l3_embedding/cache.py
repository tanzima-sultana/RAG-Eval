"""
cache.py

Saves and loads Embedded records as JSON files, one {doc_id}.json per
document holding the embeddings of all of that document's chunks, in the
embedding folder of a chunking run (see common.paths.get_embedding_path).

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written {doc_id}.json.
"""

import dataclasses
import json
import os

import numpy as np

from l3_embedding.schema import Embedded

def embedding_to_json_dict(embedding: Embedded) -> dict:
	"""
	Input: embedding, an Embedded record.
	Output: a plain dict of JSON-serializable values.
	What it does: converts the dataclass to a dict and turns the numpy
	embeddings array into a list of floats.
	"""
	data = dataclasses.asdict(embedding)
	data["embeddings"] = embedding.embeddings.tolist()
	return data

def json_dict_to_embedding(data: dict) -> Embedded:
	"""
	Input: data, a dict with chunk_id, embeddings (a list of floats) and
	normalized, as saved in a document's cache file.
	Output: the Embedded record.
	What it does: turns the embeddings list back into a numpy array and
	builds the Embedded from the dict's fields.
	"""
	embedding_vector = np.array(data["embeddings"])
	return Embedded(data["chunk_id"], embedding_vector, data["normalized"])

def _load_cached_embeddings(embedding_path: str, doc_id: str) -> dict[str, Embedded] | None:
	"""
	Input: embedding_path, the embedding folder; doc_id, the document to look up.
	Output: a dict of chunk_id to Embedded for that document's chunks, in
	the order they were saved, or None if the document has no cache file.
	What it does: reads {doc_id}.json from embedding_path if it exists.
	"""
	cache_file = os.path.join(embedding_path, f"{doc_id}.json")
	if not os.path.exists(cache_file):
		return None

	with open(cache_file, encoding="utf-8") as f:
		embedding_dicts = json.load(f)

	embeddings = {}
	for embedding_dict in embedding_dicts:
		embedding = json_dict_to_embedding(embedding_dict)
		embeddings[embedding.chunk_id] = embedding

	return embeddings

def _save_embeddings(embedding_path: str, doc_id: str, embeddings: dict[str, Embedded]) -> None:
	"""
	Input: embedding_path, the embedding folder; doc_id, the document the
	embeddings belong to; embeddings, a dict of chunk_id to Embedded for
	all of that document's chunks.
	Output: nothing.
	What it does: writes the embeddings, in order, as one JSON list to
	{doc_id}.json in embedding_path, creating the folder first if needed.
	The list is written to a temporary file and then renamed into place
	with os.replace, so {doc_id}.json is either complete or absent.
	"""
	os.makedirs(embedding_path, exist_ok=True)

	embedding_dicts = []
	for chunk_id in embeddings:
		embedding_dicts.append(embedding_to_json_dict(embeddings[chunk_id]))

	cache_file = os.path.join(embedding_path, f"{doc_id}.json")
	temporary_file = cache_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(embedding_dicts, f)
	os.replace(temporary_file, cache_file)
