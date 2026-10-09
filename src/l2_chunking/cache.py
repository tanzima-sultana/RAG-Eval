"""
cache.py

Saves and loads Chunks as JSON files, one {doc_id}.json per document
holding all of that document's chunks, in the chunk folder of a chunking
run (see common.paths.get_chunk_path).

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written {doc_id}.json
that a later run would mistake for a complete cache.
"""

import dataclasses
import json
import os

from l2_chunking.schema import Chunk

def chunk_to_json_dict(chunk: Chunk) -> dict:
	"""
	Input: chunk, a fully-populated Chunk.
	Output: a plain dict of JSON-serializable values.
	What it does: converts the Chunk dataclass to a dict. Chunk has only
	JSON-native fields, so this is a direct conversion.
	"""
	return dataclasses.asdict(chunk)

def json_dict_to_chunk(data: dict) -> Chunk:
	"""
	Input: data, a dict previously produced by chunk_to_json_dict.
	Output: the reconstructed Chunk.
	What it does: builds a Chunk from the dict's fields.
	"""
	return Chunk(**data)

def get_document_cache_file(chunk_path: str, doc_id: str) -> str:
	"""
	Input: chunk_path, the chunk folder; doc_id, a document id.
	Output: the path of that document's cache file.
	What it does: joins chunk_path and "{doc_id}.json".
	"""
	return os.path.join(chunk_path, f"{doc_id}.json")

def _load_cached_chunks(chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
	"""
	Input: chunk_path, the chunk folder; doc_id, the document to look up.
	Output: a dict of chunk_id to Chunk for that document, in the order the
	chunks were saved, or None if the document has no cache file.
	What it does: reads {doc_id}.json from chunk_path if it exists. A
	document that produced no chunks returns an empty dict, not None, so
	it still counts as cached.
	"""
	cache_file = get_document_cache_file(chunk_path, doc_id)
	if not os.path.exists(cache_file):
		return None

	with open(cache_file, encoding="utf-8") as f:
		chunk_dicts = json.load(f)

	chunks = {}
	for chunk_dict in chunk_dicts:
		chunk = json_dict_to_chunk(chunk_dict)
		chunks[chunk.id] = chunk

	return chunks

def _save_chunks(chunk_path: str, doc_id: str, chunks: dict[str, Chunk]) -> None:
	"""
	Input: chunk_path, the chunk folder; doc_id, the document the chunks
	belong to; chunks, a dict of chunk_id to Chunk for that document (may
	be empty).
	Output: nothing.
	What it does: writes all of the document's chunks, in order, as one
	JSON list to {doc_id}.json in chunk_path, creating chunk_path first if
	needed. The list is written to a temporary file and then renamed into
	place with os.replace, so {doc_id}.json is either complete or absent.
	"""
	os.makedirs(chunk_path, exist_ok=True)

	chunk_dicts = []
	for chunk_id in chunks:
		chunk_dicts.append(chunk_to_json_dict(chunks[chunk_id]))

	cache_file = get_document_cache_file(chunk_path, doc_id)
	temporary_file = cache_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(chunk_dicts, f)
	os.replace(temporary_file, cache_file)

def load_chunks(chunk_path: str, doc_ids: list[str]) -> dict[str, Chunk]:
	"""
	Input: chunk_path, the chunk folder of a chunking run; doc_ids, the
	documents whose chunks to load, for example the ids of the documents
	returned by l1 ingestion.
	Output: a dict of chunk_id to Chunk with every chunk of those
	documents, document by document in the order of doc_ids.
	What it does: load-only, never chunks anything. Reads the {doc_id}.json
	file of each document. Raises FileNotFoundError if chunk_path or a
	document's file does not exist.
	"""
	if not os.path.isdir(chunk_path):
		raise FileNotFoundError(
			f"No chunk folder at {chunk_path}. Run scripts/run.sh with the same chunking args first."
		)

	chunks = {}
	for doc_id in doc_ids:
		document_chunks = _load_cached_chunks(chunk_path, doc_id)
		if document_chunks is None:
			raise FileNotFoundError(
				f"No chunk file for document {doc_id} in {chunk_path}. "
				"Run scripts/run.sh with the same chunking args first."
			)
		chunks.update(document_chunks)

	return chunks
