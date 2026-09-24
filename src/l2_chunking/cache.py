"""
cache.py

Saves and loads Chunks as JSON files, one {chunk_id}.json per chunk, in
the chunk folder of a chunk config (see common.paths.get_chunk_path).
"""

import dataclasses
import glob
import json
import os

from l2_chunking.schema import Chunk

def chunk_to_json_dict(chunk: Chunk) -> dict:
	"""
	chunk: a fully-populated Chunk.

	Converts chunk to a plain dict of JSON-serializable values.
	Chunk has no non-JSON-native fields (no timestamp to convert),
	so this is a direct dataclass-to-dict conversion.

	Returns the JSON-serializable dict.
	"""
	return dataclasses.asdict(chunk)

def json_dict_to_chunk(data: dict) -> Chunk:
	"""
	data: a dict previously produced by chunk_to_json_dict. Chunk
		has no non-JSON-native fields (no timestamp to reparse),
		so this only reconstructs the dataclass instance.

	Returns the reconstructed Chunk.
	"""
	return Chunk(**data)

def _load_cached_chunks(chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
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
			chunks[chunk_id] = json_dict_to_chunk(json.load(f))

	return chunks

def _save_chunks(chunk_path: str, chunks: dict[str, Chunk]) -> None:
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
			json.dump(chunk_to_json_dict(chunks[chunk_id]), f)