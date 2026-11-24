import dataclasses
import json
import os

from schema.chunk import Chunk
from schema.file import File


def save_chunks(chunk_path: str, doc_id: str, chunks: dict[str, Chunk]) -> None:
	"""
	Input: chunk_path; doc_id; chunks, the document's chunks.
	Output: nothing.
	What it does: saves the document's chunks as one JSON list to {doc_id}.json.
	"""
	os.makedirs(chunk_path, exist_ok=True)

	chunk_dicts = []
	for chunk_id in chunks:
		chunk_dicts.append(dataclasses.asdict(chunks[chunk_id]))

	chunk_file = os.path.join(chunk_path, f"{doc_id}.json")
	with open(chunk_file, "w", encoding="utf-8") as f:
		json.dump(chunk_dicts, f)


def load_chunks(chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
	"""
	Input: chunk_path; doc_id.
	Output: a dict of chunk_id to Chunk, or None if the document has no saved chunks.
	What it does: loads the document's chunks from {doc_id}.json.
	"""
	chunk_file = os.path.join(chunk_path, f"{doc_id}.json")
	if not os.path.exists(chunk_file):
		return None

	with open(chunk_file, encoding="utf-8") as f:
		chunk_dicts = json.load(f)

	chunks = {}
	for chunk_dict in chunk_dicts:
		chunk = Chunk(**chunk_dict)
		chunks[chunk.id] = chunk

	return chunks


def load_saved_chunks(documents: dict[str, File], chunk_path: str) -> dict[str, Chunk]:
	"""
	Input: documents, a dict of doc_id to File; chunk_path, the chunk folder from the build manifest.
	Output: a dict of chunk_id to Chunk for every document.
	What it does: loads the chunks the build saved, and raises FileNotFoundError if any document has none. It never
	chunks.
	"""
	chunks = {}
	for doc_id in documents:
		saved_chunks = load_chunks(chunk_path, doc_id)
		if saved_chunks is None:
			raise FileNotFoundError(f"No saved chunks for document {doc_id} in {chunk_path}. Run scripts/build.sh first.")
		chunks.update(saved_chunks)

	print(f"Chunks : {len(chunks)} (all from cache)")
	return chunks
