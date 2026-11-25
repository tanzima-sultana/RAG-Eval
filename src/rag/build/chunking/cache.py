import dataclasses
import json
import os

from schema.chunk import Chunk


def save_chunks(chunk_path: str, new_chunks: dict[str, Chunk], document_ids: list[str], chunking_hash: str) -> None:
	"""
	Input: chunk_path; new_chunks, a dict of chunk_id to Chunk for the documents this run chunked; document_ids, the
	document ids of every file of this run, chunked or loaded from the cache, in order; chunking_hash, the hash of this
	chunking run.
	Output: nothing.
	What it does: groups the new chunks by their document, which is the part of the chunk id before the first "_", saves
	each document's chunks to {document_id}.json, then saves the list of document ids to
	doc_ids_{chunking_hash}.json. The documents loaded from the cache are already saved.
	"""
	os.makedirs(chunk_path, exist_ok=True)

	# 1. Group the new chunks by document, in the order the documents first appear
	new_chunks_by_document_id = {}
	for chunk_id in new_chunks:
		document_id = chunk_id.split("_", 1)[0]
		if document_id not in new_chunks_by_document_id:
			new_chunks_by_document_id[document_id] = []
		new_chunks_by_document_id[document_id].append(new_chunks[chunk_id])

	# 2. Save each new document's chunks
	for document_id in new_chunks_by_document_id:
		chunk_dicts = []
		for chunk in new_chunks_by_document_id[document_id]:
			chunk_dicts.append(dataclasses.asdict(chunk))

		chunk_file = os.path.join(chunk_path, f"{document_id}.json")
		with open(chunk_file, "w", encoding="utf-8") as f:
			json.dump(chunk_dicts, f)

	# 3. Written last, so it only lists documents whose files are saved
	doc_ids_file = os.path.join(chunk_path, f"doc_ids_{chunking_hash}.json")
	with open(doc_ids_file, "w", encoding="utf-8") as f:
		json.dump(document_ids, f)


def load_document_chunks(chunk_path: str, document_id: str) -> dict[str, Chunk] | None:
	"""
	Input: chunk_path; document_id.
	Output: a dict of chunk_id to Chunk, or None if the document has no saved chunks.
	What it does: loads the document's chunks from {document_id}.json.
	"""
	chunk_file = os.path.join(chunk_path, f"{document_id}.json")
	if not os.path.exists(chunk_file):
		return None

	with open(chunk_file, encoding="utf-8") as f:
		chunk_dicts = json.load(f)

	chunks = {}
	for chunk_dict in chunk_dicts:
		chunk = Chunk(**chunk_dict)
		chunks[chunk.chunk_id] = chunk
	return chunks


def load_chunks(chunk_path: str, chunking_hash: str) -> dict[str, Chunk] | None:
	"""
	Input: chunk_path; chunking_hash, the hash of the chunking run.
	Output: a dict of chunk_id to Chunk for every document of the run, or None if the run is not saved or one of its
	documents is missing.
	What it does: loads every document listed in doc_ids_{chunking_hash}.json and joins their chunks into one dict.
	"""
	doc_ids_file = os.path.join(chunk_path, f"doc_ids_{chunking_hash}.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		document_ids = json.load(f)

	chunks = {}
	for document_id in document_ids:
		document_chunks = load_document_chunks(chunk_path, document_id)
		if document_chunks is None:
			return None
		chunks.update(document_chunks)
	return chunks
