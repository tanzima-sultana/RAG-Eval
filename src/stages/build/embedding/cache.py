"""
cache.py

Saves and loads Embedded records as numpy .npz files, one {doc_id}.npz per
document holding the embeddings of all of that document's chunks, in the
embedding folder of a chunking run (see common.paths.get_embedding_path).

Each file holds three arrays, in the same chunk order:
chunk_ids (strings), embeddings (one float32 row per chunk) and
normalized (one bool per chunk).

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written {doc_id}.npz.
"""

import os

import numpy as np

from stages.build.embedding.schema import Embedded

def _load_cached_embeddings(embedding_path: str, doc_id: str) -> dict[str, Embedded] | None:
	"""
	Input: embedding_path, the embedding folder; doc_id, the document to look up.
	Output: a dict of chunk_id to Embedded for that document's chunks, in
	the order they were saved, or None if the document has no cache file.
	What it does: reads {doc_id}.npz from embedding_path if it exists and
	builds one Embedded per row, keeping the saved float32 vectors.
	"""
	cache_file = os.path.join(embedding_path, f"{doc_id}.npz")
	if not os.path.exists(cache_file):
		return None

	with np.load(cache_file) as saved_arrays:
		chunk_ids = saved_arrays["chunk_ids"]
		embedding_vectors = saved_arrays["embeddings"]
		normalized_flags = saved_arrays["normalized"]

	embeddings = {}
	for index in range(len(chunk_ids)):
		chunk_id = str(chunk_ids[index])
		embeddings[chunk_id] = Embedded(chunk_id, embedding_vectors[index], bool(normalized_flags[index]))

	return embeddings

def _save_embeddings(embedding_path: str, doc_id: str, embeddings: dict[str, Embedded]) -> None:
	"""
	Input: embedding_path, the embedding folder; doc_id, the document the
	embeddings belong to; embeddings, a dict of chunk_id to Embedded for
	that document's chunks.
	Output: nothing.
	What it does: writes the chunk ids, the vectors (as one float32 array)
	and the normalized flags, in order, to {doc_id}.npz in embedding_path,
	creating the folder first if needed. The file is written to a
	temporary file and then renamed into place with os.replace, so
	{doc_id}.npz is either complete or absent.
	"""
	os.makedirs(embedding_path, exist_ok=True)

	chunk_ids = []
	embedding_vectors = []
	normalized_flags = []
	for chunk_id in embeddings:
		embedding = embeddings[chunk_id]
		chunk_ids.append(chunk_id)
		embedding_vectors.append(embedding.embeddings)
		normalized_flags.append(embedding.normalized)

	cache_file = os.path.join(embedding_path, f"{doc_id}.npz")
	temporary_file = cache_file + ".tmp"
	# Written through an open file, because np.savez adds ".npz" to a
	# path that does not already end with it
	with open(temporary_file, "wb") as f:
		np.savez(
			f,
			chunk_ids=np.array(chunk_ids),
			embeddings=np.array(embedding_vectors, dtype=np.float32),
			normalized=np.array(normalized_flags),
		)
	os.replace(temporary_file, cache_file)

def load_embeddings(embedding_path: str, chunk_ids: list[str]) -> dict[str, Embedded]:
	"""
	Input: embedding_path, the embedding folder of a chunking run;
	chunk_ids, the chunk ids to load, for example the chunk ids stored in
	an index.
	Output: a dict of chunk_id to Embedded, in the order of chunk_ids.
	What it does: load-only, never embeds anything. Reads the
	{doc_id}.npz file of each chunk's document once and picks the
	requested embeddings. Raises FileNotFoundError if embedding_path or a
	document's file does not exist, and ValueError if a document's file
	does not hold a requested chunk_id.
	"""
	if not os.path.isdir(embedding_path):
		raise FileNotFoundError(
			f"No embedding folder at {embedding_path}. Run scripts/pipeline.sh with the same chunking args first."
		)

	embeddings_by_document = {}
	embeddings = {}
	for chunk_id in chunk_ids:
		doc_id = chunk_id.rsplit("_", 1)[0]
		if doc_id not in embeddings_by_document:
			document_embeddings = _load_cached_embeddings(embedding_path, doc_id)
			if document_embeddings is None:
				raise FileNotFoundError(
					f"No embedding file for document {doc_id} in {embedding_path}. "
					"Run scripts/pipeline.sh with the same chunking args first."
				)
			embeddings_by_document[doc_id] = document_embeddings

		if chunk_id not in embeddings_by_document[doc_id]:
			raise ValueError(
				f"Chunk {chunk_id} is not in the embedding file of document {doc_id} in {embedding_path}. "
				"The embeddings and the index were made with different settings."
			)
		embeddings[chunk_id] = embeddings_by_document[doc_id][chunk_id]

	return embeddings
