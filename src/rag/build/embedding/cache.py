import os

import numpy as np

from schema.embed import Embedded


def save_embeddings(embedding_path: str, doc_id: str, embeddings: dict[str, Embedded]) -> None:
	"""
	Input: embedding_path; doc_id; embeddings, the document's embeddings.
	Output: nothing.
	What it does: saves the document's chunk ids, vectors and normalized flags to {doc_id}.npz.
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

	embedding_file = os.path.join(embedding_path, f"{doc_id}.npz")
	np.savez(
		embedding_file,
		chunk_ids=np.array(chunk_ids),
		embeddings=np.array(embedding_vectors, dtype=np.float32),
		normalized=np.array(normalized_flags),
	)


def load_embeddings(embedding_path: str, doc_id: str) -> dict[str, Embedded] | None:
	"""
	Input: embedding_path; doc_id.
	Output: a dict of chunk_id to Embedded, or None if the document has no saved embeddings.
	What it does: loads the document's embeddings from {doc_id}.npz.
	"""
	embedding_file = os.path.join(embedding_path, f"{doc_id}.npz")
	if not os.path.exists(embedding_file):
		return None

	with np.load(embedding_file) as saved_arrays:
		chunk_ids = saved_arrays["chunk_ids"]
		embedding_vectors = saved_arrays["embeddings"]
		normalized_flags = saved_arrays["normalized"]

	embeddings = {}
	for index in range(len(chunk_ids)):
		chunk_id = str(chunk_ids[index])
		embeddings[chunk_id] = Embedded(chunk_id, embedding_vectors[index], bool(normalized_flags[index]))

	return embeddings
