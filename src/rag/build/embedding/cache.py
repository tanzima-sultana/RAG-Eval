import json
import os

import numpy as np

from schema.embed import Embedded


def save_embeddings(embedding_path: str, new_embeddings: dict[str, Embedded], document_ids: list[str], embedding_hash: str) -> None:
	"""
	Input: embedding_path; new_embeddings, a dict of chunk_id to Embedded for the documents this run embedded;
	document_ids, the document ids of every document of this run, embedded or loaded from the cache, in order;
	embedding_hash, the hash of this embedding run.
	Output: nothing.
	What it does: groups the new embeddings by their document, which is the part of the chunk id before the first "_",
	saves each document's chunk ids, vectors and normalized flags to {document_id}.npz, then saves the list of document
	ids to doc_ids_{embedding_hash}.json. The documents loaded from the cache are already saved.
	"""
	os.makedirs(embedding_path, exist_ok=True)

	# 1. Group the new embeddings by document, in the order the documents first appear
	new_embeddings_by_document_id = {}
	for chunk_id in new_embeddings:
		document_id = chunk_id.split("_", 1)[0]
		if document_id not in new_embeddings_by_document_id:
			new_embeddings_by_document_id[document_id] = []
		new_embeddings_by_document_id[document_id].append(new_embeddings[chunk_id])

	# 2. Save each new document's embeddings
	for document_id in new_embeddings_by_document_id:
		chunk_ids = []
		embedding_vectors = []
		normalized_flags = []
		for embedding in new_embeddings_by_document_id[document_id]:
			chunk_ids.append(embedding.chunk_id)
			embedding_vectors.append(embedding.embeddings)
			normalized_flags.append(embedding.normalized)

		embedding_file = os.path.join(embedding_path, f"{document_id}.npz")
		np.savez(
			embedding_file,
			chunk_ids=np.array(chunk_ids),
			embeddings=np.array(embedding_vectors, dtype=np.float32),
			normalized=np.array(normalized_flags),
		)

	# 3. Written last, so it only lists documents whose files are saved
	doc_ids_file = os.path.join(embedding_path, f"doc_ids_{embedding_hash}.json")
	with open(doc_ids_file, "w", encoding="utf-8") as f:
		json.dump(document_ids, f)


def load_document_embeddings(embedding_path: str, document_id: str) -> dict[str, Embedded] | None:
	"""
	Input: embedding_path; document_id.
	Output: a dict of chunk_id to Embedded, or None if the document has no saved embeddings.
	What it does: loads the document's embeddings from {document_id}.npz.
	"""
	embedding_file = os.path.join(embedding_path, f"{document_id}.npz")
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


def load_embeddings(embedding_path: str, embedding_hash: str) -> dict[str, Embedded] | None:
	"""
	Input: embedding_path; embedding_hash, the hash of the embedding run.
	Output: a dict of chunk_id to Embedded for every document of the run, or None if the run is not saved or one of its
	documents is missing.
	What it does: loads every document listed in doc_ids_{embedding_hash}.json and joins their embeddings into one dict.
	"""
	doc_ids_file = os.path.join(embedding_path, f"doc_ids_{embedding_hash}.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		document_ids = json.load(f)

	embeddings = {}
	for document_id in document_ids:
		document_embeddings = load_document_embeddings(embedding_path, document_id)
		if document_embeddings is None:
			return None
		embeddings.update(document_embeddings)
	return embeddings
