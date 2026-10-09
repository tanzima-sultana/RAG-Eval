"""
faiss.py

Builds, saves and loads the FAISS index of one indexing run. The index is
saved in the FAISS index folder (see common.paths.get_faiss_index_path) as
two files:
index.faiss, the FAISS index itself, and
chunk_ids.json, the chunk_id of each FAISS position, in order.

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file.
"""

import json
import os

import faiss
import numpy as np

from common.args import IndexArgs
from common.constants import FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX
from l3_embedding.schema import Embedded
from l4_indexing.schema import FaissIndex

def get_faiss_index(faiss_index_path: str, index_args: IndexArgs, embeddings: dict[str, Embedded]) -> FaissIndex:
	"""
	Input: faiss_index_path, the FAISS index folder; index_args, the indexing
	parameters; embeddings, a dict of chunk_id to Embedded from l3_embedding.
	Output: a FaissIndex holding the FAISS index of type faiss_index_type.
	What it does: loads the saved index from faiss_index_path if it holds
	exactly the same chunk_ids as embeddings. Otherwise builds a new index
	from all the embedding vectors with the method that matches
	faiss_index_type, saves it, and returns it. The vectors are normalized
	by l3_embedding, so inner product is the same as cosine similarity.
	"""
	if len(embeddings) == 0:
		raise ValueError("embeddings is empty. There is nothing to index.")

	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	chunk_ids_file_path = os.path.join(faiss_index_path, "chunk_ids.json")

	# 1. Collect the chunk ids and vectors in the same order
	# FAISS position i will hold vectors[i], which belongs to chunk_ids[i]
	chunk_ids = []
	vectors = []
	for chunk_id in embeddings:
		chunk_ids.append(chunk_id)
		vectors.append(embeddings[chunk_id].embeddings)
	vectors = np.array(vectors, dtype=np.float32)
	dimension = vectors.shape[1]

	# 2. Load the saved index if it holds exactly the same chunks
	# The saved chunk_ids are used, not the new ones, because they match
	# the positions in the saved index
	if os.path.exists(index_file_path) and os.path.exists(chunk_ids_file_path):
		with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
			saved_chunk_ids = json.load(f)

		if set(saved_chunk_ids) == set(chunk_ids):
			saved_index = faiss.read_index(index_file_path)
			print(f"FAISS index loaded from cache : {saved_index.ntotal} vectors")
			return FaissIndex(
				faiss_index_type=index_args.faiss_index_type,
				faiss_index=saved_index,
				chunk_ids=saved_chunk_ids,
				index_file_path=index_file_path,
				dimension=dimension
			)

	# 3. Build a new index of the requested type
	if index_args.faiss_index_type == FAISS_FLATIP_INDEX:
		new_index = build_flatip_index(vectors, dimension)
	elif index_args.faiss_index_type == FAISS_IVF_INDEX:
		new_index = build_ivf_index(vectors, dimension, index_args.ivf_nlist)
	elif index_args.faiss_index_type == FAISS_HNSW_INDEX:
		new_index = build_hnsw_index(vectors, dimension, index_args.hnsw_m, index_args.hnsw_ef_construction)
	else:
		raise ValueError(f"faiss_index_type ({index_args.faiss_index_type}) is not supported")
	print(f"FAISS index built : {new_index.ntotal} vectors")

	# 4. Save the index and its chunk ids
	os.makedirs(faiss_index_path, exist_ok=True)

	temporary_index_file = index_file_path + ".tmp"
	faiss.write_index(new_index, temporary_index_file)
	os.replace(temporary_index_file, index_file_path)

	temporary_chunk_ids_file = chunk_ids_file_path + ".tmp"
	with open(temporary_chunk_ids_file, "w", encoding="utf-8") as f:
		json.dump(chunk_ids, f)
	os.replace(temporary_chunk_ids_file, chunk_ids_file_path)

	return FaissIndex(
		faiss_index_type=index_args.faiss_index_type,
		faiss_index=new_index,
		chunk_ids=chunk_ids,
		index_file_path=index_file_path,
		dimension=dimension
	)

def build_flatip_index(vectors: np.ndarray, dimension: int) -> faiss.Index:
	"""
	Input: vectors, a float32 array with one row per chunk; dimension, the
	length of each row.
	Output: a FAISS IndexFlatIP holding all the vectors.
	What it does: builds an exact inner product index. A search compares
	the query with every vector, so results are exact but slower on large
	data.
	"""
	flatip_index = faiss.IndexFlatIP(dimension)
	flatip_index.add(vectors)
	return flatip_index

def build_ivf_index(vectors: np.ndarray, dimension: int, ivf_nlist: int) -> faiss.Index:
	"""
	Input: vectors, a float32 array with one row per chunk; dimension, the
	length of each row; ivf_nlist, the number of clusters.
	Output: a FAISS IndexIVFFlat holding all the vectors.
	What it does: groups the vectors into ivf_nlist clusters (training),
	then adds each vector to its nearest cluster. A search only looks in
	the clusters closest to the query, so it is faster but approximate.
	Raises ValueError if there are fewer vectors than clusters, because
	FAISS cannot train that many clusters.
	"""
	vector_count = vectors.shape[0]
	if vector_count < ivf_nlist:
		raise ValueError(
			f"ivf_nlist ({ivf_nlist}) must not be larger than the number of chunks ({vector_count}). "
			"Use a smaller ivf_nlist or a larger dataset."
		)

	cluster_index = faiss.IndexFlatIP(dimension)
	ivf_index = faiss.IndexIVFFlat(cluster_index, dimension, ivf_nlist, faiss.METRIC_INNER_PRODUCT)
	ivf_index.train(vectors)
	ivf_index.add(vectors)
	return ivf_index

def build_hnsw_index(vectors: np.ndarray, dimension: int, hnsw_m: int, hnsw_ef_construction: int) -> faiss.Index:
	"""
	Input: vectors, a float32 array with one row per chunk; dimension, the
	length of each row; hnsw_m, the number of neighbours per node;
	hnsw_ef_construction, the candidate list size used while building.
	Output: a FAISS IndexHNSWFlat holding all the vectors.
	What it does: builds a graph where each vector is linked to hnsw_m
	close vectors. A search walks the graph towards the query, so it is
	fast but approximate. A larger hnsw_ef_construction builds a better
	graph but takes longer.
	"""
	hnsw_index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
	hnsw_index.hnsw.efConstruction = hnsw_ef_construction
	hnsw_index.add(vectors)
	return hnsw_index
