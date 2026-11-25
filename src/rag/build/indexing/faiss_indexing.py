import math
import os
import shutil

import faiss
import numpy as np

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings

from common import constants
from common import paths
from common.args import FaissHNSWArgs
from common.args import FaissIVFArgs
from schema.chunk import Chunk
from schema.embed import Embedded
from schema.index import FaissIndex


def save_faiss_index(faiss_index_path: str, vector_store: FAISS) -> None:
	"""
	Input: faiss_index_path, the FAISS index folder; vector_store, the LangChain FAISS vector store.
	Output: nothing.
	What it does: saves index.faiss and index.pkl into a temporary folder, then moves them into faiss_index_path.
	"""
	temporary_folder_path = faiss_index_path + ".tmp"
	if os.path.exists(temporary_folder_path):
		shutil.rmtree(temporary_folder_path)
	vector_store.save_local(temporary_folder_path)

	os.makedirs(faiss_index_path, exist_ok=True)
	os.replace(os.path.join(temporary_folder_path, "index.faiss"), os.path.join(faiss_index_path, "index.faiss"))
	os.replace(os.path.join(temporary_folder_path, "index.pkl"), os.path.join(faiss_index_path, "index.pkl"))
	os.rmdir(temporary_folder_path)


def load_faiss_index(faiss_index_path: str, chunk_ids: list[str], embedding_model: Embeddings) -> FAISS | None:
	"""
	Input: faiss_index_path, the FAISS index folder; chunk_ids, the chunk ids the index must hold; embedding_model.
	Output: the saved LangChain FAISS vector store, or None if there is none or it holds different chunk ids.
	What it does: loads index.faiss and index.pkl with FAISS.load_local and compares its chunk ids with chunk_ids.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	docstore_file_path = os.path.join(faiss_index_path, "index.pkl")
	if not os.path.exists(index_file_path) or not os.path.exists(docstore_file_path):
		return None

	# index.pkl is read with pickle. It was written by this code, so allow_dangerous_deserialization is safe here
	saved_vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

	saved_chunk_ids = []
	for position in saved_vector_store.index_to_docstore_id:
		saved_chunk_ids.append(saved_vector_store.index_to_docstore_id[position])
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	return saved_vector_store


def load_saved_faiss_index(index_type: str, chunks: dict[str, Chunk], index_folder: str, embedding_model: Embeddings) -> FaissIndex:
	"""
	Input: index_type, faiss_flatip, faiss_ivf or faiss_hnsw; chunks, a dict of chunk_id to Chunk, the chunks the index
	must hold; index_folder, the index folder from the build manifest; embedding_model, the embedding model the index
	was built with.
	Output: a FaissIndex.
	What it does: loads the FAISS index the build saved, and raises FileNotFoundError if it is not saved or holds
	different chunks. It never builds.
	"""
	chunk_ids = []
	for chunk_id in chunks:
		chunk_ids.append(chunk_id)

	vector_store = load_faiss_index(index_folder, chunk_ids, embedding_model)
	if vector_store is None:
		raise FileNotFoundError(
			f"No saved {index_type} index holding these chunks in {index_folder}. Run scripts/build.sh first."
		)
	print(f"{index_type} index loaded from cache : {vector_store.index.ntotal} vectors")
	return FaissIndex(
		index_type=index_type,
		vector_store=vector_store,
		index_path=index_folder,
		dimension=vector_store.index.d
	)


def load_faiss_index_from_path(index_type: str, index_folder: str, embedding_model: Embeddings) -> FaissIndex:
	"""
	Input: index_type, faiss_flatip, faiss_ivf or faiss_hnsw; index_folder, the index folder the build saved;
	embedding_model, the embedding model the index was built with, which embeds the query.
	Output: a FaissIndex.
	What it does: loads the saved FAISS index straight from its folder with LangChain's FAISS.load_local, for serving.
	It does not check which chunks it holds, and raises FileNotFoundError if the files are not there.
	"""
	index_file_path = os.path.join(index_folder, "index.faiss")
	docstore_file_path = os.path.join(index_folder, "index.pkl")
	if not os.path.exists(index_file_path) or not os.path.exists(docstore_file_path):
		raise FileNotFoundError(f"No saved {index_type} index in {index_folder}. Run scripts/build.sh first.")
	# index.pkl is read with pickle. It was written by this code, so allow_dangerous_deserialization is safe here
	vector_store = FAISS.load_local(index_folder, embedding_model, allow_dangerous_deserialization=True)
	print(f"{index_type} index loaded : {index_folder}")
	return FaissIndex(
		index_type=index_type,
		vector_store=vector_store,
		index_path=index_folder,
		dimension=vector_store.index.d
	)


class FaissIndexing:
	def __init__(self, embedding_hash: str, embedding_model: Embeddings):
		"""
		Input: embedding_hash, the hash of the embeddings the indexes are built from, from embedding; embedding_model,
		which the FAISS vector store uses to embed queries.
		Output: nothing.
		What it does: stores the two.
		"""
		self.embedding_hash = embedding_hash
		self.embedding_model = embedding_model

	def faiss_flatip_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> tuple[FaissIndex, str, str]:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder.
		Output: (faiss_index, faiss_hash, faiss_index_path): a FaissIndex holding an exact inner product (IndexFlatIP) vector store.
		What it does: loads the saved index if it holds the same chunks, otherwise builds, wraps and saves a new one.
		"""
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# 1. Collect the chunk ids and vectors in the same order. FAISS position i holds vectors[i] of chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		vectors = np.array(vectors, dtype=np.float32)
		dimension = vectors.shape[1]

		# 2. Load the saved index if it holds exactly the same chunks
		faiss_hash = paths.get_faiss_hash(constants.FAISS_FLATIP_INDEX, None, self.embedding_hash)
		faiss_index_path = paths.get_faiss_index_path(output_path, constants.FAISS_FLATIP_INDEX, faiss_hash)
		print(f"Index folder : {faiss_index_path}")
		vector_store = load_faiss_index(faiss_index_path, chunk_ids, self.embedding_model)
		if vector_store is not None:
			print(f"FAISS flatip index loaded from cache : {vector_store.index.ntotal} vectors")
		else:
			# 3. Build the index. A search compares the query with every vector, so it is exact
			flatip_index = faiss.IndexFlatIP(dimension)
			flatip_index.add(vectors)

			# 4. Wrap it in a LangChain FAISS vector store and save it
			vector_store = self.wrap_faiss_index(flatip_index, chunk_ids, chunks)
			save_faiss_index(faiss_index_path, vector_store)
			print(f"FAISS flatip index built : {flatip_index.ntotal} vectors")

		faiss_index = FaissIndex(
			index_type=constants.FAISS_FLATIP_INDEX,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)
		return faiss_index, faiss_hash, faiss_index_path

	def faiss_ivf_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str, ivf_args: FaissIVFArgs) -> tuple[FaissIndex, str, str]:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder; ivf_args, the settings of this IVF index.
		Output: (faiss_index, faiss_hash, faiss_index_path): a FaissIndex holding an IVF (IndexIVFFlat) vector store with ivf_nlist clusters.
		What it does: computes ivf_nlist from ivf_nlist_multiplier and the number of chunks, then loads the saved index
		if it holds the same chunks, otherwise builds, wraps and saves a new one.
		"""
		if ivf_args.ivf_nlist_multiplier <= 0:
			raise ValueError(f"ivf_nlist_multiplier ({ivf_args.ivf_nlist_multiplier}) must be greater than 0.")
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# nlist = multiplier * sqrt(number of chunks), capped at number of chunks // 39, since FAISS needs about 39
		# training vectors per cluster, and at least 1
		ivf_nlist = round(ivf_args.ivf_nlist_multiplier * math.sqrt(len(embeddings)))
		max_ivf_nlist = len(embeddings) // 39
		if ivf_nlist > max_ivf_nlist:
			ivf_nlist = max_ivf_nlist
		if ivf_nlist < 1:
			ivf_nlist = 1

		# 1. Collect the chunk ids and vectors in the same order. FAISS position i holds vectors[i] of chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		vectors = np.array(vectors, dtype=np.float32)
		dimension = vectors.shape[1]

		# 2. Load the saved index if it holds exactly the same chunks
		faiss_hash = paths.get_faiss_hash(constants.FAISS_IVF_INDEX, ivf_args, self.embedding_hash)
		faiss_index_path = paths.get_faiss_index_path(output_path, constants.FAISS_IVF_INDEX, faiss_hash)
		print(f"Index folder : {faiss_index_path}")
		vector_store = load_faiss_index(faiss_index_path, chunk_ids, self.embedding_model)
		if vector_store is not None:
			print(f"FAISS ivf index loaded from cache : {vector_store.index.ntotal} vectors")
		else:
			# 3. Build the index. The vectors are grouped into ivf_nlist clusters, and a search looks only in the
			# clusters closest to the query, so it is faster but approximate
			cluster_index = faiss.IndexFlatIP(dimension)
			ivf_index = faiss.IndexIVFFlat(cluster_index, dimension, ivf_nlist, faiss.METRIC_INNER_PRODUCT)
			ivf_index.train(vectors)
			ivf_index.add(vectors)

			# 4. Wrap it in a LangChain FAISS vector store and save it
			vector_store = self.wrap_faiss_index(ivf_index, chunk_ids, chunks)
			save_faiss_index(faiss_index_path, vector_store)
			print(f"FAISS ivf index built : {ivf_index.ntotal} vectors")

		faiss_index = FaissIndex(
			index_type=constants.FAISS_IVF_INDEX,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)
		return faiss_index, faiss_hash, faiss_index_path

	def faiss_hnsw_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str, hnsw_args: FaissHNSWArgs) -> tuple[FaissIndex, str, str]:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder; hnsw_args, the settings of this HNSW index.
		Output: (faiss_index, faiss_hash, faiss_index_path): a FaissIndex holding an HNSW (IndexHNSWFlat) vector store.
		What it does: loads the saved index if it holds the same chunks, otherwise builds, wraps and saves a new one.
		"""
		if hnsw_args.hnsw_m <= 0:
			raise ValueError(f"hnsw_m ({hnsw_args.hnsw_m}) must be greater than 0.")
		if hnsw_args.hnsw_ef_construction <= 0:
			raise ValueError(f"hnsw_ef_construction ({hnsw_args.hnsw_ef_construction}) must be greater than 0.")
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# 1. Collect the chunk ids and vectors in the same order. FAISS position i holds vectors[i] of chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		vectors = np.array(vectors, dtype=np.float32)
		dimension = vectors.shape[1]

		# 2. Load the saved index if it holds exactly the same chunks
		faiss_hash = paths.get_faiss_hash(constants.FAISS_HNSW_INDEX, hnsw_args, self.embedding_hash)
		faiss_index_path = paths.get_faiss_index_path(output_path, constants.FAISS_HNSW_INDEX, faiss_hash)
		print(f"Index folder : {faiss_index_path}")
		vector_store = load_faiss_index(faiss_index_path, chunk_ids, self.embedding_model)
		if vector_store is not None:
			print(f"FAISS hnsw index loaded from cache : {vector_store.index.ntotal} vectors")
		else:
			# 3. Build the index. Each vector is linked to hnsw_m close vectors, and a search walks this graph
			# towards the query, so it is fast but approximate
			hnsw_index = faiss.IndexHNSWFlat(dimension, hnsw_args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
			hnsw_index.hnsw.efConstruction = hnsw_args.hnsw_ef_construction
			hnsw_index.add(vectors)

			# 4. Wrap it in a LangChain FAISS vector store and save it
			vector_store = self.wrap_faiss_index(hnsw_index, chunk_ids, chunks)
			save_faiss_index(faiss_index_path, vector_store)
			print(f"FAISS hnsw index built : {hnsw_index.ntotal} vectors")

		faiss_index = FaissIndex(
			index_type=constants.FAISS_HNSW_INDEX,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)
		return faiss_index, faiss_hash, faiss_index_path

	def wrap_faiss_index(self, faiss_index: faiss.Index, chunk_ids: list[str], chunks: dict[str, Chunk]) -> FAISS:
		"""
		Input: faiss_index, a built FAISS index; chunk_ids, the chunk id at each FAISS position; chunks.
		Output: a LangChain FAISS vector store around faiss_index.
		What it does: makes one LangChain Document per chunk (text as page_content, {"chunk_id": chunk_id} as
		metadata) and maps each FAISS position to its chunk_id. The vectors are normalized, so inner product is cosine.
		"""
		documents_by_chunk_id = {}
		index_to_docstore_id = {}
		for position in range(len(chunk_ids)):
			chunk_id = chunk_ids[position]
			documents_by_chunk_id[chunk_id] = LangchainDocument(
				page_content=chunks[chunk_id].chunk_texts,
				metadata={"chunk_id": chunk_id}
			)
			index_to_docstore_id[position] = chunk_id

		vector_store = FAISS(
			embedding_function=self.embedding_model,
			index=faiss_index,
			docstore=InMemoryDocstore(documents_by_chunk_id),
			index_to_docstore_id=index_to_docstore_id,
			distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
		)
		return vector_store
