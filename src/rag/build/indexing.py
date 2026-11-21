import os
import pickle
import shutil

import faiss
import numpy as np
import torch

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths
from common.args import IndexArgs
from schema.chunk import Chunk
from schema.embed import Embedded
from schema.index import FaissIndex
from schema.index import Bm25Index


def lowercase_whitespace_tokenize(text: str) -> list[str]:
	"""
	Input: text, one chunk or query text.
	Output: the list of lowercase words in text.
	What it does: lowercases text and splits it on whitespace. A named function, because pickle cannot save a lambda.
	"""
	return text.lower().split()


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


def load_faiss_index(faiss_index_path: str, chunk_ids: list[str], embedding_model: HuggingFaceEmbeddings) -> FAISS | None:
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


def save_bm25_index(bm25_index_path: str, retriever: BM25Retriever) -> None:
	"""
	Input: bm25_index_path, the BM25 index folder; retriever, the BM25Retriever.
	Output: nothing.
	What it does: saves the retriever with pickle to a temporary file, then renames it to bm25_retriever.pkl.
	"""
	os.makedirs(bm25_index_path, exist_ok=True)
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	temporary_retriever_file_path = retriever_file_path + ".tmp"
	with open(temporary_retriever_file_path, "wb") as f:
		pickle.dump(retriever, f)
	os.replace(temporary_retriever_file_path, retriever_file_path)


def load_bm25_index(bm25_index_path: str, chunk_ids: list[str]) -> BM25Retriever | None:
	"""
	Input: bm25_index_path, the BM25 index folder; chunk_ids, the chunk ids the retriever must hold.
	Output: the saved BM25Retriever, or None if there is none or it holds different chunk ids.
	What it does: loads bm25_retriever.pkl and compares its chunk ids with chunk_ids.
	"""
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	if not os.path.exists(retriever_file_path):
		return None

	# The file was written by this code, so reading it with pickle is safe here
	with open(retriever_file_path, "rb") as f:
		saved_retriever = pickle.load(f)

	saved_chunk_ids = []
	for document in saved_retriever.docs:
		saved_chunk_ids.append(document.metadata["chunk_id"])
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	return saved_retriever


class Indexing:
	def __init__(self, index_args: IndexArgs):
		"""
		Input: index_args, the indexing settings for one index type.
		Output: nothing.
		What it does: stores index_args, validates them and loads LangChain's HuggingFaceEmbeddings, which the FAISS
		vector store uses to embed queries.
		"""
		self.index_args = index_args
		self.validate()

		# Normalized like the embedding layer, so a query vector matches the chunk vectors
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = HuggingFaceEmbeddings(
			model_name=self.index_args.chunking_args.model_name,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if the index type is not supported, or if a setting of this index type is missing
		or not valid. Only the settings of index_args.index_type are checked: faiss_flatip has none.
		"""
		index_args = self.index_args

		supported_index_types = [
			constants.FAISS_FLATIP_INDEX,
			constants.FAISS_IVF_INDEX,
			constants.FAISS_HNSW_INDEX,
			constants.BM25_INDEX
		]
		if index_args.index_type not in supported_index_types:
			raise ValueError(
				f"index_type ({index_args.index_type}) is not supported. Use "
				f"\"{constants.FAISS_FLATIP_INDEX}\", \"{constants.FAISS_IVF_INDEX}\", "
				f"\"{constants.FAISS_HNSW_INDEX}\" or \"{constants.BM25_INDEX}\"."
			)

		if index_args.index_type == constants.FAISS_IVF_INDEX:
			if index_args.ivf_nlist is None:
				raise ValueError("ivf_nlist is missing. Give it in the faiss_ivf block of the indexing config.")
			if index_args.ivf_nlist <= 0:
				raise ValueError(f"ivf_nlist ({index_args.ivf_nlist}) must be greater than 0.")

		if index_args.index_type == constants.FAISS_HNSW_INDEX:
			if index_args.hnsw_m is None:
				raise ValueError("hnsw_m is missing. Give it in the faiss_hnsw block of the indexing config.")
			if index_args.hnsw_m <= 0:
				raise ValueError(f"hnsw_m ({index_args.hnsw_m}) must be greater than 0.")
			if index_args.hnsw_ef_construction is None:
				raise ValueError("hnsw_ef_construction is missing. Give it in the faiss_hnsw block of the indexing config.")
			if index_args.hnsw_ef_construction <= 0:
				raise ValueError(f"hnsw_ef_construction ({index_args.hnsw_ef_construction}) must be greater than 0.")

		if index_args.index_type == constants.BM25_INDEX:
			if index_args.bm25_k1 is None:
				raise ValueError("bm25_k1 is missing. Give it in the bm25 block of the indexing config.")
			if index_args.bm25_b is None:
				raise ValueError("bm25_b is missing. Give it in the bm25 block of the indexing config.")
			if index_args.bm25_tokenizer != "whitespace":
				raise ValueError(f"bm25_tokenizer ({index_args.bm25_tokenizer}) is not supported. Use \"whitespace\".")

	def load_saved_index(self, chunks: dict[str, Chunk], index_folder: str) -> FaissIndex | Bm25Index:
		"""
		Input: chunks, a dict of chunk_id to Chunk, the chunks the index must hold; index_folder, the index folder from
		the build manifest.
		Output: a FaissIndex or a Bm25Index.
		What it does: loads the index the build saved, and raises FileNotFoundError if it is not saved or holds
		different chunks. It never builds.
		"""
		index_type = self.index_args.index_type
		chunk_ids = []
		for chunk_id in chunks:
			chunk_ids.append(chunk_id)

		if index_type == constants.BM25_INDEX:
			retriever = load_bm25_index(index_folder, chunk_ids)
			if retriever is None:
				raise FileNotFoundError(
					f"No saved BM25 index holding these chunks in {index_folder}. Run scripts/build.sh first."
				)
			print(f"BM25 index loaded from cache : {len(chunk_ids)} chunks")
			return Bm25Index(retriever=retriever, index_path=os.path.join(index_folder, "bm25_retriever.pkl"))

		vector_store = load_faiss_index(index_folder, chunk_ids, self.embedding_model)
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

	def compute_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex | Bm25Index:
		"""
		Input: chunks, a dict of chunk_id to Chunk; embeddings, a dict of chunk_id to Embedded; output_path, the
		dataset's output folder.
		Output: a FaissIndex or a Bm25Index.
		What it does: calls the indexing method of the index type.
		"""
		index_type = self.index_args.index_type
		if index_type == constants.FAISS_FLATIP_INDEX:
			index = self.faiss_flatip_indexing(chunks, embeddings, output_path)
		elif index_type == constants.FAISS_IVF_INDEX:
			index = self.faiss_ivf_indexing(chunks, embeddings, output_path)
		elif index_type == constants.FAISS_HNSW_INDEX:
			index = self.faiss_hnsw_indexing(chunks, embeddings, output_path)
		else:
			index = self.bm25_indexing(chunks, output_path)

		return index

	def faiss_flatip_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder.
		Output: a FaissIndex holding an exact inner product (IndexFlatIP) vector store.
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
		faiss_index_path = paths.get_faiss_index_path(output_path, self.index_args)
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

		return FaissIndex(
			index_type=self.index_args.index_type,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)

	def faiss_ivf_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder.
		Output: a FaissIndex holding an IVF (IndexIVFFlat) vector store with ivf_nlist clusters.
		What it does: loads the saved index if it holds the same chunks, otherwise builds, wraps and saves a new one.
		"""
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		ivf_nlist = self.index_args.ivf_nlist
		if len(embeddings) < ivf_nlist:
			raise ValueError(
				f"ivf_nlist ({ivf_nlist}) must not be larger than the number of chunks ({len(embeddings)}). "
				"Use a smaller ivf_nlist or a larger dataset."
			)

		# 1. Collect the chunk ids and vectors in the same order. FAISS position i holds vectors[i] of chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		vectors = np.array(vectors, dtype=np.float32)
		dimension = vectors.shape[1]

		# 2. Load the saved index if it holds exactly the same chunks
		faiss_index_path = paths.get_faiss_index_path(output_path, self.index_args)
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

		return FaissIndex(
			index_type=self.index_args.index_type,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)

	def faiss_hnsw_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> FaissIndex:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder.
		Output: a FaissIndex holding an HNSW (IndexHNSWFlat) vector store.
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
		faiss_index_path = paths.get_faiss_index_path(output_path, self.index_args)
		print(f"Index folder : {faiss_index_path}")
		vector_store = load_faiss_index(faiss_index_path, chunk_ids, self.embedding_model)
		if vector_store is not None:
			print(f"FAISS hnsw index loaded from cache : {vector_store.index.ntotal} vectors")
		else:
			# 3. Build the index. Each vector is linked to hnsw_m close vectors, and a search walks this graph
			# towards the query, so it is fast but approximate
			hnsw_index = faiss.IndexHNSWFlat(dimension, self.index_args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
			hnsw_index.hnsw.efConstruction = self.index_args.hnsw_ef_construction
			hnsw_index.add(vectors)

			# 4. Wrap it in a LangChain FAISS vector store and save it
			vector_store = self.wrap_faiss_index(hnsw_index, chunk_ids, chunks)
			save_faiss_index(faiss_index_path, vector_store)
			print(f"FAISS hnsw index built : {hnsw_index.ntotal} vectors")

		return FaissIndex(
			index_type=self.index_args.index_type,
			vector_store=vector_store,
			index_path=faiss_index_path,
			dimension=dimension
		)

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
			documents_by_chunk_id[chunk_id] = Document(
				page_content=chunks[chunk_id].text,
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

	def bm25_indexing(self, chunks: dict[str, Chunk], output_path: str) -> Bm25Index:
		"""
		Input: chunks, a dict of chunk_id to Chunk; output_path, the dataset's output folder.
		Output: a Bm25Index holding a LangChain BM25Retriever.
		What it does: loads the saved retriever if it holds the same chunks, otherwise builds and saves a new one.
		"""
		if len(chunks) == 0:
			raise ValueError("chunks is empty. There is nothing to index.")

		# 1. Collect the chunk ids, texts and metadata in the same order
		chunk_ids = []
		chunk_texts = []
		chunk_metadatas = []
		for chunk_id in chunks:
			chunk_ids.append(chunk_id)
			chunk_texts.append(chunks[chunk_id].text)
			chunk_metadatas.append({"chunk_id": chunk_id})

		# 2. Load the saved retriever if it holds exactly the same chunks
		bm25_index_path = paths.get_bm25_index_path(output_path, self.index_args)
		print(f"Index folder : {bm25_index_path}")
		retriever = load_bm25_index(bm25_index_path, chunk_ids)
		if retriever is not None:
			print(f"BM25 index loaded from cache : {len(chunk_ids)} chunks")
		else:
			# 3. Build the retriever and save it
			retriever = BM25Retriever.from_texts(
				texts=chunk_texts,
				metadatas=chunk_metadatas,
				ids=chunk_ids,
				bm25_params={"k1": self.index_args.bm25_k1, "b": self.index_args.bm25_b},
				preprocess_func=lowercase_whitespace_tokenize
			)
			save_bm25_index(bm25_index_path, retriever)
			print(f"BM25 index built : {len(chunk_ids)} chunks")

		return Bm25Index(
			retriever=retriever,
			index_path=os.path.join(bm25_index_path, "bm25_retriever.pkl")
		)
