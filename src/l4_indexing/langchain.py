"""
langchain.py

Builds, saves and loads the three indexes of one indexing run with
LangChain's classes, used with --framework langchain:
FAISS (langchain_community.vectorstores.FAISS), BM25
(langchain_community.retrievers.BM25Retriever) and Qdrant
(langchain_qdrant.QdrantVectorStore).

Every index holds LangChain Documents with the chunk text as page_content
and {"chunk_id": chunk_id} as metadata. The vectors come from l3_embedding,
so no chunk is embedded again here. The HuggingFaceEmbeddings model is
loaded only because the FAISS and Qdrant vector stores need it to embed
queries at search time.
"""

import os
import pickle
import shutil

import numpy as np
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from common import paths
from common.args import ChunkingArgs
from common.args import IndexArgs
from common.constants import FAISS_FLATIP_INDEX, FAISS_IVF_INDEX, FAISS_HNSW_INDEX
from common.constants import QDRANT_URL

from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded
from l4_indexing.indexers.faiss import build_flatip_index, build_ivf_index, build_hnsw_index
from l4_indexing.schema import LangchainFaissIndex
from l4_indexing.schema import LangchainBm25Index
from l4_indexing.schema import LangchainQdrantIndex

def lowercase_whitespace_tokenize(text: str) -> list[str]:
	"""
	Input: text, one chunk or query text.
	Output: the list of lowercase words in text.
	What it does: lowercases text and splits it on spaces, tabs and new
	lines, the same as the built-in BM25 index. It is a named function,
	not written inline, because BM25Retriever keeps it and is saved with
	pickle, which cannot save an inline lambda.
	"""
	return text.lower().split()

class Langchain:
	def __init__(self, index_args: IndexArgs, chunking_args: ChunkingArgs):
		"""
		Input: index_args, the indexing settings; chunking_args, the chunking
		settings with framework "langchain".
		Output: nothing.
		What it does: stores both settings and loads LangChain's
		HuggingFaceEmbeddings with the model named by model_name, normalized
		like l3_embedding, on the GPU if there is one.
		"""
		self.index_args = index_args
		self.chunking_args = chunking_args

		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = HuggingFaceEmbeddings(
			model_name=self.chunking_args.model_name,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)

	def compute_index(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> tuple[LangchainFaissIndex, LangchainBm25Index, LangchainQdrantIndex]:
		"""
		Input: chunks, a dict of chunk_id to Chunk from l2_chunking;
		embeddings, a dict of chunk_id to Embedded from l3_embedding;
		output_path, the dataset's output folder the index folders are built under.
		Output: a tuple of (LangchainFaissIndex, LangchainBm25Index, LangchainQdrantIndex).
		What it does: gets the LangChain index folders and Qdrant collection
		name, then builds or loads all three indexes from the same chunks.
		"""
		faiss_index_path = paths.get_faiss_index_path(output_path, self.chunking_args, self.index_args)
		bm25_index_path = paths.get_bm25_index_path(output_path, self.chunking_args, self.index_args)
		qdrant_collection_name = paths.get_qdrant_collection_name(output_path, self.chunking_args)

		faiss_index = self.get_faiss_index(faiss_index_path, chunks, embeddings)
		bm25_index = self.get_bm25_index(bm25_index_path, chunks)
		qdrant_index = self.get_qdrant_index(qdrant_collection_name, chunks, embeddings)

		return faiss_index, bm25_index, qdrant_index

	def get_faiss_index(self, faiss_index_path: str, chunks: dict[str, Chunk], embeddings: dict[str, Embedded]) -> LangchainFaissIndex:
		"""
		Input: faiss_index_path, the LangChain FAISS index folder; chunks, a
		dict of chunk_id to Chunk; embeddings, a dict of chunk_id to Embedded.
		Output: a LangchainFaissIndex holding a LangChain FAISS vector store.
		What it does: loads the saved vector store from faiss_index_path if
		it holds exactly the same chunk_ids as embeddings. Otherwise builds a
		FAISS index of type faiss_index_type with the same build functions
		as the built-in FAISS index, wraps it in a LangChain FAISS vector
		store with one Document per chunk, saves it with save_local, and
		returns it. The vectors are normalized, so the vector store uses
		inner product, which is cosine similarity.
		"""
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# 1. Collect the chunk ids and vectors in the same order
		# FAISS position i will hold vectors[i], which belongs to chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		vectors = np.array(vectors, dtype=np.float32)
		dimension = vectors.shape[1]

		# 2. Load the saved vector store if it holds exactly the same chunks
		# load_local reads index.pkl with pickle; the file was written by
		# this code, so allow_dangerous_deserialization is safe here
		index_file_path = os.path.join(faiss_index_path, "index.faiss")
		docstore_file_path = os.path.join(faiss_index_path, "index.pkl")
		if os.path.exists(index_file_path) and os.path.exists(docstore_file_path):
			saved_vector_store = FAISS.load_local(
				faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True
			)
			saved_chunk_ids = []
			for position in saved_vector_store.index_to_docstore_id:
				saved_chunk_ids.append(saved_vector_store.index_to_docstore_id[position])

			if set(saved_chunk_ids) == set(chunk_ids):
				print(f"LangChain FAISS index loaded from cache : {saved_vector_store.index.ntotal} vectors")
				return LangchainFaissIndex(
					faiss_index_type=self.index_args.faiss_index_type,
					vector_store=saved_vector_store,
					index_folder_path=faiss_index_path,
					dimension=dimension
				)

		# 3. Build a new FAISS index of the requested type
		if self.index_args.faiss_index_type == FAISS_FLATIP_INDEX:
			new_faiss_index = build_flatip_index(vectors, dimension)
		elif self.index_args.faiss_index_type == FAISS_IVF_INDEX:
			new_faiss_index = build_ivf_index(vectors, dimension, self.index_args.ivf_nlist)
		elif self.index_args.faiss_index_type == FAISS_HNSW_INDEX:
			new_faiss_index = build_hnsw_index(vectors, dimension, self.index_args.hnsw_m, self.index_args.hnsw_ef_construction)
		else:
			raise ValueError(f"faiss_index_type ({self.index_args.faiss_index_type}) is not supported")

		# 4. Wrap it in a LangChain FAISS vector store
		# The docstore maps each chunk_id to its Document, and
		# index_to_docstore_id maps each FAISS position to its chunk_id
		documents_by_chunk_id = {}
		index_to_docstore_id = {}
		for position in range(len(chunk_ids)):
			chunk_id = chunk_ids[position]
			documents_by_chunk_id[chunk_id] = Document(
				page_content=chunks[chunk_id].text,
				metadata={"chunk_id": chunk_id}
			)
			index_to_docstore_id[position] = chunk_id

		new_vector_store = FAISS(
			embedding_function=self.embedding_model,
			index=new_faiss_index,
			docstore=InMemoryDocstore(documents_by_chunk_id),
			index_to_docstore_id=index_to_docstore_id,
			distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
		)
		print(f"LangChain FAISS index built : {new_faiss_index.ntotal} vectors")

		# 5. Save the vector store
		# save_local writes index.faiss and index.pkl. They are written to a
		# temporary folder first and each file is then renamed into place,
		# so a crash mid-write never leaves a half-written file.
		temporary_folder_path = faiss_index_path.rstrip("/") + ".tmp"
		if os.path.exists(temporary_folder_path):
			shutil.rmtree(temporary_folder_path)
		new_vector_store.save_local(temporary_folder_path)
		os.makedirs(faiss_index_path, exist_ok=True)
		os.replace(os.path.join(temporary_folder_path, "index.faiss"), index_file_path)
		os.replace(os.path.join(temporary_folder_path, "index.pkl"), docstore_file_path)
		os.rmdir(temporary_folder_path)

		return LangchainFaissIndex(
			faiss_index_type=self.index_args.faiss_index_type,
			vector_store=new_vector_store,
			index_folder_path=faiss_index_path,
			dimension=dimension
		)

	def get_bm25_index(self, bm25_index_path: str, chunks: dict[str, Chunk]) -> LangchainBm25Index:
		"""
		Input: bm25_index_path, the LangChain BM25 index folder; chunks, a
		dict of chunk_id to Chunk.
		Output: a LangchainBm25Index holding a LangChain BM25Retriever.
		What it does: loads the saved retriever from bm25_index_path if it
		holds exactly the same chunk_ids as chunks. Otherwise builds a
		BM25Retriever over the chunk texts with bm25_k1 and bm25_b, splitting
		text with lowercase_whitespace_tokenize, saves it with pickle, and
		returns it.
		"""
		if len(chunks) == 0:
			raise ValueError("chunks is empty. There is nothing to index.")

		if self.index_args.bm25_tokenizer != "whitespace":
			raise ValueError(
				f"bm25_tokenizer ({self.index_args.bm25_tokenizer}) is not supported. Use \"whitespace\"."
			)

		retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")

		# 1. Collect the chunk ids and texts in the same order
		chunk_ids = []
		chunk_texts = []
		chunk_metadatas = []
		for chunk_id in chunks:
			chunk_ids.append(chunk_id)
			chunk_texts.append(chunks[chunk_id].text)
			chunk_metadatas.append({"chunk_id": chunk_id})

		# 2. Load the saved retriever if it holds exactly the same chunks
		if os.path.exists(retriever_file_path):
			with open(retriever_file_path, "rb") as f:
				saved_retriever = pickle.load(f)

			saved_chunk_ids = []
			for document in saved_retriever.docs:
				saved_chunk_ids.append(document.metadata["chunk_id"])

			if set(saved_chunk_ids) == set(chunk_ids):
				print(f"LangChain BM25 index loaded from cache : {len(saved_chunk_ids)} chunks")
				return LangchainBm25Index(
					retriever=saved_retriever,
					index_file_path=retriever_file_path
				)

		# 3. Build the BM25 retriever
		new_retriever = BM25Retriever.from_texts(
			texts=chunk_texts,
			metadatas=chunk_metadatas,
			ids=chunk_ids,
			bm25_params={"k1": self.index_args.bm25_k1, "b": self.index_args.bm25_b},
			preprocess_func=lowercase_whitespace_tokenize
		)
		print(f"LangChain BM25 index built : {len(chunk_ids)} chunks")

		# 4. Save the retriever
		os.makedirs(bm25_index_path, exist_ok=True)
		temporary_retriever_file = retriever_file_path + ".tmp"
		with open(temporary_retriever_file, "wb") as f:
			pickle.dump(new_retriever, f)
		os.replace(temporary_retriever_file, retriever_file_path)

		return LangchainBm25Index(
			retriever=new_retriever,
			index_file_path=retriever_file_path
		)

	def get_qdrant_index(self, qdrant_collection_name: str, chunks: dict[str, Chunk], embeddings: dict[str, Embedded]) -> LangchainQdrantIndex:
		"""
		Input: qdrant_collection_name, the collection to use on the Qdrant
		server; chunks, a dict of chunk_id to Chunk; embeddings, a dict of
		chunk_id to Embedded.
		Output: a LangchainQdrantIndex holding a LangChain QdrantVectorStore.
		What it does: connects to the Qdrant server at QDRANT_URL. If the
		collection already exists and holds exactly the same chunk_ids as
		embeddings, wraps it as is. Otherwise deletes the old collection if
		there is one, creates a new one with cosine distance, and uploads the
		l3 vectors in batches of qdrant_batch_size with the payload layout
		QdrantVectorStore reads: {"page_content": text, "metadata":
		{"chunk_id": chunk_id}}. The points are uploaded directly, not with
		QdrantVectorStore.add_texts, because add_texts would embed every
		chunk again. Raises ConnectionError if the server cannot be reached.
		"""
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# 1. Collect the chunk ids and vectors in the same order
		# Point id i will hold vectors[i], which belongs to chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		dimension = len(vectors[0])

		# 2. Connect to the server
		# QdrantClient does not connect until the first request, so the first
		# request is made here to give a clear error when the server is down
		qdrant_client = QdrantClient(url=QDRANT_URL)
		try:
			collection_exists = qdrant_client.collection_exists(qdrant_collection_name)
		except Exception as error:
			raise ConnectionError(
				f"Qdrant server is not reachable at {QDRANT_URL}. "
				"Start it with the qdrant-docker skill."
			) from error

		# 3. Use the existing collection if it holds exactly the same chunks
		# The saved chunk ids are read page by page from the point payloads
		if collection_exists:
			saved_chunk_ids = []
			next_page_offset = None
			while True:
				points, next_page_offset = qdrant_client.scroll(
					collection_name=qdrant_collection_name,
					limit=10000,
					offset=next_page_offset,
					with_payload=["metadata"],
					with_vectors=False
				)
				for point in points:
					saved_chunk_ids.append(point.payload["metadata"]["chunk_id"])
				if next_page_offset is None:
					break

			if len(saved_chunk_ids) == len(chunk_ids) and set(saved_chunk_ids) == set(chunk_ids):
				print(f"LangChain Qdrant collection loaded from cache : {len(saved_chunk_ids)} points")
				saved_vector_store = QdrantVectorStore(
					client=qdrant_client,
					collection_name=qdrant_collection_name,
					embedding=self.embedding_model,
					distance=Distance.COSINE
				)
				return LangchainQdrantIndex(
					vector_store=saved_vector_store,
					collection_name=qdrant_collection_name,
					dimension=dimension
				)

			# The collection holds different chunks, so it is rebuilt
			qdrant_client.delete_collection(qdrant_collection_name)

		# 4. Create the collection
		qdrant_client.create_collection(
			collection_name=qdrant_collection_name,
			vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
		)

		# 5. Upload the points in batches of qdrant_batch_size
		batch_size = self.index_args.qdrant_batch_size
		for batch_start in range(0, len(chunk_ids), batch_size):
			batch_end = batch_start + batch_size
			if batch_end > len(chunk_ids):
				batch_end = len(chunk_ids)

			batch_points = []
			for position in range(batch_start, batch_end):
				chunk_id = chunk_ids[position]
				point = PointStruct(
					id=position,
					vector=vectors[position].tolist(),
					payload={"page_content": chunks[chunk_id].text, "metadata": {"chunk_id": chunk_id}}
				)
				batch_points.append(point)

			qdrant_client.upsert(collection_name=qdrant_collection_name, points=batch_points, wait=True)
		print(f"LangChain Qdrant collection built : {len(chunk_ids)} points")

		# 6. Wrap the collection in a LangChain QdrantVectorStore
		new_vector_store = QdrantVectorStore(
			client=qdrant_client,
			collection_name=qdrant_collection_name,
			embedding=self.embedding_model,
			distance=Distance.COSINE
		)

		return LangchainQdrantIndex(
			vector_store=new_vector_store,
			collection_name=qdrant_collection_name,
			dimension=dimension
		)
