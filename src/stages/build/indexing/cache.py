"""
cache.py

Saves and loads the indexes of one indexing run, the same way
stages/build/chunking/cache.py and stages/build/embedding/cache.py do for chunks and
embeddings. The indexers in indexers/ and langchain.py only build; they
call this file to check the cache and to save.

Built-in indexes:
FAISS, in the FAISS index folder (see common.paths.get_faiss_index_path):
index.faiss and chunk_ids.json, the chunk_id of each FAISS position.
BM25, in the BM25 index folder (see common.paths.get_bm25_index_path):
bm25.pkl and chunk_ids.json, the chunk_id of each BM25 position.
Qdrant, a collection on the Qdrant server at QDRANT_URL, named by
common.paths.get_qdrant_collection_name, with {"chunk_id", "text"} in
each point's payload.

LangChain indexes (--framework langchain), in the same folders under
langchain/:
FAISS, index.faiss and index.pkl written by FAISS.save_local.
BM25, bm25_retriever.pkl, the BM25Retriever saved with pickle.
Qdrant, a collection with {"page_content", "metadata": {"chunk_id"}} in
each point's payload.

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file. The
_load_cached_* functions are used while building: they return the saved
index if it holds exactly the same chunk ids, otherwise None. The load_*
functions are load-only, for the eval phase: they never build, and raise
an error if an index is missing.
"""

import json
import os
import pickle
import shutil

import faiss
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from common import constants
from common import paths
from common.args import ChunkingArgs
from common.args import IndexArgs
from common.constants import QDRANT_URL
from stages.build.indexing import validate
from stages.build.indexing.schema import FaissIndex
from stages.build.indexing.schema import Bm25Index
from stages.build.indexing.schema import QdrantIndex
from stages.build.indexing.schema import LangchainFaissIndex
from stages.build.indexing.schema import LangchainBm25Index
from stages.build.indexing.schema import LangchainQdrantIndex

# ------ Qdrant connection
def _connect_qdrant_client() -> QdrantClient:
	"""
	Input: none.
	Output: a QdrantClient connected to the Qdrant server at QDRANT_URL.
	What it does: creates the client and makes one request to it, because
	QdrantClient does not connect until the first request. Raises
	ConnectionError with a clear message if the server cannot be reached.
	"""
	qdrant_client = QdrantClient(url=QDRANT_URL)
	try:
		qdrant_client.get_collections()
	except Exception as error:
		raise ConnectionError(
			f"Qdrant server is not reachable at {QDRANT_URL}. "
			"Start it with the qdrant-docker skill."
		) from error
	return qdrant_client

# ------ Built-in FAISS
def _load_cached_faiss_index(faiss_index_path: str, faiss_index_type: str, chunk_ids: list[str]) -> FaissIndex | None:
	"""
	Input: faiss_index_path, the FAISS index folder; faiss_index_type, the
	FAISS type of this run; chunk_ids, the chunk ids the index must hold.
	Output: a FaissIndex holding the saved index, or None if there is no
	saved index or it holds different chunk ids.
	What it does: reads chunk_ids.json and, if it holds exactly the same
	chunk ids, reads index.faiss. The saved chunk_ids are returned, not the
	new ones, because they match the positions in the saved index.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	chunk_ids_file_path = os.path.join(faiss_index_path, "chunk_ids.json")
	if not os.path.exists(index_file_path) or not os.path.exists(chunk_ids_file_path):
		return None

	with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
		saved_chunk_ids = json.load(f)
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	saved_index = faiss.read_index(index_file_path)
	return FaissIndex(
		faiss_index_type=faiss_index_type,
		faiss_index=saved_index,
		chunk_ids=saved_chunk_ids,
		index_file_path=index_file_path,
		dimension=saved_index.d
	)

def _save_faiss_index(faiss_index_path: str, faiss_index: faiss.Index, chunk_ids: list[str]) -> None:
	"""
	Input: faiss_index_path, the FAISS index folder; faiss_index, the built
	FAISS index; chunk_ids, the chunk_id of each FAISS position, in order.
	Output: nothing.
	What it does: writes index.faiss and chunk_ids.json to
	faiss_index_path, creating the folder first if needed. Each file is
	written to a temporary file and then renamed into place.
	"""
	os.makedirs(faiss_index_path, exist_ok=True)
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	chunk_ids_file_path = os.path.join(faiss_index_path, "chunk_ids.json")

	temporary_index_file = index_file_path + ".tmp"
	faiss.write_index(faiss_index, temporary_index_file)
	os.replace(temporary_index_file, index_file_path)

	temporary_chunk_ids_file = chunk_ids_file_path + ".tmp"
	with open(temporary_chunk_ids_file, "w", encoding="utf-8") as f:
		json.dump(chunk_ids, f)
	os.replace(temporary_chunk_ids_file, chunk_ids_file_path)

def load_faiss_index(faiss_index_path: str, index_args: IndexArgs, dataset_size: int) -> FaissIndex:
	"""
	Input: faiss_index_path, the FAISS index folder; index_args, the
	indexing parameters; dataset_size, the number of documents the index
	must hold.
	Output: a FaissIndex holding the saved FAISS index and its chunk ids.
	What it does: load-only, never builds. Reads index.faiss and
	chunk_ids.json from faiss_index_path. Raises FileNotFoundError if
	either file is missing, and ValueError if the chunk ids do not come
	from exactly dataset_size documents, because then the index was built
	with a different dataset size.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	chunk_ids_file_path = os.path.join(faiss_index_path, "chunk_ids.json")
	if not os.path.exists(index_file_path) or not os.path.exists(chunk_ids_file_path):
		raise FileNotFoundError(
			f"No FAISS index at {faiss_index_path}. Run scripts/run.sh with the same chunking and index args first."
		)

	with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
		saved_chunk_ids = json.load(f)

	# A chunk id is "{doc_id}_{n}", so the document id is the part before the last "_"
	saved_doc_ids = set()
	for chunk_id in saved_chunk_ids:
		saved_doc_ids.add(chunk_id.rsplit("_", 1)[0])
	if len(saved_doc_ids) != dataset_size:
		raise ValueError(
			f"The FAISS index at {faiss_index_path} holds {len(saved_doc_ids)} documents, "
			f"but dataset_size is {dataset_size}. Run scripts/run.sh with --dataset_size {dataset_size} first."
		)

	saved_index = faiss.read_index(index_file_path)
	print(f"FAISS index loaded : {saved_index.ntotal} vectors")
	return FaissIndex(
		faiss_index_type=index_args.index_type,
		faiss_index=saved_index,
		chunk_ids=saved_chunk_ids,
		index_file_path=index_file_path,
		dimension=saved_index.d
	)

# ------ Built-in BM25
def _load_cached_bm25_index(bm25_index_path: str, chunk_ids: list[str]) -> Bm25Index | None:
	"""
	Input: bm25_index_path, the BM25 index folder; chunk_ids, the chunk ids
	the index must hold.
	Output: a Bm25Index holding the saved model, or None if there is no
	saved model or it holds different chunk ids.
	What it does: reads chunk_ids.json and, if it holds exactly the same
	chunk ids, reads bm25.pkl. The saved chunk_ids are returned, because
	they match the positions in the saved model.
	"""
	model_file_path = os.path.join(bm25_index_path, "bm25.pkl")
	chunk_ids_file_path = os.path.join(bm25_index_path, "chunk_ids.json")
	if not os.path.exists(model_file_path) or not os.path.exists(chunk_ids_file_path):
		return None

	with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
		saved_chunk_ids = json.load(f)
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	with open(model_file_path, "rb") as f:
		saved_model = pickle.load(f)
	return Bm25Index(
		bm25_model=saved_model,
		chunk_ids=saved_chunk_ids,
		index_file_path=model_file_path
	)

def _save_bm25_index(bm25_index_path: str, bm25_model, chunk_ids: list[str]) -> None:
	"""
	Input: bm25_index_path, the BM25 index folder; bm25_model, the built
	BM25Okapi model; chunk_ids, the chunk_id of each BM25 position, in order.
	Output: nothing.
	What it does: writes bm25.pkl (with pickle) and chunk_ids.json to
	bm25_index_path, creating the folder first if needed. Each file is
	written to a temporary file and then renamed into place.
	"""
	os.makedirs(bm25_index_path, exist_ok=True)
	model_file_path = os.path.join(bm25_index_path, "bm25.pkl")
	chunk_ids_file_path = os.path.join(bm25_index_path, "chunk_ids.json")

	temporary_model_file = model_file_path + ".tmp"
	with open(temporary_model_file, "wb") as f:
		pickle.dump(bm25_model, f)
	os.replace(temporary_model_file, model_file_path)

	temporary_chunk_ids_file = chunk_ids_file_path + ".tmp"
	with open(temporary_chunk_ids_file, "w", encoding="utf-8") as f:
		json.dump(chunk_ids, f)
	os.replace(temporary_chunk_ids_file, chunk_ids_file_path)

def load_bm25_index(bm25_index_path: str) -> Bm25Index:
	"""
	Input: bm25_index_path, the BM25 index folder.
	Output: a Bm25Index holding the saved BM25Okapi model and its chunk ids.
	What it does: load-only, never builds. Reads bm25.pkl and
	chunk_ids.json from bm25_index_path. Raises FileNotFoundError if
	either file is missing.
	"""
	model_file_path = os.path.join(bm25_index_path, "bm25.pkl")
	chunk_ids_file_path = os.path.join(bm25_index_path, "chunk_ids.json")
	if not os.path.exists(model_file_path) or not os.path.exists(chunk_ids_file_path):
		raise FileNotFoundError(
			f"No BM25 index at {bm25_index_path}. Run scripts/run.sh with the same chunking and index args first."
		)

	with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
		saved_chunk_ids = json.load(f)
	with open(model_file_path, "rb") as f:
		saved_model = pickle.load(f)

	print(f"BM25 index loaded : {len(saved_chunk_ids)} chunks")
	return Bm25Index(
		bm25_model=saved_model,
		chunk_ids=saved_chunk_ids,
		index_file_path=model_file_path
	)

# ------ Built-in Qdrant
def _load_cached_qdrant_index(qdrant_client: QdrantClient, qdrant_collection_name: str, chunk_ids: list[str]) -> QdrantIndex | None:
	"""
	Input: qdrant_client, a client connected to the Qdrant server;
	qdrant_collection_name, the collection to check; chunk_ids, the chunk
	ids the collection must hold.
	Output: a QdrantIndex for the collection, or None if the collection
	does not exist, holds different chunk ids, or has no chunk text in its
	payload.
	What it does: reads the chunk ids of every point page by page from the
	payloads and compares them with chunk_ids. A collection built before
	the text was added to the payload has no "text"; one point is enough to
	check it, because every point is uploaded with the same payload keys.
	"""
	if not qdrant_client.collection_exists(qdrant_collection_name):
		return None

	saved_chunk_ids = []
	next_page_offset = None
	while True:
		points, next_page_offset = qdrant_client.scroll(
			collection_name=qdrant_collection_name,
			limit=10000,
			offset=next_page_offset,
			with_payload=["chunk_id"],
			with_vectors=False
		)
		for point in points:
			saved_chunk_ids.append(point.payload["chunk_id"])
		if next_page_offset is None:
			break

	first_points, first_page_offset = qdrant_client.scroll(
		collection_name=qdrant_collection_name,
		limit=1,
		with_payload=True,
		with_vectors=False
	)
	saved_payload_has_text = len(first_points) > 0 and "text" in first_points[0].payload

	if not saved_payload_has_text or len(saved_chunk_ids) != len(chunk_ids) or set(saved_chunk_ids) != set(chunk_ids):
		return None

	collection_info = qdrant_client.get_collection(qdrant_collection_name)
	return QdrantIndex(
		qdrant_client=qdrant_client,
		collection_name=qdrant_collection_name,
		dimension=collection_info.config.params.vectors.size
	)

def load_qdrant_index(qdrant_collection_name: str) -> QdrantIndex:
	"""
	Input: qdrant_collection_name, the collection to open on the Qdrant server.
	Output: a QdrantIndex holding the client connected to the server and
	the collection name.
	What it does: load-only, never builds. Connects to the Qdrant server
	and checks that the collection exists and stores the chunk text in its
	payload. Raises ConnectionError if the server cannot be reached, and
	FileNotFoundError if the collection is missing or has no text.
	"""
	qdrant_client = _connect_qdrant_client()
	if not qdrant_client.collection_exists(qdrant_collection_name):
		raise FileNotFoundError(
			f"No Qdrant collection {qdrant_collection_name}. Run scripts/run.sh with the same chunking and index args first."
		)

	# Every point is uploaded with the same payload keys, so one point shows
	# whether the collection stores the chunk text
	first_points, first_page_offset = qdrant_client.scroll(
		collection_name=qdrant_collection_name,
		limit=1,
		with_payload=True,
		with_vectors=False
	)
	if len(first_points) == 0 or "text" not in first_points[0].payload:
		raise FileNotFoundError(
			f"Qdrant collection {qdrant_collection_name} is empty or has no chunk text. "
			"Run scripts/run.sh with the same chunking and index args to rebuild it."
		)

	collection_info = qdrant_client.get_collection(qdrant_collection_name)
	point_count = qdrant_client.count(qdrant_collection_name).count
	print(f"Qdrant collection loaded : {point_count} points")
	return QdrantIndex(
		qdrant_client=qdrant_client,
		collection_name=qdrant_collection_name,
		dimension=collection_info.config.params.vectors.size
	)

# ------ LangChain FAISS
def _load_cached_langchain_faiss_index(faiss_index_path: str, faiss_index_type: str, chunk_ids: list[str], embedding_model: HuggingFaceEmbeddings) -> LangchainFaissIndex | None:
	"""
	Input: faiss_index_path, the LangChain FAISS index folder;
	faiss_index_type, the FAISS type of this run; chunk_ids, the chunk ids
	the index must hold; embedding_model, the model the vector store uses
	to embed queries.
	Output: a LangchainFaissIndex holding the saved vector store, or None
	if there is no saved vector store or it holds different chunk ids.
	What it does: reads index.faiss and index.pkl with FAISS.load_local and
	compares the chunk ids in index_to_docstore_id with chunk_ids.
	load_local reads index.pkl with pickle; the file was written by this
	code, so allow_dangerous_deserialization is safe here.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	docstore_file_path = os.path.join(faiss_index_path, "index.pkl")
	if not os.path.exists(index_file_path) or not os.path.exists(docstore_file_path):
		return None

	saved_vector_store = FAISS.load_local(
		faiss_index_path, embedding_model, allow_dangerous_deserialization=True
	)
	saved_chunk_ids = []
	for position in saved_vector_store.index_to_docstore_id:
		saved_chunk_ids.append(saved_vector_store.index_to_docstore_id[position])
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	return LangchainFaissIndex(
		faiss_index_type=faiss_index_type,
		vector_store=saved_vector_store,
		index_folder_path=faiss_index_path,
		dimension=saved_vector_store.index.d
	)

def _save_langchain_faiss_index(faiss_index_path: str, vector_store: FAISS) -> None:
	"""
	Input: faiss_index_path, the LangChain FAISS index folder; vector_store,
	the built LangChain FAISS vector store.
	Output: nothing.
	What it does: save_local writes index.faiss and index.pkl. They are
	written to a temporary folder first and each file is then renamed into
	place, so a crash mid-write never leaves a half-written file.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	docstore_file_path = os.path.join(faiss_index_path, "index.pkl")

	temporary_folder_path = faiss_index_path.rstrip("/") + ".tmp"
	if os.path.exists(temporary_folder_path):
		shutil.rmtree(temporary_folder_path)
	vector_store.save_local(temporary_folder_path)
	os.makedirs(faiss_index_path, exist_ok=True)
	os.replace(os.path.join(temporary_folder_path, "index.faiss"), index_file_path)
	os.replace(os.path.join(temporary_folder_path, "index.pkl"), docstore_file_path)
	os.rmdir(temporary_folder_path)

def load_langchain_faiss_index(faiss_index_path: str, index_args: IndexArgs, dataset_size: int, embedding_model: HuggingFaceEmbeddings) -> LangchainFaissIndex:
	"""
	Input: faiss_index_path, the LangChain FAISS index folder; index_args,
	the indexing parameters; dataset_size, the number of documents the
	index must hold; embedding_model, the model the vector store uses to
	embed queries.
	Output: a LangchainFaissIndex holding the saved FAISS vector store.
	What it does: load-only, never builds. Reads index.faiss and
	index.pkl with FAISS.load_local. Raises FileNotFoundError if either
	file is missing, and ValueError if the chunk ids do not come from
	exactly dataset_size documents.
	"""
	index_file_path = os.path.join(faiss_index_path, "index.faiss")
	docstore_file_path = os.path.join(faiss_index_path, "index.pkl")
	if not os.path.exists(index_file_path) or not os.path.exists(docstore_file_path):
		raise FileNotFoundError(
			f"No LangChain FAISS index at {faiss_index_path}. Run scripts/run.sh with the same chunking and index args first."
		)

	# load_local reads index.pkl with pickle; the file was written by
	# this code, so allow_dangerous_deserialization is safe here
	saved_vector_store = FAISS.load_local(
		faiss_index_path, embedding_model, allow_dangerous_deserialization=True
	)

	# A chunk id is "{doc_id}_{n}", so the document id is the part before the last "_"
	saved_doc_ids = set()
	for position in saved_vector_store.index_to_docstore_id:
		chunk_id = saved_vector_store.index_to_docstore_id[position]
		saved_doc_ids.add(chunk_id.rsplit("_", 1)[0])
	if len(saved_doc_ids) != dataset_size:
		raise ValueError(
			f"The LangChain FAISS index at {faiss_index_path} holds {len(saved_doc_ids)} documents, "
			f"but dataset_size is {dataset_size}. Run scripts/run.sh with --dataset_size {dataset_size} first."
		)

	print(f"LangChain FAISS index loaded : {saved_vector_store.index.ntotal} vectors")
	return LangchainFaissIndex(
		faiss_index_type=index_args.index_type,
		vector_store=saved_vector_store,
		index_folder_path=faiss_index_path,
		dimension=saved_vector_store.index.d
	)

# ------ LangChain BM25
def _load_cached_langchain_bm25_index(bm25_index_path: str, chunk_ids: list[str]) -> LangchainBm25Index | None:
	"""
	Input: bm25_index_path, the LangChain BM25 index folder; chunk_ids, the
	chunk ids the retriever must hold.
	Output: a LangchainBm25Index holding the saved retriever, or None if
	there is no saved retriever or it holds different chunk ids.
	What it does: reads bm25_retriever.pkl and compares the chunk ids in
	its documents' metadata with chunk_ids.
	"""
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	if not os.path.exists(retriever_file_path):
		return None

	with open(retriever_file_path, "rb") as f:
		saved_retriever = pickle.load(f)
	saved_chunk_ids = []
	for document in saved_retriever.docs:
		saved_chunk_ids.append(document.metadata["chunk_id"])
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	return LangchainBm25Index(
		retriever=saved_retriever,
		index_file_path=retriever_file_path
	)

def _save_langchain_bm25_index(bm25_index_path: str, retriever) -> None:
	"""
	Input: bm25_index_path, the LangChain BM25 index folder; retriever, the
	built BM25Retriever.
	Output: nothing.
	What it does: writes the retriever with pickle to bm25_retriever.pkl,
	creating the folder first if needed. The file is written to a
	temporary file and then renamed into place.
	"""
	os.makedirs(bm25_index_path, exist_ok=True)
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	temporary_retriever_file = retriever_file_path + ".tmp"
	with open(temporary_retriever_file, "wb") as f:
		pickle.dump(retriever, f)
	os.replace(temporary_retriever_file, retriever_file_path)

def load_langchain_bm25_index(bm25_index_path: str) -> LangchainBm25Index:
	"""
	Input: bm25_index_path, the LangChain BM25 index folder.
	Output: a LangchainBm25Index holding the saved BM25Retriever.
	What it does: load-only, never builds. Reads bm25_retriever.pkl.
	Raises FileNotFoundError if it is missing.
	"""
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	if not os.path.exists(retriever_file_path):
		raise FileNotFoundError(
			f"No LangChain BM25 index at {bm25_index_path}. Run scripts/run.sh with the same chunking and index args first."
		)

	with open(retriever_file_path, "rb") as f:
		saved_retriever = pickle.load(f)

	print(f"LangChain BM25 index loaded : {len(saved_retriever.docs)} chunks")
	return LangchainBm25Index(
		retriever=saved_retriever,
		index_file_path=retriever_file_path
	)

# ------ LangChain Qdrant
def _load_cached_langchain_qdrant_index(qdrant_client: QdrantClient, qdrant_collection_name: str, chunk_ids: list[str], embedding_model: HuggingFaceEmbeddings) -> LangchainQdrantIndex | None:
	"""
	Input: qdrant_client, a client connected to the Qdrant server;
	qdrant_collection_name, the collection to check; chunk_ids, the chunk
	ids the collection must hold; embedding_model, the model the vector
	store uses to embed queries.
	Output: a LangchainQdrantIndex wrapping the collection in a
	QdrantVectorStore, or None if the collection does not exist or holds
	different chunk ids.
	What it does: reads the chunk ids of every point page by page from the
	"metadata" payloads and compares them with chunk_ids.
	"""
	if not qdrant_client.collection_exists(qdrant_collection_name):
		return None

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

	if len(saved_chunk_ids) != len(chunk_ids) or set(saved_chunk_ids) != set(chunk_ids):
		return None

	collection_info = qdrant_client.get_collection(qdrant_collection_name)
	saved_vector_store = QdrantVectorStore(
		client=qdrant_client,
		collection_name=qdrant_collection_name,
		embedding=embedding_model,
		distance=Distance.COSINE
	)
	return LangchainQdrantIndex(
		vector_store=saved_vector_store,
		collection_name=qdrant_collection_name,
		dimension=collection_info.config.params.vectors.size
	)

def load_langchain_qdrant_index(qdrant_collection_name: str, embedding_model: HuggingFaceEmbeddings) -> LangchainQdrantIndex:
	"""
	Input: qdrant_collection_name, the collection to open on the Qdrant
	server; embedding_model, the model the vector store uses to embed queries.
	Output: a LangchainQdrantIndex holding a QdrantVectorStore over the collection.
	What it does: load-only, never builds. Connects to the Qdrant server
	and wraps the collection in a QdrantVectorStore. Raises
	ConnectionError if the server cannot be reached, and FileNotFoundError
	if the collection does not exist.
	"""
	qdrant_client = _connect_qdrant_client()
	if not qdrant_client.collection_exists(qdrant_collection_name):
		raise FileNotFoundError(
			f"No Qdrant collection {qdrant_collection_name}. Run scripts/run.sh with the same chunking and index args first."
		)

	collection_info = qdrant_client.get_collection(qdrant_collection_name)
	point_count = qdrant_client.count(qdrant_collection_name).count
	saved_vector_store = QdrantVectorStore(
		client=qdrant_client,
		collection_name=qdrant_collection_name,
		embedding=embedding_model,
		distance=Distance.COSINE
	)

	print(f"LangChain Qdrant collection loaded : {point_count} points")
	return LangchainQdrantIndex(
		vector_store=saved_vector_store,
		collection_name=qdrant_collection_name,
		dimension=collection_info.config.params.vectors.size
	)

# ------ Load one index (eval phase)
def load_index(output_path: str, chunking_args: ChunkingArgs, index_args: IndexArgs, dataset_size: int) -> FaissIndex | Bm25Index | QdrantIndex | LangchainFaissIndex | LangchainBm25Index | LangchainQdrantIndex:
	"""
	Input: output_path, the dataset's output folder; chunking_args and index_args, the settings the index was built
	with; dataset_size, the number of documents a FAISS index must hold.
	Output: the saved index of index_type. For framework "langchain", the LangChain index.
	What it does: load-only, never builds. Raises an error if the index is missing.
	"""
	validate.validate_index_args(index_args)

	if chunking_args.framework == constants.LANGCHAIN_FRAMEWORK:
		return load_langchain_index(output_path, chunking_args, index_args, dataset_size)

	if index_args.index_type == constants.BM25_INDEX:
		bm25_index_path = paths.get_bm25_index_path(output_path, chunking_args, index_args)
		return load_bm25_index(bm25_index_path)

	if index_args.index_type == constants.QDRANT_INDEX:
		qdrant_collection_name = paths.get_qdrant_collection_name(output_path, chunking_args)
		return load_qdrant_index(qdrant_collection_name)

	faiss_index_path = paths.get_faiss_index_path(output_path, chunking_args, index_args)
	return load_faiss_index(faiss_index_path, index_args, dataset_size)

def load_langchain_index(output_path: str, chunking_args: ChunkingArgs, index_args: IndexArgs, dataset_size: int) -> LangchainFaissIndex | LangchainBm25Index | LangchainQdrantIndex:
	"""
	Input: output_path, the dataset's output folder; chunking_args and index_args, the settings the index was built
	with; dataset_size, the number of documents a FAISS index must hold.
	Output: the saved LangChain index of index_type.
	What it does: load-only, never builds. FAISS and Qdrant also need LangChain's HuggingFaceEmbeddings to embed
	queries, so it is loaded for them.
	"""
	if index_args.index_type == constants.BM25_INDEX:
		bm25_index_path = paths.get_bm25_index_path(output_path, chunking_args, index_args)
		return load_langchain_bm25_index(bm25_index_path)

	# The query embedding model, normalized like stages.build.embedding
	device = "cuda" if torch.cuda.is_available() else "cpu"
	embedding_model = HuggingFaceEmbeddings(
		model_name=chunking_args.model_name,
		model_kwargs={"device": device},
		encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
	)

	if index_args.index_type == constants.QDRANT_INDEX:
		qdrant_collection_name = paths.get_qdrant_collection_name(output_path, chunking_args)
		return load_langchain_qdrant_index(qdrant_collection_name, embedding_model)

	faiss_index_path = paths.get_faiss_index_path(output_path, chunking_args, index_args)
	return load_langchain_faiss_index(faiss_index_path, index_args, dataset_size, embedding_model)
