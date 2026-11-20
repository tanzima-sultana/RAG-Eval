import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from common import constants
from common import paths_temp
from common.args_temp import VectorDbArgs
from schema.chunk import Chunk
from schema.embed import Embedded
from schema.vectordb import VectorDb


def qdrant_collection_holds_chunks(qdrant_client: QdrantClient, collection_name: str, chunk_ids: list[str]) -> bool:
	"""
	Input: qdrant_client, a connected client; collection_name; chunk_ids, the chunk ids the collection must hold.
	Output: True if the collection exists and holds exactly chunk_ids, otherwise False.
	What it does: reads the chunk id of every point page by page and compares them with chunk_ids.
	"""
	if not qdrant_client.collection_exists(collection_name):
		return False

	saved_chunk_ids = []
	next_page_offset = None
	while True:
		points, next_page_offset = qdrant_client.scroll(
			collection_name=collection_name,
			limit=10000,
			offset=next_page_offset,
			with_payload=["metadata"],
			with_vectors=False
		)
		for point in points:
			saved_chunk_ids.append(point.payload["metadata"]["chunk_id"])
		if next_page_offset is None:
			break

	if len(saved_chunk_ids) != len(chunk_ids):
		return False
	if set(saved_chunk_ids) != set(chunk_ids):
		return False
	return True


class IndexingVectorDb:
	def __init__(self, vectordb_args: VectorDbArgs):
		"""
		Input: vectordb_args, the vector DB settings for one vector DB type.
		Output: nothing.
		What it does: stores vectordb_args, validates them and loads LangChain's HuggingFaceEmbeddings, which the
		vector store uses to embed queries.
		"""
		self.vectordb_args = vectordb_args
		self.validate()

		# Normalized like the embedding layer, so a query vector matches the chunk vectors
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = HuggingFaceEmbeddings(
			model_name=self.vectordb_args.chunking_args.model_name,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a vector DB setting is not valid.
		"""
		vectordb_args = self.vectordb_args

		if vectordb_args.vectordb_type != constants.QDRANT_INDEX:
			raise ValueError(
				f"vectordb_type ({vectordb_args.vectordb_type}) is not supported. Use \"{constants.QDRANT_INDEX}\"."
			)

		if vectordb_args.qdrant_url == "":
			raise ValueError("qdrant_url is empty.")

		if vectordb_args.qdrant_batch_size <= 0:
			raise ValueError(f"qdrant_batch_size ({vectordb_args.qdrant_batch_size}) must be greater than 0.")

	def load_saved_vectordb(self, chunks: dict[str, Chunk], collection_name: str) -> VectorDb:
		"""
		Input: chunks, a dict of chunk_id to Chunk, the chunks the collection must hold; collection_name, from the build
		manifest.
		Output: a VectorDb holding a LangChain QdrantVectorStore.
		What it does: connects to the collection the build filled, and raises FileNotFoundError if it does not exist or
		holds different chunks. It never creates or uploads.
		"""
		chunk_ids = []
		for chunk_id in chunks:
			chunk_ids.append(chunk_id)

		# 1. Connect to the server. QdrantClient does not connect until the first request, so one is made here
		qdrant_url = self.vectordb_args.qdrant_url
		qdrant_client = QdrantClient(url=qdrant_url)
		try:
			qdrant_client.get_collections()
		except Exception as error:
			raise ConnectionError(
				f"Qdrant server is not reachable at {qdrant_url}. Start it with the qdrant-docker skill."
			) from error

		# 2. The collection must hold exactly these chunks
		if not qdrant_collection_holds_chunks(qdrant_client, collection_name, chunk_ids):
			raise FileNotFoundError(
				f"No Qdrant collection {collection_name} holding these chunks. Run scripts/build_temp.sh first."
			)
		print(f"Qdrant collection loaded from cache : {len(chunk_ids)} points")

		# 3. Wrap the collection in a LangChain QdrantVectorStore
		vector_store = QdrantVectorStore(
			client=qdrant_client,
			collection_name=collection_name,
			embedding=self.embedding_model,
			distance=Distance.COSINE
		)
		dimension = qdrant_client.get_collection(collection_name).config.params.vectors.size

		return VectorDb(
			vectordb_type=self.vectordb_args.vectordb_type,
			vector_store=vector_store,
			collection_name=collection_name,
			dimension=dimension
		)

	def compute_vectordb(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> VectorDb:
		"""
		Input: chunks, a dict of chunk_id to Chunk; embeddings, a dict of chunk_id to Embedded; output_path, the
		dataset's output folder.
		Output: a VectorDb.
		What it does: calls the indexing method of the vector DB type.
		"""
		# Only qdrant for now. validate() already rejected any other type
		vectordb = self.qdrant_indexing(chunks, embeddings, output_path)
		return vectordb

	def qdrant_indexing(self, chunks: dict[str, Chunk], embeddings: dict[str, Embedded], output_path: str) -> VectorDb:
		"""
		Input: chunks; embeddings; output_path, the dataset's output folder.
		Output: a VectorDb holding a LangChain QdrantVectorStore.
		What it does: reuses the collection if it holds the same chunks, otherwise rebuilds it and uploads the
		existing vectors in batches. Raises ConnectionError if the server cannot be reached.
		"""
		if len(embeddings) == 0:
			raise ValueError("embeddings is empty. There is nothing to index.")

		# 1. Collect the chunk ids and vectors in the same order. Point id i holds vectors[i] of chunk_ids[i]
		chunk_ids = []
		vectors = []
		for chunk_id in embeddings:
			chunk_ids.append(chunk_id)
			vectors.append(embeddings[chunk_id].embeddings)
		dimension = len(vectors[0])

		# 2. Connect to the server. QdrantClient does not connect until the first request, so one is made here
		qdrant_url = self.vectordb_args.qdrant_url
		qdrant_client = QdrantClient(url=qdrant_url)
		try:
			qdrant_client.get_collections()
		except Exception as error:
			raise ConnectionError(
				f"Qdrant server is not reachable at {qdrant_url}. Start it with the qdrant-docker skill."
			) from error

		# 3. Reuse the collection if it holds exactly the same chunks, otherwise rebuild it
		collection_name = paths_temp.get_qdrant_collection_name(output_path, self.vectordb_args)
		print(f"Qdrant collection : {collection_name}")
		if qdrant_collection_holds_chunks(qdrant_client, collection_name, chunk_ids):
			print(f"Qdrant collection loaded from cache : {len(chunk_ids)} points")
		else:
			if qdrant_client.collection_exists(collection_name):
				qdrant_client.delete_collection(collection_name)

			qdrant_client.create_collection(
				collection_name=collection_name,
				vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
			)

			# 4. Upload the existing vectors in batches, in the payload layout QdrantVectorStore reads.
			# add_texts is not used, because it would embed every chunk again
			batch_size = self.vectordb_args.qdrant_batch_size
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

				qdrant_client.upsert(collection_name=collection_name, points=batch_points, wait=True)
			print(f"Qdrant collection built : {len(chunk_ids)} points")

		# 5. Wrap the collection in a LangChain QdrantVectorStore
		vector_store = QdrantVectorStore(
			client=qdrant_client,
			collection_name=collection_name,
			embedding=self.embedding_model,
			distance=Distance.COSINE
		)

		return VectorDb(
			vectordb_type=self.vectordb_args.vectordb_type,
			vector_store=vector_store,
			collection_name=collection_name,
			dimension=dimension
		)
