from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from common import constants
from common import paths
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from common.args import VectorDbArgs
from rag.build.embedding.embedding import make_embedding_model
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



def load_vectordb_from_path(collection_name: str, embedding_model: Embeddings, qdrant_url: str) -> VectorDb:
	"""
	Input: collection_name, the Qdrant collection the build filled; embedding_model, the embedding model it was built
	with, which embeds the query; qdrant_url, the Qdrant server.
	Output: a VectorDb holding a LangChain QdrantVectorStore.
	What it does: connects to the collection straight from its name, for serving, the same way the build wraps it. It
	does not check which chunks it holds, and raises if the server is not reachable or the collection does not exist.
	"""
	# 1. Connect to the server. QdrantClient does not connect until the first request, so one is made here
	qdrant_client = QdrantClient(url=qdrant_url)
	try:
		qdrant_client.get_collections()
	except Exception as error:
		raise ConnectionError(
			f"Qdrant server is not reachable at {qdrant_url}. Start it with the qdrant-docker skill."
		) from error

	if not qdrant_client.collection_exists(collection_name):
		raise FileNotFoundError(f"No Qdrant collection {collection_name}. Run scripts/build.sh first.")

	# 2. Wrap the collection in a LangChain QdrantVectorStore
	vector_store = QdrantVectorStore(
		client=qdrant_client,
		collection_name=collection_name,
		embedding=embedding_model,
		distance=Distance.COSINE
	)
	dimension = qdrant_client.get_collection(collection_name).config.params.vectors.size
	print(f"Qdrant collection loaded : {collection_name}")
	return VectorDb(
		vectordb_type=constants.QDRANT_INDEX,
		vector_store=vector_store,
		collection_name=collection_name,
		dimension=dimension
	)

class IndexingVectorDb:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs, vectordb_args: VectorDbArgs):
		"""
		Input: embedder_args, the shared embedder settings; chunking_args, the chunking settings of the chunks the
		collection holds; vectordb_args, the vector DB settings for one vector DB type.
		Output: nothing.
		What it does: stores the three args, validates them and loads the embedding model of the embedder's provider,
		which the vector store uses to embed queries.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.vectordb_args = vectordb_args
		self.validate()

		# The same embedder as the embedding layer, so a query vector matches the chunk vectors
		self.embedding_model = make_embedding_model(self.embedder_args)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if the embedding provider is not supported or a vector DB setting is not valid.
		"""
		vectordb_args = self.vectordb_args

		if self.embedder_args.embedding_provider != constants.HUGGINGFACE_EMBEDDING:
			raise ValueError(
				f"embedding_provider ({self.embedder_args.embedding_provider}) is not supported. Use "
				f"\"{constants.HUGGINGFACE_EMBEDDING}\"."
			)

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
				f"No Qdrant collection {collection_name} holding these chunks. Run scripts/build.sh first."
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
		collection_name = paths.get_qdrant_collection_name(output_path, self.embedder_args, self.chunking_args, self.vectordb_args)
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
