"""
qdrant.py

Builds and loads the Qdrant index of one indexing run. The index is a
collection on the Qdrant server at QDRANT_URL (started with the
qdrant-docker skill), named by common.paths.get_qdrant_collection_name.

Each point in the collection has an integer id, the chunk's embedding as
its vector, and {"chunk_id": chunk_id, "text": chunk text} as its payload,
so a search returns the chunk_id and its text directly. The server keeps the collection on disk,
so an existing collection with the same chunks acts as the cache.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from common.args import IndexArgs
from common.constants import QDRANT_URL
from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded
from l4_indexing.schema import QdrantIndex

def get_qdrant_index(qdrant_collection_name: str, index_args: IndexArgs, chunks: dict[str, Chunk], embeddings: dict[str, Embedded]) -> QdrantIndex:
	"""
	Input: qdrant_collection_name, the collection to use on the Qdrant
	server; index_args, the indexing parameters; chunks, a dict of
	chunk_id to Chunk from l2_chunking; embeddings, a dict of chunk_id to
	Embedded from l3_embedding.
	Output: a QdrantIndex holding the client connected to the server and
	the collection name.
	What it does: connects to the Qdrant server at QDRANT_URL. If the
	collection already exists, holds exactly the same chunk_ids as
	embeddings and stores the chunk text in its payload, uses it as is.
	Otherwise deletes the old collection if there is one, creates a new
	one with cosine distance, and uploads all embeddings with their
	chunk_id and text in batches of qdrant_batch_size.
	Raises ConnectionError if the server cannot be reached.
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
				with_payload=["chunk_id"],
				with_vectors=False
			)
			for point in points:
				saved_chunk_ids.append(point.payload["chunk_id"])
			if next_page_offset is None:
				break

		# A collection built before the text was added to the payload has
		# no "text", so it is rebuilt. Checking one point is enough, because
		# every point of a collection is uploaded with the same payload keys.
		first_points, first_page_offset = qdrant_client.scroll(
			collection_name=qdrant_collection_name,
			limit=1,
			with_payload=True,
			with_vectors=False
		)
		saved_payload_has_text = len(first_points) > 0 and "text" in first_points[0].payload

		if saved_payload_has_text and len(saved_chunk_ids) == len(chunk_ids) and set(saved_chunk_ids) == set(chunk_ids):
			print(f"Qdrant collection loaded from cache : {len(saved_chunk_ids)} points")
			return QdrantIndex(
				qdrant_client=qdrant_client,
				collection_name=qdrant_collection_name,
				dimension=dimension
			)

		# The collection holds different chunks or no text, so it is rebuilt
		qdrant_client.delete_collection(qdrant_collection_name)

	# 4. Create the collection
	# The vectors are normalized by l3_embedding, so cosine distance gives
	# the same ranking as inner product in FAISS
	qdrant_client.create_collection(
		collection_name=qdrant_collection_name,
		vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
	)

	# 5. Upload the points in batches of qdrant_batch_size
	batch_size = index_args.qdrant_batch_size
	for batch_start in range(0, len(chunk_ids), batch_size):
		batch_end = batch_start + batch_size
		if batch_end > len(chunk_ids):
			batch_end = len(chunk_ids)

		batch_points = []
		for position in range(batch_start, batch_end):
			point = PointStruct(
				id=position,
				vector=vectors[position].tolist(),
				payload={"chunk_id": chunk_ids[position], "text": chunks[chunk_ids[position]].text}
			)
			batch_points.append(point)

		qdrant_client.upsert(collection_name=qdrant_collection_name, points=batch_points, wait=True)
	print(f"Qdrant collection built : {len(chunk_ids)} points")

	return QdrantIndex(
		qdrant_client=qdrant_client,
		collection_name=qdrant_collection_name,
		dimension=dimension
	)
