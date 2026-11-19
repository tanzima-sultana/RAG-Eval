import dataclasses

from langchain_qdrant import QdrantVectorStore

# One record per vector DB collection, not per chunk: all chunks of one chunking run go into one collection


@dataclasses.dataclass
class VectorDb:
	# qdrant
	vectordb_type: str
	vector_store: QdrantVectorStore
	collection_name: str
	dimension: int
