import dataclasses


@dataclasses.dataclass
class ChunkEntry:
	# The chunks and embeddings the build saved for one chunk type
	# fixed, sentence or semantic
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float
	# The folder of the saved chunks, one file per document
	chunk_path: str
	# The folder of the saved embeddings, one file per document
	embedding_path: str
	chunk_count: int


@dataclasses.dataclass
class IndexEntry:
	# One index the build saved
	# The chunk type whose chunks the index holds
	chunk_type: str
	# faiss_flatip, faiss_ivf, faiss_hnsw, bm25 or qdrant
	index_type: str
	# The settings of this index type: ivf_nlist, hnsw_m and hnsw_ef_construction, or bm25_k1, bm25_b and
	# bm25_tokenizer. Empty for faiss_flatip and qdrant
	index_settings: dict
	# Where the index is: a folder for FAISS and BM25, the collection name for Qdrant
	location: str
	# The vectors or chunks the index holds
	count: int


@dataclasses.dataclass
class BuildManifest:
	# What one build saved and where, so the later steps only load it
	dataset_name: str
	dataset_size: int
	# The folder of the saved documents
	documents_path: str
	# The model the chunks were embedded with
	embedding_model: str
	# When the build finished, in ISO format
	built_at: str
	chunk_entries: list[ChunkEntry]
	index_entries: list[IndexEntry]
