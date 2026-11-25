import dataclasses

from common.args import IngestionArgs, ChunkingArgs, EmbeddingArgs, EmbedderArgs, Bm25IndexingArgs, FaissIVFArgs, FaissHNSWArgs, QdrantArgs

@dataclasses.dataclass
class IngestionEntry:
	args: IngestionArgs 
	hash: str 
	path: str

@dataclasses.dataclass
class ChunkEntry:
	args: ChunkingArgs 
	hash: str 
	path: str

@dataclasses.dataclass
class EmbeddingEntry:
	args: EmbeddingArgs 
	hash: str 
	path: str

@dataclasses.dataclass
class Bm25Entry:
	args: Bm25IndexingArgs 
	hash: str 
	path: str

@dataclasses.dataclass
class FaissEntry:
	faiss_type: str 
	faiss_args: None | FaissIVFArgs | FaissHNSWArgs
	hash: str 
	path: str

@dataclasses.dataclass
class VectorDbEntry:
	vectordb_type: str
	args: QdrantArgs 
	hash: str 
	path: str

@dataclasses.dataclass
class BuildManifest:
	dataset_name: str 
	dataset_size: int 
	build_hash: str 
	ingestion_entry: IngestionEntry 
	chunk_entry: ChunkEntry 
	embedder: EmbedderArgs
	embedding_entry: EmbeddingEntry 
	bm25_index_entry: Bm25Entry 
	faiss_index_entry: FaissEntry | None
	vectordb_index_entry: VectorDbEntry | None
