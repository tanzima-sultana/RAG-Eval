import dataclasses

from common.chunk_config import ChunkConfig

# Unlike l2_chunking's Chunk and l3_embedding's Embedded, which are
# both one record per chunk_id (a straight one-to-one transformation
# of a single chunk), Index is one record per completed index build:
# many chunks go into a single FAISS index file, BM25 structure, or
# Qdrant collection, so there is no single chunk_id to key this
# record by.
#
# chunk_config identifies which chunk build (dataset_name/chunk_type/
# min_chunk_size/max_chunk_size/fixed_chunk_overlap/semantic_threshold)
# this index was built from -- the same ChunkConfig l2_chunking and
# l3_embedding already carry, reused here instead of repeating those
# six fields flat on Index (see common/chunk_config.py for why they
# are bundled together at all).
#
# chunk_ids holds the ordered list of every chunk_id folded into this
# build. It is needed regardless of backend because none of FAISS
# (sequential int64 positions), BM25 (corpus position), or Qdrant
# (uint64/UUID point ids) accept the string chunk_id directly as
# their own id -- this list is what lets indexing.py map a backend
# hit back to a real chunk_id, and check whether a chunk_id is
# already covered by this index without querying a live service.
#
# path and collection_name are backend-specific and mutually
# exclusive: path is the saved index file for FAISS/BM25, and
# collection_name is the Qdrant collection this build was pushed to.
@dataclasses.dataclass
class Index:
    index_type: str
    chunk_config: ChunkConfig
    chunk_ids: list[str]
    vector_count: int
    path: str | None = None
    collection_name: str | None = None
