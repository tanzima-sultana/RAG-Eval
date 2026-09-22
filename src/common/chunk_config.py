"""
chunk_config.py

Declares ChunkConfig: the dataset_name/chunk_type/min_chunk_size/
max_chunk_size/fixed_chunk_overlap/semantic_threshold tuple that
identifies one specific chunk build. This exact combination of values
travels together across l2_chunking (which produces the chunks),
l3_embedding (which reads those chunks back to embed them), and
l4_indexing (which reads the resulting embeddings back to index them)
-- every one of those layers uses it to build the same
common.paths.get_chunk_path/get_embeddings_path/get_index_path-style
directory name. Bundling it into one dataclass, instead of passing the
six values as separate same-typed positional arguments at every one of
those call sites, is what prevents two of them (most easily
min_chunk_size/max_chunk_size, both plain ints) from being silently
swapped by position.
"""

import dataclasses


@dataclasses.dataclass
class ChunkConfig:
    dataset_name: str
    chunk_type: str
    min_chunk_size: int
    max_chunk_size: int
    fixed_chunk_overlap: int
    semantic_threshold: float
