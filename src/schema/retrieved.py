import dataclasses

from schema.setup import Setup


@dataclasses.dataclass
class Retrieved:
	# The chunks found for one eval question with one setup
	# Same as the EvalSample and Query id, linking the result to its eval question
	id: str
	# The setup that found the chunks: chunk type, index type, query type, retrieval type and reranking
	setup: Setup
	top_k: int
	# Chunks kept after reranking. 0 when not reranked
	rerank_top_k: int
	# The chunks, best first
	chunk_ids: list[str]
	# The document of each chunk: the part of the chunk id before the last "_"
	doc_ids: list[str]
