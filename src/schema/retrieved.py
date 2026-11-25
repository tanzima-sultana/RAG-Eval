import dataclasses

from schema.chunk import Chunk

@dataclasses.dataclass
class Retrieved:
	question_id: str
	reranking: bool
	top_k: int
	rerank_top_k: int 
	chunks: list[Chunk]
