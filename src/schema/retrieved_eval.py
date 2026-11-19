import dataclasses

from schema.retrieved import Retrieved


@dataclasses.dataclass
class Eval:
	# The scores of one Retrieved at one level, chunk or doc
	# 1 if at least one chunk is relevant, else 0
	hit: int
	# Share of the k chunks that are relevant
	precision: float
	# Share of the relevant items found
	recall: float
	# 1 / rank of the first relevant chunk. 0 for a miss
	mrr: float
	# Rewards relevant chunks near the top
	ndcg: float
	# Rank of the first relevant chunk, from 1. 0 for a miss
	first_relevant_rank: int


@dataclasses.dataclass
class RetrievedEval:
	# The Retrieved that was scored: its setup and its chunks
	retrieved: Retrieved
	# A chunk is relevant if it is from a Qna doc and overlaps a Qna context passage.
	# Recall is the share of context passages found
	chunk_eval: Eval
	# A chunk is relevant if it is from a Qna doc. Recall is the share of Qna docs found
	doc_eval: Eval
