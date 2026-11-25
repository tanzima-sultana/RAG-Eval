import dataclasses

from schema.retrieved import Retrieved
from schema.setup import Setup


@dataclasses.dataclass
class RetrievedEval:
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