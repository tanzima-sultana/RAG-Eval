import dataclasses

from common.args import QueryArgs, RetrievalArgs

@dataclasses.dataclass
class SetupScore:
	# The val queries the means are taken over
	queries_count: int
	mean_hit: float
	mean_recall: float
	mean_mrr: float
	mean_ndcg: float

@dataclasses.dataclass
class Setup:
	build_hash: str
	query_args: QueryArgs
	retrieve_args: RetrievalArgs
	score: SetupScore