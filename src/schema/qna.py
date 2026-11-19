import dataclasses


@dataclasses.dataclass
class Qna:
	# One generated question-answer item used for evaluation. The fields match the saved QnA JSON
	id: str
	# RAGAS user_input
	qus: str
	# RAGAS reference, the ground truth answer
	ans: str
	# RAGAS reference_contexts, the source passages the question was made from
	context: list[str]
	# RAGAS synthesizer_name, e.g. single_hop_specific_query_synthesizer
	qus_type: str
	# The documents the context came from. A list, as a multi-hop question can span several documents
	doc_ids: list[str]
	doc_paths: list[str]
	# The provider and model that wrote the question and answer, e.g. anthropic and claude-sonnet-5-5
	llm: str
	llm_model: str
