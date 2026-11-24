import dataclasses


@dataclasses.dataclass
class Qus:
	# One question text to search with. The retriever embeds and tokenizes it itself
	qus: str


@dataclasses.dataclass
class Ans:
	# One hypothetical answer passage (HyDE). Searched as text by the dense retriever only
	ans: str


@dataclasses.dataclass
class Query:
	# Same as the EvalSample id, linking the query to its eval question
	id: str
	# The question as it is. Always made, not set by query_types
	original_qus: Qus
	# Each extra query type is None when it is not in query_types of this run
	rewritten_qus: Qus | None
	# n_multi_qus phrasings. Empty if the LLM did not return exactly n_multi_qus of them
	multi_qus: list[Qus] | None
	# Its ans is I_DONT_KNOW if the LLM did not know
	hyde_ans: Ans | None
	# The provider and model that wrote rewritten_qus, multi_qus and hyde_ans, e.g. ollama and llama3.1:8b
	llm: str
	llm_model: str
