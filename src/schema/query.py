import dataclasses

import numpy as np


@dataclasses.dataclass
class Qus:
	# One searchable question of a query
	qus: str
	# qus lowercased and split on whitespace, the same as the BM25 chunks
	tokens: list[str]
	# qus embedded with the chunking embedding model
	embeddings: np.ndarray


@dataclasses.dataclass
class Ans:
	# One hypothetical answer passage (HyDE). Searched by its embedding only, so it has no BM25 tokens
	ans: str
	embeddings: np.ndarray


@dataclasses.dataclass
class Query:
	# Same as the EvalSample id, linking the query to its eval question
	id: str
	# Each query type is None when it is not in query_types of this run
	original_qus: Qus | None
	rewritten_qus: Qus | None
	# n_multi_qus phrasings. Empty if the LLM did not return exactly n_multi_qus of them
	multi_qus: list[Qus] | None
	# Its ans is I_DONT_KNOW if the LLM did not know
	hyde_ans: Ans | None
	normalized: bool
	embedding_model: str
	# The provider and model that wrote rewritten_qus, multi_qus and hyde_ans, e.g. ollama and llama3.1:8b
	llm: str
	llm_model: str
