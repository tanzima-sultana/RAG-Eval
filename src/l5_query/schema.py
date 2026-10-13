import dataclasses
import numpy as np

@dataclasses.dataclass
class Qus:
	"""
	One searchable question of a query, in the forms each index needs.

	Fields:
		qus:        The question that is searched.
		tokens:     qus lowercased and split on whitespace, the same as the l4 BM25 chunks. Used by BM25.
		embeddings: qus embedded with the l3 embedding model. Used by FAISS and Qdrant.
	"""
	qus: str
	tokens: list[str]
	embeddings: np.ndarray

@dataclasses.dataclass
class Ans:
	"""
	One hypothetical answer passage of a query (HyDE). Searched by its embedding only, so it has no BM25 tokens.

	Fields:
		ans:        The answer passage written by the LLM.
		embeddings: ans embedded with the l3 embedding model. Used by FAISS and Qdrant.
	"""
	ans: str
	embeddings: np.ndarray

@dataclasses.dataclass
class Query:
	"""
	One query per eval question, holding the question and each transformation of it.

	Fields:
		id:              Same as the EvalQnA id, linking the query to its eval question.
		original_qus:    The eval question as it is.
		rewritten_qus:   The question rewritten by the LLM.
		multi_qus:       n_multi_qus phrasings of the question written by the LLM. Empty if the LLM did not return them (I don't know).
		hyde_ans:        A hypothetical answer passage written by the LLM (HyDE). Its ans is I_DONT_KNOW if the LLM does not know.
		normalized:      True if every embedding is normalized to length 1.
		embedding_model: The embedding model that made the embeddings, the same as l3.
		llm_model:       The LLM that wrote rewritten_qus, multi_qus and hyde_ans.
	"""
	id: str
	original_qus: Qus
	rewritten_qus: Qus
	multi_qus: list[Qus]
	hyde_ans: Ans
	normalized: bool
	embedding_model: str
	llm_model: str
