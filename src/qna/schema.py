import dataclasses

@dataclasses.dataclass
class EvalQnA:
	"""
	One generated question-answer item used for evaluation.

	Some fields are filled from the RAGAS test generator output, the rest by
	this project. The RAGAS field each one comes from is shown in brackets.

	Fields from RAGAS:
		qus:          The generated question (RAGAS user_input).
		ans:          The reference (ground truth) answer (RAGAS reference).
		context: 	  The source text passages used to build the question
		              (RAGAS reference_contexts).
		qus_type:     The question type, e.g. single_hop_specific_query_synthesizer
		              or multi_hop_abstract_query_synthesizer (RAGAS synthesizer_name).

	Project fields:
		id:        Unique identifier of this item.
		doc_ids:   Ids of the l1 documents the context came from.
		           A list, because a multi-hop question can span several documents.
		doc_paths: Paths of those same documents, in the same order as doc_ids.
		llm:       The LLM provider that wrote the question and answer, e.g. claude.
		llm_model: The model that wrote the question and answer (CLAUDE_QUESTION_MODEL). The knowledge graph
		           model (CLAUDE_TRANSFORMS_MODEL) is not recorded, since its output is not in the item.
	"""
	id: str
	qus: str
	ans: str
	context: list[str]
	qus_type: str
	doc_ids: list[str]
	doc_paths: list[str]
	llm: str
	llm_model: str
