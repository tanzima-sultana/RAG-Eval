import dataclasses

@dataclasses.dataclass
class Evaluation:
	"""
	The retrieval evaluation of one Retrieved at one level, chunk or doc. Which chunks are relevant and what recall
	counts depend on the level, see RetrievalEvaluation.

	Fields:
		hit:                 1 if at least one chunk is relevant, else 0.
		precision:           Share of the k chunks that are relevant.
		recall:              Share of the relevant items found in the k chunks.
		mrr:                 1 / rank of the first relevant chunk. 0 for a miss.
		ndcg:                nDCG of the k chunks, rewarding relevant chunks near the top.
		first_relevant_rank: Rank of the first relevant chunk, starting at 1. 0 for a miss.
	"""
	hit: int
	precision: float
	recall: float
	mrr: float
	ndcg: float
	first_relevant_rank: int

@dataclasses.dataclass
class RetrievalEvaluation:
	"""
	The retrieval evaluation of one Retrieved against its eval question.

	Fields:
		k:             Number of chunks scored: top_k, or rerank_top_k for a reranked Retrieved.
		chunk_evaluation: The evaluation at chunk level. A chunk is relevant if it comes from one of the EvalQnA doc_ids and
		               its text overlaps one of the EvalQnA context passages. Recall is the share of context passages found.
		doc_evaluation:   The evaluation at doc level. A chunk is relevant if it comes from one of the EvalQnA doc_ids.
		               Recall is the share of EvalQnA doc_ids found.
	"""
	k: int
	chunk_evaluation: Evaluation
	doc_evaluation: Evaluation

@dataclasses.dataclass
class AnswerEvaluation:
	"""
	The answer evaluation of one Generated against its eval question.
	The answer evaluation list is in the same order as the generated list, so answer_evaluations[i] belongs to
	generated_list[i], context_list[i] and retrieved_list[i], which holds its setup.

	Fields:
		i_dont_know:  1 if the answer is I_DONT_KNOW, else 0.
		correctness:  How well the answer matches the EvalQnA reference answer, from 0 to 1.
		faithfulness: Share of the answer's claims that the context supports, from 0 to 1.
		relevance:    How well the answer addresses the question, from 0 to 1.
		framework:    The framework that scored the answer, e.g. ragas.
		llm:          The LLM provider that judged the answer, e.g. claude.
		llm_model:    The model that judged the answer.
	"""
	i_dont_know: int
	correctness: float
	faithfulness: float
	relevance: float
	framework: str
	llm: str
	llm_model: str
