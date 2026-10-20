import dataclasses

from stages.evaluation.retrieval.schema import Retrieved
from stages.evaluation.generation.schema import Generated
from stages.question.qna.schema import EvalQnA

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
		              Each of the three is None when the judge could not give it, e.g. when the provider's content
		              filter blocked the judge's reply. A None score is left out of the averages.
		framework:    The framework that scored the answer, e.g. ragas.
		llm:          The LLM provider that judged the answer, e.g. claude.
		llm_model:    The model that judged the answer.
	"""
	i_dont_know: int
	correctness: float | None
	faithfulness: float | None
	relevance: float | None
	framework: str
	llm: str
	llm_model: str

@dataclasses.dataclass
class EvalResults:
	"""
	Everything one eval run hands to the analysis stage, saved as one file per settings.
	retrieved_list, retrieval_evaluation_list, generated_list and answer_evaluations are in the same order, so their
	i-th items belong to the same kept setup and eval question.

	Fields:
		settings:                  Every setting of the run, as {setting name: value}, with the system prompt and user
		                           prompt template added.
		settings_hash:             The sha256 of settings, the same for two runs with the same settings.
		retrieval_evaluations:     Every (Retrieved, RetrievalEvaluation) of the run, one per eval question and setup.
		retrieved_list:            The Retrieved of the kept setups, the ones answers were generated for.
		retrieval_evaluation_list: The RetrievalEvaluation of each of those Retrieved.
		generated_list:            The answer of each of those Retrieved.
		answer_evaluations:        The AnswerEvaluation of each of those answers.
		eval_qna:                  The eval questions.
	"""
	settings: dict
	settings_hash: str
	retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]]
	retrieved_list: list[Retrieved]
	retrieval_evaluation_list: list[RetrievalEvaluation]
	generated_list: list[Generated]
	answer_evaluations: list[AnswerEvaluation]
	eval_qna: list[EvalQnA]
