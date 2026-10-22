from common.args import AnswerEvalArgs
from common.constants import RAGAS_FRAMEWORK

def validate_n_retrieval_evaluation(n_retrieval_evaluation: int):
	"""
	Input: n_retrieval_evaluation, the number of best plain and best reranked setups the filter keeps.
	Output: nothing. Raises ValueError if it is less than 1.
	"""
	if n_retrieval_evaluation < 1:
		raise ValueError(
			f"n_retrieval_evaluation ({n_retrieval_evaluation}) must be at least 1"
		)

def validate_answer_eval_args(answer_eval_args: AnswerEvalArgs):
	"""
	Input: answer_eval_args, the framework, judge LLM and embedding model.
	Output: nothing. Raises ValueError if the framework is not supported. The judge LLM is checked by
	llm.validate.validate_llm.
	"""
	if answer_eval_args.ans_eval_framework != RAGAS_FRAMEWORK:
		raise ValueError(
			f"ans_eval_framework ({answer_eval_args.ans_eval_framework}) is not supported. Use \"{RAGAS_FRAMEWORK}\"."
		)
