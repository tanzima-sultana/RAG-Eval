"""
answer_evaluation.py

Measures how good each generated answer is: correctness against the eval question's reference answer, faithfulness
to the context the LLM was given and relevance to the question.
"""

from common import constants
from common.args import AnswerEvalArgs
from l7_context.schema import Context
from l9_generation.schema import Generated
from l10_evaluation.schema import AnswerEvaluation
from qna.schema import EvalQnA

def load_answer_evaluation_cache(answer_evaluation_file: str) -> dict:
	"""
	Input: answer_evaluation_file, the answer evaluation cache file of one framework and judge model.
	Output: the cached {answer_key: AnswerEvaluation fields} dict, or an empty dict if the file does not exist.
	What it does: reads the file.
	"""
	# TODO: read the JSON file, like l9_generation.cache.load_generated_cache
	raise NotImplementedError

def save_answer_evaluation_cache(answer_evaluation_file: str, answer_evaluation_cache: dict) -> None:
	"""
	Input: answer_evaluation_file, the answer evaluation cache file; answer_evaluation_cache, its {answer_key: AnswerEvaluation fields} dict.
	Output: nothing.
	What it does: writes the dict as JSON to answer_evaluation_file, through a temporary file.
	"""
	# TODO: write the JSON file, like l9_generation.cache.save_generated_cache
	raise NotImplementedError

def evaluate_answers_ragas(eval_qna_by_id: dict[str, EvalQnA], context_list: list[Context], generated_list: list[Generated], answer_eval_args: AnswerEvalArgs, answer_evaluation_cache: dict) -> None:
	"""
	Input: eval_qna_by_id, the eval questions by id; context_list and generated_list, in the same order;
	answer_eval_args, the framework and judge LLM; answer_evaluation_cache, the cached {answer_key: AnswerEvaluation fields} dict.
	Output: nothing. The new scores are added to answer_evaluation_cache.
	What it does: scores every answer not in the cache with RAGAS correctness, faithfulness and relevance.
	I_DONT_KNOW answers are skipped, since they get 0 without a judge call.
	"""
	# 1. TODO: collect the answers not in the cache and not I_DONT_KNOW, once per answer_key
	# 2. TODO: build the RAGAS samples: question (generated.qus), response (generated.answer),
	#    retrieved contexts (context.context), reference (eval_qna_by_id[generated.id].ans)
	# 3. TODO: run the RAGAS metrics with the judge LLM
	# 4. TODO: add each sample's scores to answer_evaluation_cache by its answer_key
	raise NotImplementedError

def get_answer_evaluations(output_path: str, eval_qna: list[EvalQnA], context_list: list[Context], generated_list: list[Generated], answer_eval_args: AnswerEvalArgs) -> list[AnswerEvaluation]:
	"""
	Input: output_path, the dataset's output folder; eval_qna, the eval questions; context_list and generated_list,
	the l7 contexts and l9 answers, in the same order; answer_eval_args, the framework and judge LLM.
	Output: one AnswerEvaluation per Generated, in the same order, so answer_evaluations[i] belongs to generated_list[i],
	context_list[i] and retrieved_list[i].
	What it does: loads the cached scores, scores the answers not in the cache, saves the cache, then builds the
	AnswerEvaluation list in order.
	"""
	# 1. Find each answer's eval question by its id. Many answers share one id, one per kept setup
	eval_qna_by_id = {}
	for eval_qna_item in eval_qna:
		eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	# 2. TODO: load the cache from the answer evaluation path (built in common.paths from output_path, framework and
	#    judge model)

	# 3. TODO: score the answers not in the cache, with the method of the chosen framework, then save the cache

	# 4. TODO: build one AnswerEvaluation per Generated, in order: all 0 and i_dont_know 1 for an I_DONT_KNOW answer,
	#    otherwise from the cache
	raise NotImplementedError
