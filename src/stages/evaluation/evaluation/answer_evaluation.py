"""
answer_evaluation.py

Measures how good each generated answer is: correctness against the eval question's reference answer, faithfulness
to the context the LLM was given and relevance to the question.

The scores of one framework and judge model are cached in one JSON file (see common.paths.get_answer_evaluation_path):

answer_evaluation.json: {answer_key: {"correctness": ..., "faithfulness": ..., "relevance": ...}}, where answer_key is
                        the sha256 of the question, answer, context and reference answer. Two setups that gave the same
                        answer from the same context share one entry, and a changed answer or context gets a new key.
                        A score the judge could not give is stored as null (None), see evaluate_answers_ragas.
"""

import hashlib
import json
import math
import os

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset
from ragas import RunConfig
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerCorrectness
from ragas.metrics import Faithfulness
from ragas.metrics import ResponseRelevancy

from common import constants
from common import paths
from common.args import AnswerEvalArgs
from stages.evaluation.context.schema import Context
from stages.evaluation.generation.schema import Generated
from stages.evaluation.evaluation.schema import AnswerEvaluation
from stages.evaluation.evaluation.validate import validate_answer_eval_args
from llm.llm import get_llm
from llm.validate import validate_llm
from stages.question.qna.schema import EvalQnA

def load_answer_evaluation_cache(answer_evaluation_file: str) -> dict:
	"""
	Input: answer_evaluation_file, the answer evaluation cache file of one framework and judge model.
	Output: the cached {answer_key: {"correctness", "faithfulness", "relevance"}} dict, or an empty dict if the file
	does not exist.
	What it does: reads the file.
	"""
	if not os.path.exists(answer_evaluation_file):
		return {}

	with open(answer_evaluation_file, encoding="utf-8") as f:
		answer_evaluation_cache = json.load(f)

	return answer_evaluation_cache

def save_answer_evaluation_cache(answer_evaluation_file: str, answer_evaluation_cache: dict) -> None:
	"""
	Input: answer_evaluation_file, the answer evaluation cache file; answer_evaluation_cache, its
	{answer_key: {"correctness", "faithfulness", "relevance"}} dict.
	Output: nothing.
	What it does: writes the dict as JSON to answer_evaluation_file through a temporary file, creating its folder first.
	"""
	os.makedirs(os.path.dirname(answer_evaluation_file), exist_ok=True)

	temporary_answer_evaluation_file = answer_evaluation_file + ".tmp"
	with open(temporary_answer_evaluation_file, "w", encoding="utf-8") as f:
		json.dump(answer_evaluation_cache, f, indent=2)
	os.replace(temporary_answer_evaluation_file, answer_evaluation_file)

def get_answer_key(generated: Generated, context: Context, eval_qna_item: EvalQnA) -> str:
	"""
	Input: generated, one generated answer; context, the context it was given; eval_qna_item, its eval question.
	Output: the answer_key, a sha256 hex string.
	What it does: hashes the question, answer, context and reference answer, the four inputs every score depends on.
	"""
	key_text = json.dumps([generated.qus, generated.answer, context.context, eval_qna_item.ans])
	answer_key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
	return answer_key

def evaluate_answers_ragas(eval_qna_by_id: dict[str, EvalQnA], context_list: list[Context], generated_list: list[Generated], answer_eval_args: AnswerEvalArgs, answer_evaluation_cache: dict) -> int:
	"""
	Input: eval_qna_by_id, the eval questions by id; context_list and generated_list, in the same order;
	answer_eval_args, the framework, judge LLM and embedding model; answer_evaluation_cache, the cached
	{answer_key: {"correctness", "faithfulness", "relevance"}} dict.
	Output: the number of answers with at least one score RAGAS could not give. The new scores are added to
	answer_evaluation_cache.
	What it does: scores every answer not in the cache with the RAGAS answer_correctness, faithfulness and
	answer_relevancy metrics. I_DONT_KNOW answers are skipped, since they get 0 without a judge call.
	"""
	# 1. Collect the answers not in the cache and not I_DONT_KNOW, once per answer_key
	answer_keys_to_score = []
	samples = []
	for index in range(len(generated_list)):
		generated = generated_list[index]
		if generated.answer == constants.I_DONT_KNOW:
			continue
		context = context_list[index]
		eval_qna_item = eval_qna_by_id[generated.id]
		answer_key = get_answer_key(generated, context, eval_qna_item)
		if answer_key in answer_evaluation_cache:
			continue
		if answer_key in answer_keys_to_score:
			continue

		# 2. One RAGAS sample: the question, the answer, the context the LLM was given and the reference answer
		sample = SingleTurnSample(
			user_input=generated.qus,
			response=generated.answer,
			retrieved_contexts=[context.context],
			reference=eval_qna_item.ans
		)
		answer_keys_to_score.append(answer_key)
		samples.append(sample)
	print(f"Answers to score with RAGAS : {len(samples)}")

	# The LLM and embedding model are built only when there is something to score, so a fully cached run needs no
	# API keys, ALLOW_PAID_API or Ollama server
	if len(samples) == 0:
		return 0

	# 3. Get the judge LLM and the embedding model. RAGAS parses every reply as JSON, so it uses the LLM's
	# json_chat_model. RAGAS sets temperature 0.01 by default, which the Claude 5.5 models reject, so the model's own
	# temperature is kept
	validate_llm(answer_eval_args.llm_args)
	judge_llm = get_llm(answer_eval_args.llm_args)
	ragas_judge_llm = LangchainLLMWrapper(judge_llm.json_chat_model, bypass_temperature=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	huggingface_embeddings = HuggingFaceEmbeddings(
		model_name=answer_eval_args.embedding_model,
		model_kwargs={"device": device},
		encode_kwargs={"normalize_embeddings": True}
	)
	ragas_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)

	# 4. Run the three metrics. A sample whose judge reply fails gets NaN instead of stopping the run
	evaluation_result = evaluate(
		dataset=EvaluationDataset(samples=samples),
		metrics=[AnswerCorrectness(), Faithfulness(), ResponseRelevancy()],
		llm=ragas_judge_llm,
		embeddings=ragas_embeddings,
		run_config=RunConfig(timeout=answer_eval_args.llm_args.timeout_seconds),
		raise_exceptions=False
	)
	judge_llm.compute_cost()

	# 5. Add every sample to the cache. RAGAS already retries a failed judge call (RunConfig max_retries), so a score
	# that is still NaN cannot be given, e.g. the provider's content filter blocked the judge's reply. It is stored as
	# None, so the run goes on and a run again does not pay for it again
	unscored_count = 0
	for index in range(len(answer_keys_to_score)):
		answer_key = answer_keys_to_score[index]
		sample_scores = evaluation_result.scores[index]
		scores = {}
		has_unscored = False
		for score_name, metric_name in (("correctness", "answer_correctness"), ("faithfulness", "faithfulness"), ("relevance", "answer_relevancy")):
			score = sample_scores[metric_name]
			if math.isnan(score):
				scores[score_name] = None
				has_unscored = True
			else:
				scores[score_name] = score
		if has_unscored:
			unscored_count = unscored_count + 1
		answer_evaluation_cache[answer_key] = scores

	return unscored_count

def get_answer_evaluations(output_path: str, eval_qna: list[EvalQnA], context_list: list[Context], generated_list: list[Generated], answer_eval_args: AnswerEvalArgs) -> list[AnswerEvaluation]:
	"""
	Input: output_path, the dataset's output folder; eval_qna, the eval questions; context_list and generated_list,
	the contexts and generated answers, in the same order; answer_eval_args, the framework, judge LLM and embedding model.
	Output: one AnswerEvaluation per Generated, in the same order, so answer_evaluations[i] belongs to generated_list[i],
	context_list[i] and retrieved_list[i].
	What it does: loads the cached scores, scores the answers not in the cache, saves the cache, then builds the
	AnswerEvaluation list in order.
	"""
	validate_answer_eval_args(answer_eval_args)

	# 1. Find each answer's eval question by its id. Many answers share one id, one per kept setup
	eval_qna_by_id = {}
	for eval_qna_item in eval_qna:
		eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	# 2. Load the cached scores of this framework and judge model
	answer_evaluation_file = paths.get_answer_evaluation_path(output_path, answer_eval_args.framework, answer_eval_args.llm_args.llm_model)
	answer_evaluation_cache = load_answer_evaluation_cache(answer_evaluation_file)
	print(f"Cached answer evaluations : {len(answer_evaluation_cache)} in {answer_evaluation_file}")

	# 3. Score the answers not in the cache, then save the cache. RAGAS is the only framework, checked by
	# validate_answer_eval_args
	unscored_count = evaluate_answers_ragas(eval_qna_by_id, context_list, generated_list, answer_eval_args, answer_evaluation_cache)
	save_answer_evaluation_cache(answer_evaluation_file, answer_evaluation_cache)

	# 4. Answers with a score the judge could not give keep None for it and are left out of that score's averages
	print(f"Newly scored answers with a score the judge could not give : {unscored_count}")

	# 5. Build one AnswerEvaluation per Generated, in order: all 0 for an I_DONT_KNOW answer, otherwise from the cache
	answer_evaluations = []
	i_dont_know_count = 0
	for index in range(len(generated_list)):
		generated = generated_list[index]
		if generated.answer == constants.I_DONT_KNOW:
			i_dont_know = 1
			correctness = 0.0
			faithfulness = 0.0
			relevance = 0.0
			i_dont_know_count = i_dont_know_count + 1
		else:
			answer_key = get_answer_key(generated, context_list[index], eval_qna_by_id[generated.id])
			scores = answer_evaluation_cache[answer_key]
			i_dont_know = 0
			correctness = scores["correctness"]
			faithfulness = scores["faithfulness"]
			relevance = scores["relevance"]

		answer_evaluation = AnswerEvaluation(
			i_dont_know=i_dont_know,
			correctness=correctness,
			faithfulness=faithfulness,
			relevance=relevance,
			framework=answer_eval_args.framework,
			llm=answer_eval_args.llm_args.llm,
			llm_model=answer_eval_args.llm_args.llm_model
		)
		answer_evaluations.append(answer_evaluation)
	print(f"I don't know answers, scored 0 : {i_dont_know_count} of {len(generated_list)}")

	return answer_evaluations
