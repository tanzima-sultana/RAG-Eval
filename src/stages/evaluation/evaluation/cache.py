"""
cache.py

Saves and loads the results of one eval run as one JSON file per settings, at
common.paths.get_eval_results_path: {output_path}/eval_results/{settings_hash}.json. This file is the hand-off from
the evaluation stage to the analysis stage, so the analysis can run on its own without redoing the evaluation.

A run with the same settings overwrites the file. It is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file.
"""

import dataclasses
import hashlib
import json
import os

from common import paths
from stages.evaluation.prompt.prompt_template import SYSTEM_PROMPT
from stages.evaluation.prompt.prompt_template import USER_PROMPT_TEMPLATE
from stages.evaluation.retrieval.schema import Retrieved
from stages.evaluation.generation.schema import Generated
from stages.evaluation.evaluation.schema import AnswerEvaluation
from stages.evaluation.evaluation.schema import EvalResults
from stages.evaluation.evaluation.schema import Evaluation
from stages.evaluation.evaluation.schema import RetrievalEvaluation
from stages.question.qna.schema import EvalQnA

def get_eval_settings(args) -> dict:
	"""
	Input: args, the parsed eval arguments.
	Output: the run's settings, as {setting name: value}.
	What it does: copies every argument and adds the system prompt and user prompt template, since they are not in the
	config, so a run with another prompt counts as a different run. The analysis settings n_compare_runs and
	n_worst_answers are left out, since they do not change the eval results, so the analysis can change them and
	still find the results.
	"""
	settings = dict(vars(args))
	settings["system_prompt"] = SYSTEM_PROMPT
	settings["user_prompt_template"] = USER_PROMPT_TEMPLATE
	del settings["n_compare_runs"]
	del settings["n_worst_answers"]
	return settings

def get_settings_hash(settings: dict) -> str:
	"""
	Input: settings, the run's settings.
	Output: the sha256 hex string of the settings.
	What it does: hashes the settings written as sorted JSON. default=str writes any setting JSON cannot hold as text.
	"""
	settings_text = json.dumps(settings, sort_keys=True, default=str)
	return hashlib.sha256(settings_text.encode("utf-8")).hexdigest()

def save_eval_results(output_path: str, eval_results: EvalResults) -> str:
	"""
	Input: output_path, the dataset's output folder; eval_results, the run's results.
	Output: the path of the saved file.
	What it does: writes eval_results as JSON to {output_path}/eval_results/{settings_hash}.json, through a temporary
	file. Each (Retrieved, RetrievalEvaluation) pair is written as {"retrieved": ..., "retrieval_evaluation": ...}.
	"""
	retrieval_evaluation_dicts = []
	for retrieved, retrieval_evaluation in eval_results.retrieval_evaluations:
		retrieval_evaluation_dicts.append({
			"retrieved": dataclasses.asdict(retrieved),
			"retrieval_evaluation": dataclasses.asdict(retrieval_evaluation),
		})

	retrieved_dicts = []
	for retrieved in eval_results.retrieved_list:
		retrieved_dicts.append(dataclasses.asdict(retrieved))
	kept_retrieval_evaluation_dicts = []
	for retrieval_evaluation in eval_results.retrieval_evaluation_list:
		kept_retrieval_evaluation_dicts.append(dataclasses.asdict(retrieval_evaluation))
	generated_dicts = []
	for generated in eval_results.generated_list:
		generated_dicts.append(dataclasses.asdict(generated))
	answer_evaluation_dicts = []
	for answer_evaluation in eval_results.answer_evaluations:
		answer_evaluation_dicts.append(dataclasses.asdict(answer_evaluation))
	eval_qna_dicts = []
	for eval_qna_item in eval_results.eval_qna:
		eval_qna_dicts.append(dataclasses.asdict(eval_qna_item))

	eval_results_data = {
		"settings": eval_results.settings,
		"settings_hash": eval_results.settings_hash,
		"retrieval_evaluations": retrieval_evaluation_dicts,
		"retrieved_list": retrieved_dicts,
		"retrieval_evaluation_list": kept_retrieval_evaluation_dicts,
		"generated_list": generated_dicts,
		"answer_evaluations": answer_evaluation_dicts,
		"eval_qna": eval_qna_dicts,
	}

	eval_results_file = paths.get_eval_results_path(output_path, eval_results.settings_hash)
	os.makedirs(os.path.dirname(eval_results_file), exist_ok=True)
	temporary_file = eval_results_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(eval_results_data, f, default=str)
	os.replace(temporary_file, eval_results_file)

	return eval_results_file

def json_dict_to_retrieval_evaluation(data: dict) -> RetrievalEvaluation:
	"""
	Input: data, a dict produced by dataclasses.asdict of a RetrievalEvaluation.
	Output: the rebuilt RetrievalEvaluation.
	What it does: rebuilds the nested chunk and doc Evaluation, which JSON holds as dicts, then the RetrievalEvaluation.
	"""
	return RetrievalEvaluation(
		k=data["k"],
		chunk_evaluation=Evaluation(**data["chunk_evaluation"]),
		doc_evaluation=Evaluation(**data["doc_evaluation"]),
	)

def load_eval_results(output_path: str, settings_hash: str) -> EvalResults:
	"""
	Input: output_path, the dataset's output folder; settings_hash, the sha256 of the run's settings.
	Output: the saved EvalResults of the run with those settings.
	What it does: reads {output_path}/eval_results/{settings_hash}.json and rebuilds every dataclass, including the
	nested chunk and doc Evaluation of each RetrievalEvaluation. Raises FileNotFoundError if no run with these
	settings was saved.
	"""
	eval_results_file = paths.get_eval_results_path(output_path, settings_hash)
	if not os.path.exists(eval_results_file):
		raise FileNotFoundError(
			f"No eval results at {eval_results_file}. Run scripts/eval.sh with the same settings first."
		)

	with open(eval_results_file, encoding="utf-8") as f:
		eval_results_data = json.load(f)

	retrieval_evaluations = []
	for pair_data in eval_results_data["retrieval_evaluations"]:
		retrieved = Retrieved(**pair_data["retrieved"])
		retrieval_evaluation = json_dict_to_retrieval_evaluation(pair_data["retrieval_evaluation"])
		retrieval_evaluations.append((retrieved, retrieval_evaluation))

	retrieved_list = []
	for retrieved_data in eval_results_data["retrieved_list"]:
		retrieved_list.append(Retrieved(**retrieved_data))
	retrieval_evaluation_list = []
	for retrieval_evaluation_data in eval_results_data["retrieval_evaluation_list"]:
		retrieval_evaluation_list.append(json_dict_to_retrieval_evaluation(retrieval_evaluation_data))
	generated_list = []
	for generated_data in eval_results_data["generated_list"]:
		generated_list.append(Generated(**generated_data))
	answer_evaluations = []
	for answer_evaluation_data in eval_results_data["answer_evaluations"]:
		answer_evaluations.append(AnswerEvaluation(**answer_evaluation_data))
	eval_qna = []
	for eval_qna_data in eval_results_data["eval_qna"]:
		eval_qna.append(EvalQnA(**eval_qna_data))

	eval_results = EvalResults(
		settings=eval_results_data["settings"],
		settings_hash=eval_results_data["settings_hash"],
		retrieval_evaluations=retrieval_evaluations,
		retrieved_list=retrieved_list,
		retrieval_evaluation_list=retrieval_evaluation_list,
		generated_list=generated_list,
		answer_evaluations=answer_evaluations,
		eval_qna=eval_qna,
	)
	return eval_results
