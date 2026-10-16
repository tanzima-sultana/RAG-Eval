"""
diagnosis.py

Explains where answers lose points: answer scores split by whether retrieval found the source, and by question type,
and the lowest scoring answers next to their reference answers.
"""

from analysis.report import get_setup_cells
from analysis.summary import get_setup_key
from common import constants
from l6_retrieval.schema import Retrieved
from l9_generation.schema import Generated
from l10_evaluation.schema import AnswerEvaluation
from l10_evaluation.schema import RetrievalEvaluation
from qna.schema import EvalQnA

def get_answer_score_cells(answer_evaluation_list: list[AnswerEvaluation]) -> str:
	"""
	Input: answer_evaluation_list, the answer evaluations of one table row.
	Output: the row's last cells, " n | idk | correct | faithful | relevant |", as Markdown.
	What it does: counts the answers and averages the I_DONT_KNOW share and the three scores.
	"""
	answer_count = len(answer_evaluation_list)
	i_dont_know_sum = 0.0
	correctness_sum = 0.0
	faithfulness_sum = 0.0
	relevance_sum = 0.0
	for answer_evaluation in answer_evaluation_list:
		i_dont_know_sum = i_dont_know_sum + answer_evaluation.i_dont_know
		correctness_sum = correctness_sum + answer_evaluation.correctness
		faithfulness_sum = faithfulness_sum + answer_evaluation.faithfulness
		relevance_sum = relevance_sum + answer_evaluation.relevance

	answer_score_cells = (
		f" {answer_count} | {i_dont_know_sum / answer_count:.3f} | {correctness_sum / answer_count:.3f} |"
		f" {faithfulness_sum / answer_count:.3f} | {relevance_sum / answer_count:.3f} |"
	)
	return answer_score_cells

def get_diagnosis(retrieved_list: list[Retrieved], retrieval_evaluation_list: list[RetrievalEvaluation], answer_evaluations: list[AnswerEvaluation], eval_qna: list[EvalQnA]) -> str:
	"""
	Input: retrieved_list, retrieval_evaluation_list and answer_evaluations, in the same order; eval_qna, the eval
	questions.
	Output: the diagnosis section of the report, as Markdown text.
	What it does: per kept setup, averages the answer scores of the answers whose retrieval hit the source and of those
	whose retrieval missed it, and averages the answer scores per question type.
	"""
	eval_qna_by_id = {}
	for eval_qna_item in eval_qna:
		eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	# 1. Group the answers by setup and chunk-level hit or miss, and by setup and question type.
	# A chunk-level hit means at least one retrieved chunk overlaps the source passage the question was written from
	evaluations_by_setup_and_hit = {}
	evaluations_by_setup_and_qus_type = {}
	for index in range(len(retrieved_list)):
		setup_key = get_setup_key(retrieved_list[index])
		answer_evaluation = answer_evaluations[index]

		if retrieval_evaluation_list[index].chunk_evaluation.hit == 1:
			hit_text = "hit"
		else:
			hit_text = "miss"
		hit_key = (setup_key, hit_text)
		if hit_key not in evaluations_by_setup_and_hit:
			evaluations_by_setup_and_hit[hit_key] = []
		evaluations_by_setup_and_hit[hit_key].append(answer_evaluation)

		qus_type = eval_qna_by_id[retrieved_list[index].id].qus_type
		qus_type_key = (setup_key, qus_type)
		if qus_type_key not in evaluations_by_setup_and_qus_type:
			evaluations_by_setup_and_qus_type[qus_type_key] = []
		evaluations_by_setup_and_qus_type[qus_type_key].append(answer_evaluation)

	# 2. The table of answer scores by retrieval hit or miss, in setup order
	lines = []
	lines.append("## Diagnosis")
	lines.append("")
	lines.append("### Answer scores by retrieval hit")
	lines.append("")
	lines.append("A hit means a retrieved chunk overlaps the passage the question was written from; on a miss the LLM never saw it. I don't know answers count as 0.")
	lines.append("")
	lines.append("| Chunk | Retrieval | Index | Query | Rerank | k | Source | n | idk | correct | faithful | relevant |")
	lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
	for hit_key in sorted(evaluations_by_setup_and_hit):
		setup_key, hit_text = hit_key
		row = get_setup_cells(setup_key) + f" {hit_text} |" + get_answer_score_cells(evaluations_by_setup_and_hit[hit_key])
		lines.append(row)

	# 3. The table of answer scores by question type, in setup order
	lines.append("")
	lines.append("### Answer scores by question type")
	lines.append("")
	lines.append("| Chunk | Retrieval | Index | Query | Rerank | k | Question type | n | idk | correct | faithful | relevant |")
	lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
	for qus_type_key in sorted(evaluations_by_setup_and_qus_type):
		setup_key, qus_type = qus_type_key
		row = get_setup_cells(setup_key) + f" {qus_type} |" + get_answer_score_cells(evaluations_by_setup_and_qus_type[qus_type_key])
		lines.append(row)

	diagnosis = "\n".join(lines)
	return diagnosis

def get_worst_answers(retrieved_list: list[Retrieved], generated_list: list[Generated], answer_evaluations: list[AnswerEvaluation], eval_qna: list[EvalQnA], n_worst_answers: int) -> str:
	"""
	Input: retrieved_list, generated_list and answer_evaluations, in the same order; eval_qna, the eval questions;
	n_worst_answers, how many answers to list.
	Output: the worst answers section of the report, as Markdown text.
	What it does: lists the n_worst_answers answers with the lowest correctness, then faithfulness, each with its setup,
	question, answer, reference answer and scores. I_DONT_KNOW answers are left out, since the idk column already
	counts them and they would fill the list with 0 scores.
	"""
	eval_qna_by_id = {}
	for eval_qna_item in eval_qna:
		eval_qna_by_id[eval_qna_item.id] = eval_qna_item

	# 1. Sort the answers by correctness, then faithfulness, lowest first, and keep n_worst_answers
	scored_answers = []
	for index in range(len(generated_list)):
		if generated_list[index].answer == constants.I_DONT_KNOW:
			continue
		answer_evaluation = answer_evaluations[index]
		scored_answers.append((answer_evaluation.correctness, answer_evaluation.faithfulness, index))
	worst_scored_answers = sorted(scored_answers)[:n_worst_answers]

	# 2. Write each with its setup, question, answer, reference answer and scores
	lines = []
	lines.append("## Worst answers")
	lines.append("")
	lines.append(f"The {len(worst_scored_answers)} answers with the lowest correctness, then faithfulness. I don't know answers are left out.")
	rank = 0
	for correctness, faithfulness, index in worst_scored_answers:
		rank = rank + 1
		retrieved = retrieved_list[index]
		generated = generated_list[index]
		answer_evaluation = answer_evaluations[index]
		eval_qna_item = eval_qna_by_id[generated.id]
		setup_text = get_setup_cells(get_setup_key(retrieved))

		lines.append("")
		lines.append(f"### {rank}. correctness {correctness:.3f}, faithfulness {faithfulness:.3f}, relevance {answer_evaluation.relevance:.3f}")
		lines.append("")
		lines.append(f"- **Setup** (chunk, retrieval, index, query, rerank, k): {setup_text}")
		lines.append(f"- **Question type:** {eval_qna_item.qus_type}")
		lines.append(f"- **Question:** {generated.qus}")
		lines.append(f"- **Answer:** {generated.answer}")
		lines.append(f"- **Reference answer:** {eval_qna_item.ans}")

	worst_answers = "\n".join(lines)
	return worst_answers
