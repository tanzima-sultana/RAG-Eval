"""
analysis.py

Analysis orchestrator. Runs on the results the evaluation stage saved: summarises the run's retrieval and answer
evaluations, diagnoses where answers lose points, lists the worst answers, saves the run and compares it with the
most recent earlier runs, then writes everything to the run's report.
"""

import datetime

from stages.analysis.cache import load_analysis_runs
from stages.analysis.cache import save_analysis_run
from stages.analysis.comparison import get_comparison
from stages.analysis.diagnosis import get_diagnosis
from stages.analysis.diagnosis import get_worst_answers
from stages.analysis.report import get_answer_summary_table
from stages.analysis.report import get_report
from stages.analysis.report import get_retrieval_summary_table
from stages.analysis.schema import AnalysisRun
from stages.analysis.summary import get_answer_summaries
from stages.analysis.summary import get_retrieval_summaries
from stages.analysis.validate import validate_analysis_args
from common import paths
from common.args import AnalysisArgs
from stages.evaluation.evaluation.schema import EvalResults

def analyze_run(output_path: str, analysis_args: AnalysisArgs, eval_results: EvalResults) -> AnalysisRun:
	"""
	Input: output_path, the dataset's output folder; analysis_args, n_compare_runs and n_worst_answers; eval_results,
	the saved results of one eval run: its settings, every (Retrieved, RetrievalEvaluation), the kept setups' Retrieved,
	RetrievalEvaluation, answers and answer evaluations in the same order, and the eval questions.
	Output: the run's AnalysisRun.
	What it does: builds the summaries, diagnosis, worst answers and comparison, saves the run with its report and
	prints the report.
	"""
	validate_analysis_args(analysis_args)
	settings = eval_results.settings
	settings_hash = eval_results.settings_hash
	retrieval_evaluations = eval_results.retrieval_evaluations
	retrieved_list = eval_results.retrieved_list
	retrieval_evaluation_list = eval_results.retrieval_evaluation_list
	generated_list = eval_results.generated_list
	answer_evaluations = eval_results.answer_evaluations
	eval_qna = eval_results.eval_qna

	# 1. Summaries: one RetrievalSummary per retrieval setup, one AnswerSummary per kept setup
	retrieval_summaries = get_retrieval_summaries(retrieval_evaluations)
	answer_summaries = get_answer_summaries(retrieved_list, answer_evaluations)
	print(f"Retrieval summaries : {len(retrieval_summaries)}")
	print(f"Answer summaries : {len(answer_summaries)}")

	# 2. The run record. The run id comes from the current time. Two runs with the same settings get the same
	# settings_hash, made by the evaluation stage
	now = datetime.datetime.now()
	run_id = now.strftime("%Y%m%d-%H%M%S")
	created_at = now.isoformat(timespec="seconds")

	# Every answer of one run is judged by the same framework and model
	if len(answer_evaluations) > 0:
		judge = f"{answer_evaluations[0].framework} {answer_evaluations[0].llm_model}"
	else:
		judge = ""

	analysis_run = AnalysisRun(
		run_id=run_id,
		created_at=created_at,
		settings=settings,
		settings_hash=settings_hash,
		judge=judge,
		retrieval_summaries=retrieval_summaries,
		answer_summaries=answer_summaries
	)
	print(f"Analysis run : {run_id}, judge : {judge}, settings hash : {settings_hash[:12]}")

	# 3. Diagnosis and worst answers, as Markdown sections of the report
	diagnosis = get_diagnosis(retrieved_list, retrieval_evaluation_list, answer_evaluations, eval_qna)
	worst_answers = get_worst_answers(retrieved_list, generated_list, answer_evaluations, eval_qna, analysis_args.n_worst_answers)

	# 4. Comparison with the n_compare_runs - 1 newest earlier runs. A saved run with the same settings_hash is this
	# run's earlier version, which this run replaces, so it is not compared
	saved_analysis_runs = load_analysis_runs(paths.get_analysis_runs_path(output_path))
	compared_analysis_runs = [analysis_run]
	for saved_analysis_run in saved_analysis_runs:
		if saved_analysis_run.settings_hash == analysis_run.settings_hash:
			continue
		if len(compared_analysis_runs) >= analysis_args.n_compare_runs:
			break
		compared_analysis_runs.append(saved_analysis_run)
	comparison = get_comparison(compared_analysis_runs)
	print(f"Runs compared : {len(compared_analysis_runs)}, the current run included")

	# 5. The report. The retrieval summary of the kept setups, the ones answers were generated for, comes first; every
	# setup's retrieval summary comes last, since it is long
	kept_retrieval_summaries = []
	for retrieval_summary in retrieval_summaries:
		for answer_summary in answer_summaries:
			if retrieval_summary.setup == answer_summary.setup:
				kept_retrieval_summaries.append(retrieval_summary)
				break

	sections = []
	sections.append(get_answer_summary_table(answer_summaries, judge))
	sections.append(get_retrieval_summary_table(kept_retrieval_summaries, "Retrieval evaluation of the kept setups"))
	sections.append(comparison)
	sections.append(diagnosis)
	sections.append(worst_answers)
	sections.append(get_retrieval_summary_table(retrieval_summaries, "Retrieval evaluation of every setup"))
	report = get_report(analysis_run, sections)

	# 6. Save the run with its report, then print the report
	run_folder = save_analysis_run(paths.get_analysis_runs_path(output_path), analysis_run, report)
	print("\n" + report)
	print(f"\nAnalysis run saved : {run_folder}")

	return analysis_run
