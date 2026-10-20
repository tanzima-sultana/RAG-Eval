import time

from common import paths
from common.args import parse_eval_args
from common.args import make_ingest_args
from common.args import make_analysis_args
from stages.analysis.analysis import analyze_run
from stages.analysis.schema import AnalysisRun
from stages.evaluation.evaluation import cache as eval_cache

def run_analysis(args) -> AnalysisRun:
	"""
	Input: args, the parsed eval arguments, holding the settings of the eval run to analyse and the analysis settings.
	Output: the run's AnalysisRun.
	What it does: loads the results the evaluation stage saved for these settings, then summarises, diagnoses and
	compares the run with the most recent earlier runs, and saves it with its report.
	"""
	ingest_args = make_ingest_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	analysis_args = make_analysis_args(args)

	# 1. Load the eval results of the run with these settings
	print("\n\nLoading eval results")
	s1 = time.time()
	settings = eval_cache.get_eval_settings(args)
	eval_results = eval_cache.load_eval_results(output_path, eval_cache.get_settings_hash(settings))
	print(f"Eval results loaded : {paths.get_eval_results_path(output_path, eval_results.settings_hash)}")
	print(f"Retrieval evaluations : {len(eval_results.retrieval_evaluations)}, answer evaluations : {len(eval_results.answer_evaluations)}")
	print("\nLoading eval results time : ", time.time() - s1)

	# 2. Analysis: summaries, diagnosis, worst answers and comparison with earlier runs, saved with the run's report
	print("\n\nAnalysing the run")
	s2 = time.time()
	analysis_run = analyze_run(output_path, analysis_args, eval_results)
	print("\nAnalysing the run time : ", time.time() - s2)

	return analysis_run

if __name__ == "__main__":
	print("\n ------------- RAG Analysis -------- \n")
	start = time.time()

	args = parse_eval_args()
	print(args)

	run_analysis(args)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
