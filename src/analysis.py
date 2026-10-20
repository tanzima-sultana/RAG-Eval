import time

from common import paths
from common.args import parse_eval_args
from common.args import make_ingest_args
from common.args import make_analysis_args
from stages.analysis.cache import load_analysis_runs
from stages.analysis.comparison import get_comparison
from stages.analysis.validate import validate_analysis_args

def run_analysis(args) -> str:
	"""
	Input: args, the parsed eval arguments, holding the build settings and n_compare_runs.
	Output: the comparison section, as Markdown text.
	What it does: loads the runs eval saved under {output_path}/analysis/runs, compares the n_compare_runs newest
	ones, the newest first, and prints the comparison. Raises ValueError if no run is saved yet.
	"""
	ingest_args = make_ingest_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	analysis_args = make_analysis_args(args)
	validate_analysis_args(analysis_args)

	# 1. Load the saved runs, newest first
	analysis_runs_path = paths.get_analysis_runs_path(output_path)
	saved_analysis_runs = load_analysis_runs(analysis_runs_path)
	print(f"Analysis runs folder : {analysis_runs_path}")
	print(f"Saved runs : {len(saved_analysis_runs)}")
	if len(saved_analysis_runs) == 0:
		raise ValueError(f"No saved runs in {analysis_runs_path}. Run scripts/eval.sh first.")

	# 2. Compare the n_compare_runs newest runs
	compared_analysis_runs = []
	for saved_analysis_run in saved_analysis_runs:
		if len(compared_analysis_runs) >= analysis_args.n_compare_runs:
			break
		compared_analysis_runs.append(saved_analysis_run)
	comparison = get_comparison(compared_analysis_runs)
	print(f"Runs compared : {len(compared_analysis_runs)}")
	print("\n" + comparison)

	return comparison

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
