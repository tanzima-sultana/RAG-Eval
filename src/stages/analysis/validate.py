from common.args import AnalysisArgs

def validate_analysis_args(analysis_args: AnalysisArgs):
	"""
	Input: analysis_args, n_compare_runs and n_worst_answers.
	Output: nothing. Raises ValueError if a setting is invalid.
	"""

	# 1. Check n_compare_runs. The comparison always holds the current run
	if analysis_args.n_compare_runs < 1:
		raise ValueError(
			f"n_compare_runs ({analysis_args.n_compare_runs}) must be at least 1, the current run"
		)

	# 2. Check n_worst_answers. 0 lists no worst answers
	if analysis_args.n_worst_answers < 0:
		raise ValueError(
			f"n_worst_answers ({analysis_args.n_worst_answers}) must not be negative"
		)
