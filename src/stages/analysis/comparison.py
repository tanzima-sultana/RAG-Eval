"""
comparison.py

Compares the current run with the most recent earlier runs: the best setup of each run, and the settings that
changed between them.
"""

from stages.analysis.report import get_setup_cells
from stages.analysis.schema import AnalysisRun

def get_comparison(analysis_runs: list[AnalysisRun]) -> str:
	"""
	Input: analysis_runs, the current run first, then the earlier runs, newest first.
	Output: the comparison section of the report, as Markdown text. With only the current run, its best setup alone.
	What it does: picks the best setup of each run, by mean correctness then faithfulness, or by chunk recall then MRR
	for a run without answers, and lists them with each run's judge and the settings that differ from the current run.
	"""
	current_run = analysis_runs[0]

	lines = []
	lines.append("## Comparison with earlier runs")
	lines.append("")
	if len(analysis_runs) == 1:
		lines.append("There is no earlier run with different settings to compare with. The best setup of this run:")
	else:
		lines.append(f"The best setup of this run and of the {len(analysis_runs) - 1} most recent earlier runs, newest first. Answer scores from a different judge are marked, since they are not directly comparable.")
	lines.append("")
	lines.append("| Run | Judge | Chunk | Retrieval | Index | Query | Rerank | k | c_recall | c_mrr | n | idk | correct | faithful | relevant |")
	lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

	# 1. One row per run with its best setup
	for analysis_run in analysis_runs:
		if analysis_run.run_id == current_run.run_id:
			run_text = f"{analysis_run.run_id} (current)"
		else:
			run_text = analysis_run.run_id

		if analysis_run.judge == current_run.judge:
			judge_text = analysis_run.judge
		else:
			judge_text = f"{analysis_run.judge} (different judge)"

		# Pick the best setup: by correctness then faithfulness, or by chunk recall then MRR when the run has no answers
		if len(analysis_run.answer_summaries) > 0:
			best_answer_summary = analysis_run.answer_summaries[0]
			for answer_summary in analysis_run.answer_summaries:
				if (answer_summary.correctness, answer_summary.faithfulness) > (best_answer_summary.correctness, best_answer_summary.faithfulness):
					best_answer_summary = answer_summary
			best_setup = best_answer_summary.setup
			answer_cells = (
				f" {best_answer_summary.n} | {best_answer_summary.i_dont_know:.3f} | {best_answer_summary.correctness:.3f} |"
				f" {best_answer_summary.faithfulness:.3f} | {best_answer_summary.relevance:.3f} |"
			)
		else:
			best_retrieval_summary = analysis_run.retrieval_summaries[0]
			for retrieval_summary in analysis_run.retrieval_summaries:
				if (retrieval_summary.chunk_recall, retrieval_summary.chunk_mrr) > (best_retrieval_summary.chunk_recall, best_retrieval_summary.chunk_mrr):
					best_retrieval_summary = retrieval_summary
			best_setup = best_retrieval_summary.setup
			answer_cells = " - | - | - | - | - |"

		# The retrieval scores of the best setup, from the run's retrieval summaries
		retrieval_cells = " - | - |"
		for retrieval_summary in analysis_run.retrieval_summaries:
			if retrieval_summary.setup == best_setup:
				retrieval_cells = f" {retrieval_summary.chunk_recall:.3f} | {retrieval_summary.chunk_mrr:.3f} |"

		setup_key = (best_setup.chunk_type, best_setup.retrieval_type, best_setup.index_type, best_setup.query_type, best_setup.reranking, best_setup.k)
		row = f"| {run_text} | {judge_text} " + get_setup_cells(setup_key) + retrieval_cells + answer_cells
		lines.append(row)

	# 2. The settings of each earlier run that differ from the current run's
	if len(analysis_runs) > 1:
		lines.append("")
		lines.append("### Settings that differ from the current run")
		for analysis_run in analysis_runs[1:]:
			setting_names = set(current_run.settings)
			for setting_name in analysis_run.settings:
				setting_names.add(setting_name)

			differences = []
			for setting_name in sorted(setting_names):
				current_value = current_run.settings.get(setting_name, "(not set)")
				earlier_value = analysis_run.settings.get(setting_name, "(not set)")
				if current_value != earlier_value:
					# A prompt spans several lines, which would break the Markdown list, so its lines are joined with " / "
					earlier_text = str(earlier_value).replace("\n", " / ")
					current_text = str(current_value).replace("\n", " / ")
					differences.append(f"`{setting_name}`: {earlier_text} → {current_text}")

			lines.append("")
			if len(differences) == 0:
				lines.append(f"- **{analysis_run.run_id}:** no setting differs.")
			else:
				lines.append(f"- **{analysis_run.run_id}:** " + "; ".join(differences))

	comparison = "\n".join(lines)
	return comparison
