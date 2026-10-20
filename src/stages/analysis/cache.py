"""
cache.py

Saves and loads the analysed runs, one folder per run under common.paths.get_analysis_runs_path:

{run_id}/run.json:  the AnalysisRun, read back by later runs to compare against.
{run_id}/report.md: the run's readable report.

A run with the same settings_hash as a saved one replaces it, so the history holds only different experiments.
"""

import dataclasses
import json
import os
import shutil

from stages.analysis.schema import AnalysisRun
from stages.analysis.schema import AnswerSummary
from stages.analysis.schema import RetrievalSummary
from stages.analysis.schema import Setup

def load_analysis_runs(analysis_runs_path: str) -> list[AnalysisRun]:
	"""
	Input: analysis_runs_path, the folder holding one subfolder per analysed run.
	Output: every saved AnalysisRun, newest first, or an empty list if the folder does not exist.
	What it does: reads each {run_id}/run.json back into an AnalysisRun, with its nested Setup and summary dataclasses.
	"""
	if not os.path.exists(analysis_runs_path):
		return []

	analysis_runs = []
	for run_folder_name in os.listdir(analysis_runs_path):
		run_file = os.path.join(analysis_runs_path, run_folder_name, "run.json")
		if not os.path.exists(run_file):
			continue
		with open(run_file, encoding="utf-8") as f:
			run_data = json.load(f)

		# JSON holds the nested dataclasses as dicts, so each summary and its Setup are rebuilt
		retrieval_summaries = []
		for retrieval_summary_data in run_data["retrieval_summaries"]:
			retrieval_summary_data["setup"] = Setup(**retrieval_summary_data["setup"])
			retrieval_summaries.append(RetrievalSummary(**retrieval_summary_data))
		answer_summaries = []
		for answer_summary_data in run_data["answer_summaries"]:
			answer_summary_data["setup"] = Setup(**answer_summary_data["setup"])
			# Runs saved before unscored existed had every answer scored
			if "unscored" not in answer_summary_data:
				answer_summary_data["unscored"] = 0
			answer_summaries.append(AnswerSummary(**answer_summary_data))

		analysis_run = AnalysisRun(
			run_id=run_data["run_id"],
			created_at=run_data["created_at"],
			settings=run_data["settings"],
			settings_hash=run_data["settings_hash"],
			judge=run_data["judge"],
			retrieval_summaries=retrieval_summaries,
			answer_summaries=answer_summaries
		)
		analysis_runs.append(analysis_run)

	# Newest first. created_at is an ISO date and time, so sorting the text sorts by time
	sort_keys = []
	for index in range(len(analysis_runs)):
		sort_keys.append((analysis_runs[index].created_at, index))
	newest_first_runs = []
	for created_at, index in sorted(sort_keys, reverse=True):
		newest_first_runs.append(analysis_runs[index])

	return newest_first_runs

def save_analysis_run(analysis_runs_path: str, analysis_run: AnalysisRun, report: str) -> str:
	"""
	Input: analysis_runs_path, the folder holding one subfolder per analysed run; analysis_run, the current run;
	report, its Markdown report.
	Output: the folder the run was saved to.
	What it does: removes a saved run with the same settings_hash, then writes run.json and report.md to
	{analysis_runs_path}/{run_id}/, each through a temporary file.
	"""
	# 1. Remove the folder of a saved run with the same settings_hash, the earlier version of this run
	for saved_analysis_run in load_analysis_runs(analysis_runs_path):
		if saved_analysis_run.settings_hash == analysis_run.settings_hash:
			shutil.rmtree(os.path.join(analysis_runs_path, saved_analysis_run.run_id))
			print(f"Replaced the earlier run with the same settings : {saved_analysis_run.run_id}")

	# 2. Write run.json and report.md. default=str writes any setting JSON cannot hold as text
	run_folder = os.path.join(analysis_runs_path, analysis_run.run_id)
	os.makedirs(run_folder, exist_ok=True)

	run_file = os.path.join(run_folder, "run.json")
	temporary_run_file = run_file + ".tmp"
	with open(temporary_run_file, "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(analysis_run), f, indent=2, default=str)
	os.replace(temporary_run_file, run_file)

	report_file = os.path.join(run_folder, "report.md")
	temporary_report_file = report_file + ".tmp"
	with open(temporary_report_file, "w", encoding="utf-8") as f:
		f.write(report)
	os.replace(temporary_report_file, report_file)

	return run_folder
