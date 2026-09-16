"""
timing.py

A minimal, process-wide stopwatch for profiling the l1_doc_processing
pipeline. Every stage (downloading, raw caching, parsing, text
cleanup, metadata extraction, dedup, ...) records how many seconds it
spent into one shared set of totals, so main.py can print a single
breakdown at the end of a run without threading timing arguments
through every function call.
"""

import time

_total_seconds_by_stage: dict[str, float] = {}
_call_count_by_stage: dict[str, int] = {}


def record(stage_name: str, elapsed_seconds: float) -> None:
	"""
	stage_name: which pipeline stage this measurement belongs to
		(e.g. "download_row", "parse_pdf", "text_cleanup").
	elapsed_seconds: how many seconds this single occurrence of the
		stage took.

	Adds elapsed_seconds to stage_name's running total, and
	increments stage_name's running call count. Both start at zero
	the first time a given stage_name is recorded.

	Returns nothing.
	"""
	if stage_name not in _total_seconds_by_stage:
		_total_seconds_by_stage[stage_name] = 0.0
		_call_count_by_stage[stage_name] = 0

	_total_seconds_by_stage[stage_name] += elapsed_seconds
	_call_count_by_stage[stage_name] += 1


def reset() -> None:
	"""
	Clears every previously recorded stage, so a fresh run starts
	from an empty breakdown instead of accumulating across runs
	within the same process.

	Returns nothing.
	"""
	_total_seconds_by_stage.clear()
	_call_count_by_stage.clear()


def print_summary() -> None:
	"""
	Prints every recorded stage's total elapsed time, call count, and
	average time per call, ordered from the slowest stage's total
	time to the fastest, so the biggest contributor to the run's
	total time is easiest to spot.

	Returns nothing.
	"""
	stage_totals = []
	for stage_name in _total_seconds_by_stage:
		total_seconds = _total_seconds_by_stage[stage_name]
		stage_totals.append((total_seconds, stage_name))
	stage_totals.sort(reverse=True)

	print("\n----- Timing breakdown (slowest stage first) -----")
	for total_seconds, stage_name in stage_totals:
		call_count = _call_count_by_stage[stage_name]
		average_seconds = total_seconds / call_count
		print(
			f"{stage_name:24s} total={total_seconds:9.3f}s  "
			f"calls={call_count:6d}  avg={average_seconds:9.5f}s"
		)
	print("----------------------------------------------------\n")
