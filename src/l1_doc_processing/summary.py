"""
summary.py

Counts and prints the end-of-run summary of doc processing: documents
loaded, newly processed and skipped by reason, and document parse times.
"""

import dataclasses


@dataclasses.dataclass
class SummaryCounts:
	"""
	Counts of rows and documents skipped during one process() run, by reason.
	"""
	filtered: int = 0
	decode_error: int = 0
	parse_error: int = 0
	skipped: int = 0
	duplicate: int = 0


def print_run_summary(cached_count: int, total_count: int, summary_counts: SummaryCounts) -> None:
	"""
	cached_count: documents loaded from disk at the start of the run.
	total_count: documents held at the end of the run.
	summary_counts: rows and documents skipped in this run, by reason.

	Prints how many documents were loaded from cache, newly processed,
	and skipped by reason.

	Returns nothing.
	"""
	print(f"Loaded {cached_count} already processed document(s).")
	print(f"Newly processed {total_count - cached_count} document(s). Total: {total_count}.")
	print(f"Skipped {summary_counts.filtered} row(s) not matching the dataset's row_filter.")
	print(f"Skipped {summary_counts.decode_error} row(s) due to decode or download errors.")
	print(f"Skipped {summary_counts.parse_error} document(s) due to parse errors.")
	print(f"Skipped {summary_counts.skipped} document(s) that failed in an earlier run.")
	print(f"Skipped {summary_counts.duplicate} document(s) due to duplicate detection.")


def print_parse_time_summary(parse_times: list[tuple[str, float]]) -> None:
	"""
	parse_times: (doc_id, seconds) for every document parsed successfully
		in this run.

	Prints how many documents were parsed and their total, average,
	fastest and slowest parse time, with the fastest and slowest doc_id.

	Returns nothing.
	"""
	print("\n----- Document parse time -----")
	if len(parse_times) == 0:
		print("No documents were parsed in this run.")
		return

	total_seconds = 0.0
	fastest_doc_id, fastest_seconds = parse_times[0]
	slowest_doc_id, slowest_seconds = parse_times[0]
	for doc_id, seconds in parse_times:
		total_seconds += seconds
		if seconds < fastest_seconds:
			fastest_doc_id = doc_id
			fastest_seconds = seconds
		if seconds > slowest_seconds:
			slowest_doc_id = doc_id
			slowest_seconds = seconds
	average_seconds = total_seconds / len(parse_times)

	print(f"Documents parsed : {len(parse_times)}")
	print(f"Total            : {total_seconds:.3f}s")
	print(f"Average          : {average_seconds:.3f}s")
	print(f"Fastest          : {fastest_seconds:.3f}s ({fastest_doc_id})")
	print(f"Slowest          : {slowest_seconds:.3f}s ({slowest_doc_id})")
