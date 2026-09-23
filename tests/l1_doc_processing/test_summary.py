"""
test_summary.py

Tests for l1_doc_processing.summary: the run summary and the document
parse time summary.
"""

from l1_doc_processing.summary import RunCounts, print_parse_time_summary, print_run_summary


def test_print_run_summary_reports_every_count(capsys):
    """
    Input: cached and total counts, and a RunCounts with every field set.
    Output: the printed summary.
    """
    run_counts = RunCounts(filtered=5, decode_error=4, previously_skipped=3, parse_error=2, duplicate=1)

    print_run_summary(10, 25, run_counts)

    output = capsys.readouterr().out
    assert "Loaded 10 already processed document(s)." in output
    assert "Newly processed 15 document(s). Total: 25." in output
    assert "Skipped 5 row(s) not matching the dataset's row_filter." in output
    assert "Skipped 4 row(s) due to decode or download errors." in output
    assert "Skipped 3 document(s) that failed in an earlier run." in output
    assert "Skipped 2 document(s) due to parse errors." in output
    assert "Skipped 1 document(s) due to duplicate detection." in output


def test_print_parse_time_summary_reports_count_total_average_fastest_and_slowest(capsys):
    """
    Input: three (doc_id, seconds) parse times.
    Output: the printed summary.
    """
    print_parse_time_summary([("pdf_a", 0.2), ("pdf_b", 0.5), ("pdf_c", 0.2)])

    output = capsys.readouterr().out
    assert "Documents parsed : 3" in output
    assert "Total            : 0.900s" in output
    assert "Average          : 0.300s" in output
    assert "Fastest          : 0.200s (pdf_a)" in output
    assert "Slowest          : 0.500s (pdf_b)" in output


def test_print_parse_time_summary_handles_no_parsed_documents(capsys):
    """
    Input: an empty list of parse times.
    Output: the printed summary.
    """
    print_parse_time_summary([])

    assert "No documents were parsed in this run." in capsys.readouterr().out
