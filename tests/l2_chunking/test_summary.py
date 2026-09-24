"""
test_summary.py

Tests for l2_chunking.summary: the run summary, the sentence embedding
and document chunk time summaries, and the chunk statistics.
"""

import os

from l2_chunking.schema import Chunk
from l2_chunking.summary import (
    print_chunk_statistics,
    print_chunk_time_summary,
    print_run_summary,
    print_sentence_embedding_summary,
)


def test_print_run_summary_reports_cached_and_new_counts(capsys):
    """
    Input: 10 documents, 4 of them with 12 cached chunks, and 20 new chunks.
    Output: the printed summary.
    """
    print_run_summary(10, 4, 12, 20)

    output = capsys.readouterr().out
    assert "Loaded 10 processed document(s)." in output
    assert "Loaded cached chunks for 4 document(s): 12 chunk(s)." in output
    assert "Newly chunked 6 document(s): 20 chunk(s)." in output
    assert "Total chunks: 32." in output


def test_print_sentence_embedding_summary_reports_count_and_time(capsys):
    """
    Input: 150 sentences embedded in 2.5 seconds.
    Output: the printed summary.
    """
    print_sentence_embedding_summary(150, 2.5)

    output = capsys.readouterr().out
    assert "Sentences embedded : 150" in output
    assert "Total              : 2.500s" in output


def test_print_chunk_time_summary_reports_count_total_average_fastest_and_slowest(capsys):
    """
    Input: three (doc_id, seconds) chunk times.
    Output: the printed summary.
    """
    print_chunk_time_summary([("pdf_a", 0.2), ("pdf_b", 0.5), ("pdf_c", 0.2)])

    output = capsys.readouterr().out
    assert "Documents chunked : 3" in output
    assert "Total             : 0.900s" in output
    assert "Average           : 0.300s" in output
    assert "Fastest           : 0.200s (pdf_a)" in output
    assert "Slowest           : 0.500s (pdf_b)" in output


def test_print_chunk_time_summary_handles_no_chunked_documents(capsys):
    """
    Input: an empty list of chunk times.
    Output: the printed summary.
    """
    print_chunk_time_summary([])

    assert "No documents were chunked in this run." in capsys.readouterr().out


def test_print_chunk_statistics_reports_sizes_and_file_bytes(isolated_data_dir, capsys):
    """
    Input: two chunks of 2 and 4 tokens, and two chunk files of 10 and
    30 bytes.
    Output: the printed statistics.
    """
    chunk_path = str(isolated_data_dir / "chunks_out")
    os.makedirs(chunk_path)
    with open(os.path.join(chunk_path, "doc1_0.json"), "w", encoding="utf-8") as f:
        f.write("x" * 10)
    with open(os.path.join(chunk_path, "doc1_1.json"), "w", encoding="utf-8") as f:
        f.write("x" * 30)
    chunks = {
        "doc1_0": Chunk(id="doc1_0", text="a b", size=2, type="fixed"),
        "doc1_1": Chunk(id="doc1_1", text="a b c d", size=4, type="fixed"),
    }

    print_chunk_statistics(chunks, chunk_path)

    output = capsys.readouterr().out
    assert "Total chunks            : 2" in output
    assert "Average chunk size      : 3.0 tokens" in output
    assert "Total chunk file size   : 40 bytes" in output
    assert "Average chunk file size : 20.0 bytes" in output


def test_print_chunk_statistics_handles_no_chunks_and_no_folder(isolated_data_dir, capsys):
    """
    Input: no chunks and a chunk folder that does not exist.
    Output: the printed statistics.
    """
    print_chunk_statistics({}, str(isolated_data_dir / "missing"))

    output = capsys.readouterr().out
    assert "Total chunks            : 0" in output
    assert "Total chunk file size   : 0 bytes" in output
