"""
test_main.py

Tests for stages.build.embedding.main's CLI argument parsing. The
if __name__ == "__main__" block itself is not covered here, since it
only runs the pipeline end-to-end (already covered at the Embedding
level in test_embedding.py) and is not exercised on import.

Unlike stages.build.chunking.main, stages.build.embedding.main has no validate(): chunk_type/
semantic_threshold consistency was already enforced by whichever
stages.build.chunking run produced the chunks being embedded here.
"""

import pytest

from stages.build.embedding.main import parse_args


def test_parse_args_parses_required_and_optional_arguments(monkeypatch):
    """
    Input: sys.argv set to a full set of CLI flags.
    Output: parse_args()'s return value.

    Checks that every flag lands on the right attribute, with the
    right type conversion (dataset_size/max_chunk_size/
    fixed_chunk_overlap become int, semantic_threshold becomes float).
    """
    argv = [
        "main.py",
        "--dataset_name", "myset",
        "--dataset_size", "100",
        "--chunk_type", "fixed",
        "--min_chunk_size", "128",
        "--max_chunk_size", "256",
        "--fixed_chunk_overlap", "32",
        "--model_name", "all-MiniLM-L6-v2",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.dataset_name == "myset"
    assert args.dataset_size == 100
    assert args.chunk_type == "fixed"
    assert args.min_chunk_size == 128
    assert args.max_chunk_size == 256
    assert args.fixed_chunk_overlap == 32
    assert args.semantic_threshold == 0.0
    assert args.model_name == "all-MiniLM-L6-v2"


def test_parse_args_parses_semantic_threshold_when_given(monkeypatch):
    """
    Input: sys.argv including an explicit --semantic_threshold flag.
    Output: parse_args()'s return value.
    """
    argv = [
        "main.py",
        "--dataset_name", "myset",
        "--dataset_size", "100",
        "--chunk_type", "semantic",
        "--min_chunk_size", "128",
        "--max_chunk_size", "256",
        "--fixed_chunk_overlap", "32",
        "--semantic_threshold", "0.4",
        "--model_name", "all-MiniLM-L6-v2",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.semantic_threshold == 0.4


def test_parse_args_exits_when_a_required_argument_is_missing(monkeypatch):
    """
    Input: sys.argv missing the required --model_name flag.
    Output: none; parse_args() is expected to call sys.exit via
    argparse's own error handling.
    """
    argv = [
        "main.py",
        "--dataset_name", "myset",
        "--dataset_size", "100",
        "--chunk_type", "fixed",
        "--min_chunk_size", "128",
        "--max_chunk_size", "256",
        "--fixed_chunk_overlap", "32",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        parse_args()
