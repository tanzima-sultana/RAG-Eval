"""
test_main.py

Tests for l2_chunking.main's CLI wiring: argument parsing and the
semantic-threshold-required-for-semantic-chunking check. The
if __name__ == "__main__" block itself is not covered here, since it
only runs the pipeline end-to-end (already covered at the Chunking
level in test_chunking.py) and is not exercised on import.
"""

import argparse

import pytest

from l2_chunking.main import parse_args, validate


# ---------- validate ----------


def test_validate_passes_when_chunk_type_is_fixed_and_semantic_threshold_is_default():
    """
    Input: parsed args with chunk_type "fixed" and the default
    semantic_threshold (0.0).
    Output: none; validate() is expected not to raise/exit.

    Checks that the semantic-threshold requirement only applies to
    semantic chunking.
    """
    args = argparse.Namespace(chunk_type="fixed", semantic_threshold=0.0)

    validate(args)


def test_validate_passes_when_chunk_type_is_semantic_and_threshold_is_nonzero():
    """
    Input: parsed args with chunk_type "semantic" and a nonzero
    semantic_threshold.
    Output: none; validate() is expected not to raise/exit.
    """
    args = argparse.Namespace(chunk_type="semantic", semantic_threshold=0.5)

    validate(args)


def test_validate_exits_when_chunk_type_is_semantic_and_threshold_is_default():
    """
    Input: parsed args with chunk_type "semantic" and the default
    semantic_threshold (0.0), i.e. never actually provided.
    Output: none; validate() is expected to call sys.exit.
    """
    args = argparse.Namespace(chunk_type="semantic", semantic_threshold=0.0)

    with pytest.raises(SystemExit):
        validate(args)


# ---------- parse_args ----------


def test_parse_args_parses_required_and_optional_arguments(monkeypatch):
    """
    Input: sys.argv set to a full set of CLI flags.
    Output: parse_args()'s return value.

    Checks that every flag lands on the right attribute, with the
    right type conversion (max_chunk_size/fixed_chunk_overlap become
    int, semantic_threshold becomes float).
    """
    argv = [
        "main.py",
        "--chunk_type", "fixed",
        "--max_chunk_size", "256",
        "--fixed_chunk_overlap", "32",
        "--model_name", "all-MiniLM-L6-v2",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.chunk_type == "fixed"
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
        "--chunk_type", "semantic",
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
        "--chunk_type", "fixed",
        "--max_chunk_size", "256",
        "--fixed_chunk_overlap", "32",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        parse_args()
