"""
test_main.py

Tests for l1_doc_processing.main's CLI wiring: argument parsing and
the dataset_url/dataset_path mutual-exclusivity check. The
if __name__ == "__main__" block itself is not covered here, since it
only runs the pipeline end-to-end (already covered at the DocProcessing
level in test_doc_processing.py) and is not exercised on import.
"""

import argparse

import pytest

from l1_doc_processing.main import parse_args, validate


# ---------- validate ----------


def test_validate_passes_when_only_dataset_url_is_given():
    """
    Input: parsed args with dataset_url set and dataset_path None.
    Output: none; validate() is expected not to raise/exit.
    """
    args = argparse.Namespace(dataset_url="some-url", dataset_path=None)

    validate(args)


def test_validate_passes_when_only_dataset_path_is_given():
    """
    Input: parsed args with dataset_path set and dataset_url None.
    Output: none; validate() is expected not to raise/exit.
    """
    args = argparse.Namespace(dataset_url=None, dataset_path="/some/path")

    validate(args)


def test_validate_exits_when_both_are_given():
    """
    Input: parsed args with both dataset_url and dataset_path set.
    Output: none; validate() is expected to call sys.exit.
    """
    args = argparse.Namespace(dataset_url="some-url", dataset_path="/some/path")

    with pytest.raises(SystemExit):
        validate(args)


def test_validate_exits_when_neither_is_given():
    """
    Input: parsed args with both dataset_url and dataset_path None.
    Output: none; validate() is expected to call sys.exit.
    """
    args = argparse.Namespace(dataset_url=None, dataset_path=None)

    with pytest.raises(SystemExit):
        validate(args)


# ---------- parse_args ----------


def test_parse_args_parses_required_and_optional_arguments(monkeypatch):
    """
    Input: sys.argv set to a full set of CLI flags.
    Output: parse_args()'s return value.

    Checks that every flag lands on the right attribute, with the
    right type conversion (dataset_size becomes an int).
    """
    argv = [
        "main.py",
        "--doc_type", "pdf",
        "--dataset_name", "CUAD",
        "--dataset_size", "20",
        "--dataset_url", "some-org/some-dataset",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.doc_type == "pdf"
    assert args.dataset_name == "CUAD"
    assert args.dataset_size == 20
    assert args.dataset_url == "some-org/some-dataset"
    assert args.dataset_path is None


def test_parse_args_exits_when_a_required_argument_is_missing(monkeypatch):
    """
    Input: sys.argv missing the required --doc_type flag.
    Output: none; parse_args() is expected to call sys.exit via
    argparse's own error handling.
    """
    argv = [
        "main.py",
        "--dataset_name", "CUAD",
        "--dataset_size", "20",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        parse_args()
