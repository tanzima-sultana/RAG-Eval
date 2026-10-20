"""
test_validate.py

Tests for stages.analysis.validate.validate_analysis_args.
"""

import pytest

from stages.analysis.validate import validate_analysis_args
from common.args import AnalysisArgs


def test_valid_args_pass():
    """
    Input: n_compare_runs 5 and n_worst_answers 10.
    Output: nothing raised.
    What it does: checks valid settings pass.
    """
    validate_analysis_args(AnalysisArgs(n_compare_runs=5, n_worst_answers=10))


def test_smallest_valid_args_pass():
    """
    Input: n_compare_runs 1 and n_worst_answers 0.
    Output: nothing raised.
    What it does: checks comparing only the current run and listing no worst answers are valid.
    """
    validate_analysis_args(AnalysisArgs(n_compare_runs=1, n_worst_answers=0))


@pytest.mark.parametrize("n_compare_runs", [0, -1])
def test_n_compare_runs_below_one_raises(n_compare_runs):
    """
    Input: n_compare_runs 0 or -1.
    Output: ValueError naming n_compare_runs.
    What it does: checks the comparison always holds the current run.
    """
    with pytest.raises(ValueError, match="n_compare_runs"):
        validate_analysis_args(AnalysisArgs(n_compare_runs=n_compare_runs, n_worst_answers=10))


def test_negative_n_worst_answers_raises():
    """
    Input: n_worst_answers -1.
    Output: ValueError naming n_worst_answers.
    What it does: checks a negative count is caught.
    """
    with pytest.raises(ValueError, match="n_worst_answers"):
        validate_analysis_args(AnalysisArgs(n_compare_runs=5, n_worst_answers=-1))
