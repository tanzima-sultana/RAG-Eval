"""
test_validate.py

Tests for stages.evaluation.generation.validate.validate_generation_args.
"""

import pytest

from common.args import GenerationArgs
from llm.schema import LLMArgs
from stages.evaluation.generation.validate import validate_generation_args


def make_generation_args(batch_poll_seconds, batch_timeout_seconds):
    """
    Input: the two batch settings.
    Output: a GenerationArgs with a fixed Ollama LLM and the given batch settings.
    What it does: builds the args the tests check.
    """
    llm_args = LLMArgs(
        llm="ollama",
        llm_model="llama3.1:8b",
        input_price=0.0,
        output_price=0.0,
        cache_write_price=0.0,
        cache_read_price=0.0,
        max_tokens=1000,
        timeout_seconds=60,
        cache_ttl="5m"
    )
    return GenerationArgs(llm_args=llm_args, batch_poll_seconds=batch_poll_seconds, batch_timeout_seconds=batch_timeout_seconds, framework="")


def test_valid_args_pass():
    """
    Input: batch_poll_seconds 60 and batch_timeout_seconds 86400.
    Output: nothing raised.
    What it does: checks valid settings pass.
    """
    validate_generation_args(make_generation_args(60, 86400))


def test_timeout_equal_to_poll_passes():
    """
    Input: batch_timeout_seconds equal to batch_poll_seconds.
    Output: nothing raised.
    What it does: checks a timeout of one poll still checks the batch once.
    """
    validate_generation_args(make_generation_args(60, 60))


@pytest.mark.parametrize("batch_poll_seconds", [0, -1])
def test_poll_below_one_raises(batch_poll_seconds):
    """
    Input: batch_poll_seconds 0 or -1.
    Output: ValueError naming batch_poll_seconds.
    What it does: checks the batch status is never checked without a pause.
    """
    with pytest.raises(ValueError, match="batch_poll_seconds"):
        validate_generation_args(make_generation_args(batch_poll_seconds, 86400))


def test_timeout_below_poll_raises():
    """
    Input: batch_timeout_seconds 30 and batch_poll_seconds 60.
    Output: ValueError naming batch_timeout_seconds.
    What it does: checks the run does not give up before its first status check.
    """
    with pytest.raises(ValueError, match="batch_timeout_seconds"):
        validate_generation_args(make_generation_args(60, 30))
