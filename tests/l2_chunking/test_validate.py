"""
test_validate.py

Tests for l2_chunking.validate.validate_chunking_args.
"""

import pytest

from l2_chunking.validate import validate_chunking_args


@pytest.mark.parametrize("chunk_type", ["fixed", "sentence", "semantic"])
def test_valid_args_pass_for_each_chunk_type(make_chunking_args, chunk_type):
    """
    Input: valid ChunkingArgs for each supported chunk_type.
    Output: nothing raised.
    What it does: checks valid args pass every check.
    """
    validate_chunking_args(make_chunking_args(chunk_type=chunk_type))


def test_langchain_framework_passes(make_chunking_args):
    """
    Input: framework "langchain".
    Output: nothing raised.
    What it does: checks "langchain" is an accepted framework.
    """
    validate_chunking_args(make_chunking_args(framework="langchain"))


def test_unknown_framework_raises(make_chunking_args):
    """
    Input: framework "llamaindex".
    Output: ValueError mentioning framework.
    What it does: checks only "" and "langchain" are accepted.
    """
    with pytest.raises(ValueError, match="framework"):
        validate_chunking_args(make_chunking_args(framework="llamaindex"))


@pytest.mark.parametrize("model_name", ["", "   "])
def test_empty_model_name_raises(make_chunking_args, model_name):
    """
    Input: an empty or whitespace-only model_name.
    Output: ValueError mentioning model_name.
    What it does: checks a model name must be given.
    """
    with pytest.raises(ValueError, match="model_name"):
        validate_chunking_args(make_chunking_args(model_name=model_name))


def test_unknown_chunk_type_raises(make_chunking_args):
    """
    Input: chunk_type "paragraph".
    Output: ValueError mentioning chunk_type.
    What it does: checks only fixed, sentence and semantic are accepted.
    """
    with pytest.raises(ValueError, match="chunk_type"):
        validate_chunking_args(make_chunking_args(chunk_type="paragraph"))


@pytest.mark.parametrize("min_chunk_size", [6, 7])
def test_min_chunk_size_not_smaller_than_max_raises(make_chunking_args, min_chunk_size):
    """
    Input: min_chunk_size equal to or larger than max_chunk_size (6).
    Output: ValueError mentioning min_chunk_size.
    What it does: checks min_chunk_size must be smaller than max_chunk_size.
    """
    with pytest.raises(ValueError, match="min_chunk_size"):
        validate_chunking_args(make_chunking_args(min_chunk_size=min_chunk_size, max_chunk_size=6))


@pytest.mark.parametrize("fixed_chunk_overlap", [6, 7])
def test_overlap_not_smaller_than_max_raises(make_chunking_args, fixed_chunk_overlap):
    """
    Input: fixed_chunk_overlap equal to or larger than max_chunk_size (6).
    Output: ValueError mentioning fixed_chunk_overlap.
    What it does: checks the overlap must be smaller than max_chunk_size.
    """
    with pytest.raises(ValueError, match="fixed_chunk_overlap"):
        validate_chunking_args(make_chunking_args(fixed_chunk_overlap=fixed_chunk_overlap, max_chunk_size=6))


def test_semantic_with_zero_threshold_raises(make_chunking_args):
    """
    Input: chunk_type "semantic" with semantic_threshold 0.0.
    Output: ValueError mentioning the semantic threshold.
    What it does: checks semantic chunking needs a non-zero threshold.
    """
    with pytest.raises(ValueError, match="semantic threshold"):
        validate_chunking_args(make_chunking_args(chunk_type="semantic", semantic_threshold=0.0))


def test_fixed_with_zero_threshold_passes(make_chunking_args):
    """
    Input: chunk_type "fixed" with semantic_threshold 0.0.
    Output: nothing raised.
    What it does: checks the threshold is only required for semantic.
    """
    validate_chunking_args(make_chunking_args(chunk_type="fixed", semantic_threshold=0.0))
