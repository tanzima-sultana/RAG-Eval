"""
test_validate.py

Tests for stages.question.query.validate.validate_query_args.
"""

import pytest

from stages.question.query.validate import validate_query_args


def test_valid_args_pass(make_query_args):
    """
    Input: valid QueryArgs.
    Output: nothing raised.
    What it does: checks valid settings pass. The query LLM is checked by llm.validate.validate_llm.
    """
    validate_query_args(make_query_args())


@pytest.mark.parametrize("embedding_model", ["", "   "])
def test_empty_embedding_model_raises(make_query_args, embedding_model):
    """
    Input: an empty or whitespace-only embedding_model.
    Output: ValueError mentioning embedding_model.
    What it does: checks the build's embedding model must be given.
    """
    with pytest.raises(ValueError, match="embedding_model"):
        validate_query_args(make_query_args(embedding_model=embedding_model))


def test_n_multi_qus_zero_raises(make_query_args):
    """
    Input: n_multi_qus 0.
    Output: ValueError mentioning n_multi_qus.
    What it does: checks at least one phrasing is asked for.
    """
    with pytest.raises(ValueError, match="n_multi_qus"):
        validate_query_args(make_query_args(n_multi_qus=0))
