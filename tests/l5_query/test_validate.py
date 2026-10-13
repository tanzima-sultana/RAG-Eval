"""
test_validate.py

Tests for l5_query.validate.validate_query_args.
"""

import pytest

from l5_query.validate import validate_query_args


@pytest.mark.parametrize("query_llm", ["ollama", "claude"])
def test_valid_args_pass_for_each_query_llm(make_query_args, query_llm):
    """
    Input: valid QueryArgs for each supported query_llm.
    Output: nothing raised.
    What it does: checks valid args pass every check.
    """
    validate_query_args(make_query_args(query_llm=query_llm))


@pytest.mark.parametrize("embedding_model", ["", "   "])
def test_empty_embedding_model_raises(make_query_args, embedding_model):
    """
    Input: an empty or whitespace-only embedding_model.
    Output: ValueError mentioning embedding_model.
    What it does: checks the build's embedding model must be given.
    """
    with pytest.raises(ValueError, match="embedding_model"):
        validate_query_args(make_query_args(embedding_model=embedding_model))


def test_unknown_query_llm_raises(make_query_args):
    """
    Input: query_llm "gemini".
    Output: ValueError mentioning query_llm.
    What it does: checks only ollama and claude are accepted.
    """
    with pytest.raises(ValueError, match="query_llm"):
        validate_query_args(make_query_args(query_llm="gemini"))


def test_empty_query_llm_model_raises(make_query_args):
    """
    Input: a whitespace-only query_llm_model.
    Output: ValueError mentioning query_llm_model.
    What it does: checks a model name must be given.
    """
    with pytest.raises(ValueError, match="query_llm_model"):
        validate_query_args(make_query_args(query_llm_model="  "))


def test_n_multi_qus_zero_raises(make_query_args):
    """
    Input: n_multi_qus 0.
    Output: ValueError mentioning n_multi_qus.
    What it does: checks at least one phrasing is asked for.
    """
    with pytest.raises(ValueError, match="n_multi_qus"):
        validate_query_args(make_query_args(n_multi_qus=0))
