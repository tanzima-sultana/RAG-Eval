"""
test_validate.py

Tests for l6_retrieval.validate.validate_retrieval_args.
"""

import pytest

from l6_retrieval.validate import validate_retrieval_args


def test_valid_args_pass(make_retrieval_args):
    """
    Input: all three retrieval types, top_k 3 and rerank_top_k 2.
    Output: nothing raised.
    What it does: checks valid settings pass.
    """
    validate_retrieval_args(make_retrieval_args())


def test_rerank_top_k_equal_to_top_k_passes(make_retrieval_args):
    """
    Input: rerank_top_k equal to top_k.
    Output: nothing raised.
    What it does: checks the reranker may keep every chunk the retriever returns.
    """
    validate_retrieval_args(make_retrieval_args(top_k=3, rerank_top_k=3))


def test_unknown_retrieval_type_raises(make_retrieval_args):
    """
    Input: retrieval_types with "sparse".
    Output: ValueError naming the supported types.
    What it does: checks a misspelled retrieval type is caught.
    """
    with pytest.raises(ValueError, match="sparse"):
        validate_retrieval_args(make_retrieval_args(retrieval_types=["dense", "sparse"]))


def test_unknown_query_type_raises(make_retrieval_args):
    """
    Input: query_types with "expanded".
    Output: ValueError naming the supported types.
    What it does: checks a misspelled query type is caught.
    """
    with pytest.raises(ValueError, match="expanded"):
        validate_retrieval_args(make_retrieval_args(query_types=["original", "expanded"]))


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_below_one_raises(make_retrieval_args, top_k):
    """
    Input: top_k of 0 or below.
    Output: ValueError mentioning top_k.
    What it does: checks every retriever returns at least one chunk.
    """
    with pytest.raises(ValueError, match="top_k"):
        validate_retrieval_args(make_retrieval_args(top_k=top_k, rerank_top_k=1))


@pytest.mark.parametrize("rerank_top_k", [0, -1])
def test_rerank_top_k_below_one_raises(make_retrieval_args, rerank_top_k):
    """
    Input: rerank_top_k of 0 or below.
    Output: ValueError mentioning rerank_top_k.
    What it does: checks the reranker keeps at least one chunk.
    """
    with pytest.raises(ValueError, match="rerank_top_k"):
        validate_retrieval_args(make_retrieval_args(rerank_top_k=rerank_top_k))


def test_rerank_top_k_above_top_k_raises(make_retrieval_args):
    """
    Input: top_k 3 and rerank_top_k 5.
    Output: ValueError saying rerank_top_k must not be more than top_k.
    What it does: checks the reranker is never asked for more chunks than it gets.
    """
    with pytest.raises(ValueError, match="must not be more than top_k"):
        validate_retrieval_args(make_retrieval_args(top_k=3, rerank_top_k=5))
