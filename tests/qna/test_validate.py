"""
test_validate.py

Tests for qna.validate.validate_eval_qna_args.
"""

import pytest

from qna.validate import validate_eval_qna_args


def test_valid_args_pass(make_eval_qna_args):
    """
    Input: valid EvalQnAArgs.
    Output: nothing raised.
    What it does: checks valid settings pass. The QnA LLMs are checked by llm.validate.validate_llm.
    """
    validate_eval_qna_args(make_eval_qna_args())


@pytest.mark.parametrize("eval_qna_model", ["", "   "])
def test_empty_eval_qna_model_raises(make_eval_qna_args, eval_qna_model):
    """
    Input: an empty or whitespace-only eval_qna_model.
    Output: ValueError mentioning eval_qna_model.
    What it does: checks the embedding model RAGAS uses must be given.
    """
    with pytest.raises(ValueError, match="eval_qna_model"):
        validate_eval_qna_args(make_eval_qna_args(eval_qna_model=eval_qna_model))


@pytest.mark.parametrize("n_sample_documents", [0, -1])
def test_n_sample_documents_not_positive_raises(make_eval_qna_args, n_sample_documents):
    """
    Input: n_sample_documents of 0 or below.
    Output: ValueError mentioning n_sample_documents.
    What it does: checks at least one document must be sampled.
    """
    with pytest.raises(ValueError, match="n_sample_documents"):
        validate_eval_qna_args(make_eval_qna_args(n_sample_documents=n_sample_documents))


@pytest.mark.parametrize("n_sample_evals", [0, -1])
def test_n_sample_evals_not_positive_raises(make_eval_qna_args, n_sample_evals):
    """
    Input: n_sample_evals of 0 or below.
    Output: ValueError mentioning n_sample_evals.
    What it does: checks at least one eval QnA item must be asked for.
    """
    with pytest.raises(ValueError, match="n_sample_evals"):
        validate_eval_qna_args(make_eval_qna_args(n_sample_evals=n_sample_evals))
