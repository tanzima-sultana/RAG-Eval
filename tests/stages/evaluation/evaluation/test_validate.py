"""
test_validate.py

Tests for stages.evaluation.evaluation.validate.validate_n_retrieval_evaluation and validate_answer_eval_args.
"""

import pytest

from common.args import AnswerEvalArgs
from llm.schema import LLMArgs
from stages.evaluation.evaluation.validate import validate_answer_eval_args
from stages.evaluation.evaluation.validate import validate_n_retrieval_evaluation


def make_answer_eval_args(framework):
    """
    Input: framework, the answer evaluation framework.
    Output: an AnswerEvalArgs with a fixed Ollama judge and embedding model.
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
    return AnswerEvalArgs(framework=framework, llm_args=llm_args, embedding_model="all-MiniLM-L6-v2")


def test_n_retrieval_evaluation_of_one_passes():
    """
    Input: n_retrieval_evaluation 1.
    Output: nothing raised.
    What it does: checks keeping one setup of each kind is valid.
    """
    validate_n_retrieval_evaluation(1)


@pytest.mark.parametrize("n_retrieval_evaluation", [0, -1])
def test_n_retrieval_evaluation_below_one_raises(n_retrieval_evaluation):
    """
    Input: n_retrieval_evaluation 0 or -1.
    Output: ValueError naming n_retrieval_evaluation.
    What it does: checks the filter always keeps at least one setup.
    """
    with pytest.raises(ValueError, match="n_retrieval_evaluation"):
        validate_n_retrieval_evaluation(n_retrieval_evaluation)


def test_ragas_framework_passes():
    """
    Input: framework ragas.
    Output: nothing raised.
    What it does: checks the supported framework is valid.
    """
    validate_answer_eval_args(make_answer_eval_args("ragas"))


def test_unknown_framework_raises():
    """
    Input: framework deepeval.
    Output: ValueError naming the framework.
    What it does: checks an unsupported framework is caught.
    """
    with pytest.raises(ValueError, match="deepeval"):
        validate_answer_eval_args(make_answer_eval_args("deepeval"))
