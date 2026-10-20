"""
test_prompt.py

Tests for stages.question.query.prompt: each prompt fills with str.format() using only
the keys query.py passes.
"""

from stages.question.query.prompt import HYDE_PROMPT
from stages.question.query.prompt import MULTI_QUERY_PROMPT
from stages.question.query.prompt import REWRITE_PROMPT


def test_rewrite_prompt_fills_with_qus():
    """
    Input: REWRITE_PROMPT and a question.
    Output: the filled prompt.
    What it does: checks the prompt needs only {qus} and ends with the question.
    """
    filled_prompt = REWRITE_PROMPT.format(qus="What is FAISS?")

    assert filled_prompt.endswith("Question: What is FAISS?")


def test_multi_query_prompt_fills_with_qus_and_count():
    """
    Input: MULTI_QUERY_PROMPT, a question and n_multi_qus 3.
    Output: the filled prompt.
    What it does: checks the prompt needs only {qus} and {n_multi_qus}, and asks for 3 phrasings.
    """
    filled_prompt = MULTI_QUERY_PROMPT.format(qus="What is FAISS?", n_multi_qus=3)

    assert "exactly 3 different phrasings" in filled_prompt
    assert filled_prompt.endswith("Question: What is FAISS?")


def test_hyde_prompt_fills_with_qus():
    """
    Input: HYDE_PROMPT and a question.
    Output: the filled prompt.
    What it does: checks the prompt needs only {qus} and ends with the question.
    """
    filled_prompt = HYDE_PROMPT.format(qus="What is FAISS?")

    assert filled_prompt.endswith("Question: What is FAISS?")
