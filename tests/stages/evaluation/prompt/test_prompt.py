"""
test_prompt.py

Tests for stages.evaluation.prompt.prompt.get_prompt_list and the templates in stages.evaluation.prompt.prompt_template.
"""

import hashlib

import numpy as np
import pytest

from common.constants import I_DONT_KNOW
from stages.question.query.schema import Ans
from stages.question.query.schema import Qus
from stages.question.query.schema import Query
from stages.evaluation.context.schema import Context
from stages.evaluation.prompt.prompt import get_prompt_list
from stages.evaluation.prompt.prompt_template import SYSTEM_PROMPT
from stages.evaluation.prompt.prompt_template import USER_PROMPT_TEMPLATE


def make_query(query_id, question):
    """
    Input: query_id and the original question.
    Output: a Query whose rewritten question and HyDE answer differ from the original.
    What it does: builds the query get_prompt_list reads the original question from.
    """
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return Query(
        id=query_id,
        original_qus=Qus(qus=question, tokens=question.lower().split(), embeddings=vector),
        rewritten_qus=Qus(qus="REWRITTEN " + question, tokens=[], embeddings=vector),
        multi_qus=[],
        hyde_ans=Ans(ans="A hypothetical answer.", embeddings=vector),
        normalized=True,
        embedding_model="all-MiniLM-L6-v2",
        llm="ollama",
        llm_model="llama3.1:8b"
    )


@pytest.fixture
def queries():
    """
    Input: none.
    Output: two tuning queries, eval_qna_0 and eval_qna_1.
    What it does: gives the tests the original questions to put in the prompts.
    """
    return [make_query("eval_qna_0", "What is the refund window?"), make_query("eval_qna_1", "Who pays for shipping?")]


def expected_hash(user_prompt):
    """
    Input: user_prompt.
    Output: the sha256 of SYSTEM_PROMPT, a blank line and user_prompt.
    What it does: works out the prompt hash the way l9 expects it.
    """
    return hashlib.sha256((SYSTEM_PROMPT + "\n\n" + user_prompt).encode("utf-8")).hexdigest()


# ------ Templates

def test_system_prompt_asks_for_exactly_i_dont_know():
    """
    Input: SYSTEM_PROMPT.
    Output: it holds the I_DONT_KNOW constant.
    What it does: checks the LLM is told the exact phrase l9 and l10 look for.
    """
    assert "reply with exactly: " + I_DONT_KNOW in SYSTEM_PROMPT


def test_user_prompt_template_has_the_context_and_the_question():
    """
    Input: USER_PROMPT_TEMPLATE filled with a context and a question.
    Output: "Context:", the context, a blank line, then "Question: " and the question.
    What it does: checks the layout the LLM gets.
    """
    assert USER_PROMPT_TEMPLATE.format(context="CTX", question="Q?") == "Context:\nCTX\n\nQuestion: Q?"


# ------ get_prompt_list

def test_prompt_holds_the_context_and_the_original_question(queries):
    """
    Input: one Context for eval_qna_0.
    Output: one Prompt with the original question (not the rewritten one), SYSTEM_PROMPT, and the user prompt
    filled with the context and that question.
    What it does: checks every query type is answered for the eval question itself.
    """
    prompt_list = get_prompt_list([Context(id="eval_qna_0", context="Refunds take thirty days.", token_count=4)], queries)

    prompt = prompt_list[0]
    assert prompt.id == "eval_qna_0"
    assert prompt.qus == "What is the refund window?"
    assert prompt.system_prompt == SYSTEM_PROMPT
    assert prompt.user_prompt == "Context:\nRefunds take thirty days.\n\nQuestion: What is the refund window?"


def test_token_count_adds_the_system_prompt_and_question_words(queries):
    """
    Input: a Context with token count 40, and the 5-word question of eval_qna_0.
    Output: token_count 40 + the words of SYSTEM_PROMPT + 5.
    What it does: checks the estimate of the prompt size.
    """
    prompt_list = get_prompt_list([Context(id="eval_qna_0", context="Some text.", token_count=40)], queries)

    assert prompt_list[0].token_count == 40 + len(SYSTEM_PROMPT.split()) + 5


def test_prompt_hash_is_the_sha256_of_the_system_and_user_prompt(queries):
    """
    Input: one Context.
    Output: prompt_hash equals sha256(SYSTEM_PROMPT + blank line + user_prompt).
    What it does: checks the hash l9 uses as its cache key and batch request id.
    """
    prompt = get_prompt_list([Context(id="eval_qna_0", context="Refunds take thirty days.", token_count=4)], queries)[0]

    assert prompt.prompt_hash == expected_hash(prompt.user_prompt)
    assert len(prompt.prompt_hash) == 64


def test_same_context_and_question_give_the_same_hash(queries):
    """
    Input: two Contexts for eval_qna_0 with the same text (two setups that found the same chunks), and one with
    other text.
    Output: the first two prompts share a hash; the third has another.
    What it does: checks identical prompts share one hash, so l9 sends them once, and a different context does not.
    """
    context_list = [
        Context(id="eval_qna_0", context="Refunds take thirty days.", token_count=4),
        Context(id="eval_qna_0", context="Refunds take thirty days.", token_count=4),
        Context(id="eval_qna_0", context="Keep the receipt.", token_count=3),
    ]

    prompt_list = get_prompt_list(context_list, queries)

    assert prompt_list[0].prompt_hash == prompt_list[1].prompt_hash
    assert prompt_list[0].prompt_hash != prompt_list[2].prompt_hash


def test_same_context_with_another_question_gives_another_hash(queries):
    """
    Input: the same context text for eval_qna_0 and eval_qna_1.
    Output: two different hashes.
    What it does: checks the question is part of the hash, so one question's answer is never reused for another.
    """
    context_list = [
        Context(id="eval_qna_0", context="Shared text.", token_count=2),
        Context(id="eval_qna_1", context="Shared text.", token_count=2),
    ]

    prompt_list = get_prompt_list(context_list, queries)

    assert prompt_list[0].prompt_hash != prompt_list[1].prompt_hash


def test_empty_context_gives_an_empty_prompt_in_its_place(queries):
    """
    Input: an empty Context between two with text.
    Output: three Prompts; the middle one keeps its id, question and system prompt, with user_prompt "",
    token_count 0 and prompt_hash "".
    What it does: checks an empty context is kept in place and marked, so l9 answers it I_DONT_KNOW without a call.
    """
    context_list = [
        Context(id="eval_qna_0", context="Refunds take thirty days.", token_count=4),
        Context(id="eval_qna_1", context="", token_count=0),
        Context(id="eval_qna_0", context="Keep the receipt.", token_count=3),
    ]

    prompt_list = get_prompt_list(context_list, queries)

    assert len(prompt_list) == 3
    empty_prompt = prompt_list[1]
    assert (empty_prompt.id, empty_prompt.qus, empty_prompt.system_prompt) == ("eval_qna_1", "Who pays for shipping?", SYSTEM_PROMPT)
    assert (empty_prompt.user_prompt, empty_prompt.token_count, empty_prompt.prompt_hash) == ("", 0, "")
    assert prompt_list[2].user_prompt.startswith("Context:\nKeep the receipt.")


def test_one_prompt_per_context_in_the_same_order(queries):
    """
    Input: Contexts for eval_qna_1, eval_qna_0 and eval_qna_1 again.
    Output: Prompts with the same ids and each id's own question, in the same order.
    What it does: checks prompt_list[i] belongs to context_list[i].
    """
    context_list = [
        Context(id="eval_qna_1", context="A.", token_count=1),
        Context(id="eval_qna_0", context="B.", token_count=1),
        Context(id="eval_qna_1", context="C.", token_count=1),
    ]

    prompt_list = get_prompt_list(context_list, queries)

    ids_and_questions = []
    for prompt in prompt_list:
        ids_and_questions.append((prompt.id, prompt.qus))
    assert ids_and_questions == [
        ("eval_qna_1", "Who pays for shipping?"),
        ("eval_qna_0", "What is the refund window?"),
        ("eval_qna_1", "Who pays for shipping?"),
    ]


def test_braces_in_the_context_are_kept_as_text(queries):
    """
    Input: a context holding "{question}" and "{0}", as code or JSON in a document might.
    Output: the user prompt holds them unchanged.
    What it does: checks str.format only fills the template's own fields, never text inside the context.
    """
    prompt = get_prompt_list([Context(id="eval_qna_0", context="Use {question} and {0} here.", token_count=5)], queries)[0]

    assert "Use {question} and {0} here." in prompt.user_prompt


def test_context_whose_question_is_missing_raises(queries):
    """
    Input: a Context for eval_qna_9, which is not a tuning query.
    Output: KeyError.
    What it does: checks a context from a held-out or unknown question stops the run instead of getting a wrong
    question.
    """
    with pytest.raises(KeyError):
        get_prompt_list([Context(id="eval_qna_9", context="Text.", token_count=1)], queries)
