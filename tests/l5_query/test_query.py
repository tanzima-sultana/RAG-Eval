"""
test_query.py

Tests for l5_query.query: generate_queries turning each eval question
into a rewrite, multi-query phrasings and a HyDE answer with tokens and
embeddings, and get_queries using the cache. The query LLM and the
embedding model are faked (see conftest.py), so nothing is downloaded and
no LLM is called.
"""

import os

import numpy as np
import pytest

from common import paths
from common.constants import I_DONT_KNOW
from l5_query.query import generate_queries
from l5_query.query import get_queries
from common.args import make_llm_args

# The query LLM settings every test uses. The LLM itself is faked, so no call is made
QUERY_LLM_ARGS = make_llm_args("ollama", "llama3.1:8b", 0, 0, 0, 0, 16000, 180, "5m")


def test_generate_queries_builds_one_query_per_question(fake_query_llm, make_query_args, make_eval_qna):
    """
    Input: two eval questions.
    Output: the generated queries and the fake LLM.
    What it does: checks each query has the eval id, the stripped rewrite, 3 phrasings, the HyDE passage,
    the settings, and that the cost was printed.
    """
    created_llms, replies = fake_query_llm
    eval_qna = make_eval_qna(["What is FAISS?", "How does BM25 rank?"])

    queries = generate_queries(eval_qna, make_query_args(), QUERY_LLM_ARGS)

    assert len(queries) == 2
    assert queries[0].id == "qna0"
    assert queries[1].id == "qna1"
    assert queries[0].original_qus.qus == "What is FAISS?"
    assert queries[0].rewritten_qus.qus == "Rewritten What is FAISS?"
    assert len(queries[0].multi_qus) == 3
    assert queries[0].multi_qus[0].qus == "First What is FAISS?"
    assert queries[0].hyde_ans.ans == "A passage answering What is FAISS?"
    assert queries[0].normalized is True
    assert queries[0].embedding_model == "fake-model"
    assert queries[0].llm_model == "llama3.1:8b"
    assert len(created_llms[0].prompts) == 6
    assert created_llms[0].cost_computed is True


def test_tokens_are_lowercase_words(fake_query_llm, make_query_args, make_eval_qna):
    """
    Input: the question "What Is FAISS?".
    Output: the tokens of the original, rewritten and multi qus.
    What it does: checks tokens are split the same way as the BM25 chunks.
    """
    queries = generate_queries(make_eval_qna(["What Is FAISS?"]), make_query_args(), QUERY_LLM_ARGS)

    assert queries[0].original_qus.tokens == ["what", "is", "faiss?"]
    assert queries[0].rewritten_qus.tokens == ["rewritten", "what", "is", "faiss?"]
    assert queries[0].multi_qus[0].tokens == ["first", "what", "is", "faiss?"]


def test_each_text_gets_its_own_embedding(fake_query_llm, make_query_args, make_eval_qna, fake_embed_function):
    """
    Input: two questions, the first with no phrasings kept and the second with 3.
    Output: the embeddings of every qus and ans.
    What it does: checks each vector belongs to its own text, so the flat multi-query list is split back correctly.
    """
    created_llms, replies = fake_query_llm
    replies["multi"]["Short?"] = "only one line"
    eval_qna = make_eval_qna(["Short?", "A much longer second question?"])

    queries = generate_queries(eval_qna, make_query_args(), QUERY_LLM_ARGS)

    for query in queries:
        assert np.allclose(query.original_qus.embeddings, fake_embed_function(query.original_qus.qus))
        assert np.allclose(query.rewritten_qus.embeddings, fake_embed_function(query.rewritten_qus.qus))
        assert np.allclose(query.hyde_ans.embeddings, fake_embed_function(query.hyde_ans.ans))
        for multi_qus in query.multi_qus:
            assert np.allclose(multi_qus.embeddings, fake_embed_function(multi_qus.qus))
    assert len(queries[0].multi_qus) == 0
    assert len(queries[1].multi_qus) == 3


def test_blank_lines_between_phrasings_are_skipped(fake_query_llm, make_query_args, make_eval_qna):
    """
    Input: a multi-query reply with blank lines and spaces around 3 phrasings.
    Output: the multi_qus texts.
    What it does: checks the 3 phrasings are kept, stripped, and the blank lines dropped.
    """
    created_llms, replies = fake_query_llm
    replies["multi"]["What is FAISS?"] = "\n  one  \n\n two\n   \nthree\n"

    queries = generate_queries(make_eval_qna(["What is FAISS?"]), make_query_args(), QUERY_LLM_ARGS)

    multi_texts = []
    for multi_qus in queries[0].multi_qus:
        multi_texts.append(multi_qus.qus)
    assert multi_texts == ["one", "two", "three"]


@pytest.mark.parametrize("multi_reply", [
    "only one line",
    "one\ntwo\nthree\nfour",
    "I could not find any information about this.",
])
def test_wrong_number_of_phrasings_keeps_none(fake_query_llm, make_query_args, make_eval_qna, multi_reply, capsys):
    """
    Input: a multi-query reply with fewer or more lines than n_multi_qus.
    Output: the multi_qus of the query and the printed count.
    What it does: checks the reply is treated as I don't know: no phrasings are kept, it is counted, and the run goes on.
    """
    created_llms, replies = fake_query_llm
    replies["multi"]["What is FAISS?"] = multi_reply

    queries = generate_queries(make_eval_qna(["What is FAISS?"]), make_query_args(), QUERY_LLM_ARGS)

    assert queries[0].multi_qus == []
    assert "Multi-query I don't know : 1 of 1" in capsys.readouterr().out


@pytest.mark.parametrize("hyde_reply", [
    "I don't know.",
    "I do not know the answer to that.",
    "i don’t know, sorry",
])
def test_hyde_refusal_is_stored_as_i_dont_know(fake_query_llm, make_query_args, make_eval_qna, hyde_reply, capsys):
    """
    Input: a HyDE reply that starts with a form of "I don't know".
    Output: the hyde_ans of the query and the printed count.
    What it does: checks the reply is stored as exactly I_DONT_KNOW and counted, so l6 can skip it.
    """
    created_llms, replies = fake_query_llm
    replies["hyde"]["What is FAISS?"] = hyde_reply

    queries = generate_queries(make_eval_qna(["What is FAISS?"]), make_query_args(), QUERY_LLM_ARGS)

    assert queries[0].hyde_ans.ans == I_DONT_KNOW
    assert "HyDE I don't know : 1 of 1" in capsys.readouterr().out


def test_hyde_passage_mentioning_not_knowing_later_is_kept(fake_query_llm, make_query_args, make_eval_qna):
    """
    Input: a HyDE passage that says "I don't know" only after its first words.
    Output: the hyde_ans of the query.
    What it does: checks only a reply that starts with it is treated as a refusal.
    """
    created_llms, replies = fake_query_llm
    passage = "FAISS is a library. Some say I don't know how it works."
    replies["hyde"]["What is FAISS?"] = passage

    queries = generate_queries(make_eval_qna(["What is FAISS?"]), make_query_args(), QUERY_LLM_ARGS)

    assert queries[0].hyde_ans.ans == passage


def test_get_queries_generates_and_saves_when_not_cached(tmp_path, fake_query_llm, make_query_args, make_eval_qna_args, make_eval_qna):
    """
    Input: an output folder with no query cache.
    Output: the queries and the query cache folder.
    What it does: checks the queries are generated and saved under get_query_path.
    """
    created_llms, replies = fake_query_llm
    query_args = make_query_args()
    eval_qna_args = make_eval_qna_args()
    output_path = str(tmp_path / "myset")

    queries = get_queries(output_path, make_eval_qna(["What is FAISS?"]), eval_qna_args, query_args, QUERY_LLM_ARGS)

    query_path = paths.get_query_path(output_path, eval_qna_args, "claude-sonnet-5-5", query_args, "llama3.1:8b")
    assert len(queries) == 1
    assert len(created_llms) == 1
    assert os.path.exists(os.path.join(query_path, "queries.json"))
    assert os.path.exists(os.path.join(query_path, "embeddings.npz"))


def test_get_queries_uses_the_cache_without_calling_the_llm(tmp_path, fake_query_llm, make_query_args, make_eval_qna_args, make_eval_qna):
    """
    Input: the same get_queries call made twice.
    Output: the second call's queries and the number of LLMs created.
    What it does: checks the second call loads the cache and never gets an LLM.
    """
    created_llms, replies = fake_query_llm
    query_args = make_query_args()
    eval_qna_args = make_eval_qna_args()
    output_path = str(tmp_path / "myset")
    eval_qna = make_eval_qna(["What is FAISS?"])
    get_queries(output_path, eval_qna, eval_qna_args, query_args, QUERY_LLM_ARGS)

    cached_queries = get_queries(output_path, eval_qna, eval_qna_args, query_args, QUERY_LLM_ARGS)

    assert len(created_llms) == 1
    assert cached_queries[0].rewritten_qus.qus == "Rewritten What is FAISS?"


def test_get_queries_with_invalid_args_raises_before_the_llm(tmp_path, fake_query_llm, make_query_args, make_eval_qna_args, make_eval_qna):
    """
    Input: n_multi_qus 0.
    Output: ValueError mentioning n_multi_qus.
    What it does: checks the args are validated before any LLM is created.
    """
    created_llms, replies = fake_query_llm

    with pytest.raises(ValueError, match="n_multi_qus"):
        get_queries(str(tmp_path), make_eval_qna(["What is FAISS?"]), make_eval_qna_args(), make_query_args(n_multi_qus=0), QUERY_LLM_ARGS)

    assert len(created_llms) == 0
