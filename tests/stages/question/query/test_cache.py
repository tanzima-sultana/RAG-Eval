"""
test_cache.py

Tests for stages.question.query.cache: saving the Query list as queries.json and
embeddings.npz, and loading it back.
"""

import json
import os

import numpy as np
import pytest

from stages.question.query import cache


def assert_qus_equal(loaded_qus, saved_qus):
    """
    Input: loaded_qus and saved_qus, two Qus.
    Output: nothing. Fails the test if they differ.
    What it does: compares the text, tokens and vector.
    """
    assert loaded_qus.qus == saved_qus.qus
    assert loaded_qus.tokens == saved_qus.tokens
    assert np.allclose(loaded_qus.embeddings, saved_qus.embeddings)


def assert_queries_equal(loaded_queries, saved_queries):
    """
    Input: loaded_queries and saved_queries, two Query lists.
    Output: nothing. Fails the test if they differ.
    What it does: compares every field of every Query, vectors with np.allclose.
    """
    assert len(loaded_queries) == len(saved_queries)
    for position in range(len(saved_queries)):
        loaded_query = loaded_queries[position]
        saved_query = saved_queries[position]
        assert loaded_query.id == saved_query.id
        assert_qus_equal(loaded_query.original_qus, saved_query.original_qus)
        assert_qus_equal(loaded_query.rewritten_qus, saved_query.rewritten_qus)
        assert len(loaded_query.multi_qus) == len(saved_query.multi_qus)
        for multi_position in range(len(saved_query.multi_qus)):
            assert_qus_equal(loaded_query.multi_qus[multi_position], saved_query.multi_qus[multi_position])
        assert loaded_query.hyde_ans.ans == saved_query.hyde_ans.ans
        assert np.allclose(loaded_query.hyde_ans.embeddings, saved_query.hyde_ans.embeddings)
        assert loaded_query.normalized == saved_query.normalized
        assert loaded_query.embedding_model == saved_query.embedding_model
        assert loaded_query.llm_model == saved_query.llm_model


def test_save_then_load_returns_the_same_queries(tmp_path, make_query):
    """
    Input: three queries with different numbers of multi-query phrasings, one with none.
    Output: the queries loaded back.
    What it does: checks every text, token list and vector comes back on the right query.
    """
    queries = []
    queries.append(make_query("qna0", ["one phrasing", "a second phrasing"]))
    queries.append(make_query("qna1", []))
    queries.append(make_query("qna2", ["only this phrasing here"], hyde_text="I don't know"))

    cache.save_queries(str(tmp_path), queries)
    loaded_queries = cache.load_cached_queries(str(tmp_path))

    assert_queries_equal(loaded_queries, queries)


def test_save_writes_both_files_and_no_temporary_files(tmp_path, make_query):
    """
    Input: one query saved to a folder that does not exist yet.
    Output: the files in the folder.
    What it does: checks the folder is created and only the two final files remain.
    """
    query_path = str(tmp_path / "queries" / "run")

    cache.save_queries(query_path, [make_query("qna0", ["a phrasing"])])

    assert sorted(os.listdir(query_path)) == ["embeddings.npz", "queries.json"]


def test_queries_json_holds_texts_and_tokens_but_no_vectors(tmp_path, make_query):
    """
    Input: one query saved.
    Output: queries.json.
    What it does: checks the readable file holds the texts and tokens, and the vectors are kept out of it.
    """
    cache.save_queries(str(tmp_path), [make_query("qna0", ["a phrasing"])])

    with open(os.path.join(str(tmp_path), "queries.json"), encoding="utf-8") as f:
        query_dicts = json.load(f)

    assert query_dicts[0]["original_qus"] == {"qus": "Original qna0", "tokens": ["original", "qna0"]}
    assert query_dicts[0]["hyde_ans"] == {"ans": "A passage"}


def test_load_returns_none_when_a_file_is_missing(tmp_path, make_query):
    """
    Input: an empty folder, then a folder with only queries.json.
    Output: None both times.
    What it does: checks a partial cache is treated as no cache.
    """
    assert cache.load_cached_queries(str(tmp_path)) is None

    cache.save_queries(str(tmp_path), [make_query("qna0", ["a phrasing"])])
    os.remove(os.path.join(str(tmp_path), "embeddings.npz"))

    assert cache.load_cached_queries(str(tmp_path)) is None


def test_load_with_files_out_of_order_raises(tmp_path, make_query):
    """
    Input: a saved cache whose queries.json entries are swapped.
    Output: ValueError mentioning out of order.
    What it does: checks texts are never joined with another query's vectors.
    """
    queries = []
    queries.append(make_query("qna0", ["a phrasing"]))
    queries.append(make_query("qna1", ["another phrasing"]))
    cache.save_queries(str(tmp_path), queries)

    queries_file = os.path.join(str(tmp_path), "queries.json")
    with open(queries_file, encoding="utf-8") as f:
        query_dicts = json.load(f)
    swapped_query_dicts = []
    swapped_query_dicts.append(query_dicts[1])
    swapped_query_dicts.append(query_dicts[0])
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(swapped_query_dicts, f)

    with pytest.raises(ValueError, match="out of order"):
        cache.load_cached_queries(str(tmp_path))
