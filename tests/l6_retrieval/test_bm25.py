"""
test_bm25.py

Tests for l6_retrieval.retrievars.bm25: searching a real in-memory BM25 index with each query type's tokens.
"""

import pytest

from l6_retrieval.retrievars import bm25

# BM25 gives a word in half or more of the chunks an IDF of 0 or less, so "refund" is in only 2 of 6 chunks
CHUNK_IDS = ["doc_a_0", "doc_a_1", "doc_b_0", "doc_c_0", "doc_d_0", "doc_e_0"]
CHUNK_TEXTS = [
    "refund policy refund within thirty days",
    "shipping takes five days",
    "the refund needs a receipt",
    "office hours are nine to five",
    "parking is behind the building",
    "the cafe opens at eight",
]


@pytest.fixture
def bm25_index(make_bm25_index):
    """
    Input: the make_bm25_index fixture.
    Output: a Bm25Index of the six chunks.
    What it does: gives the BM25 tests one small index.
    """
    return make_bm25_index(CHUNK_IDS, CHUNK_TEXTS)


def test_one_output_per_query_type_without_hyde(bm25_index, make_query):
    """
    Input: one query.
    Output: 3 Retrieved, original, rewritten and multi, marked bm25 and not reranked.
    What it does: checks HyDE, which has no tokens, is not searched by BM25.
    """
    bm25_outputs = bm25.get_bm25_outputs([make_query()], "fixed", "bm25", bm25_index, 3, ["original", "rewritten", "multi", "hyde"])

    query_types = []
    for retrieved in bm25_outputs:
        query_types.append(retrieved.query_type)
        assert (retrieved.retrieval_type, retrieved.index_type, retrieved.reranking) == ("bm25", "bm25", False)
    assert query_types == ["original", "rewritten", "multi"]


def test_only_the_given_query_types_are_searched(bm25_index, make_query):
    """
    Input: one query and query_types ["original", "hyde"].
    Output: 1 Retrieved, the original.
    What it does: checks query types not in query_types are not searched, and HyDE is still skipped.
    """
    bm25_outputs = bm25.get_bm25_outputs([make_query()], "fixed", "bm25", bm25_index, 3, ["original", "hyde"])

    query_types = []
    for retrieved in bm25_outputs:
        query_types.append(retrieved.query_type)
    assert query_types == ["original"]


def test_chunk_with_more_query_words_ranks_first(bm25_index, make_query):
    """
    Input: the tokens "refund policy", top_k 3.
    Output: doc_a_0 (has refund twice and policy) then doc_b_0 (has refund); no other chunk.
    What it does: checks BM25 ranks by the query words, and chunks with no query word (score 0) are dropped
    even when top_k leaves room.
    """
    original_output = bm25.get_bm25_outputs([make_query(original_tokens=("refund", "policy"))], "fixed", "bm25", bm25_index, 3, ["original", "rewritten", "multi", "hyde"])[0]

    assert original_output.chunk_ids == ["doc_a_0", "doc_b_0"]
    assert original_output.doc_ids == ["doc_a", "doc_b"]
    assert original_output.chunk_scores[0] > original_output.chunk_scores[1] > 0


def test_query_with_no_known_word_finds_nothing(bm25_index, make_query):
    """
    Input: the tokens "pizza".
    Output: an original output with no chunks.
    What it does: checks a query that shares no word with any chunk returns nothing, not random chunks.
    """
    original_output = bm25.get_bm25_outputs([make_query(original_tokens=("pizza",))], "fixed", "bm25", bm25_index, 3, ["original", "rewritten", "multi", "hyde"])[0]

    assert original_output.chunk_ids == []


def test_multi_phrasings_are_merged_and_cut_to_top_k(bm25_index, make_query):
    """
    Input: three multi phrasings: "shipping", "office" and "receipt", top_k 2.
    Output: one multi output with 2 of those 3 chunks, each once, best first.
    What it does: checks each phrasing is searched and their hits are merged into one list and cut to top_k.
    """
    query = make_query(multi=((1, 0, 0), (0, 1, 0), (0, 0, 1)), multi_tokens=(("shipping",), ("office",), ("receipt",)))

    multi_output = bm25.get_bm25_outputs([query], "fixed", "bm25", bm25_index, 2, ["original", "rewritten", "multi", "hyde"])[2]

    assert len(multi_output.chunk_ids) == 2
    assert set(multi_output.chunk_ids) <= {"doc_a_1", "doc_c_0", "doc_b_0"}
    assert multi_output.chunk_scores[0] >= multi_output.chunk_scores[1]


def test_no_multi_phrasings_gives_an_empty_output(bm25_index, make_query):
    """
    Input: a query with no multi phrasings.
    Output: its multi output holds no chunks.
    What it does: checks a query type with nothing to search still gets an output, so it counts as a miss.
    """
    multi_output = bm25.get_bm25_outputs([make_query(multi=(), multi_tokens=())], "fixed", "bm25", bm25_index, 3, ["original", "rewritten", "multi", "hyde"])[2]

    assert (multi_output.query_type, multi_output.chunk_ids) == ("multi", [])
