"""
test_dense.py

Tests for stages.evaluation.retrieval.retrievars.dense: searching a real in-memory FAISS index and a fake Qdrant collection
with each query type's embeddings.
"""

import pytest

from common.constants import I_DONT_KNOW
from stages.evaluation.retrieval.retrievars import dense

# Four chunks: three along an axis each, one in between. doc_my_file has "_" in its id
CHUNK_IDS = ["doc_a_0", "doc_a_1", "doc_b_0", "doc_my_file_2"]
CHUNK_VECTORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]


@pytest.fixture
def faiss_index(make_faiss_index):
    """
    Input: the make_faiss_index fixture.
    Output: a FaissIndex of the four chunks.
    What it does: gives the dense tests one small index.
    """
    return make_faiss_index(CHUNK_IDS, CHUNK_VECTORS)


def test_one_output_per_query_type_in_order(faiss_index, make_query):
    """
    Input: two queries.
    Output: 8 Retrieved: original, rewritten, multi and hyde for the first query, then the same for the second,
    each marked dense, not reranked, with the chunk type, index type and top_k.
    What it does: checks every query type of every query gets one output, in a fixed order.
    """
    queries = [make_query(query_id="eval_qna_0"), make_query(query_id="eval_qna_1")]

    dense_outputs = dense.get_dense_outputs(queries, "fixed", "faiss_flatip", faiss_index, 2, ["original", "rewritten", "multi", "hyde"])

    keys = []
    for retrieved in dense_outputs:
        keys.append((retrieved.id, retrieved.query_type))
        assert (retrieved.chunk_type, retrieved.retrieval_type, retrieved.index_type) == ("fixed", "dense", "faiss_flatip")
        assert (retrieved.top_k, retrieved.reranking, retrieved.rerank_top_k) == (2, False, 0)
    assert keys == [
        ("eval_qna_0", "original"), ("eval_qna_0", "rewritten"), ("eval_qna_0", "multi"), ("eval_qna_0", "hyde"),
        ("eval_qna_1", "original"), ("eval_qna_1", "rewritten"), ("eval_qna_1", "multi"), ("eval_qna_1", "hyde"),
    ]


def test_only_the_given_query_types_are_searched(faiss_index, make_query):
    """
    Input: two queries and query_types ["original"].
    Output: 2 Retrieved, the original of each query.
    What it does: checks query types not in query_types are not searched.
    """
    queries = [make_query(query_id="eval_qna_0"), make_query(query_id="eval_qna_1")]

    dense_outputs = dense.get_dense_outputs(queries, "fixed", "faiss_flatip", faiss_index, 2, ["original"])

    keys = []
    for retrieved in dense_outputs:
        keys.append((retrieved.id, retrieved.query_type))
    assert keys == [("eval_qna_0", "original"), ("eval_qna_1", "original")]


def test_original_query_finds_the_closest_chunks_best_first(faiss_index, make_query):
    """
    Input: an original question along the first axis, top_k 2.
    Output: doc_a_0 (cosine 1.0) then doc_my_file_2 (cosine 1/sqrt(3)), with their documents.
    What it does: checks the cosine scores, the order, the top_k cut, and that a doc id with "_" is kept whole.
    """
    dense_outputs = dense.get_dense_outputs([make_query(original=(1, 0, 0))], "fixed", "faiss_flatip", faiss_index, 2, ["original", "rewritten", "multi", "hyde"])
    original_output = dense_outputs[0]

    assert original_output.chunk_ids == ["doc_a_0", "doc_my_file_2"]
    assert original_output.chunk_scores == pytest.approx([1.0, 3 ** -0.5], abs=1e-5)
    assert original_output.doc_ids == ["doc_a", "doc_my_file"]


def test_multi_phrasings_are_merged_keeping_each_chunks_best_score(faiss_index, make_query):
    """
    Input: two multi phrasings, one along the first axis and one along the second, top_k 3.
    Output: one multi output holding doc_a_0 and doc_a_1 (each 1.0) then doc_my_file_2, each chunk once.
    What it does: checks the phrasings' hits are merged into one list, a chunk found by both keeps its best score.
    """
    query = make_query(multi=((1, 0, 0), (0, 1, 0)), multi_tokens=(("a",), ("b",)))

    multi_output = dense.get_dense_outputs([query], "fixed", "faiss_flatip", faiss_index, 3, ["original", "rewritten", "multi", "hyde"])[2]

    assert multi_output.query_type == "multi"
    assert sorted(multi_output.chunk_ids[:2]) == ["doc_a_0", "doc_a_1"]
    assert multi_output.chunk_ids[2] == "doc_my_file_2"
    assert multi_output.chunk_scores == pytest.approx([1.0, 1.0, 3 ** -0.5], abs=1e-5)


def test_refused_hyde_and_missing_multi_give_empty_outputs(faiss_index, make_query):
    """
    Input: a query whose HyDE answer is I_DONT_KNOW and that has no multi phrasings.
    Output: its multi and hyde outputs exist but hold no chunks.
    What it does: checks a query type with nothing to search still gets an output, so it counts as a miss.
    """
    query = make_query(multi=(), multi_tokens=(), hyde_text=I_DONT_KNOW)

    dense_outputs = dense.get_dense_outputs([query], "fixed", "faiss_flatip", faiss_index, 2, ["original", "rewritten", "multi", "hyde"])

    assert (dense_outputs[2].query_type, dense_outputs[2].chunk_ids) == ("multi", [])
    assert (dense_outputs[3].query_type, dense_outputs[3].chunk_ids) == ("hyde", [])


def test_top_k_above_the_number_of_chunks_returns_every_chunk_once(faiss_index, make_query):
    """
    Input: top_k 10 on an index of 4 chunks.
    Output: 4 chunks.
    What it does: checks the -1 positions FAISS returns for missing results are skipped.
    """
    original_output = dense.get_dense_outputs([make_query()], "fixed", "faiss_flatip", faiss_index, 10, ["original", "rewritten", "multi", "hyde"])[0]

    assert sorted(original_output.chunk_ids) == sorted(CHUNK_IDS)


def test_qdrant_is_searched_once_per_embedding(make_qdrant_index, make_query):
    """
    Input: a Qdrant collection that returns doc_b_0 (0.9) and doc_a_1 (0.4), and a query with two multi phrasings.
    Output: the outputs hold those chunks best first; Qdrant got 1 + 1 + 2 + 1 searches with limit top_k, the
    vector as a list, and only the chunk_id payload.
    What it does: checks the Qdrant path searches each embedding and reads the chunk id from the payload.
    """
    qdrant_index = make_qdrant_index([("doc_b_0", 0.9), ("doc_a_1", 0.4)])
    query = make_query(multi=((1, 0, 0), (0, 1, 0)), multi_tokens=(("a",), ("b",)))

    dense_outputs = dense.get_dense_outputs([query], "fixed", "qdrant", qdrant_index, 2, ["original", "rewritten", "multi", "hyde"])

    assert dense_outputs[0].chunk_ids == ["doc_b_0", "doc_a_1"]
    assert dense_outputs[0].chunk_scores == pytest.approx([0.9, 0.4])
    assert dense_outputs[0].index_type == "qdrant"
    searches = qdrant_index.qdrant_client.searches
    assert len(searches) == 5
    assert searches[0]["collection_name"] == "test_collection"
    assert searches[0]["limit"] == 2
    assert searches[0]["with_payload"] == ["chunk_id"]
    assert isinstance(searches[0]["query"], list)
