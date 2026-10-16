"""
test_hybrid.py

Tests for l6_retrieval.retrievars.hybrid: fusing dense and BM25 outputs with reciprocal rank fusion.
"""

import pytest

from common.constants import RRF_K
from l6_retrieval.retrievars import hybrid


def test_rrf_adds_one_over_k_plus_rank_from_each_list():
    """
    Input: dense [x, y] and BM25 [y, z].
    Output: x = 1/(K+1); y = 1/(K+2) + 1/(K+1); z = 1/(K+2).
    What it does: checks the fused score uses ranks starting at 1 and adds up across both lists.
    """
    fused_scores = hybrid.fuse_with_rrf(["x", "y"], ["y", "z"])

    assert fused_scores["x"] == pytest.approx(1 / (RRF_K + 1))
    assert fused_scores["y"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused_scores["z"] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_with_empty_lists_is_empty():
    """
    Input: two empty lists.
    Output: an empty dict.
    What it does: checks an output with no chunks fuses to nothing.
    """
    assert hybrid.fuse_with_rrf([], []) == {}


def test_chunk_found_by_both_ranks_first(make_retrieved):
    """
    Input: dense [doc_a_0, doc_b_0] and BM25 [doc_c_0, doc_b_0] for the same query and type, top_k 2.
    Output: doc_b_0 first (in both lists), then one of the chunks found once; index type faiss_flatip+bm25.
    What it does: checks fusion rewards agreement between the retrievers, cuts to top_k and names both indexes.
    """
    dense_outputs = [make_retrieved(chunk_ids=["doc_a_0", "doc_b_0"], chunk_scores=[0.9, 0.8])]
    bm25_outputs = [make_retrieved(retrieval_type="bm25", index_type="bm25", chunk_ids=["doc_c_0", "doc_b_0"], chunk_scores=[7.0, 5.0])]

    hybrid_outputs = hybrid.get_hybrid_outputs("fixed", "faiss_flatip", dense_outputs, bm25_outputs, 2)

    assert len(hybrid_outputs) == 1
    hybrid_output = hybrid_outputs[0]
    assert hybrid_output.chunk_ids[0] == "doc_b_0"
    assert len(hybrid_output.chunk_ids) == 2
    assert hybrid_output.doc_ids[0] == "doc_b"
    assert (hybrid_output.retrieval_type, hybrid_output.index_type) == ("hybrid", "faiss_flatip+bm25")
    assert (hybrid_output.top_k, hybrid_output.reranking, hybrid_output.rerank_top_k) == (2, False, 0)


def test_outputs_are_paired_by_query_id_and_type(make_retrieved):
    """
    Input: dense outputs for (q0, original), (q0, hyde) and (q1, original); BM25 outputs in another order for
    (q1, original) and (q0, original).
    Output: two hybrid outputs, (q0, original) and (q1, original), each fused with its own BM25 output.
    What it does: checks dense and BM25 outputs are matched by key, not position, and HyDE, which has no BM25
    output, gets no hybrid output.
    """
    dense_outputs = [
        make_retrieved(id="q0", query_type="original", chunk_ids=["doc_a_0"], chunk_scores=[0.9]),
        make_retrieved(id="q0", query_type="hyde", chunk_ids=["doc_a_1"], chunk_scores=[0.9]),
        make_retrieved(id="q1", query_type="original", chunk_ids=["doc_b_0"], chunk_scores=[0.9]),
    ]
    bm25_outputs = [
        make_retrieved(id="q1", query_type="original", retrieval_type="bm25", index_type="bm25", chunk_ids=["doc_b_1"], chunk_scores=[3.0]),
        make_retrieved(id="q0", query_type="original", retrieval_type="bm25", index_type="bm25", chunk_ids=["doc_a_2"], chunk_scores=[3.0]),
    ]

    hybrid_outputs = hybrid.get_hybrid_outputs("fixed", "qdrant", dense_outputs, bm25_outputs, 5)

    keys_and_chunks = []
    for hybrid_output in hybrid_outputs:
        keys_and_chunks.append((hybrid_output.id, hybrid_output.query_type, sorted(hybrid_output.chunk_ids)))
    assert keys_and_chunks == [("q0", "original", ["doc_a_0", "doc_a_2"]), ("q1", "original", ["doc_b_0", "doc_b_1"])]
    assert hybrid_outputs[0].index_type == "qdrant+bm25"
