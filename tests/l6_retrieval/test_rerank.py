"""
test_rerank.py

Tests for l6_retrieval.rerank.get_reranked_outputs, with a fake cross-encoder.
"""

from l6_retrieval import rerank


def test_chunks_are_sorted_by_cross_encoder_score_and_cut(make_query, make_chunk, make_retrieved, fake_cross_encoder):
    """
    Input: one output with chunks doc_a_0, doc_b_1, doc_c_2; the cross-encoder scores them 0.1, 5.0 and 2.0;
    rerank_top_k 2.
    Output: doc_b_1 then doc_c_2, with their cross-encoder scores and documents, marked reranked with
    rerank_top_k 2, and every other field kept.
    What it does: checks the reranker reorders by its own score and keeps the best rerank_top_k.
    """
    chunks = {
        "doc_a_0": make_chunk("doc_a_0", "text a"),
        "doc_b_1": make_chunk("doc_b_1", "text b"),
        "doc_c_2": make_chunk("doc_c_2", "text c"),
    }
    retrieved = make_retrieved(query_type="multi", retrieval_type="hybrid", index_type="faiss_hnsw+bm25", top_k=3,
                               chunk_ids=["doc_a_0", "doc_b_1", "doc_c_2"], chunk_scores=[0.03, 0.02, 0.01])
    cross_encoder = fake_cross_encoder({"text a": 0.1, "text b": 5.0, "text c": 2.0})

    reranked_outputs = rerank.get_reranked_outputs([make_query()], chunks, [retrieved], cross_encoder, 2)

    reranked = reranked_outputs[0]
    assert reranked.chunk_ids == ["doc_b_1", "doc_c_2"]
    assert reranked.chunk_scores == [5.0, 2.0]
    assert reranked.doc_ids == ["doc_b", "doc_c"]
    assert (reranked.reranking, reranked.rerank_top_k) == (True, 2)
    assert (reranked.id, reranked.query_type, reranked.chunk_type) == ("eval_qna_0", "multi", "fixed")
    assert (reranked.retrieval_type, reranked.index_type, reranked.top_k) == ("hybrid", "faiss_hnsw+bm25", 3)


def test_every_query_type_is_scored_against_the_original_question(make_query, make_chunk, make_retrieved, fake_cross_encoder):
    """
    Input: a rewritten and a hyde output of the same query.
    Output: one predict call whose pairs all use the original question, in output order.
    What it does: checks all pairs are scored in one call, against the question the answer must match.
    """
    chunks = {"doc_a_0": make_chunk("doc_a_0", "text a"), "doc_b_1": make_chunk("doc_b_1", "text b")}
    retrieved_outputs = [
        make_retrieved(query_type="rewritten", chunk_ids=["doc_a_0"], chunk_scores=[0.9]),
        make_retrieved(query_type="hyde", chunk_ids=["doc_b_1", "doc_a_0"], chunk_scores=[0.8, 0.7]),
    ]
    cross_encoder = fake_cross_encoder({"text a": 1.0, "text b": 2.0})

    rerank.get_reranked_outputs([make_query(question="What is the refund policy?")], chunks, retrieved_outputs, cross_encoder, 2)

    assert cross_encoder.predict_calls == [[
        ("What is the refund policy?", "text a"),
        ("What is the refund policy?", "text b"),
        ("What is the refund policy?", "text a"),
    ]]


def test_each_output_gets_its_own_scores(make_query, make_chunk, make_retrieved, fake_cross_encoder):
    """
    Input: two outputs from two queries, each with two chunks.
    Output: two reranked outputs, each sorted by its own chunks' scores.
    What it does: checks the flat list of scores is split back to the right output.
    """
    chunks = {}
    for chunk_id, text in (("doc_a_0", "a0"), ("doc_a_1", "a1"), ("doc_b_0", "b0"), ("doc_b_1", "b1")):
        chunks[chunk_id] = make_chunk(chunk_id, text)
    retrieved_outputs = [
        make_retrieved(id="q0", chunk_ids=["doc_a_0", "doc_a_1"], chunk_scores=[0.9, 0.8]),
        make_retrieved(id="q1", chunk_ids=["doc_b_0", "doc_b_1"], chunk_scores=[0.9, 0.8]),
    ]
    queries = [make_query(query_id="q0", question="first?"), make_query(query_id="q1", question="second?")]
    cross_encoder = fake_cross_encoder({"a0": 1.0, "a1": 3.0, "b0": 4.0, "b1": 2.0})

    reranked_outputs = rerank.get_reranked_outputs(queries, chunks, retrieved_outputs, cross_encoder, 2)

    assert reranked_outputs[0].chunk_ids == ["doc_a_1", "doc_a_0"]
    assert reranked_outputs[1].chunk_ids == ["doc_b_0", "doc_b_1"]
    assert cross_encoder.predict_calls[0][2] == ("second?", "b0")


def test_outputs_with_no_chunks_do_not_call_the_cross_encoder(make_query, make_retrieved, fake_cross_encoder):
    """
    Input: one output with no chunks.
    Output: one reranked output with no chunks, and no predict call.
    What it does: checks an empty output is kept, reranked, without running the model.
    """
    cross_encoder = fake_cross_encoder({})

    reranked_outputs = rerank.get_reranked_outputs([make_query()], {}, [make_retrieved(chunk_ids=[], chunk_scores=[])], cross_encoder, 2)

    assert reranked_outputs[0].chunk_ids == []
    assert reranked_outputs[0].reranking is True
    assert cross_encoder.predict_calls == []
