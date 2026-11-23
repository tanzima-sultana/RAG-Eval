"""
test_retrieval_evaluation.py

Tests for evaluation.retrieval_evaluation on small hand-built results, at chunk level and at doc level.
"""

import math

import pytest

from evaluation.retrieval_evaluation import RetrievalEvaluation
from schema.chunk import Document
from schema.eval_sample import EvalSample
from schema.retrieved import Retrieved
from schema.setup import Setup


def make_eval_sample(doc_ids, context):
    """
    Input: doc_ids, the EvalSample docs; context, the EvalSample context passages.
    Output: an EvalSample with id q1.
    What it does: builds one eval question with placeholder text fields.
    """
    return EvalSample(
        id="q1",
        qus="What is X?",
        ans="X.",
        context=context,
        qus_type="single_hop_specific_query_synthesizer",
        doc_ids=doc_ids,
        doc_paths=["unused.pdf"],
        llm="anthropic",
        llm_model="claude-sonnet-5-5",
    )


def make_retrieved(chunk_ids, top_k=4, reranking=False, rerank_top_k=0):
    """
    Input: chunk_ids, best first; top_k; reranking; rerank_top_k.
    Output: a Retrieved for q1 whose doc ids come from the chunk ids.
    What it does: builds one result to score.
    """
    chunk_scores = []
    doc_ids = []
    for chunk_id in chunk_ids:
        chunk_scores.append(1.0)
        doc_ids.append(chunk_id.rsplit("_", 1)[0])
    return Retrieved(
        id="q1",
        setup=Setup(chunk_type="fixed", index_type="faiss_flatip", query_type="original", retrieval_type="dense", reranking=reranking),
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        chunk_ids=chunk_ids,
        chunk_scores=chunk_scores,
        doc_ids=doc_ids,
    )


def make_chunks():
    """
    Input: none.
    Output: a dict of chunk_id to Document.
    What it does: doc1_0 matches the first passage, doc1_1 the second, doc1_2 is from doc1 but matches neither,
    and doc2_0 matches the first passage but is from another doc.
    """
    texts = {
        "doc1_0": "alpha beta gamma delta",
        "doc1_1": "one two three four",
        "doc1_2": "nothing in common here",
        "doc2_0": "alpha beta gamma delta",
    }
    chunks = {}
    for chunk_id in texts:
        chunks[chunk_id] = Document(id=chunk_id, text=texts[chunk_id], size=4, type="fixed")
    return chunks


PASSAGES = ["Alpha beta gamma delta epsilon", "one two three four five"]


def test_compute_retrieved_evals_keeps_the_order_and_the_retrieved():
    """
    Input: two results.
    Output: their RetrievedEvals.
    What it does: checks there is one RetrievedEval per Retrieved, in the same order, holding that Retrieved.
    """
    first = make_retrieved(["doc1_0"])
    second = make_retrieved(["doc2_0"])

    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1"], PASSAGES)]).compute_retrieved_evals([first, second], make_chunks())

    assert len(retrieved_evals) == 2
    assert retrieved_evals[0].retrieved is first
    assert retrieved_evals[1].retrieved is second


def test_a_miss_scores_zero_at_both_levels():
    """
    Input: a result holding only a chunk of another doc.
    Output: its RetrievedEval.
    What it does: checks every score is 0 at both levels.
    """
    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1"], PASSAGES)]).compute_retrieved_evals(
        [make_retrieved(["doc2_0"])], make_chunks()
    )

    for level_eval in [retrieved_evals[0].chunk_eval, retrieved_evals[0].doc_eval]:
        assert level_eval.hit == 0
        assert level_eval.precision == 0.0
        assert level_eval.recall == 0.0
        assert level_eval.mrr == 0.0
        assert level_eval.ndcg == 0.0
        assert level_eval.first_relevant_rank == 0


def test_chunk_eval_needs_the_eval_sample_doc_and_a_passage_overlap():
    """
    Input: a result with a chunk of another doc first, then a doc1 chunk matching no passage, then one matching.
    Output: its chunk level Eval.
    What it does: checks the other doc and the non-matching chunk are not relevant, so the first hit is at rank 3.
    """
    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1"], PASSAGES)]).compute_retrieved_evals(
        [make_retrieved(["doc2_0", "doc1_2", "doc1_0"])], make_chunks()
    )

    chunk_eval = retrieved_evals[0].chunk_eval
    assert chunk_eval.hit == 1
    assert chunk_eval.first_relevant_rank == 3
    assert chunk_eval.mrr == pytest.approx(1.0 / 3)
    assert chunk_eval.precision == pytest.approx(1.0 / 4)
    # One of the two passages is found
    assert chunk_eval.recall == pytest.approx(0.5)
    # One relevant chunk at rank 3, ideally at rank 1
    assert chunk_eval.ndcg == pytest.approx((1.0 / math.log2(4)) / (1.0 / math.log2(2)))


def test_chunk_eval_recall_counts_each_passage_found():
    """
    Input: a result holding a chunk of each passage.
    Output: its chunk level Eval.
    What it does: checks both passages are found and a perfect order gives nDCG 1.
    """
    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1"], PASSAGES)]).compute_retrieved_evals(
        [make_retrieved(["doc1_0", "doc1_1"])], make_chunks()
    )

    chunk_eval = retrieved_evals[0].chunk_eval
    assert chunk_eval.recall == 1.0
    assert chunk_eval.first_relevant_rank == 1
    assert chunk_eval.ndcg == pytest.approx(1.0)
    # 2 relevant chunks out of the k = 4 asked for, though only 2 were returned
    assert chunk_eval.precision == pytest.approx(0.5)


def test_doc_eval_counts_every_chunk_of_an_eval_sample_doc():
    """
    Input: a two-doc question and a result with a doc2 chunk at rank 2 and doc1 chunks at ranks 3 and 4.
    Output: its doc level Eval.
    What it does: checks a chunk of an EvalSample doc is relevant whatever its text, and recall is the share of EvalSample docs found.
    """
    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1", "doc3"], PASSAGES)]).compute_retrieved_evals(
        [make_retrieved(["doc2_0", "doc1_2", "doc1_0"])], make_chunks()
    )

    doc_eval = retrieved_evals[0].doc_eval
    assert doc_eval.hit == 1
    assert doc_eval.first_relevant_rank == 2
    assert doc_eval.mrr == pytest.approx(0.5)
    assert doc_eval.precision == pytest.approx(2.0 / 4)
    # doc1 is found, doc3 is not
    assert doc_eval.recall == pytest.approx(0.5)


def test_a_reranked_result_is_scored_at_rerank_top_k():
    """
    Input: a reranked result with rerank_top_k 2 out of top_k 4.
    Output: its RetrievedEval.
    What it does: checks precision divides by rerank_top_k, not top_k.
    """
    retrieved = make_retrieved(["doc1_0", "doc1_2"], top_k=4, reranking=True, rerank_top_k=2)

    retrieved_evals = RetrievalEvaluation([make_eval_sample(["doc1"], PASSAGES)]).compute_retrieved_evals([retrieved], make_chunks())

    assert retrieved_evals[0].chunk_eval.precision == pytest.approx(1.0 / 2)
    assert retrieved_evals[0].doc_eval.precision == pytest.approx(2.0 / 2)
