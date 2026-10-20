"""
test_context.py

Tests for stages.evaluation.context.context.get_context_list: turning each Retrieved into the context text the LLM gets.
"""

import pytest

from stages.build.chunking.schema import Chunk
from stages.evaluation.retrieval.schema import Retrieved
from stages.evaluation.context.context import get_context_list
from stages.evaluation.context.schema import Context


def make_retrieved(query_id, chunk_ids, query_type="original"):
    """
    Input: query_id, chunk_ids (best first) and query_type.
    Output: a Retrieved of a dense flatip search with those chunks.
    What it does: builds the retrieval output get_context_list reads; only id and chunk_ids matter to it.
    """
    chunk_scores = []
    doc_ids = []
    for chunk_id in chunk_ids:
        chunk_scores.append(0.5)
        doc_ids.append(chunk_id.rsplit("_", 1)[0])
    return Retrieved(id=query_id, query_type=query_type, chunk_type="fixed", retrieval_type="dense", index_type="faiss_flatip",
                     top_k=5, reranking=False, rerank_top_k=0, chunk_ids=list(chunk_ids), chunk_scores=chunk_scores, doc_ids=doc_ids)


@pytest.fixture
def chunks():
    """
    Input: none.
    Output: three chunks by id, with sizes 4, 3 and 5.
    What it does: gives the tests chunk texts and sizes to build contexts from.
    """
    chunk_list = [
        Chunk(id="doc_a_0", text="Refunds take thirty days.", size=4, type="fixed"),
        Chunk(id="doc_a_1", text="Keep the receipt.", size=3, type="fixed"),
        Chunk(id="doc_b_0", text="Shipping is free over fifty.", size=5, type="fixed"),
    ]
    chunk_by_id = {}
    for chunk in chunk_list:
        chunk_by_id[chunk.id] = chunk
    return chunk_by_id


def test_chunk_texts_are_joined_best_first_with_a_blank_line(chunks):
    """
    Input: one Retrieved with doc_b_0 ranked before doc_a_0.
    Output: one Context holding doc_b_0's text, a blank line, then doc_a_0's text, with token count 5 + 4.
    What it does: checks the retriever's order is kept, the chunks are separated, and the token count is the sum
    of the chunk sizes.
    """
    context_list = get_context_list([make_retrieved("eval_qna_0", ["doc_b_0", "doc_a_0"])], chunks)

    assert context_list == [Context(id="eval_qna_0", context="Shipping is free over fifty.\n\nRefunds take thirty days.", token_count=9)]


def test_one_context_per_retrieved_in_the_same_order(chunks):
    """
    Input: three Retrieved: eval_qna_1, eval_qna_0, then eval_qna_1 again with other chunks (another setup).
    Output: three Contexts with the same ids in the same order, each with its own chunks.
    What it does: checks context_list[i] belongs to retrieved_list[i], even when an id comes twice.
    """
    retrieved_outputs = [
        make_retrieved("eval_qna_1", ["doc_a_1"]),
        make_retrieved("eval_qna_0", ["doc_a_0"]),
        make_retrieved("eval_qna_1", ["doc_b_0"], query_type="hyde"),
    ]

    context_list = get_context_list(retrieved_outputs, chunks)

    ids_and_contexts = []
    for context in context_list:
        ids_and_contexts.append((context.id, context.context))
    assert ids_and_contexts == [
        ("eval_qna_1", "Keep the receipt."),
        ("eval_qna_0", "Refunds take thirty days."),
        ("eval_qna_1", "Shipping is free over fifty."),
    ]


def test_retrieved_with_no_chunks_keeps_its_place_with_an_empty_context(chunks):
    """
    Input: a Retrieved with no chunks between two with chunks.
    Output: three Contexts; the middle one is "" with token count 0.
    What it does: checks an empty result is not dropped, so the lists stay lined up and prompt and generation can mark it.
    """
    retrieved_outputs = [
        make_retrieved("eval_qna_0", ["doc_a_0"]),
        make_retrieved("eval_qna_1", []),
        make_retrieved("eval_qna_2", ["doc_a_1"]),
    ]

    context_list = get_context_list(retrieved_outputs, chunks)

    assert len(context_list) == 3
    assert context_list[1] == Context(id="eval_qna_1", context="", token_count=0)
    assert context_list[2].context == "Keep the receipt."


def test_no_retrieved_gives_no_contexts(chunks):
    """
    Input: an empty Retrieved list.
    Output: an empty Context list.
    What it does: checks a chunk type the filter left nothing for gives nothing.
    """
    assert get_context_list([], chunks) == []


def test_chunk_missing_from_the_chunks_raises(chunks):
    """
    Input: a Retrieved with a chunk id that is not in the chunks given.
    Output: KeyError.
    What it does: checks a Retrieved paired with another chunk type's chunks stops the run instead of making a
    wrong context.
    """
    with pytest.raises(KeyError):
        get_context_list([make_retrieved("eval_qna_0", ["doc_z_9"])], chunks)
