"""
test_fixed.py

Tests for l2_chunking.fixed.FixedChunking: the fixed-window chunking
strategy, which slides a max_chunk_size window across a document's
whole token stream.
"""

from l2_chunking.chunkers.fixed import FixedChunking


def test_chunk_slides_a_window_with_overlap_across_the_whole_document(
    make_fake_chunking, document_factory
):
    """
    Input: a document with 7 whitespace-separated tokens,
    max_chunk_size 4, overlap 1.
    Output: chunk()'s return value.

    Checks the sliding-window split end to end: window text, size,
    and chunk_type stamping.
    """
    chunker = FixedChunking(make_fake_chunking(max_chunk_size=4, fixed_chunk_overlap=1))
    document = document_factory("doc1", "one two three four five six seven")

    chunks = chunker.chunk(document)

    assert chunks["doc1_0"].text == "one two three four"
    assert chunks["doc1_1"].text == "four five six seven"
    assert chunks["doc1_2"].text == "seven"
    assert chunks["doc1_2"].size == 1
    assert all(chunk.type == "fixed" for chunk in chunks.values())


def test_chunk_produces_a_single_chunk_when_text_fits_within_max_chunk_size(
    make_fake_chunking, document_factory
):
    """
    Input: a document with 2 tokens and max_chunk_size 10.
    Output: chunk()'s return value.
    """
    chunker = FixedChunking(make_fake_chunking(max_chunk_size=10, fixed_chunk_overlap=2))
    document = document_factory("doc1", "short text")

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks["doc1_0"].text == "short text"
    assert chunks["doc1_0"].size == 2


def test_chunk_returns_empty_dict_for_empty_text(make_fake_chunking, document_factory):
    """
    Input: a document with empty text.
    Output: chunk()'s return value.
    """
    chunker = FixedChunking(make_fake_chunking(max_chunk_size=4, fixed_chunk_overlap=0))
    document = document_factory("doc1", "")

    chunks = chunker.chunk(document)

    assert chunks == {}
