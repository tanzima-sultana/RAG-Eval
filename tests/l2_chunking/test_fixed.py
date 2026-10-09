"""
test_fixed.py

Tests for l2_chunking.chunkers.fixed.FixedChunking.
"""

from l2_chunking.chunkers.fixed import FixedChunking


def get_chunk_texts(chunks):
    """
    Input: chunks, a dict of chunk_id to Chunk.
    Output: the list of chunk texts in order.
    What it does: collects each chunk's text.
    """
    chunk_texts = []
    for chunk_id in chunks:
        chunk_texts.append(chunks[chunk_id].text)
    return chunk_texts


def test_fixed_splits_document_into_overlapping_windows(make_chunker, make_chunking_args, make_document):
    """
    Input: a 10-word document with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks the windows, sizes and fixed type.
    """
    chunker = make_chunker(FixedChunking, make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1))
    document = make_document("doc", "a b c d e f g h i j")

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["a b c d", "d e f g", "g h i j"]
    assert chunks["doc_0"].size == 4
    assert chunks["doc_2"].size == 4
    assert chunks["doc_0"].type == "fixed"


def test_fixed_short_document_gives_one_chunk(make_chunker, make_chunking_args, make_document):
    """
    Input: a 3-word document with max_chunk_size 6.
    Output: the chunks.
    What it does: checks the whole document becomes one chunk.
    """
    chunker = make_chunker(FixedChunking, make_chunking_args(max_chunk_size=6))
    document = make_document("doc", "a b c")

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["a b c"]


def test_fixed_empty_document_gives_no_chunks(make_chunker, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the chunks.
    What it does: checks no chunks are made.
    """
    chunker = make_chunker(FixedChunking, make_chunking_args())
    document = make_document("doc", "")

    assert chunker.chunk(document) == {}
