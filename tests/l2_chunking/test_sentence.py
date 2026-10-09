"""
test_sentence.py

Tests for l2_chunking.chunkers.sentence.SentenceChunking. Token counts use
the conftest FakeTokenizer: one token per word, punctuation included.
"""

from l2_chunking.chunkers.sentence import SentenceChunking


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


def test_sentence_packs_whole_sentences_up_to_max_size(make_chunker, make_chunking_args, make_document):
    """
    Input: sentences of 3, 3 and 1 words with max_chunk_size 6.
    Output: the chunks.
    What it does: checks the first two sentences fill one chunk and the
    third, which would go over 6 tokens, starts a new one.
    """
    chunker = make_chunker(SentenceChunking, make_chunking_args(max_chunk_size=6))
    document = make_document("doc", "One two three. Four five six. Seven.")

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["One two three. Four five six.", "Seven."]
    assert chunks["doc_0"].size == 6
    assert chunks["doc_1"].size == 1
    assert chunks["doc_0"].type == "sentence"


def test_sentence_splits_a_too_long_sentence_with_fixed_windows(make_chunker, make_chunking_args, make_document):
    """
    Input: a 2-word sentence, then a 10-word sentence, with max_chunk_size 4 (step 3)
    and overlap 1.
    Output: the chunks.
    What it does: checks the short sentence is flushed first, then the
    long sentence is split into overlapping windows with continued ids.
    """
    chunker = make_chunker(SentenceChunking, make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1))
    document = make_document("doc", "Short one. a b c d e f g h i j.")

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["Short one.", "a b c d", "d e f g", "g h i j."]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2", "doc_3"]


def test_sentence_empty_document_gives_no_chunks(make_chunker, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the chunks.
    What it does: checks no chunks are made.
    """
    chunker = make_chunker(SentenceChunking, make_chunking_args())
    document = make_document("doc", "")

    assert chunker.chunk(document) == {}
