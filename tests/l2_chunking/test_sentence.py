"""
test_sentence.py

Tests for l2_chunking.sentence.SentenceChunking: sentence-aware
chunking, which accumulates whole sentences up to max_chunk_size and
falls back to a fixed window for any single sentence too long to fit
in a chunk on its own.

Several of these are regression tests for bugs found and fixed
earlier while building this strategy:
  - test_chunk_does_not_drop_the_sentence_that_triggers_a_flush guards
    against the sentence that pushes a chunk over budget being
    silently dropped instead of starting the next chunk.
  - test_chunk_final_trailing_chunk_text_is_a_string_not_a_list guards
    against the last chunk's text being left as the raw list of
    sentences instead of being joined into a string.
  - test_chunk_respects_the_real_token_count_not_the_sentence_count
    guards against the token budget being compared against how many
    sentences have accumulated instead of how many tokens they hold.
"""

from l2_chunking.sentence import SentenceChunking


def test_chunk_accumulates_sentences_until_the_token_budget_is_exceeded(
    make_fake_chunking, document_factory
):
    """
    Input: a document with three 8-token sentences, max_chunk_size 10.
    Output: chunk()'s return value.

    Checks that each sentence becomes its own chunk once accumulating
    another would exceed max_chunk_size.
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=10, fixed_chunk_overlap=0))
    text = (
        "w1 w2 w3 w4 w5 w6 w7 w8. "
        "w9 w10 w11 w12 w13 w14 w15 w16. "
        "w17 w18 w19 w20 w21 w22 w23 w24."
    )
    document = document_factory("doc1", text)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    for chunk in chunks.values():
        assert chunk.size == 8
        assert isinstance(chunk.text, str)


def test_chunk_respects_the_real_token_count_not_the_sentence_count(
    make_fake_chunking, document_factory
):
    """
    Input: three 8-token sentences (24 tokens total), max_chunk_size
    10.
    Output: chunk()'s return value.

    Regression test: the token-budget check must count actual tokens
    across accumulated sentences, not how many sentences have been
    accumulated. All three sentences merging into one chunk here
    would mean the budget is being measured in sentences (3 <= 10)
    instead of tokens (24 > 10).
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=10, fixed_chunk_overlap=0))
    text = (
        "w1 w2 w3 w4 w5 w6 w7 w8. "
        "w9 w10 w11 w12 w13 w14 w15 w16. "
        "w17 w18 w19 w20 w21 w22 w23 w24."
    )
    document = document_factory("doc1", text)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert all(chunk.size <= 10 for chunk in chunks.values())


def test_chunk_does_not_drop_the_sentence_that_triggers_a_flush(
    make_fake_chunking, document_factory
):
    """
    Input: three short sentences where the third one exceeds the
    budget together with the first two.
    Output: chunk()'s return value.

    Regression test: the sentence that pushes the running chunk over
    max_chunk_size must start the next chunk, not be silently dropped.
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=5, fixed_chunk_overlap=0))
    document = document_factory("doc1", "A A A. B B B. C C C C C.")

    chunks = chunker.chunk(document)

    texts = sorted(chunk.text for chunk in chunks.values())
    assert texts == ["A A A.", "B B B.", "C C C C C."]


def test_chunk_falls_back_to_a_fixed_window_for_a_sentence_longer_than_max_chunk_size(
    make_fake_chunking, document_factory
):
    """
    Input: a document whose single sentence has 6 tokens, but
    max_chunk_size is only 3.
    Output: chunk()'s return value.

    Checks that an oversized sentence is split via the fixed-window
    fallback rather than kept whole or dropped.
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=3, fixed_chunk_overlap=0))
    document = document_factory("doc1", "one two three four five six.")

    chunks = chunker.chunk(document)

    texts = [chunk.text for _, chunk in sorted(chunks.items())]
    assert texts == ["one two three", "four five six."]


def test_chunk_final_trailing_chunk_text_is_a_string_not_a_list(
    make_fake_chunking, document_factory
):
    """
    Input: a document short enough that everything accumulates into
    one chunk, only finalized after the sentence loop ends.
    Output: chunk()'s return value.

    Regression test: the trailing chunk built after the main loop
    must join accumulated sentence texts into a string, not pass the
    raw list of sentences through as Chunk.text.
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=100, fixed_chunk_overlap=0))
    document = document_factory("doc1", "First sentence. Second sentence.")

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    chunk = list(chunks.values())[0]
    assert isinstance(chunk.text, str)
    assert chunk.text == "First sentence. Second sentence."


def test_chunk_returns_empty_dict_for_empty_text(make_fake_chunking, document_factory):
    """
    Input: a document with empty text.
    Output: chunk()'s return value.
    """
    chunker = SentenceChunking(make_fake_chunking(max_chunk_size=10, fixed_chunk_overlap=0))
    document = document_factory("doc1", "")

    chunks = chunker.chunk(document)

    assert chunks == {}
