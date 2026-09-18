"""
test_chunker.py

Tests for l2_chunking.chunker: the Chunker abstract base class (its
shared __init__, _finalize_chunk, and _split_tokens_into_chunks
helpers, used by every concrete strategy) and the get_chunker registry
lookup.
"""

import pytest

from l2_chunking.chunker import Chunker, get_chunker
from l2_chunking.chunkers.fixed import FixedChunking
from l2_chunking.chunkers.sentence import SentenceChunking
from l2_chunking.chunkers.semantic import SemanticChunking
from l2_chunking.schema import Chunk


class _MinimalChunker(Chunker):
    """
    Smallest possible concrete Chunker, used to exercise the shared
    base-class machinery (__init__, _finalize_chunk,
    _split_tokens_into_chunks) directly, without going through a real
    strategy's own chunk() logic.
    """

    chunk_type = "minimal_test"

    def chunk(self, document):
        return {}


# ---------- get_chunker ----------


def test_get_chunker_returns_fixed_chunking_class_for_fixed_chunk_type():
    """
    Input: chunk_type "fixed".
    Output: get_chunker's return value.
    """
    assert get_chunker("fixed") is FixedChunking


def test_get_chunker_returns_sentence_chunking_class_for_sentence_chunk_type():
    """
    Input: chunk_type "sentence".
    Output: get_chunker's return value.
    """
    assert get_chunker("sentence") is SentenceChunking


def test_get_chunker_returns_semantic_chunking_class_for_semantic_chunk_type():
    """
    Input: chunk_type "semantic".
    Output: get_chunker's return value.
    """
    assert get_chunker("semantic") is SemanticChunking


def test_get_chunker_raises_value_error_for_unknown_chunk_type():
    """
    Input: a chunk_type with no registered strategy.
    Output: none; get_chunker is expected to raise.
    """
    with pytest.raises(ValueError):
        get_chunker("unknown-strategy")


# ---------- Chunker (abstract) ----------


def test_chunker_cannot_be_instantiated_directly(make_fake_chunking):
    """
    Input: none.
    Output: none; Chunker(...) is expected to raise.

    Checks that Chunker is an abstract base class: it declares chunk
    as abstract, so it cannot be instantiated on its own even though
    __init__ itself is concrete.
    """
    with pytest.raises(TypeError):
        Chunker(make_fake_chunking())


# ---------- Chunker.__init__ ----------


def test_chunker_init_copies_config_and_model_state_from_chunking(
    make_fake_chunking, fake_tokenizer
):
    """
    Input: a fake Chunking-like object with distinct, recognizable
    values for every attribute Chunker.__init__ copies.
    Output: the constructed _MinimalChunker's own attributes.

    Checks that every attribute Chunker.__init__ documents copying is
    actually copied onto the strategy instance.
    """
    fake_chunking = make_fake_chunking(
        max_chunk_size=42,
        fixed_chunk_overlap=7,
        semantic_threshold=0.3,
        model="fake-model-object",
        all_embeddings="fake-embeddings",
        embeddings_offset={"doc1": (0, 1)},
    )

    chunker = _MinimalChunker(fake_chunking)

    assert chunker.max_chunk_size == 42
    assert chunker.fixed_chunk_overlap == 7
    assert chunker.semantic_threshold == 0.3
    assert chunker.model == "fake-model-object"
    assert chunker.tokenizer is fake_tokenizer
    assert chunker.all_embeddings == "fake-embeddings"
    assert chunker.embeddings_offset == {"doc1": (0, 1)}


# ---------- _finalize_chunk ----------


def test_finalize_chunk_builds_chunk_with_id_from_doc_id_and_chunk_idx(make_fake_chunking):
    """
    Input: a doc_id, a chunk_idx, chunk text, and a token list.
    Output: _finalize_chunk's return value.

    Checks the id format, size accounting, and chunk_type stamping.
    """
    chunker = _MinimalChunker(make_fake_chunking())

    chunk = chunker._finalize_chunk("doc1", 3, "some text", ["t1", "t2", "t3"])

    assert chunk == Chunk(id="doc1_3", text="some text", size=3, type="minimal_test")


# ---------- _split_tokens_into_chunks ----------


def test_split_tokens_into_chunks_slides_a_window_with_overlap(make_fake_chunking):
    """
    Input: 7 whitespace-separated tokens, max_chunk_size 4, overlap 1.
    Output: _split_tokens_into_chunks's return value.

    Checks the sliding-window math: window size, step size, and the
    final, shorter trailing window.
    """
    chunker = _MinimalChunker(make_fake_chunking(max_chunk_size=4, fixed_chunk_overlap=1))
    tokens = "one two three four five six seven".split()

    chunks, next_idx = chunker._split_tokens_into_chunks("doc1", tokens)

    assert next_idx == 3
    assert chunks["doc1_0"].text == "one two three four"
    assert chunks["doc1_1"].text == "four five six seven"
    assert chunks["doc1_2"].text == "seven"
    assert chunks["doc1_2"].size == 1


def test_split_tokens_into_chunks_starts_numbering_from_given_chunk_idx(make_fake_chunking):
    """
    Input: 4 tokens, max_chunk_size 2, overlap 0, chunk_idx starting
    at 5.
    Output: _split_tokens_into_chunks's return value.

    Checks that a caller resuming numbering after already-emitted
    chunks (e.g. sentence/semantic's long-sentence fallback) gets
    chunk ids that continue the sequence instead of restarting at 0.
    """
    chunker = _MinimalChunker(make_fake_chunking(max_chunk_size=2, fixed_chunk_overlap=0))
    tokens = "a b c d".split()

    chunks, next_idx = chunker._split_tokens_into_chunks("doc1", tokens, chunk_idx=5)

    assert set(chunks.keys()) == {"doc1_5", "doc1_6"}
    assert next_idx == 7


def test_split_tokens_into_chunks_returns_empty_dict_for_empty_tokens(make_fake_chunking):
    """
    Input: an empty token list.
    Output: _split_tokens_into_chunks's return value.
    """
    chunker = _MinimalChunker(make_fake_chunking(max_chunk_size=4, fixed_chunk_overlap=0))

    chunks, next_idx = chunker._split_tokens_into_chunks("doc1", [])

    assert chunks == {}
    assert next_idx == 0
