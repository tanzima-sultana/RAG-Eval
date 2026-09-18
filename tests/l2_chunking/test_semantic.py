"""
test_semantic.py

Tests for l2_chunking.semantic.SemanticChunking: sentence-aware
chunking that also splits on a semantic dissimilarity boundary between
consecutive sentences, using precomputed sentence embeddings.
"""

import numpy as np

from common.constants import SEMANTIC_CHUNKING
from l2_chunking.chunkers.semantic import SemanticChunking
from l2_chunking.schema import Chunk


def test_chunk_splits_at_a_dissimilar_sentence_boundary(make_fake_chunking, document_factory):
    """
    Input: 4 sentences whose embeddings form two dissimilar pairs
    (sentences 0-1 similar, 2-3 similar, but 1-2 dissimilar), a token
    budget large enough that only the dissimilarity check can trigger
    a split.
    Output: chunk()'s return value.
    """
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    fake_chunking = make_fake_chunking(
        max_chunk_size=100,
        fixed_chunk_overlap=0,
        semantic_threshold=0.5,
        all_embeddings=embeddings,
        embeddings_offset={"doc1": (0, 4)},
    )
    chunker = SemanticChunking(fake_chunking)
    document = document_factory("doc1", "alpha one. alpha two. beta one. beta two.")

    chunks = chunker.chunk(document)

    texts = [chunk.text for _, chunk in sorted(chunks.items())]
    assert texts == ["alpha one. alpha two.", "beta one. beta two."]


def test_chunk_splits_on_token_budget_when_sentences_are_all_similar(
    make_fake_chunking, document_factory
):
    """
    Input: 3 sentences with identical embeddings (never dissimilar),
    each with 3 tokens, max_chunk_size 5.
    Output: chunk()'s return value.

    Checks that the token-budget split still works independently of
    the dissimilarity check.
    """
    embeddings = np.ones((3, 2))
    fake_chunking = make_fake_chunking(
        max_chunk_size=5,
        fixed_chunk_overlap=0,
        semantic_threshold=0.5,
        all_embeddings=embeddings,
        embeddings_offset={"doc1": (0, 3)},
    )
    chunker = SemanticChunking(fake_chunking)
    document = document_factory("doc1", "w1 w2 w3. w4 w5 w6. w7 w8 w9.")

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    for chunk in chunks.values():
        assert chunk.size == 3


def test_chunk_falls_back_to_a_fixed_window_for_an_oversized_sentence(
    make_fake_chunking, document_factory
):
    """
    Input: a document with a short sentence, a long sentence (way
    over max_chunk_size), and another short sentence, all with
    identical embeddings.
    Output: chunk()'s return value.

    Checks that the oversized middle sentence is split via the
    fixed-window fallback, and the short sentences on either side are
    left as their own chunks.
    """
    embeddings = np.ones((3, 2))
    fake_chunking = make_fake_chunking(
        max_chunk_size=3,
        fixed_chunk_overlap=0,
        semantic_threshold=0.5,
        all_embeddings=embeddings,
        embeddings_offset={"doc1": (0, 3)},
    )
    chunker = SemanticChunking(fake_chunking)
    text = "short one. this sentence has way more than three tokens in it. short two."
    document = document_factory("doc1", text)

    chunks = chunker.chunk(document)

    texts = [chunk.text for _, chunk in sorted(chunks.items())]
    assert texts[0] == "short one."
    assert texts[-1] == "short two."
    assert len(texts) == 6


def test_is_dissimilar_returns_false_when_either_embedding_is_a_zero_vector(make_fake_chunking):
    """
    Input: two embeddings where one is the zero vector.
    Output: _is_dissimilar's return value.

    Checks the divide-by-zero guard: a zero-norm embedding must not
    force a split (or crash), regardless of semantic_threshold.
    """
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
    chunker = SemanticChunking(
        make_fake_chunking(semantic_threshold=0.9, all_embeddings=embeddings, embeddings_offset={})
    )

    assert chunker._is_dissimilar(embeddings, 0, 1) is False


def test_chunk_returns_empty_dict_for_empty_text(make_fake_chunking, document_factory):
    """
    Input: a document with empty text, and an embeddings_offset entry
    pointing at an empty embeddings slice.
    Output: chunk()'s return value.
    """
    fake_chunking = make_fake_chunking(
        max_chunk_size=10,
        fixed_chunk_overlap=0,
        all_embeddings=np.empty((0, 0)),
        embeddings_offset={"doc1": (0, 0)},
    )
    chunker = SemanticChunking(fake_chunking)
    document = document_factory("doc1", "")

    chunks = chunker.chunk(document)

    assert chunks == {}


# ---------- merge_chunks ----------


def _chunk(chunk_id, text, size):
    """
    Builds a Chunk directly, for tests that exercise merge_chunks on
    its own rather than through chunk()'s sentence-splitting logic.
    """
    return Chunk(id=chunk_id, text=text, size=size, type=SEMANTIC_CHUNKING)


def test_merge_chunks_merges_consecutive_chunks_under_min_size_that_fit_the_budget(
    make_fake_chunking,
):
    """
    Input: three chunks, each below min_chunk_size, whose combined
    size still fits max_chunk_size.
    Output: merge_chunks's return value.

    Checks the basic case: a run of undersized chunks collapses into
    one.
    """
    chunker = SemanticChunking(make_fake_chunking(min_chunk_size=128, max_chunk_size=256))
    chunks = {
        "doc1_0": _chunk("doc1_0", "AAAA", 50),
        "doc1_1": _chunk("doc1_1", "BBBB", 50),
        "doc1_2": _chunk("doc1_2", "CCCC", 50),
    }

    merged = chunker.merge_chunks(chunks)

    assert len(merged) == 1
    merged_chunk = list(merged.values())[0]
    assert merged_chunk.text == "AAAA BBBB CCCC"
    assert merged_chunk.size == 150


def test_merge_chunks_merges_into_the_actual_immediately_preceding_chunk(make_fake_chunking):
    """
    Input: chunk A (too big to merge with B), chunk B (small, but its
    combined size with A exceeds max_chunk_size so it stays
    unmerged), chunk C (small enough to merge with B).
    Output: merge_chunks's return value.

    Regression test: C must merge into B, its real predecessor in the
    output sequence -- not into A, an earlier chunk that never
    actually preceded C once B was kept separate.
    """
    chunker = SemanticChunking(make_fake_chunking(min_chunk_size=128, max_chunk_size=256))
    chunks = {
        "doc1_0": _chunk("doc1_0", "AAAA", 200),
        "doc1_1": _chunk("doc1_1", "BBBB", 100),
        "doc1_2": _chunk("doc1_2", "CCCC", 50),
    }

    merged = chunker.merge_chunks(chunks)

    assert len(merged) == 2
    assert merged["doc1_0"].text == "AAAA"
    assert merged["doc1_0"].size == 200
    assert merged["doc1_1"].text == "BBBB CCCC"
    assert merged["doc1_1"].size == 150


def test_merge_chunks_does_not_mutate_the_input_chunks_dict(make_fake_chunking):
    """
    Input: two chunks that merge_chunks will merge.
    Output: merge_chunks's return value, and the original chunks dict
    passed in.

    Checks that merge_chunks builds new Chunk objects rather than
    mutating the caller's original Chunk objects in place.
    """
    chunker = SemanticChunking(make_fake_chunking(min_chunk_size=128, max_chunk_size=256))
    original_a = _chunk("doc1_0", "AAAA", 200)
    chunks = {"doc1_0": original_a, "doc1_1": _chunk("doc1_1", "BBBB", 50)}

    chunker.merge_chunks(chunks)

    assert chunks["doc1_0"] is original_a
    assert original_a.text == "AAAA"
    assert original_a.size == 200


def test_merge_chunks_leaves_chunks_at_or_above_min_size_alone(make_fake_chunking):
    """
    Input: two chunks, neither below min_chunk_size.
    Output: merge_chunks's return value.
    """
    chunker = SemanticChunking(make_fake_chunking(min_chunk_size=128, max_chunk_size=256))
    chunks = {
        "doc1_0": _chunk("doc1_0", "AAAA", 200),
        "doc1_1": _chunk("doc1_1", "BBBB", 200),
    }

    merged = chunker.merge_chunks(chunks)

    assert merged == chunks
