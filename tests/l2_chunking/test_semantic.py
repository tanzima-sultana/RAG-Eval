"""
test_semantic.py

Tests for l2_chunking.semantic.SemanticChunking: sentence-aware
chunking that also splits on a semantic dissimilarity boundary between
consecutive sentences, using precomputed sentence embeddings.
"""

import numpy as np

from l2_chunking.semantic import SemanticChunking


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
