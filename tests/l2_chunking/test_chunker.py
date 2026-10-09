"""
test_chunker.py

Tests for l2_chunking.chunker: the shared Chunker base class helpers and
the get_chunker registry lookup.
"""

import numpy as np
import pytest

from l2_chunking.chunker import Chunker
from l2_chunking.chunker import get_chunker
from l2_chunking.chunkers.fixed import FixedChunking
from l2_chunking.chunkers.semantic import SemanticChunking
from l2_chunking.chunkers.sentence import SentenceChunking


class MinimalChunker(Chunker):
    """
    Smallest concrete Chunker, used to test the base class helpers
    without any real strategy's chunk() logic.
    """

    chunk_type = "minimal"

    def chunk(self, document):
        """
        Input: document (ignored).
        Output: an empty dict.
        What it does: nothing; only here so the class can be created.
        """
        return {}


# ---------- get_chunker ----------


@pytest.mark.parametrize(
    "chunk_type, expected_class",
    [("fixed", FixedChunking), ("sentence", SentenceChunking), ("semantic", SemanticChunking)],
)
def test_get_chunker_returns_registered_class(chunk_type, expected_class):
    """
    Input: each supported chunk_type.
    Output: the class get_chunker returns.
    What it does: checks each chunk_type maps to its strategy class.
    """
    assert get_chunker(chunk_type) is expected_class


def test_get_chunker_raises_for_unknown_chunk_type():
    """
    Input: chunk_type "paragraph".
    Output: ValueError mentioning the unknown type.
    What it does: checks an unregistered chunk_type is rejected.
    """
    with pytest.raises(ValueError, match="paragraph"):
        get_chunker("paragraph")


# ---------- Chunker.__init__ ----------


def test_init_copies_settings_model_and_embeddings(make_chunker, make_chunking_args):
    """
    Input: ChunkingArgs and semantic embeddings passed through a stand-in Chunking.
    Output: the Chunker's attributes.
    What it does: checks sizes, threshold, tokenizer and embeddings are copied.
    """
    chunking_args = make_chunking_args(min_chunk_size=3, max_chunk_size=9, fixed_chunk_overlap=1, semantic_threshold=0.7)
    all_embeddings = np.zeros((2, 2))
    embeddings_offset = {"doc": (0, 2)}

    chunker = make_chunker(MinimalChunker, chunking_args, all_embeddings, embeddings_offset)

    assert chunker.min_chunk_size == 3
    assert chunker.max_chunk_size == 9
    assert chunker.fixed_chunk_overlap == 1
    assert chunker.semantic_threshold == 0.7
    assert chunker.tokenizer is chunker.model.tokenizer
    assert chunker.all_embeddings is all_embeddings
    assert chunker.embeddings_offset is embeddings_offset


# ---------- _finalize_chunk ----------


def test_finalize_chunk_builds_id_size_and_type(make_chunker, make_chunking_args):
    """
    Input: doc_id "doc", chunk_idx 3, a text and 4 tokens.
    Output: the Chunk built.
    What it does: checks the id is "doc_3", size is the token count and
    type is the chunker's chunk_type.
    """
    chunker = make_chunker(MinimalChunker, make_chunking_args())

    chunk = chunker._finalize_chunk("doc", 3, "a b c d", ["a", "b", "c", "d"])

    assert chunk.id == "doc_3"
    assert chunk.text == "a b c d"
    assert chunk.size == 4
    assert chunk.type == "minimal"


# ---------- _split_tokens_into_chunks ----------


def test_split_tokens_slides_window_with_overlap(make_chunker, make_chunking_args):
    """
    Input: 10 tokens with max_chunk_size 4 and overlap 1 (step 3).
    Output: the chunks and the next chunk index.
    What it does: checks windows start at 0, 3, 6 and 9, share one token
    with the window before, and the next index is 4.
    """
    chunker = make_chunker(MinimalChunker, make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1))
    tokens = "a b c d e f g h i j".split()

    chunks, next_chunk_idx = chunker._split_tokens_into_chunks("doc", tokens)

    chunk_texts = []
    for chunk_id in chunks:
        chunk_texts.append(chunks[chunk_id].text)
    assert chunk_texts == ["a b c d", "d e f g", "g h i j", "j"]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2", "doc_3"]
    assert next_chunk_idx == 4


def test_split_tokens_continues_numbering_from_given_index(make_chunker, make_chunking_args):
    """
    Input: 6 tokens with max_chunk_size 4, overlap 1, starting at chunk_idx 5.
    Output: the chunk ids and the next chunk index.
    What it does: checks numbering starts at 5 instead of 0.
    """
    chunker = make_chunker(MinimalChunker, make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1))
    tokens = "a b c d e f".split()

    chunks, next_chunk_idx = chunker._split_tokens_into_chunks("doc", tokens, 5)

    assert list(chunks.keys()) == ["doc_5", "doc_6"]
    assert next_chunk_idx == 7


def test_split_tokens_with_no_tokens_returns_nothing(make_chunker, make_chunking_args):
    """
    Input: an empty token list starting at chunk_idx 2.
    Output: the chunks and the next chunk index.
    What it does: checks no chunks are made and the index is unchanged.
    """
    chunker = make_chunker(MinimalChunker, make_chunking_args())

    chunks, next_chunk_idx = chunker._split_tokens_into_chunks("doc", [], 2)

    assert chunks == {}
    assert next_chunk_idx == 2
