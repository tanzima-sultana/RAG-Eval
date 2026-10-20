"""
test_bm25.py

Tests for stages.build.indexing.indexers.bm25.get_bm25_index: building the BM25Okapi
model, saving it and reusing the saved one.
"""

import os

import numpy as np
import pytest

from stages.build.chunking.schema import Chunk
from stages.build.indexing.indexers import bm25 as bm25_indexer


def test_get_bm25_index_builds_and_saves(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: 6 chunks, with no saved model.
    Output: the Bm25Index and the files in the index folder.
    What it does: checks the model is built with every chunk id and saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    bm25_index_path = str(tmp_path / "bm25")

    bm25_index = bm25_indexer.get_bm25_index(bm25_index_path, make_index_args(index_type="bm25"), chunks)

    assert bm25_index.chunk_ids == list(chunks.keys())
    assert bm25_index.bm25_model.corpus_size == 6
    assert os.path.exists(os.path.join(bm25_index_path, "bm25.pkl"))
    assert os.path.exists(os.path.join(bm25_index_path, "chunk_ids.json"))


def test_bm25_model_ranks_the_matching_chunk_first(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: 6 chunks "word{i} shared" and the query ["word3"].
    Output: the position of the best score.
    What it does: checks the chunk holding the query word scores highest.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)

    bm25_index = bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)
    scores = bm25_index.bm25_model.get_scores(["word3"])

    best_position = int(np.argmax(scores))
    assert bm25_index.chunk_ids[best_position] == "doc1_3"


def test_bm25_chunks_are_lowercased(tmp_path, make_index_args):
    """
    Input: chunks with upper case text and the lowercase query ["apple"].
    Output: the score of each chunk.
    What it does: checks chunk text is lowercased, so "APPLE" matches "apple".
    """
    chunks = {}
    chunks["doc0_0"] = Chunk("doc0_0", "APPLE pie", 2, "fixed")
    chunks["doc0_1"] = Chunk("doc0_1", "banana bread", 2, "fixed")
    chunks["doc0_2"] = Chunk("doc0_2", "cherry tart", 2, "fixed")

    bm25_index = bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)
    scores = bm25_index.bm25_model.get_scores(["apple"])

    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_get_bm25_index_reuses_saved_model(tmp_path, make_index_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same chunks indexed twice.
    Output: the second Bm25Index.
    What it does: checks the second call loads the saved model and never builds.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)

    def fail_build(*args, **kwargs):
        """
        Input: the build arguments.
        Output: never returns.
        What it does: fails the test if a build happens.
        """
        raise AssertionError("the saved model should have been reused")

    monkeypatch.setattr(bm25_indexer, "BM25Okapi", fail_build)
    reloaded_index = bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)

    assert reloaded_index.chunk_ids == list(chunks.keys())


def test_get_bm25_index_rebuilds_when_chunks_change(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: 6 chunks indexed, then 4 chunks indexed in the same folder.
    Output: the second Bm25Index.
    What it does: checks a saved model with other chunk ids is replaced.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)

    fewer_chunks, fewer_embeddings = make_chunks_and_embeddings(4)
    rebuilt_index = bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), fewer_chunks)

    assert rebuilt_index.chunk_ids == list(fewer_chunks.keys())
    assert rebuilt_index.bm25_model.corpus_size == 4


def test_unsupported_tokenizer_raises(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: bm25_tokenizer "nltk".
    Output: ValueError mentioning bm25_tokenizer.
    What it does: checks only the whitespace tokenizer is accepted.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)

    with pytest.raises(ValueError, match="bm25_tokenizer"):
        bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25", bm25_tokenizer="nltk"), chunks)


def test_get_bm25_index_with_no_chunks_raises(tmp_path, make_index_args):
    """
    Input: an empty chunks dict.
    Output: ValueError mentioning chunks.
    What it does: checks there must be something to index.
    """
    with pytest.raises(ValueError, match="chunks is empty"):
        bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), {})
