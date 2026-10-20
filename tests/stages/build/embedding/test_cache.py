"""
test_cache.py

Tests for stages.build.embedding.cache: saving a document's embeddings as one
{doc_id}.npz file, loading them back, and loading chosen chunk ids with
load_embeddings.
"""

import os

import numpy as np
import pytest

from stages.build.embedding import cache
from stages.build.embedding.schema import Embedded


def make_embeddings(doc_id, chunk_count):
    """
    Input: doc_id and chunk_count.
    Output: a dict of chunk_id to Embedded with chunk_count embeddings for doc_id.
    What it does: builds embeddings with ids "{doc_id}_0", "{doc_id}_1", ... and the vector [index, 1].
    """
    embeddings = {}
    for chunk_index in range(chunk_count):
        chunk_id = f"{doc_id}_{chunk_index}"
        embeddings[chunk_id] = Embedded(chunk_id, np.array([float(chunk_index), 1.0]), True)
    return embeddings


def test_load_cached_embeddings_returns_none_when_no_file(tmp_path):
    """
    Input: an empty embedding folder.
    Output: _load_cached_embeddings' return value.
    What it does: checks None is returned for a document with no file.
    """
    assert cache._load_cached_embeddings(str(tmp_path), "doc1") is None


def test_save_and_load_embeddings_round_trip(tmp_path):
    """
    Input: three embeddings of one document, saved then loaded.
    Output: the loaded embeddings and the folder's files.
    What it does: checks the ids, order, float32 vectors and normalized flags come back, and no
    temporary file is left behind.
    """
    embeddings = make_embeddings("doc1", 3)

    cache._save_embeddings(str(tmp_path), "doc1", embeddings)
    loaded_embeddings = cache._load_cached_embeddings(str(tmp_path), "doc1")

    assert os.listdir(tmp_path) == ["doc1.npz"]
    assert list(loaded_embeddings.keys()) == ["doc1_0", "doc1_1", "doc1_2"]
    assert loaded_embeddings["doc1_2"].embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded_embeddings["doc1_2"].embeddings, [2.0, 1.0])
    assert loaded_embeddings["doc1_2"].normalized is True


def test_load_embeddings_returns_the_requested_chunks_in_order(tmp_path):
    """
    Input: two saved documents, and chunk ids from both in a mixed order.
    Output: load_embeddings' return value.
    What it does: checks only the requested chunks are returned, in the requested order.
    """
    cache._save_embeddings(str(tmp_path), "doc1", make_embeddings("doc1", 2))
    cache._save_embeddings(str(tmp_path), "doc2", make_embeddings("doc2", 2))

    embeddings = cache.load_embeddings(str(tmp_path), ["doc2_1", "doc1_0"])

    assert list(embeddings.keys()) == ["doc2_1", "doc1_0"]
    np.testing.assert_array_equal(embeddings["doc2_1"].embeddings, [1.0, 1.0])


def test_load_embeddings_raises_when_the_folder_does_not_exist(tmp_path):
    """
    Input: an embedding folder that was never created.
    Output: the raised error.
    What it does: checks a FileNotFoundError is raised.
    """
    with pytest.raises(FileNotFoundError, match="No embedding folder"):
        cache.load_embeddings(str(tmp_path / "missing"), ["doc1_0"])


def test_load_embeddings_raises_when_a_document_file_is_missing(tmp_path):
    """
    Input: a folder holding doc1 only, and a chunk id of doc2.
    Output: the raised error.
    What it does: checks a FileNotFoundError names the missing document.
    """
    cache._save_embeddings(str(tmp_path), "doc1", make_embeddings("doc1", 1))

    with pytest.raises(FileNotFoundError, match="doc2"):
        cache.load_embeddings(str(tmp_path), ["doc2_0"])


def test_load_embeddings_raises_when_a_chunk_is_not_in_its_document_file(tmp_path):
    """
    Input: doc1 saved with one chunk, and a request for its second chunk.
    Output: the raised error.
    What it does: checks a ValueError names the missing chunk.
    """
    cache._save_embeddings(str(tmp_path), "doc1", make_embeddings("doc1", 1))

    with pytest.raises(ValueError, match="doc1_1"):
        cache.load_embeddings(str(tmp_path), ["doc1_1"])
