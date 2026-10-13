"""
test_faiss.py

Tests for l4_indexing.indexers.faiss: building the flat, IVF and HNSW
indexes, and get_faiss_index's build, save and cache reuse.
"""

import os

import numpy as np
import pytest

from l4_indexing.indexers import faiss as faiss_indexer


def make_vectors(embeddings):
    """
    Input: embeddings, a dict of chunk_id to Embedded.
    Output: a float32 array with one row per chunk, in dict order.
    What it does: stacks the embedding vectors.
    """
    vectors = []
    for chunk_id in embeddings:
        vectors.append(embeddings[chunk_id].embeddings)
    return np.array(vectors, dtype=np.float32)


def test_flatip_index_finds_each_vector_itself(make_chunks_and_embeddings):
    """
    Input: 10 unit vectors in a flat inner product index.
    Output: the top hit of searching with each vector.
    What it does: checks each vector's nearest neighbour is itself, with score 1.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    vectors = make_vectors(embeddings)

    flatip_index = faiss_indexer.build_flatip_index(vectors, vectors.shape[1])
    scores, positions = flatip_index.search(vectors, 1)

    assert flatip_index.ntotal == 10
    for row in range(10):
        assert positions[row][0] == row
        assert scores[row][0] == pytest.approx(1.0, abs=1e-5)


def test_ivf_index_holds_every_vector(make_chunks_and_embeddings):
    """
    Input: 10 vectors and ivf_nlist 2.
    Output: the IVF index.
    What it does: checks the index is trained and holds all 10 vectors.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    vectors = make_vectors(embeddings)

    ivf_index = faiss_indexer.build_ivf_index(vectors, vectors.shape[1], 2)

    assert ivf_index.is_trained
    assert ivf_index.ntotal == 10


def test_ivf_index_with_more_clusters_than_vectors_raises(make_chunks_and_embeddings):
    """
    Input: 3 vectors and ivf_nlist 4.
    Output: ValueError mentioning ivf_nlist.
    What it does: checks FAISS is never asked to train more clusters than vectors.
    """
    chunks, embeddings = make_chunks_and_embeddings(3)
    vectors = make_vectors(embeddings)

    with pytest.raises(ValueError, match="ivf_nlist"):
        faiss_indexer.build_ivf_index(vectors, vectors.shape[1], 4)


def test_hnsw_index_holds_every_vector_with_ef_construction(make_chunks_and_embeddings):
    """
    Input: 10 vectors, hnsw_m 4 and hnsw_ef_construction 8.
    Output: the HNSW index.
    What it does: checks the index holds all vectors and keeps efConstruction.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    vectors = make_vectors(embeddings)

    hnsw_index = faiss_indexer.build_hnsw_index(vectors, vectors.shape[1], 4, 8)

    assert hnsw_index.ntotal == 10
    assert hnsw_index.hnsw.efConstruction == 8


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_get_faiss_index_builds_and_saves(tmp_path, make_index_args, make_chunks_and_embeddings, index_type):
    """
    Input: 10 embeddings and each FAISS index type, with no saved index.
    Output: the FaissIndex and the files in the index folder.
    What it does: checks the index is built with every chunk id and saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index_path = str(tmp_path / "faiss")

    faiss_index = faiss_indexer.get_faiss_index(faiss_index_path, make_index_args(index_type=index_type), embeddings)

    assert faiss_index.faiss_index_type == index_type
    assert faiss_index.faiss_index.ntotal == 10
    assert faiss_index.chunk_ids == list(embeddings.keys())
    assert faiss_index.dimension == 4
    assert os.path.exists(os.path.join(faiss_index_path, "index.faiss"))
    assert os.path.exists(os.path.join(faiss_index_path, "chunk_ids.json"))


def test_get_faiss_index_reuses_saved_index(tmp_path, make_index_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same embeddings indexed twice.
    Output: the second FaissIndex.
    What it does: checks the second call loads the saved index and never builds.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index_path = str(tmp_path / "faiss")
    faiss_indexer.get_faiss_index(faiss_index_path, make_index_args(), embeddings)

    def fail_build(vectors, dimension):
        """
        Input: the build arguments.
        Output: never returns.
        What it does: fails the test if a build happens.
        """
        raise AssertionError("the saved index should have been reused")

    monkeypatch.setattr(faiss_indexer, "build_flatip_index", fail_build)
    reloaded_index = faiss_indexer.get_faiss_index(faiss_index_path, make_index_args(), embeddings)

    assert reloaded_index.faiss_index.ntotal == 10
    assert reloaded_index.chunk_ids == list(embeddings.keys())


def test_get_faiss_index_rebuilds_when_chunks_change(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: 10 embeddings indexed, then 6 embeddings indexed in the same folder.
    Output: the second FaissIndex.
    What it does: checks a saved index with other chunk ids is replaced.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index_path = str(tmp_path / "faiss")
    faiss_indexer.get_faiss_index(faiss_index_path, make_index_args(), embeddings)

    fewer_chunks, fewer_embeddings = make_chunks_and_embeddings(6)
    rebuilt_index = faiss_indexer.get_faiss_index(faiss_index_path, make_index_args(), fewer_embeddings)

    assert rebuilt_index.faiss_index.ntotal == 6
    assert rebuilt_index.chunk_ids == list(fewer_embeddings.keys())


def test_get_faiss_index_with_no_embeddings_raises(tmp_path, make_index_args):
    """
    Input: an empty embeddings dict.
    Output: ValueError mentioning embeddings.
    What it does: checks there must be something to index.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        faiss_indexer.get_faiss_index(str(tmp_path), make_index_args(), {})
