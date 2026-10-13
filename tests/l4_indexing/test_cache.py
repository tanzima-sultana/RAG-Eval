"""
test_cache.py

Tests for l4_indexing.cache: saving the built-in indexes, the
_load_cached_* checks used while building, and the load-only functions
used by eval, including load_index choosing the right loader.
"""

import json
import os

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from l4_indexing import cache
from l4_indexing.indexers import bm25 as bm25_indexer
from l4_indexing.indexers import faiss as faiss_indexer
from l4_indexing.indexing import Indexing
from l4_indexing.schema import FaissIndex
from l4_indexing.schema import Bm25Index
from l4_indexing.schema import QdrantIndex
from l4_indexing.schema import LangchainFaissIndex
from l4_indexing.schema import LangchainBm25Index
from l4_indexing.schema import LangchainQdrantIndex


def save_flatip_index(faiss_index_path, embeddings):
    """
    Input: faiss_index_path and embeddings.
    Output: nothing.
    What it does: builds a flat index from embeddings and saves it.
    """
    chunk_ids = []
    vectors = []
    for chunk_id in embeddings:
        chunk_ids.append(chunk_id)
        vectors.append(embeddings[chunk_id].embeddings)
    vectors = np.array(vectors, dtype=np.float32)
    flatip_index = faiss_indexer.build_flatip_index(vectors, vectors.shape[1])
    cache._save_faiss_index(faiss_index_path, flatip_index, chunk_ids)


# ---------- Built-in FAISS ----------


def test_saved_faiss_chunk_ids_keep_their_order(tmp_path, make_chunks_and_embeddings):
    """
    Input: a flat index saved from 4 embeddings.
    Output: chunk_ids.json.
    What it does: checks the chunk ids are written in FAISS position order.
    """
    chunks, embeddings = make_chunks_and_embeddings(4)
    save_flatip_index(str(tmp_path), embeddings)

    with open(os.path.join(str(tmp_path), "chunk_ids.json"), encoding="utf-8") as f:
        saved_chunk_ids = json.load(f)

    assert saved_chunk_ids == list(embeddings.keys())
    assert not os.path.exists(os.path.join(str(tmp_path), "chunk_ids.json.tmp"))


def test_load_cached_faiss_index_returns_none_when_missing(tmp_path):
    """
    Input: an empty folder.
    Output: None.
    What it does: checks a missing index is not an error while building.
    """
    assert cache._load_cached_faiss_index(str(tmp_path), "faiss_flatip", ["doc0_0"]) is None


def test_load_cached_faiss_index_returns_none_for_other_chunks(tmp_path, make_chunks_and_embeddings):
    """
    Input: an index saved from 4 embeddings, checked against other chunk ids.
    Output: None.
    What it does: checks an index holding other chunks is not reused.
    """
    chunks, embeddings = make_chunks_and_embeddings(4)
    save_flatip_index(str(tmp_path), embeddings)

    assert cache._load_cached_faiss_index(str(tmp_path), "faiss_flatip", ["doc0_0", "doc1_1"]) is None


def test_load_cached_faiss_index_keeps_saved_order(tmp_path, make_chunks_and_embeddings):
    """
    Input: an index saved from 4 embeddings, checked against the same ids in reverse order.
    Output: the FaissIndex.
    What it does: checks the saved chunk ids are returned, since they match the FAISS positions.
    """
    chunks, embeddings = make_chunks_and_embeddings(4)
    save_flatip_index(str(tmp_path), embeddings)
    reversed_chunk_ids = list(reversed(list(embeddings.keys())))

    cached_index = cache._load_cached_faiss_index(str(tmp_path), "faiss_flatip", reversed_chunk_ids)

    assert cached_index.chunk_ids == list(embeddings.keys())


def test_load_faiss_index_with_matching_dataset_size(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: an index saved from chunks of 2 documents, loaded with dataset_size 2.
    Output: the FaissIndex.
    What it does: checks load-only returns the saved index.
    """
    chunks, embeddings = make_chunks_and_embeddings(4, doc_count=2)
    save_flatip_index(str(tmp_path), embeddings)

    faiss_index = cache.load_faiss_index(str(tmp_path), make_index_args(), 2)

    assert faiss_index.faiss_index.ntotal == 4
    assert faiss_index.faiss_index_type == "faiss_flatip"


def test_load_faiss_index_with_other_dataset_size_raises(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: an index saved from chunks of 2 documents, loaded with dataset_size 3.
    Output: ValueError mentioning dataset_size.
    What it does: checks an index built with another dataset size is not used.
    """
    chunks, embeddings = make_chunks_and_embeddings(4, doc_count=2)
    save_flatip_index(str(tmp_path), embeddings)

    with pytest.raises(ValueError, match="dataset_size"):
        cache.load_faiss_index(str(tmp_path), make_index_args(), 3)


def test_load_faiss_index_missing_raises(tmp_path, make_index_args):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks load-only never builds a missing index.
    """
    with pytest.raises(FileNotFoundError):
        cache.load_faiss_index(str(tmp_path), make_index_args(), 2)


# ---------- Built-in BM25 ----------


def test_load_bm25_index_returns_the_saved_model(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: a BM25 model built from 6 chunks.
    Output: the Bm25Index loaded back.
    What it does: checks the chunk ids and model scores survive saving.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    built_index = bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)

    loaded_index = cache.load_bm25_index(str(tmp_path))

    assert loaded_index.chunk_ids == built_index.chunk_ids
    assert list(loaded_index.bm25_model.get_scores(["word2"])) == list(built_index.bm25_model.get_scores(["word2"]))


def test_load_bm25_index_missing_raises(tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks load-only never builds a missing index.
    """
    with pytest.raises(FileNotFoundError):
        cache.load_bm25_index(str(tmp_path))


def test_load_cached_bm25_index_returns_none_for_other_chunks(tmp_path, make_index_args, make_chunks_and_embeddings):
    """
    Input: a BM25 model saved from 6 chunks, checked against other chunk ids.
    Output: None.
    What it does: checks a model holding other chunks is not reused.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    bm25_indexer.get_bm25_index(str(tmp_path), make_index_args(index_type="bm25"), chunks)

    assert cache._load_cached_bm25_index(str(tmp_path), ["doc0_0"]) is None


# ---------- Built-in Qdrant ----------


def test_load_qdrant_index_missing_collection_raises(memory_qdrant_client):
    """
    Input: a collection name that does not exist.
    Output: FileNotFoundError.
    What it does: checks load-only never creates a collection.
    """
    with pytest.raises(FileNotFoundError):
        cache.load_qdrant_index("missing_collection")


def test_load_qdrant_index_without_text_raises(memory_qdrant_client):
    """
    Input: a collection whose points have only a chunk_id in their payload.
    Output: FileNotFoundError mentioning the chunk text.
    What it does: checks a collection from before the text was stored is rebuilt.
    """
    memory_qdrant_client.create_collection(
        collection_name="old_collection",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    memory_qdrant_client.upsert(
        collection_name="old_collection",
        points=[PointStruct(id=0, vector=[1.0, 0.0, 0.0, 0.0], payload={"chunk_id": "doc0_0"})]
    )

    with pytest.raises(FileNotFoundError, match="no chunk text"):
        cache.load_qdrant_index("old_collection")


def test_load_cached_qdrant_index_returns_none_without_text(memory_qdrant_client):
    """
    Input: a collection with the right chunk id but no text in its payload.
    Output: None.
    What it does: checks such a collection is not reused while building.
    """
    memory_qdrant_client.create_collection(
        collection_name="old_collection",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    memory_qdrant_client.upsert(
        collection_name="old_collection",
        points=[PointStruct(id=0, vector=[1.0, 0.0, 0.0, 0.0], payload={"chunk_id": "doc0_0"})]
    )

    assert cache._load_cached_qdrant_index(memory_qdrant_client, "old_collection", ["doc0_0"]) is None


def test_connect_qdrant_client_raises_when_server_is_down(monkeypatch):
    """
    Input: QDRANT_URL pointing at a port nothing listens on.
    Output: ConnectionError mentioning Qdrant.
    What it does: checks a missing server gives a clear error.
    """
    monkeypatch.setattr(cache, "QDRANT_URL", "http://localhost:1")

    with pytest.raises(ConnectionError, match="Qdrant"):
        cache._connect_qdrant_client()


# ---------- load_index ----------


@pytest.mark.parametrize("index_type, index_class", [
    ("faiss_flatip", FaissIndex),
    ("faiss_ivf", FaissIndex),
    ("faiss_hnsw", FaissIndex),
    ("bm25", Bm25Index),
    ("qdrant", QdrantIndex),
])
def test_load_index_loads_each_built_in_index(tmp_path, memory_qdrant_client, make_index_args, make_chunking_args, make_chunks_and_embeddings, index_type, index_class):
    """
    Input: each built-in index type, built first with Indexing.
    Output: the index load_index returns.
    What it does: checks load_index finds the index where Indexing saved it.
    """
    chunks, embeddings = make_chunks_and_embeddings(10, doc_count=2)
    output_path = str(tmp_path / "myset")
    index_args = make_index_args(index_type=index_type)
    chunking_args = make_chunking_args()
    Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)

    loaded_index = cache.load_index(output_path, chunking_args, index_args, 2)

    assert isinstance(loaded_index, index_class)


@pytest.mark.parametrize("index_type, index_class", [
    ("faiss_flatip", LangchainFaissIndex),
    ("bm25", LangchainBm25Index),
    ("qdrant", LangchainQdrantIndex),
])
def test_load_index_loads_each_langchain_index(tmp_path, memory_qdrant_client, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, index_type, index_class):
    """
    Input: framework "langchain" and each index kind, built first with Indexing.
    Output: the index load_index returns.
    What it does: checks load_index returns the LangChain index for framework "langchain".
    """
    chunks, embeddings = make_chunks_and_embeddings(10, doc_count=2)
    output_path = str(tmp_path / "myset")
    index_args = make_index_args(index_type=index_type)
    chunking_args = make_chunking_args(framework="langchain")
    Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)

    loaded_index = cache.load_index(output_path, chunking_args, index_args, 2)

    assert isinstance(loaded_index, index_class)


def test_load_index_with_invalid_args_raises(tmp_path, make_index_args, make_chunking_args):
    """
    Input: index_type "annoy".
    Output: ValueError mentioning index_type.
    What it does: checks load_index validates its args first.
    """
    with pytest.raises(ValueError, match="index_type"):
        cache.load_index(str(tmp_path), make_chunking_args(), make_index_args(index_type="annoy"), 2)
