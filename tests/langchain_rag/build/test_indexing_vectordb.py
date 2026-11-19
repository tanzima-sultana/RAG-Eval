"""
test_indexing_vectordb.py

Tests for langchain_rag.build.indexing_vectordb. Qdrant runs in memory and HuggingFaceEmbeddings is replaced with a fake,
so no server is needed and no model is downloaded.
"""

import pytest
from qdrant_client import QdrantClient

from langchain_rag.build import indexing_vectordb as indexing_vectordb_module
from langchain_rag.build.indexing_vectordb import IndexingVectorDb
from langchain_rag.build.indexing_vectordb import qdrant_collection_holds_chunks

OUTPUT_PATH = "out/langchain/dataset"


@pytest.fixture
def patched_embeddings(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes IndexingVectorDb load the fake instead of the real model.
    """
    monkeypatch.setattr(indexing_vectordb_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return fake_huggingface_embeddings


@pytest.fixture
def memory_qdrant_client(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the one in-memory QdrantClient every connection in the test gets.
    What it does: makes QdrantClient(url=...) return an in-memory client, so no server is needed.
    """
    qdrant_client = QdrantClient(":memory:")

    def connect_memory_qdrant_client(url=None):
        """
        Input: url (ignored).
        Output: the in-memory QdrantClient.
        What it does: stands in for QdrantClient(url=...).
        """
        return qdrant_client

    monkeypatch.setattr(indexing_vectordb_module, "QdrantClient", connect_memory_qdrant_client)
    return qdrant_client


def count_upserted_points(qdrant_client, monkeypatch):
    """
    Input: qdrant_client; pytest's monkeypatch fixture.
    Output: a list that gets the number of points of every upsert call.
    What it does: wraps qdrant_client.upsert so each call's point count is recorded.
    """
    upserted_point_counts = []
    real_upsert = qdrant_client.upsert

    def recording_upsert(collection_name, points, wait=True):
        """
        Input: the arguments of QdrantClient.upsert.
        Output: what the real upsert returns.
        What it does: records len(points), then calls the real upsert.
        """
        upserted_point_counts.append(len(points))
        return real_upsert(collection_name=collection_name, points=points, wait=wait)

    monkeypatch.setattr(qdrant_client, "upsert", recording_upsert)
    return upserted_point_counts


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"vectordb_type": "chroma"}, "vectordb_type"),
    ({"qdrant_url": ""}, "qdrant_url"),
    ({"qdrant_batch_size": 0}, "qdrant_batch_size"),
])
def test_init_raises_for_invalid_setting(patched_embeddings, make_vectordb_args, overrides, message):
    """
    Input: VectorDbArgs with one invalid setting.
    Output: ValueError.
    What it does: checks IndexingVectorDb validates the settings when it is created, and names the invalid one.
    """
    with pytest.raises(ValueError, match=message):
        IndexingVectorDb(make_vectordb_args(**overrides))


# ---------- qdrant_indexing ----------


def test_qdrant_builds_collection_with_langchain_payload(patched_embeddings, memory_qdrant_client, make_vectordb_args, make_chunks_and_embeddings):
    """
    Input: 10 chunks, with no collection on the server.
    Output: the VectorDb and the collection.
    What it does: checks every chunk is uploaded with its text and chunk_id, under the expected collection name.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    vectordb = IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)

    assert vectordb.vectordb_type == "qdrant"
    assert vectordb.collection_name == "langchain_dataset_fixed_fake-model_2_6_2_0.5"
    assert vectordb.dimension == 4
    assert memory_qdrant_client.count(vectordb.collection_name).count == 10
    assert qdrant_collection_holds_chunks(memory_qdrant_client, vectordb.collection_name, list(chunks.keys()))

    points, next_page_offset = memory_qdrant_client.scroll(vectordb.collection_name, limit=10, with_payload=True)
    payloads_by_chunk_id = {}
    for point in points:
        payloads_by_chunk_id[point.payload["metadata"]["chunk_id"]] = point.payload
    assert payloads_by_chunk_id["doc0_2"]["page_content"] == chunks["doc0_2"].text


def test_qdrant_uploads_in_batches_of_batch_size(patched_embeddings, memory_qdrant_client, make_vectordb_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: 10 chunks with qdrant_batch_size 3.
    Output: the number of points of each upsert call.
    What it does: checks the points go up in batches of 3, with the rest in the last batch.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    upserted_point_counts = count_upserted_points(memory_qdrant_client, monkeypatch)

    IndexingVectorDb(make_vectordb_args(qdrant_batch_size=3)).compute_vectordb(chunks, embeddings, OUTPUT_PATH)

    assert upserted_point_counts == [3, 3, 3, 1]


def test_qdrant_search_with_a_chunk_vector_finds_that_chunk(patched_embeddings, memory_qdrant_client, make_vectordb_args, make_chunks_and_embeddings):
    """
    Input: 10 chunks, searched with chunk "doc1_3"'s vector.
    Output: the top hit and its score.
    What it does: checks the existing vectors are uploaded for the right chunks and the score is cosine similarity.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    vectordb = IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)

    documents_and_scores = vectordb.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc1_3"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert document.metadata["chunk_id"] == "doc1_3"
    assert document.page_content == chunks["doc1_3"].text
    assert score == pytest.approx(1.0, abs=1e-5)


def test_qdrant_reuses_collection_with_same_chunks(patched_embeddings, memory_qdrant_client, make_vectordb_args, make_chunks_and_embeddings, monkeypatch, capsys):
    """
    Input: the same chunks indexed twice.
    Output: the second run's upsert calls and what it printed.
    What it does: checks the second run reuses the collection and uploads nothing.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)
    upserted_point_counts = count_upserted_points(memory_qdrant_client, monkeypatch)
    capsys.readouterr()

    IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)

    assert upserted_point_counts == []
    assert "Qdrant collection loaded from cache : 10 points" in capsys.readouterr().out


def test_qdrant_rebuilds_collection_when_chunks_change(patched_embeddings, memory_qdrant_client, make_vectordb_args, make_chunks_and_embeddings, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings.
    Output: what the second run printed and the collection.
    What it does: checks a collection with different chunk ids is replaced and holds only the new chunks.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    vectordb = IndexingVectorDb(make_vectordb_args()).compute_vectordb(more_chunks, more_embeddings, OUTPUT_PATH)

    assert "Qdrant collection built : 12 points" in capsys.readouterr().out
    assert memory_qdrant_client.count(vectordb.collection_name).count == 12
    assert qdrant_collection_holds_chunks(memory_qdrant_client, vectordb.collection_name, list(more_chunks.keys()))


def test_qdrant_collection_holds_chunks_is_false_for_missing_or_different_collection(memory_qdrant_client, patched_embeddings, make_vectordb_args, make_chunks_and_embeddings):
    """
    Input: a collection of 10 chunks, checked against other chunk ids, and a missing collection.
    Output: the check results.
    What it does: checks only a collection with exactly the same chunk ids counts as cached.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    vectordb = IndexingVectorDb(make_vectordb_args()).compute_vectordb(chunks, embeddings, OUTPUT_PATH)

    other_chunk_ids = list(chunks.keys())[:9]
    assert qdrant_collection_holds_chunks(memory_qdrant_client, vectordb.collection_name, other_chunk_ids) is False
    assert qdrant_collection_holds_chunks(memory_qdrant_client, "missing", list(chunks.keys())) is False


def test_qdrant_raises_connection_error_when_server_is_down(patched_embeddings, make_vectordb_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: a Qdrant client whose first request fails.
    Output: ConnectionError.
    What it does: checks a down server gives a clear error that names the URL.
    """
    class UnreachableQdrantClient:
        """
        Stand-in for a QdrantClient whose server is down.
        """

        def __init__(self, url=None):
            """
            Input: url.
            Output: nothing.
            What it does: stores url.
            """
            self.url = url

        def get_collections(self):
            """
            Input: none.
            Output: nothing.
            What it does: raises, as a client does when the server cannot be reached.
            """
            raise RuntimeError("connection refused")

    monkeypatch.setattr(indexing_vectordb_module, "QdrantClient", UnreachableQdrantClient)
    chunks, embeddings = make_chunks_and_embeddings(10)
    indexing_vectordb = IndexingVectorDb(make_vectordb_args(qdrant_url="http://localhost:9999"))

    with pytest.raises(ConnectionError, match="http://localhost:9999"):
        indexing_vectordb.compute_vectordb(chunks, embeddings, OUTPUT_PATH)


def test_qdrant_raises_for_no_embeddings(patched_embeddings, memory_qdrant_client, make_vectordb_args):
    """
    Input: no chunks and no embeddings.
    Output: ValueError.
    What it does: checks an empty collection is never built.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        IndexingVectorDb(make_vectordb_args()).compute_vectordb({}, {}, OUTPUT_PATH)
