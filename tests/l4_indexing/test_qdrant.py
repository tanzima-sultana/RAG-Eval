"""
test_qdrant.py

Tests for l4_indexing.indexers.qdrant.get_qdrant_index: creating the
collection, uploading in batches, and reusing an existing collection.
Qdrant runs in memory (see memory_qdrant_client in conftest.py).
"""

import pytest

from l4_indexing.indexers import qdrant as qdrant_indexer


def read_payloads(qdrant_client, collection_name):
    """
    Input: qdrant_client and collection_name.
    Output: a dict of chunk_id to payload for every point.
    What it does: reads every point's payload.
    """
    points, next_page_offset = qdrant_client.scroll(collection_name=collection_name, limit=1000, with_payload=True)
    payloads = {}
    for point in points:
        payloads[point.payload["chunk_id"]] = point.payload
    return payloads


def test_get_qdrant_index_uploads_every_chunk_with_text(memory_qdrant_client, make_index_args, make_chunks_and_embeddings):
    """
    Input: 5 chunks and embeddings, with no collection.
    Output: the QdrantIndex and the points in the collection.
    What it does: checks every chunk is uploaded with its chunk_id and text.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)

    qdrant_index = qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), chunks, embeddings)
    payloads = read_payloads(memory_qdrant_client, "test_collection")

    assert qdrant_index.collection_name == "test_collection"
    assert qdrant_index.dimension == 4
    assert len(payloads) == 5
    for chunk_id in chunks:
        assert payloads[chunk_id]["text"] == chunks[chunk_id].text


def test_get_qdrant_index_uploads_in_batches(memory_qdrant_client, make_index_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: 5 chunks and qdrant_batch_size 2.
    Output: the number of points in each upsert call.
    What it does: checks the points are sent as batches of 2, 2 and 1.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    batch_sizes = []
    real_upsert = memory_qdrant_client.upsert

    def recording_upsert(collection_name, points, wait=True):
        """
        Input: the upsert arguments.
        Output: what the real upsert returns.
        What it does: records the batch size, then uploads.
        """
        batch_sizes.append(len(points))
        return real_upsert(collection_name=collection_name, points=points, wait=wait)

    monkeypatch.setattr(memory_qdrant_client, "upsert", recording_upsert)
    qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant", qdrant_batch_size=2), chunks, embeddings)

    assert batch_sizes == [2, 2, 1]


def test_search_finds_the_chunk_of_the_query_vector(memory_qdrant_client, make_index_args, make_chunks_and_embeddings):
    """
    Input: 5 chunks, searched with chunk "doc0_2"'s vector.
    Output: the top hit.
    What it does: checks the vectors are stored with the right chunk_id.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), chunks, embeddings)

    response = memory_qdrant_client.query_points(
        collection_name="test_collection",
        query=embeddings["doc0_2"].embeddings.tolist(),
        limit=1,
        with_payload=True
    )

    assert response.points[0].payload["chunk_id"] == "doc0_2"


def test_get_qdrant_index_reuses_collection_with_same_chunks(memory_qdrant_client, make_index_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same chunks indexed twice.
    Output: the second QdrantIndex.
    What it does: checks the second call uses the existing collection and uploads nothing.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), chunks, embeddings)

    def fail_upsert(*args, **kwargs):
        """
        Input: the upsert arguments.
        Output: never returns.
        What it does: fails the test if anything is uploaded.
        """
        raise AssertionError("the existing collection should have been reused")

    monkeypatch.setattr(memory_qdrant_client, "upsert", fail_upsert)
    reused_index = qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), chunks, embeddings)

    assert reused_index.collection_name == "test_collection"


def test_get_qdrant_index_rebuilds_when_chunks_change(memory_qdrant_client, make_index_args, make_chunks_and_embeddings):
    """
    Input: 5 chunks indexed, then 3 chunks indexed in the same collection.
    Output: the points in the collection.
    What it does: checks the old collection is deleted and only the 3 new chunks remain.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), chunks, embeddings)

    fewer_chunks, fewer_embeddings = make_chunks_and_embeddings(3)
    qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), fewer_chunks, fewer_embeddings)
    payloads = read_payloads(memory_qdrant_client, "test_collection")

    assert sorted(payloads.keys()) == sorted(fewer_chunks.keys())


def test_get_qdrant_index_with_no_embeddings_raises(memory_qdrant_client, make_index_args):
    """
    Input: an empty embeddings dict.
    Output: ValueError mentioning embeddings.
    What it does: checks there must be something to index.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        qdrant_indexer.get_qdrant_index("test_collection", make_index_args(index_type="qdrant"), {}, {})
