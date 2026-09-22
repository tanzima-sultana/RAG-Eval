"""
test_embedding.py

Tests for l3_embedding.embedding.Embedding: the orchestrator that
loads a dataset's cached chunks, embeds every chunk that has no
embedding cached yet, and saves/caches the resulting vectors.

Every test here uses isolated_data_dir (so writes to a dataset's
data/processed/, chunks/, and embeddings/ land in a temporary
directory, never this repo's real files) and
fake_sentence_transformer_class (so constructing an Embedding never
downloads or runs a real embedding model). FakeSentenceTransformer
also records every encode() call it receives, so the compute_embeddings
tests below can check chunks are batched into as few encode() calls as
possible instead of one call per chunk.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from common import paths
from l2_chunking.schema import Chunk
from l3_embedding.embedding import Embedding
from l3_embedding.schema import Embedded


def _make_embedding(
    dataset_size=1, min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.0
):
    """
    dataset_size, min_chunk_size, max_chunk_size, fixed_chunk_overlap,
        semantic_threshold: overridable Embedding constructor
        arguments; each defaults to a value that keeps chunk/embedding
        paths short and predictable in tests that do not care about
        it specifically.

    Returns an Embedding built for dataset_name "myset" and
    model_name "fake" (fake_sentence_transformer_class must already be
    active in the calling test, so this never loads a real model).
    """
    return Embedding(
        dataset_name="myset",
        dataset_size=dataset_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        fixed_chunk_overlap=fixed_chunk_overlap,
        semantic_threshold=semantic_threshold,
        model_name="fake",
    )


def _touch_processed_document(data_path, doc_id):
    """
    data_path: directory to create the placeholder processed-document
        file in (created if missing).
    doc_id: identifies the file to create.

    Creates an empty {doc_id}.json file under data_path.
    Embedding.load_document_ids never reads a processed document's
    contents, only its file name, so an empty placeholder is enough to
    stand in for a real l1_doc_processing output file.

    Returns nothing.
    """
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, f"{doc_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("{}")


def _write_chunk_file(chunk_path, chunk):
    """
    chunk_path: directory to write the chunk JSON file into (created
        if missing).
    chunk: the Chunk to persist, written as {chunk.id}.json in the
        same shape l2_chunking.chunking.Chunking._save_chunks writes,
        so Embedding._load_cached_chunks can read it back.

    Returns nothing.
    """
    os.makedirs(chunk_path, exist_ok=True)
    file_path = os.path.join(chunk_path, f"{chunk.id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(chunk), f)


# ---------- __init__ ----------


def test_init_stores_every_constructor_argument(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: an Embedding constructed with the SentenceTransformer class
    faked out.
    Output: every attribute __init__ is expected to set.

    Checks that __init__ stores every constructor argument as-is, and
    loads a model via SentenceTransformer(model_name, ...).
    """
    embedding = Embedding(
        dataset_name="myset",
        dataset_size=5,
        min_chunk_size=1,
        max_chunk_size=10,
        fixed_chunk_overlap=2,
        semantic_threshold=0.3,
        model_name="fake-model",
    )

    assert embedding.dataset_name == "myset"
    assert embedding.dataset_size == 5
    assert embedding.min_chunk_size == 1
    assert embedding.max_chunk_size == 10
    assert embedding.fixed_chunk_overlap == 2
    assert embedding.semantic_threshold == 0.3
    assert embedding.model_name == "fake-model"
    assert embedding.model.model_name == "fake-model"


# ---------- load_document_ids ----------


def test_load_document_ids_raises_file_not_found_when_data_path_does_not_exist(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a data_path that has never been created.
    Output: none; load_document_ids is expected to raise
    FileNotFoundError.
    """
    embedding = _make_embedding()
    missing_path = str(isolated_data_dir / "never_created")

    with pytest.raises(FileNotFoundError):
        embedding.load_document_ids(missing_path, 1)


def test_load_document_ids_raises_value_error_when_data_path_is_empty(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a data_path that exists but holds no files.
    Output: none; load_document_ids is expected to raise ValueError.
    """
    embedding = _make_embedding()
    empty_path = str(isolated_data_dir / "empty")
    os.makedirs(empty_path)

    with pytest.raises(ValueError):
        embedding.load_document_ids(empty_path, 1)


def test_load_document_ids_raises_value_error_when_dataset_size_exceeds_available_files(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a data_path holding one processed document, and a
    dataset_size of 2.
    Output: none; load_document_ids is expected to raise ValueError.
    """
    embedding = _make_embedding()
    data_path = str(isolated_data_dir / "docs")
    _touch_processed_document(data_path, "doc1")

    with pytest.raises(ValueError):
        embedding.load_document_ids(data_path, 2)


def test_load_document_ids_returns_first_dataset_size_ids_in_sorted_order(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: three processed-document files written out of order, and a
    dataset_size of 2.
    Output: load_document_ids's return value.

    Checks that ids come back sorted by file name and truncated to
    dataset_size, regardless of the order the files were created in.
    """
    embedding = _make_embedding()
    data_path = str(isolated_data_dir / "docs")
    _touch_processed_document(data_path, "doc2")
    _touch_processed_document(data_path, "doc1")
    _touch_processed_document(data_path, "doc3")

    doc_ids = embedding.load_document_ids(data_path, 2)

    assert doc_ids == ["doc1", "doc2"]


# ---------- json_dict_to_chunk ----------


def test_json_dict_to_chunk_reconstructs_a_chunk(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: a dict shaped like dataclasses.asdict(chunk).
    Output: embedding.json_dict_to_chunk's return value.
    """
    embedding = _make_embedding()
    data = {"id": "doc1_0", "text": "hello world", "size": 2, "type": "fixed"}

    chunk = embedding.json_dict_to_chunk(data)

    assert chunk == Chunk(id="doc1_0", text="hello world", size=2, type="fixed")


# ---------- _load_cached_chunks ----------


def test_load_cached_chunks_returns_none_when_chunk_path_does_not_exist(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a chunk_path that has never been created.
    Output: embedding._load_cached_chunks's return value.
    """
    embedding = _make_embedding()
    chunk_path = str(isolated_data_dir / "never_created")

    assert embedding._load_cached_chunks(chunk_path, "doc1") is None


def test_load_cached_chunks_returns_none_when_doc_id_has_no_cached_files(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: chunk_path exists and holds another doc_id's chunk, but
    none for the doc_id being looked up.
    Output: embedding._load_cached_chunks's return value.
    """
    embedding = _make_embedding()
    chunk_path = str(isolated_data_dir / "chunks_out")
    _write_chunk_file(chunk_path, Chunk(id="doc2_0", text="x", size=1, type="fixed"))

    assert embedding._load_cached_chunks(chunk_path, "doc1") is None


def test_load_cached_chunks_reads_every_matching_chunk_file(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: chunk_path holding two chunks for doc1 and one for doc2.
    Output: embedding._load_cached_chunks's return value for doc1.

    Checks that only doc1's chunks are returned, keyed by chunk_id.
    """
    embedding = _make_embedding()
    chunk_path = str(isolated_data_dir / "chunks_out")
    _write_chunk_file(chunk_path, Chunk(id="doc1_0", text="a", size=1, type="fixed"))
    _write_chunk_file(chunk_path, Chunk(id="doc1_1", text="b", size=1, type="fixed"))
    _write_chunk_file(chunk_path, Chunk(id="doc2_0", text="c", size=1, type="fixed"))

    chunks = embedding._load_cached_chunks(chunk_path, "doc1")

    assert chunks == {
        "doc1_0": Chunk(id="doc1_0", text="a", size=1, type="fixed"),
        "doc1_1": Chunk(id="doc1_1", text="b", size=1, type="fixed"),
    }


# ---------- embedding_to_json_dict / json_dict_to_embedding ----------


def test_embedding_to_json_dict_and_back_round_trips(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: a fully-populated Embedded.
    Output: embedding.embedding_to_json_dict then
    embedding.json_dict_to_embedding, applied in sequence.

    Checks embeddings is stored as a JSON-serializable list in
    between, and comes back as an equivalent ndarray.
    """
    embedding = _make_embedding()
    embedded = Embedded(chunk_id="doc1_0", embeddings=np.array([1.0, 2.0, 3.0]), normalized=True)

    data = embedding.embedding_to_json_dict(embedded)
    restored = embedding.json_dict_to_embedding(data)

    assert data["embeddings"] == [1.0, 2.0, 3.0]
    assert restored.chunk_id == "doc1_0"
    assert np.array_equal(restored.embeddings, embedded.embeddings)
    assert restored.normalized is True


# ---------- _save_embeddings / _load_cached_embeddings ----------


def test_load_cached_embeddings_returns_none_when_file_does_not_exist(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: an embedding_path that has never been written to.
    Output: embedding._load_cached_embeddings's return value.
    """
    embedding = _make_embedding()
    embedding_path = str(isolated_data_dir / "never_created")

    assert embedding._load_cached_embeddings(embedding_path, "doc1_0") is None


def test_save_and_load_embeddings_round_trips(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: an Embedded saved via embedding._save_embeddings.
    Output: embedding._load_cached_embeddings's return value for the
    same chunk_id.
    """
    embedding = _make_embedding()
    embedding_path = str(isolated_data_dir / "embeddings_out")
    embedded = Embedded(chunk_id="doc1_0", embeddings=np.array([1.0, 2.0]), normalized=True)

    embedding._save_embeddings(embedding_path, "doc1_0", embedded)
    loaded = embedding._load_cached_embeddings(embedding_path, "doc1_0")

    assert loaded.chunk_id == "doc1_0"
    assert np.array_equal(loaded.embeddings, embedded.embeddings)
    assert loaded.normalized is True


# ---------- compute_embeddings ----------


def test_compute_embeddings_raises_value_error_for_unknown_chunk_type_before_loading_anything(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: an unknown chunk_type, and no processed-documents
    directory created at all.
    Output: none; compute_embeddings is expected to raise ValueError.

    Checks that chunk_type is validated against the l2_chunking
    strategy registry before load_document_ids ever runs, so a
    missing processed-documents directory never gets a chance to
    raise its own FileNotFoundError first.
    """
    embedding = _make_embedding(dataset_size=1)

    with pytest.raises(ValueError):
        embedding.compute_embeddings("totally-bogus-chunk-type")


def test_compute_embeddings_end_to_end_embeds_and_persists_new_chunks(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document with two cached chunks, and no
    embeddings cached yet.
    Output: embedding.compute_embeddings("fixed")'s return value, and
    the files written under paths.get_embeddings_path for this test's
    params.

    Checks the whole pipeline end to end: chunks are loaded, embedded,
    and both returned and persisted to disk.
    """
    data_path = paths.get_processed_dataset_path("myset")
    _touch_processed_document(data_path, "doc1")
    chunk_path = paths.get_chunk_path("myset", "fixed", 1, 10, 0, 0.0)
    _write_chunk_file(chunk_path, Chunk(id="doc1_0", text="hello world", size=2, type="fixed"))
    _write_chunk_file(chunk_path, Chunk(id="doc1_1", text="goodbye", size=1, type="fixed"))
    embedding = _make_embedding(dataset_size=1)

    embeddings = embedding.compute_embeddings("fixed")

    assert set(embeddings.keys()) == {"doc1_0", "doc1_1"}
    assert embeddings["doc1_0"].chunk_id == "doc1_0"
    assert np.array_equal(embeddings["doc1_0"].embeddings, np.array([float(len("hello world")), 1.0]))
    assert np.array_equal(embeddings["doc1_1"].embeddings, np.array([float(len("goodbye")), 1.0]))
    assert embeddings["doc1_1"].normalized is True
    output_path = paths.get_embeddings_path("myset", "fixed", 1, 10, 0, 0.0)
    assert sorted(os.listdir(output_path)) == ["doc1_0.json", "doc1_1.json"]


def test_compute_embeddings_batches_all_pending_chunks_into_one_encode_call(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: two processed documents, each with one cached chunk, none
    of which have embeddings cached yet.
    Output: embedding.model.encode_calls after compute_embeddings
    runs.

    Checks that every not-yet-cached chunk across every document is
    embedded in a single model.encode call, in doc_id order, instead
    of one call per chunk.
    """
    data_path = paths.get_processed_dataset_path("myset")
    _touch_processed_document(data_path, "doc1")
    _touch_processed_document(data_path, "doc2")
    chunk_path = paths.get_chunk_path("myset", "fixed", 1, 10, 0, 0.0)
    _write_chunk_file(chunk_path, Chunk(id="doc1_0", text="a", size=1, type="fixed"))
    _write_chunk_file(chunk_path, Chunk(id="doc2_0", text="bb", size=1, type="fixed"))
    embedding = _make_embedding(dataset_size=2)

    embedding.compute_embeddings("fixed")

    assert embedding.model.encode_calls == [["a", "bb"]]


def test_compute_embeddings_skips_encode_call_when_every_chunk_already_cached(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document with one cached chunk, whose
    embedding is already cached on disk.
    Output: embedding.model.encode_calls, and
    compute_embeddings("fixed")'s return value.

    Checks the per-chunk cache-hit branch: once a chunk's embedding is
    already on disk, compute_embeddings must reuse it straight from
    disk instead of calling model.encode at all.
    """
    data_path = paths.get_processed_dataset_path("myset")
    _touch_processed_document(data_path, "doc1")
    chunk_path = paths.get_chunk_path("myset", "fixed", 1, 10, 0, 0.0)
    _write_chunk_file(chunk_path, Chunk(id="doc1_0", text="hello", size=1, type="fixed"))
    output_path = paths.get_embeddings_path("myset", "fixed", 1, 10, 0, 0.0)
    embedding = _make_embedding(dataset_size=1)
    cached_embedded = Embedded(chunk_id="doc1_0", embeddings=np.array([9.0, 9.0]), normalized=True)
    embedding._save_embeddings(output_path, "doc1_0", cached_embedded)

    embeddings = embedding.compute_embeddings("fixed")

    assert embedding.model.encode_calls == []
    assert np.array_equal(embeddings["doc1_0"].embeddings, np.array([9.0, 9.0]))


def test_compute_embeddings_only_embeds_the_not_yet_cached_chunks(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document with two cached chunks, one of
    which already has a cached embedding and one of which does not.
    Output: embedding.model.encode_calls, and
    compute_embeddings("fixed")'s return value.

    Checks that the already-cached chunk's text is never sent to
    model.encode, while the not-yet-cached chunk still gets a freshly
    computed embedding, in the same run.
    """
    data_path = paths.get_processed_dataset_path("myset")
    _touch_processed_document(data_path, "doc1")
    chunk_path = paths.get_chunk_path("myset", "fixed", 1, 10, 0, 0.0)
    _write_chunk_file(chunk_path, Chunk(id="doc1_0", text="cached chunk", size=2, type="fixed"))
    _write_chunk_file(chunk_path, Chunk(id="doc1_1", text="new", size=1, type="fixed"))
    output_path = paths.get_embeddings_path("myset", "fixed", 1, 10, 0, 0.0)
    embedding = _make_embedding(dataset_size=1)
    cached_embedded = Embedded(chunk_id="doc1_0", embeddings=np.array([9.0, 9.0]), normalized=True)
    embedding._save_embeddings(output_path, "doc1_0", cached_embedded)

    embeddings = embedding.compute_embeddings("fixed")

    assert embedding.model.encode_calls == [["new"]]
    assert np.array_equal(embeddings["doc1_0"].embeddings, np.array([9.0, 9.0]))
    assert np.array_equal(embeddings["doc1_1"].embeddings, np.array([float(len("new")), 1.0]))


def test_compute_embeddings_skips_document_ids_with_no_cached_chunks(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: two processed documents, only one of which has any cached
    chunks.
    Output: compute_embeddings("fixed")'s return value.

    Checks that a document with no chunks cached yet under chunk_path
    is silently skipped, and does not stop the other document's
    chunks from being embedded.
    """
    data_path = paths.get_processed_dataset_path("myset")
    _touch_processed_document(data_path, "doc1")
    _touch_processed_document(data_path, "doc2")
    chunk_path = paths.get_chunk_path("myset", "fixed", 1, 10, 0, 0.0)
    _write_chunk_file(chunk_path, Chunk(id="doc2_0", text="only doc2 has chunks", size=4, type="fixed"))
    embedding = _make_embedding(dataset_size=2)

    embeddings = embedding.compute_embeddings("fixed")

    assert set(embeddings.keys()) == {"doc2_0"}
