"""
test_chunking.py

Tests for l2_chunking.chunking.Chunking: the orchestrator that loads
processed documents, dispatches to a registered chunking strategy,
and saves/caches the resulting chunks.

Every test here uses isolated_data_dir (so writes to a dataset's
data/processed/ and chunks/ land in a temporary directory, never this
repo's real files) and fake_sentence_transformer_class (so
constructing a Chunking never downloads or runs a real embedding
model).

Chunking takes dataset_name and dataset_size directly as constructor
arguments (the way a real run's l2_chunking.main wires them in from
its own --dataset_name/--dataset_size CLI args). Tests that call
compute_chunks construct Chunking with the same dataset_name they
write processed documents under.
"""

import dataclasses
import datetime
import json
import os

import numpy as np
import pytest

from common import paths
from l1_doc_processing.schema import Document
from l2_chunking.chunking import Chunking
from l2_chunking.schema import Chunk


def _write_processed_document(docs_dir, doc_id, document):
    """
    docs_dir: directory to write the processed document JSON file
        into (created if missing).
    doc_id: identifies the file to write.
    document: the Document to persist.

    Writes document as {doc_id}.json in the same shape
    l1_doc_processing.doc_processing.DocProcessing.document_to_json_dict
    produces, so Chunking.load_documents can read it back.

    Returns nothing.
    """
    os.makedirs(docs_dir, exist_ok=True)
    data = dataclasses.asdict(document)
    data["timestamp"] = document.timestamp.isoformat()
    with open(os.path.join(docs_dir, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_document(doc_id, text):
    """
    doc_id: id to give the Document.
    text: the Document's text content.

    Returns a minimally-populated Document, matching
    conftest.document_factory but usable outside a fixture (some
    tests here build several documents at once, keyed by doc_id).
    """
    return Document(
        id=doc_id,
        text=text,
        title=None,
        path="irrelevant",
        format="fake",
        metadata={},
        content_hash="irrelevant",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


# ---------- __init__ ----------


def test_init_sets_tokenizer_from_the_loaded_model(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a Chunking constructed with the SentenceTransformer class
    faked out.
    Output: chunking.tokenizer.

    Checks that __init__ wires tokenizer up from the loaded model.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )

    assert chunking.tokenizer is chunking.model.tokenizer


def test_init_leaves_embeddings_state_unset_until_compute_chunks_runs(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a freshly constructed Chunking.
    Output: chunking.all_embeddings and chunking.embeddings_offset.

    Checks that embeddings-related state starts out unset, since it
    is only ever needed (and only ever computed) for semantic
    chunking, inside compute_chunks.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )

    assert chunking.all_embeddings is None
    assert chunking.embeddings_offset is None


# ---------- validate ----------


def test_validate_raises_when_overlap_is_not_smaller_than_max_chunk_size(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a Chunking with fixed_chunk_overlap equal to max_chunk_size.
    Output: none; validate() is expected to raise.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=10, semantic_threshold=0.5, model_name="fake"
    )

    with pytest.raises(ValueError):
        chunking.validate()


def test_validate_passes_when_overlap_is_smaller_than_max_chunk_size(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a Chunking with fixed_chunk_overlap smaller than
    max_chunk_size.
    Output: none; validate() is expected not to raise.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=2, semantic_threshold=0.5, model_name="fake"
    )

    chunking.validate()


# ---------- chunk_to_json_dict / json_dict_to_chunk ----------


def test_chunk_to_json_dict_and_back_round_trips(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: a fully-populated Chunk.
    Output: chunking.chunk_to_json_dict then chunking.json_dict_to_chunk,
    applied in sequence.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )
    chunk = Chunk(id="doc1_0", text="hello world", size=2, type="fixed")

    data = chunking.chunk_to_json_dict(chunk)
    restored = chunking.json_dict_to_chunk(data)

    assert restored == chunk


# ---------- _save_chunks / _load_cached_chunks ----------


def test_save_and_load_chunks_round_trips(isolated_data_dir, fake_sentence_transformer_class):
    """
    Input: two Chunks belonging to the same doc_id, saved in one call
    via chunking._save_chunks.
    Output: chunking._load_cached_chunks's return value for that
    doc_id.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )
    chunk_path = str(isolated_data_dir / "chunks_out")
    chunk_a = Chunk(id="doc1_0", text="a", size=1, type="fixed")
    chunk_b = Chunk(id="doc1_1", text="b", size=1, type="fixed")
    chunking._save_chunks(chunk_path, {"doc1_0": chunk_a, "doc1_1": chunk_b})

    loaded = chunking._load_cached_chunks(chunk_path, "doc1")

    assert loaded == {"doc1_0": chunk_a, "doc1_1": chunk_b}


def test_load_cached_chunks_returns_none_when_chunk_path_does_not_exist(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a chunk_path that has never been created.
    Output: chunking._load_cached_chunks's return value.

    Checks the "nothing cached yet at all" branch, which must not
    raise even though chunk_path does not exist on disk.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )
    chunk_path = str(isolated_data_dir / "never_created")

    assert chunking._load_cached_chunks(chunk_path, "doc1") is None


def test_load_cached_chunks_returns_none_when_doc_id_has_no_cached_files(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: chunk_path exists and holds another doc_id's chunks, but
    none for the doc_id being looked up.
    Output: chunking._load_cached_chunks's return value.

    Checks that a partially-populated chunk_path (some other document
    already cached) does not get mistaken for a cache hit on a
    different, not-yet-chunked doc_id.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )
    chunk_path = str(isolated_data_dir / "chunks_out")
    chunking._save_chunks(chunk_path, {"doc1_0": Chunk(id="doc1_0", text="a", size=1, type="fixed")})

    assert chunking._load_cached_chunks(chunk_path, "doc2") is None


# ---------- load_documents ----------


def test_load_documents_reads_every_processed_document_file(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: two processed Document JSON files written under a
    directory.
    Output: chunking.load_documents's return value.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )
    docs_dir = str(isolated_data_dir / "docs")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "hello world"))
    _write_processed_document(docs_dir, "doc2", _make_document("doc2", "goodbye world"))

    documents = chunking.load_documents(docs_dir, dataset_size=2)

    assert set(documents.keys()) == {"doc1", "doc2"}
    assert documents["doc1"].text == "hello world"
    assert documents["doc2"].text == "goodbye world"


# ---------- compute_chunks ----------


def test_compute_chunks_fixed_end_to_end_writes_and_returns_chunks(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document written to the path implied by
    dataset_name "myset".
    Output: chunking.compute_chunks("fixed")'s return value, and the
    files written under paths.get_chunk_path for this test's params.

    Checks the whole pipeline end to end: documents are loaded,
    chunked, and both returned and persisted to disk.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(
        docs_dir, "doc1", _make_document("doc1", "one two three four five six seven")
    )
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1, semantic_threshold=0.0, model_name="fake"
    )

    chunks = chunking.compute_chunks("fixed")

    assert len(chunks) == 3
    assert chunks["doc1_0"].text == "one two three four"
    output_path = paths.get_chunk_path("myset", "fixed", 1, 4, 1, 0.0)
    assert sorted(os.listdir(output_path)) == ["doc1_0.json", "doc1_1.json", "doc1_2.json"]


def test_compute_chunks_returns_cached_chunks_without_recomputing_them(
    isolated_data_dir, fake_sentence_transformer_class, monkeypatch
):
    """
    Input: a first compute_chunks("fixed") call that writes doc1's
    chunks to disk under its per-document cache, followed by a second
    compute_chunks("fixed") call with FixedChunking.chunk patched to
    raise if it is invoked again.
    Output: both calls' return values.

    Checks the per-document cache-hit branch added by
    Chunking._load_cached_chunks: once a document's chunks are already
    on disk, a second run must reuse them straight from disk instead
    of re-running the chunking strategy on that document. (Per-document
    caching still needs the processed-documents directory to be
    present, unlike the old whole-run cache, since compute_chunks
    loads every document up front to know which doc_ids to check the
    cache for.)
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "one two three four"))
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0, semantic_threshold=0.0, model_name="fake"
    )

    first_run = chunking.compute_chunks("fixed")

    def fail_if_called(self, document):
        raise AssertionError("chunk() should not run again once chunks are cached")

    monkeypatch.setattr(
        "l2_chunking.chunkers.fixed.FixedChunking.chunk", fail_if_called
    )

    second_run = chunking.compute_chunks("fixed")

    assert second_run == first_run


def test_compute_chunks_semantic_builds_embeddings_and_offsets(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document with two sentences, under
    dataset_name "myset".
    Output: chunking.compute_chunks("semantic")'s return value, and
    chunking.all_embeddings / chunking.embeddings_offset afterward.

    Checks that the semantic branch actually computes embeddings for
    every sentence and records each document's slice into them.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(
        docs_dir, "doc1", _make_document("doc1", "w1 w2 w3 w4 w5. w6 w7 w8 w9 w10.")
    )
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=5, fixed_chunk_overlap=0, semantic_threshold=0.5, model_name="fake"
    )

    chunks = chunking.compute_chunks("semantic")

    assert chunking.embeddings_offset == {"doc1": (0, 2)}
    assert isinstance(chunking.all_embeddings, np.ndarray)
    assert chunking.all_embeddings.shape[0] == 2
    assert len(chunks) == 2
    for chunk in chunks.values():
        assert chunk.size == 5


def test_compute_chunks_raises_value_error_for_unknown_chunk_type_before_loading_documents(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: an unknown chunk_type, and no processed-documents
    directory created at all.
    Output: none; compute_chunks is expected to raise ValueError.

    Per compute_chunks's own docstring: "Looks up chunk_type in the
    chunking strategy registry first, raising ValueError immediately
    if it is not a registered, implemented strategy, before any
    document loading happens." This checks that contract directly: if
    chunk_type validation ran before document loading as documented,
    a missing documents directory would never be touched, so the
    failure here must be ValueError, not a filesystem error from
    trying to list a directory that doesn't exist.
    """
    chunking = Chunking(
        dataset_name="myset", dataset_size=1,
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.0, model_name="fake"
    )

    with pytest.raises(ValueError):
        chunking.compute_chunks("totally-bogus-chunk-type")
