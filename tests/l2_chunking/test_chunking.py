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

Chunking takes a ChunkConfig (dataset_name, chunk_type,
min_chunk_size, max_chunk_size, fixed_chunk_overlap,
semantic_threshold bundled together -- see common.chunk_config) plus
dataset_size and model_name as constructor arguments, the way a real
run's l2_chunking.main wires them in from its own CLI args. Tests that
call compute_chunks construct Chunking with a ChunkConfig whose
dataset_name matches the one they write processed documents under,
and whose chunk_type is the strategy that test means to run.
"""

import dataclasses
import datetime
import json
import os

import numpy as np
import pytest

from common.chunk_config import ChunkConfig
from common import paths
from l2_chunking.chunking import Chunking


def _write_processed_document(docs_dir, doc_id, document):
    """
    docs_dir: directory to write the processed document JSON file
        into (created if missing).
    doc_id: identifies the file to write.
    document: the Document to persist.

    Writes document as {doc_id}.json in the same shape
    l1_doc_processing.doc_processing.DocProcessing.document_to_json_dict
    produces, so l1_doc_processing.cache.load_documents can read it back.

    Returns nothing.
    """
    os.makedirs(docs_dir, exist_ok=True)
    data = dataclasses.asdict(document)
    data["timestamp"] = document.timestamp.isoformat()
    with open(os.path.join(docs_dir, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_progress(dataset_name, processed_doc_ids):
    """
    dataset_name: dataset whose progress.json to write.
    processed_doc_ids: doc_ids to list as processed.

    Writes the progress.json that l1_doc_processing leaves behind, which
    compute_chunks requires.

    Returns nothing.
    """
    progress = Progress("someone/dataset", "train", "pdf", len(processed_doc_ids), processed_doc_ids, [])
    save_progress(dataset_name, progress)


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
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

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
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.5,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    assert chunking.all_embeddings is None
    assert chunking.embeddings_offset is None


# ---------- compute_chunks ----------


def test_compute_chunks_fixed_end_to_end_writes_and_returns_chunks(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document written to the path implied by
    dataset_name "myset".
    Output: chunking.compute_chunks()'s return value, and the files
    written under paths.get_chunk_path for this test's chunk_config.

    Checks the whole pipeline end to end: documents are loaded,
    chunked, and both returned and persisted to disk.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(
        docs_dir, "doc1", _make_document("doc1", "one two three four five six seven")
    )
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    chunks = chunking.compute_chunks()

    assert len(chunks) == 3
    assert chunks["doc1_0"].text == "one two three four"
    output_path = paths.get_chunk_path(chunk_config)
    assert sorted(os.listdir(output_path)) == ["doc1_0.json", "doc1_1.json", "doc1_2.json"]


def test_compute_chunks_returns_cached_chunks_without_recomputing_them(
    isolated_data_dir, fake_sentence_transformer_class, monkeypatch
):
    """
    Input: a first compute_chunks() call that writes doc1's chunks to
    disk under its per-document cache, followed by a second
    compute_chunks() call with FixedChunking.chunk patched to raise if
    it is invoked again.
    Output: both calls' return values.

    Checks the per-document cache-hit branch added by
    l2_chunking.cache._load_cached_chunks: once a document's chunks are already
    on disk, a second run must reuse them straight from disk instead
    of re-running the chunking strategy on that document. (Per-document
    caching still needs the processed-documents directory to be
    present, unlike the old whole-run cache, since compute_chunks
    loads every document up front to know which doc_ids to check the
    cache for.)
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "one two three four"))
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    first_run = chunking.compute_chunks()

    def fail_if_called(self, document):
        raise AssertionError("chunk() should not run again once chunks are cached")

    monkeypatch.setattr(
        "l2_chunking.chunkers.fixed.FixedChunking.chunk", fail_if_called
    )

    second_run = chunking.compute_chunks()

    assert second_run == first_run


def test_compute_chunks_semantic_builds_embeddings_and_offsets(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document with two sentences, under
    dataset_name "myset".
    Output: chunking.compute_chunks()'s return value, and
    chunking.all_embeddings / chunking.embeddings_offset afterward.

    Checks that the semantic branch actually computes embeddings for
    every sentence and records each document's slice into them.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(
        docs_dir, "doc1", _make_document("doc1", "w1 w2 w3 w4 w5. w6 w7 w8 w9 w10.")
    )
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="semantic",
        min_chunk_size=1, max_chunk_size=5, fixed_chunk_overlap=0, semantic_threshold=0.5,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    chunks = chunking.compute_chunks()

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
    Input: a ChunkConfig with an unknown chunk_type, and no
    processed-documents directory created at all.
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
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="totally-bogus-chunk-type",
        min_chunk_size=1, max_chunk_size=10, fixed_chunk_overlap=0, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    with pytest.raises(ValueError):
        chunking.compute_chunks()


def test_compute_chunks_raises_for_an_invalid_chunk_config_before_loading_documents(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: a ChunkConfig whose overlap is not smaller than max_chunk_size,
    and no processed documents on disk.
    Output: none; ValueError is expected.
    """
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=4, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    with pytest.raises(ValueError):
        chunking.compute_chunks()


def test_compute_chunks_raises_when_progress_json_is_missing(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document on disk but no progress.json.
    Output: none; FileNotFoundError is expected.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "one two three four"))
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    with pytest.raises(FileNotFoundError):
        chunking.compute_chunks()


def test_compute_chunks_raises_when_dataset_size_exceeds_processed_documents(
    isolated_data_dir, fake_sentence_transformer_class
):
    """
    Input: one processed document and its progress.json, dataset_size 2.
    Output: none; ValueError is expected and no chunks are written.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "one two three four"))
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=2, model_name="fake")

    with pytest.raises(ValueError):
        chunking.compute_chunks()

    assert not os.path.exists(paths.get_chunk_path(chunk_config))


def test_compute_chunks_prints_new_then_cached_counts_and_chunk_time(
    isolated_data_dir, fake_sentence_transformer_class, capsys
):
    """
    Input: one processed document chunked twice with the same config.
    Output: the printed summaries of both runs.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "one two three four five six seven"))
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="fixed",
        min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1, semantic_threshold=0.0,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    chunking.compute_chunks()
    first_output = capsys.readouterr().out
    chunking.compute_chunks()
    second_output = capsys.readouterr().out

    assert "Newly chunked 1 document(s): 3 chunk(s)." in first_output
    assert "Documents chunked : 1" in first_output
    assert "Loaded cached chunks for 1 document(s): 3 chunk(s)." in second_output
    assert "No documents were chunked in this run." in second_output


def test_compute_chunks_prints_sentence_embedding_time_for_semantic(
    isolated_data_dir, fake_sentence_transformer_class, capsys
):
    """
    Input: one processed document with two sentences, semantic chunking.
    Output: the printed sentence embedding summary.
    """
    docs_dir = paths.get_processed_dataset_path("myset")
    _write_processed_document(docs_dir, "doc1", _make_document("doc1", "w1 w2 w3 w4 w5. w6 w7 w8 w9 w10."))
    _write_progress("myset", ["doc1"])
    chunk_config = ChunkConfig(
        dataset_name="myset", chunk_type="semantic",
        min_chunk_size=1, max_chunk_size=5, fixed_chunk_overlap=0, semantic_threshold=0.5,
    )
    chunking = Chunking(chunk_config, dataset_size=1, model_name="fake")

    chunking.compute_chunks()

    output = capsys.readouterr().out
    assert "----- Sentence embedding time (semantic) -----" in output
    assert "Sentences embedded : 2" in output
