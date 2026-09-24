"""
test_cache.py

Tests for l1_doc_processing.cache: converting Documents to and from
JSON, saving/loading processed Document files, and loading a dataset's
first processed documents in progress.json order.
"""

import datetime
import os

import pytest

from l1_doc_processing.cache import (
    document_to_json_dict,
    json_dict_to_document,
    load_document_ids,
    load_documents,
    load_processed_document,
    save_processed_document,
)
from l1_doc_processing.progress import Progress, save_progress
from l1_doc_processing.schema import Document


DATASET_NAME = "myset"


def _make_document(doc_id):
    """
    Returns a small, fully-populated Document with the given doc_id.
    """
    return Document(
        id=doc_id,
        text="hello",
        title=None,
        path="some/path",
        format="pdf",
        metadata={"page_count": 1},
        content_hash="hash",
        timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def test_document_to_json_dict_and_back_round_trips():
    """
    Input: a Document converted to a JSON dict and back.
    Output: the rebuilt Document.
    """
    document = _make_document("pdf_x")

    data = document_to_json_dict(document)
    rebuilt = json_dict_to_document(data)

    assert isinstance(data["timestamp"], str)
    assert rebuilt == document


def test_save_and_load_processed_document_round_trips(isolated_data_dir):
    """
    Input: a Document saved with save_processed_document.
    Output: load_processed_document's return value and the file.
    """
    document = _make_document("pdf_x")

    save_processed_document(DATASET_NAME, "pdf_x", document)
    loaded = load_processed_document(DATASET_NAME, "pdf_x")

    assert loaded == document
    assert os.path.isfile(f"{DATASET_NAME}/data/processed/pdf_x.json")


def test_load_processed_document_returns_none_when_missing(isolated_data_dir):
    """
    Input: a doc_id with no processed file.
    Output: load_processed_document's return value.
    """
    assert load_processed_document(DATASET_NAME, "pdf_missing") is None


def _save_documents_and_progress(processed_doc_ids):
    """
    processed_doc_ids: doc_ids to save as processed documents and list,
        in this order, in progress.json.

    Returns nothing.
    """
    for doc_id in processed_doc_ids:
        save_processed_document(DATASET_NAME, doc_id, _make_document(doc_id))
    save_progress(DATASET_NAME, Progress("someone/dataset", "train", "pdf", len(processed_doc_ids), processed_doc_ids, []))


def test_load_document_ids_returns_the_first_ids_in_progress_order(isolated_data_dir):
    """
    Input: progress.json listing three doc_ids out of name order, dataset_size 2.
    Output: load_document_ids's return value.
    """
    _save_documents_and_progress(["pdf_c", "pdf_a", "pdf_b"])

    assert load_document_ids(DATASET_NAME, 2) == ["pdf_c", "pdf_a"]


def test_load_documents_follows_progress_order_and_stops_at_dataset_size(isolated_data_dir):
    """
    Input: three processed documents listed out of name order in
    progress.json, dataset_size 2.
    Output: load_documents's return value.
    """
    _save_documents_and_progress(["pdf_c", "pdf_a", "pdf_b"])

    documents = load_documents(DATASET_NAME, 2)

    assert list(documents.keys()) == ["pdf_c", "pdf_a"]
    assert documents["pdf_c"] == _make_document("pdf_c")


def test_load_documents_raises_when_a_listed_document_file_is_missing(isolated_data_dir):
    """
    Input: progress.json listing a doc_id whose processed file was deleted.
    Output: none; FileNotFoundError is expected.
    """
    _save_documents_and_progress(["pdf_a", "pdf_b"])
    os.remove(f"{DATASET_NAME}/data/processed/pdf_b.json")

    with pytest.raises(FileNotFoundError):
        load_documents(DATASET_NAME, 2)
