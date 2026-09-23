"""
test_document_store.py

Tests for l1_doc_processing.document_store: converting Documents to and
from JSON, and saving/loading processed Document files.
"""

import datetime
import os

from l1_doc_processing.document_store import (
    document_to_json_dict,
    json_dict_to_document,
    load_processed_document,
    save_processed_document,
)
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
