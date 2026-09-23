"""
test_schema.py

Tests for l1_doc_processing.schema: Document and Progress, the plain dataclasses
shared across the l1_doc_processing pipeline.
"""

import dataclasses
import datetime

from l1_doc_processing.schema import Document, Progress


def test_document_holds_every_field_it_is_constructed_with():
    """
    Input: one value per Document field.
    Output: the constructed Document.

    Checks that Document is a plain data container: every field is
    stored as-is and readable back by attribute name.
    """
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

    document = Document(
        id="pdf_abc123",
        text="hello world",
        title="A Title",
        path="data/raw/pdf_abc123.pdf",
        format="pdf",
        metadata={"page_count": 1},
        content_hash="deadbeef",
        timestamp=timestamp,
    )

    assert document.id == "pdf_abc123"
    assert document.text == "hello world"
    assert document.title == "A Title"
    assert document.path == "data/raw/pdf_abc123.pdf"
    assert document.format == "pdf"
    assert document.metadata == {"page_count": 1}
    assert document.content_hash == "deadbeef"
    assert document.timestamp == timestamp


def test_document_title_accepts_none():
    """
    Input: a Document constructed with title=None.
    Output: the constructed Document.

    Checks that title is allowed to be None, as its "str | None" type
    hint declares (e.g. a PDF with no title metadata).
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    document = Document(
        id="pdf_abc123",
        text="hello world",
        title=None,
        path="data/raw/pdf_abc123.pdf",
        format="pdf",
        metadata={},
        content_hash="deadbeef",
        timestamp=timestamp,
    )

    assert document.title is None


def test_document_converts_to_a_plain_dict_via_dataclasses_asdict():
    """
    Input: a fully-populated Document.
    Output: dataclasses.asdict(document).

    Checks that Document round-trips through dataclasses.asdict as a
    plain dict, which document_to_json_dict relies on.
    """
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

    document = Document(
        id="pdf_abc123",
        text="hello world",
        title=None,
        path="data/raw/pdf_abc123.pdf",
        format="pdf",
        metadata={"page_count": 1},
        content_hash="deadbeef",
        timestamp=timestamp,
    )

    data = dataclasses.asdict(document)

    assert data["id"] == "pdf_abc123"
    assert data["metadata"] == {"page_count": 1}
    assert data["timestamp"] == timestamp


def test_progress_holds_every_field_and_converts_to_a_plain_dict():
    """
    Input: a fully-populated Progress.
    Output: its fields and dataclasses.asdict(progress).

    save_progress writes dataclasses.asdict(progress) as JSON, so every
    field must be plain JSON data.
    """
    progress = Progress(
        dataset_identifier="someone/dataset",
        split="train",
        doc_type="pdf",
        rows_consumed=12,
        processed_doc_ids=["pdf_a", "pdf_b"],
        skipped_doc_ids=["pdf_c"],
    )

    data = dataclasses.asdict(progress)

    assert data == {
        "dataset_identifier": "someone/dataset",
        "split": "train",
        "doc_type": "pdf",
        "rows_consumed": 12,
        "processed_doc_ids": ["pdf_a", "pdf_b"],
        "skipped_doc_ids": ["pdf_c"],
    }
