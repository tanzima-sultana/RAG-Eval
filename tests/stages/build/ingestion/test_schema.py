"""
test_schema.py

Tests for stages.build.ingestion.schema: the Document, Metadata and DocumentType
models this project uses for ingested documents.
"""

import pydantic
import pytest

from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def make_metadata():
    """
    Input: none.
    Output: a Metadata with only the required fields set.
    What it does: builds a minimal valid Metadata for the tests below.
    """
    metadata = Metadata(source="file.pdf", file_size_bytes=10, content_hash="hash")
    return metadata


def test_document_type_values_are_plain_strings():
    """
    Input: DocumentType.PDF and DocumentType.TXT.
    Output: their string values.
    What it does: checks the enum values compare equal to plain strings.
    """
    assert DocumentType.PDF == "pdf"
    assert DocumentType.TXT == "txt"


def test_metadata_optional_fields_default_to_none():
    """
    Input: a Metadata built with only the required fields.
    Output: its optional fields.
    What it does: checks every optional field defaults to None.
    """
    metadata = make_metadata()

    assert metadata.minhash_signature is None
    assert metadata.page_count is None
    assert metadata.parser is None
    assert metadata.author is None
    assert metadata.title is None
    assert metadata.creation_date is None


def test_metadata_rejects_negative_file_size():
    """
    Input: file_size_bytes of -1.
    Output: a pydantic ValidationError.
    What it does: checks the ge=0 rule on file_size_bytes.
    """
    with pytest.raises(pydantic.ValidationError):
        Metadata(source="file.pdf", file_size_bytes=-1, content_hash="hash")


def test_document_accepts_type_as_string():
    """
    Input: a Document built with type "pdf" as a plain string.
    Output: the Document's type.
    What it does: checks the string is turned into DocumentType.PDF.
    """
    document = Document(id="doc1", type="pdf", text="text", metadata=make_metadata())

    assert document.type is DocumentType.PDF


def test_document_rejects_unknown_type():
    """
    Input: a Document built with type "exe".
    Output: a pydantic ValidationError.
    What it does: checks only DocumentType values are accepted.
    """
    with pytest.raises(pydantic.ValidationError):
        Document(id="doc1", type="exe", text="text", metadata=make_metadata())
