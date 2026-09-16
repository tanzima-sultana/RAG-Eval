"""
test_metadata.py

Tests for l1_doc_processing.metadata.extract_metadata, the
format-agnostic metadata builder that only looks at a document's raw
bytes.
"""

from l1_doc_processing.metadata import extract_metadata


def test_extract_metadata_returns_doc_type_as_format_and_byte_size():
    """
    Input: some raw bytes and doc_type "pdf".
    Output: the metadata dict.

    Checks that "format" is set to doc_type and "raw_bytes_size" is set
    to the length of raw_bytes.
    """
    raw_bytes = b"hello world"

    metadata = extract_metadata(raw_bytes, "pdf")

    assert metadata["format"] == "pdf"
    assert metadata["raw_bytes_size"] == len(raw_bytes)


def test_extract_metadata_handles_empty_raw_bytes():
    """
    Input: empty raw bytes and doc_type "pdf".
    Output: the metadata dict.

    Checks that an empty raw_bytes value produces raw_bytes_size 0
    rather than raising.
    """
    metadata = extract_metadata(b"", "pdf")

    assert metadata["raw_bytes_size"] == 0
    assert metadata["format"] == "pdf"
