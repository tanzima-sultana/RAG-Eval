"""
test_pdf.py

Tests for l1_doc_processing.parsers.pdf: the private extraction
helpers (_extract_text, _extract_title, _extract_pdf_metadata), tested
directly against lightweight fake pypdf.PdfReader stand-ins so every
branch (missing metadata, missing title, blank title) is reachable
deterministically; and PDFParser.parse, tested end-to-end against a
real, minimal PDF built with pypdf's own writer.
"""

import datetime
import io

import pypdf
import pytest

from common.identifiers import make_content_hash
from l1_doc_processing.parsers.pdf import (
    PDFParser,
    _extract_pdf_metadata,
    _extract_text,
    _extract_title,
)


class _FakePage:
    """
    Stand-in for a pypdf page object: extract_text() returns whatever
    fixed string it was built with.
    """

    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeMetadata:
    """
    Stand-in for a pypdf DocumentInformation object: exposes only the
    title/author attributes _extract_title and _extract_pdf_metadata
    read.
    """

    def __init__(self, title=None, author=None):
        self.title = title
        self.author = author


class _FakeReader:
    """
    Stand-in for a pypdf.PdfReader: exposes only the pages/metadata
    attributes the extraction helpers read.
    """

    def __init__(self, pages, metadata=None):
        self.pages = pages
        self.metadata = metadata


def _build_minimal_pdf_bytes(title=None, author=None):
    """
    title: PDF /Title metadata to set, or None to leave it unset.
    author: PDF /Author metadata to set, or None to leave it unset.

    Builds a real, valid, single blank-page PDF using pypdf's own
    writer, so PDFParser can be tested end-to-end without depending on
    hand-built PDF byte layouts.

    Returns the PDF's raw bytes.
    """
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)

    document_info = {}
    if title is not None:
        document_info["/Title"] = title
    if author is not None:
        document_info["/Author"] = author
    if len(document_info) > 0:
        writer.add_metadata(document_info)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ---------- _extract_text ----------


def test_extract_text_joins_multiple_pages_with_blank_line():
    """
    Input: a fake reader with two pages of text.
    Output: _extract_text's return value.

    Checks that page text is joined in page order with a blank line
    between pages.
    """
    reader = _FakeReader(pages=[_FakePage("page one"), _FakePage("page two")])

    text = _extract_text(reader)

    assert text == "page one\n\npage two"


def test_extract_text_single_page_has_no_separator():
    """
    Input: a fake reader with a single page of text.
    Output: _extract_text's return value.

    Checks that a single page's text is returned as-is, with no
    trailing separator.
    """
    reader = _FakeReader(pages=[_FakePage("only page")])

    text = _extract_text(reader)

    assert text == "only page"


def test_extract_text_no_pages_returns_empty_string():
    """
    Input: a fake reader with no pages.
    Output: _extract_text's return value.

    Checks the degenerate case of a PDF with zero pages.
    """
    reader = _FakeReader(pages=[])

    text = _extract_text(reader)

    assert text == ""


# ---------- _extract_title ----------


def test_extract_title_returns_none_when_metadata_is_none():
    """
    Input: a fake reader whose metadata attribute is None.
    Output: _extract_title's return value.

    Checks the case of a PDF with no document information dictionary
    at all.
    """
    reader = _FakeReader(pages=[], metadata=None)

    assert _extract_title(reader) is None


def test_extract_title_returns_none_when_title_is_none():
    """
    Input: a fake reader with metadata present but title unset.
    Output: _extract_title's return value.
    """
    reader = _FakeReader(pages=[], metadata=_FakeMetadata(title=None))

    assert _extract_title(reader) is None


def test_extract_title_returns_none_when_title_is_empty_string():
    """
    Input: a fake reader whose title metadata is an empty string.
    Output: _extract_title's return value.

    Checks that an empty title is treated the same as a missing title.
    """
    reader = _FakeReader(pages=[], metadata=_FakeMetadata(title=""))

    assert _extract_title(reader) is None


def test_extract_title_returns_none_when_title_is_whitespace_only():
    """
    Input: a fake reader whose title metadata is whitespace only.
    Output: _extract_title's return value.

    Checks that a whitespace-only title is treated the same as a
    missing title.
    """
    reader = _FakeReader(pages=[], metadata=_FakeMetadata(title="   "))

    assert _extract_title(reader) is None


def test_extract_title_returns_title_when_present():
    """
    Input: a fake reader with a real title set.
    Output: _extract_title's return value.
    """
    reader = _FakeReader(pages=[], metadata=_FakeMetadata(title="My Contract"))

    assert _extract_title(reader) == "My Contract"


# ---------- _extract_pdf_metadata ----------


def test_extract_pdf_metadata_returns_page_count_and_author():
    """
    Input: a fake reader with three pages and an author set.
    Output: _extract_pdf_metadata's return value.
    """
    pages = [_FakePage(""), _FakePage(""), _FakePage("")]
    reader = _FakeReader(pages=pages, metadata=_FakeMetadata(author="Jane Doe"))

    pdf_metadata = _extract_pdf_metadata(reader)

    assert pdf_metadata == {"page_count": 3, "author": "Jane Doe"}


def test_extract_pdf_metadata_author_is_none_when_metadata_missing():
    """
    Input: a fake reader with one page and metadata set to None.
    Output: _extract_pdf_metadata's return value.

    Checks that a PDF with no document information dictionary still
    produces a metadata dict, with author set to None.
    """
    reader = _FakeReader(pages=[_FakePage("")], metadata=None)

    pdf_metadata = _extract_pdf_metadata(reader)

    assert pdf_metadata == {"page_count": 1, "author": None}


# ---------- PDFParser.parse (end-to-end, real PDF bytes) ----------


def test_pdf_parser_parse_builds_a_fully_populated_document():
    """
    Input: raw bytes for a real, valid, single blank-page PDF with
    title and author metadata set.
    Output: PDFParser().parse's return value.

    Checks the full wiring of parse(): the doc_id/data_path passed in
    are carried straight through, title/author/page_count come from
    the PDF's metadata, format is "pdf", content_hash matches the
    cleaned (here: empty, since the page is blank) extracted text, and
    timestamp is a timezone-aware UTC datetime.
    """
    raw_bytes = _build_minimal_pdf_bytes(title="My Contract", author="Jane Doe")

    document = PDFParser().parse("pdf_abc123", raw_bytes, "data/raw/pdf_abc123.pdf")

    assert document.id == "pdf_abc123"
    assert document.path == "data/raw/pdf_abc123.pdf"
    assert document.format == "pdf"
    assert document.title == "My Contract"
    assert document.text == ""
    assert document.content_hash == make_content_hash("")
    assert document.metadata["format"] == "pdf"
    assert document.metadata["raw_bytes_size"] == len(raw_bytes)
    assert document.metadata["page_count"] == 1
    assert document.metadata["author"] == "Jane Doe"
    assert isinstance(document.timestamp, datetime.datetime)
    assert document.timestamp.tzinfo is not None


def test_pdf_parser_parse_title_is_none_when_pdf_has_no_metadata():
    """
    Input: raw bytes for a real, valid, single blank-page PDF with no
    title/author metadata set.
    Output: PDFParser().parse's return value.

    Checks that a PDF with no metadata still parses successfully, with
    title and author left as None.
    """
    raw_bytes = _build_minimal_pdf_bytes()

    document = PDFParser().parse("pdf_no_meta", raw_bytes, "data/raw/pdf_no_meta.pdf")

    assert document.title is None
    assert document.metadata["author"] is None


def test_pdf_parser_parse_raises_on_invalid_pdf_bytes():
    """
    Input: bytes that are not a valid PDF.
    Output: none; parse is expected to raise.

    Checks that garbage input fails loudly (via pypdf) rather than
    silently producing a bogus Document. process_raw_data relies on
    this to count and skip unparseable documents.
    """
    with pytest.raises(pypdf.errors.PyPdfError):
        PDFParser().parse("pdf_bad", b"not a pdf at all", "data/raw/pdf_bad.pdf")
