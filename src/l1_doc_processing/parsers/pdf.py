"""
pdf.py

Implements Parser for PDF documents. Extracts text and metadata from a
document's raw bytes with pypdf, and builds a Document. This is the
only file allowed to know PDF-specific concerns (pypdf, page
extraction).
"""

import datetime
import io
import time

import pypdf

from common.identifiers import make_content_hash
from l1_doc_processing import timing
from l1_doc_processing.metadata import extract_metadata
from l1_doc_processing.parser import Parser
from l1_doc_processing.schema import Document
from l1_doc_processing.text_cleanup import clean_text


def _extract_text(reader: pypdf.PdfReader) -> str:
    """
    reader: a pypdf.PdfReader already opened on the document's bytes.

    Extracts the text of every page, in page order, and joins them into
    a single string with a blank line between pages.

    Returns the extracted text.
    """
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if text != "":
            text += "\n\n"
        text += page_text
    return text


def _extract_title(reader: pypdf.PdfReader) -> str | None:
    """
    reader: a pypdf.PdfReader already opened on the document's bytes.

    Looks up the PDF's Title entry in its document information
    dictionary, if the PDF has one set.

    Returns the title, or None if the PDF has no metadata or no title.
    """
    metadata = reader.metadata
    if metadata is None:
        return None

    title = metadata.title
    if title is None:
        return None
    if title.strip() == "":
        return None

    return title


def _extract_pdf_metadata(reader: pypdf.PdfReader) -> dict:
    """
    reader: a pypdf.PdfReader already opened on the document's bytes.

    Extracts page_count (from len(reader.pages)) and author (from
    reader.metadata) directly from the PDF object. Many PDFs have no
    author metadata, so author is set to None rather than failing or
    being left out.

    Returns a dict with "page_count" and "author".
    """
    page_count = len(reader.pages)

    author = None
    pdf_info = reader.metadata
    if pdf_info is not None:
        author = pdf_info.author

    return {"page_count": page_count, "author": author}


class PDFParser(Parser):
    def parse(self, doc_id: str, raw_bytes: bytes, data_path: str) -> Document:
        """
        doc_id: this document's identifier, already computed by the
            caller. Set directly on the returned Document rather than
            recomputed here.
        raw_bytes: the PDF's raw, undecoded bytes.
        data_path: the original raw data path of the document.

        Reads the PDF's bytes, extracts and cleans its text, extracts
        title/metadata with pypdf, and builds the content_hash, and source path needed for caching, dedup, and
        cross-layer mapping.

        Returns a fully-populated Document.
        """
        open_reader_start = time.perf_counter()
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        timing.record("pdf_open_reader", time.perf_counter() - open_reader_start)

        # Text cleanup
        extract_text_start = time.perf_counter()
        text = _extract_text(reader)
        timing.record("pdf_extract_text", time.perf_counter() - extract_text_start)

        text_cleanup_start = time.perf_counter()
        text = clean_text(text)
        timing.record("text_cleanup", time.perf_counter() - text_cleanup_start)

        extract_title_start = time.perf_counter()
        title = _extract_title(reader)
        timing.record("pdf_extract_title", time.perf_counter() - extract_title_start)

        # Metadata: generic fields (from raw_bytes), then PDF-specific fields on top
        extract_metadata_start = time.perf_counter()
        metadata = extract_metadata(raw_bytes, "pdf")
        metadata.update(_extract_pdf_metadata(reader))
        timing.record("extract_metadata", time.perf_counter() - extract_metadata_start)

        content_hash_start = time.perf_counter()
        content_hash = make_content_hash(text)
        timing.record("make_content_hash", time.perf_counter() - content_hash_start)

        timestamp = datetime.datetime.now(datetime.timezone.utc)

        return Document(
            id=doc_id,
            text=text,
            title=title,
            path=data_path,
            format="pdf",
            metadata=metadata,
            content_hash=content_hash,
            timestamp=timestamp,
        )
