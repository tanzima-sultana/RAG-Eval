"""
test_docx.py

Tests for l1_doc_processing.parsers.docx. Real .docx bytes are built in
memory with zipfile (a .docx is a zip of XML files), so no sample files
or extra libraries are needed.
"""

import datetime
import io
import zipfile

import pytest

from l1_doc_processing.parsers.docx import DocxParser


_DOCUMENT_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document'
    ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
    "<w:body>{body}</w:body>"
    "</w:document>"
)

_CORE_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    "<cp:coreProperties"
    ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
    ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
    "{fields}"
    "</cp:coreProperties>"
)

_APP_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
    "{fields}"
    "</Properties>"
)


def _paragraph(text):
    """
    text: the paragraph's text.

    Returns the XML of a one-run Word paragraph holding text.
    """
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _build_docx_bytes(body, core_fields=None, app_fields=None):
    """
    body: XML placed inside <w:body> of word/document.xml.
    core_fields: XML placed in docProps/core.xml, or None to leave
        that file out.
    app_fields: XML placed in docProps/app.xml, or None to leave that
        file out.

    Returns the bytes of a .docx zip holding those files.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as docx_zip:
        docx_zip.writestr("word/document.xml", _DOCUMENT_TEMPLATE.format(body=body))
        if core_fields is not None:
            docx_zip.writestr("docProps/core.xml", _CORE_TEMPLATE.format(fields=core_fields))
        if app_fields is not None:
            docx_zip.writestr("docProps/app.xml", _APP_TEMPLATE.format(fields=app_fields))
    return buffer.getvalue()


def test_parse_builds_a_fully_populated_document():
    """
    Input: a .docx with two paragraphs, a title, an author and a page count.
    Output: the Document returned by DocxParser.parse.
    """
    raw_bytes = _build_docx_bytes(
        _paragraph("First paragraph.") + _paragraph("Second paragraph."),
        core_fields="<dc:title>Contract</dc:title><dc:creator>Jane</dc:creator>",
        app_fields="<Pages>3</Pages>",
    )

    document = DocxParser().parse("docx_abc", raw_bytes, "data/raw/docx_abc.docx")

    assert document.id == "docx_abc"
    assert document.text == "First paragraph.\nSecond paragraph."
    assert document.title == "Contract"
    assert document.path == "data/raw/docx_abc.docx"
    assert document.format == "docx"
    assert document.metadata == {
        "format": "docx",
        "raw_bytes_size": len(raw_bytes),
        "page_count": 3,
        "author": "Jane",
    }
    assert document.content_hash == DocxParser().make_content_hash(document.text)
    assert isinstance(document.timestamp, datetime.datetime)


def test_parse_keeps_tabs_and_line_breaks_inside_a_paragraph():
    """
    Input: one paragraph with a tab and a line break between runs.
    Output: the parsed Document's text.
    """
    body = "<w:p><w:r><w:t>Name</w:t><w:tab/><w:t>Value</w:t><w:br/><w:t>Next line</w:t></w:r></w:p>"
    raw_bytes = _build_docx_bytes(body)

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.text == "Name Value\nNext line"


def test_parse_extracts_table_cell_text_in_order():
    """
    Input: a paragraph, then a 1x2 table.
    Output: the parsed Document's text.
    """
    table = (
        "<w:tbl><w:tr>"
        "<w:tc>" + _paragraph("Cell one") + "</w:tc>"
        "<w:tc>" + _paragraph("Cell two") + "</w:tc>"
        "</w:tr></w:tbl>"
    )
    raw_bytes = _build_docx_bytes(_paragraph("Intro") + table)

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.text == "Intro\nCell one\nCell two"


def test_parse_skips_fallback_copies_of_text():
    """
    Input: a paragraph whose text box appears in both mc:Choice and
    mc:Fallback, as Word saves it.
    Output: the parsed Document's text.
    """
    body = (
        "<w:p><w:r><mc:AlternateContent>"
        "<mc:Choice><w:t>Box text</w:t></mc:Choice>"
        "<mc:Fallback><w:t>Box text</w:t></mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
    )
    raw_bytes = _build_docx_bytes(body)

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.text == "Box text"


def test_parse_sets_title_author_and_page_count_to_none_when_missing():
    """
    Input: a .docx with no docProps files.
    Output: the parsed Document's title and metadata.
    """
    raw_bytes = _build_docx_bytes(_paragraph("Body"))

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.title is None
    assert document.metadata["author"] is None
    assert document.metadata["page_count"] is None


def test_parse_treats_a_blank_title_as_none():
    """
    Input: a .docx whose title is only whitespace.
    Output: the parsed Document's title.
    """
    raw_bytes = _build_docx_bytes(_paragraph("Body"), core_fields="<dc:title>   </dc:title>")

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.title is None


def test_parse_raises_on_bytes_that_are_not_a_zip():
    """
    Input: raw bytes that are not a .docx at all.
    Output: none; zipfile.BadZipFile is expected.
    """
    with pytest.raises(zipfile.BadZipFile):
        DocxParser().parse("docx_abc", b"not a docx file", "path")


def test_parse_raises_when_document_xml_is_missing():
    """
    Input: a valid zip with no word/document.xml.
    Output: none; KeyError is expected.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as docx_zip:
        docx_zip.writestr("something_else.xml", "<x/>")

    with pytest.raises(KeyError):
        DocxParser().parse("docx_abc", buffer.getvalue(), "path")


def test_parse_keeps_repeated_table_cells():
    """
    Input: a table whose cells repeat "Yes" three times.
    Output: the parsed Document's text.

    Checks that repeated lines are not removed as if they were page
    headers/footers, which a .docx body never contains.
    """
    table = "<w:tbl>"
    for question in ["Signed", "Dated", "Witnessed"]:
        table += "<w:tr><w:tc>" + _paragraph(question) + "</w:tc><w:tc>" + _paragraph("Yes") + "</w:tc></w:tr>"
    table += "</w:tbl>"
    raw_bytes = _build_docx_bytes(table)

    document = DocxParser().parse("docx_abc", raw_bytes, "path")

    assert document.text == "Signed\nYes\nDated\nYes\nWitnessed\nYes"
