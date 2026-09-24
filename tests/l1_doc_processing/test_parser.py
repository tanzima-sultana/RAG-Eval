"""
test_parser.py

Tests for l1_doc_processing.parser: the Parser interface and the
get_parser registry lookup.
"""

import pytest

from l1_doc_processing.parser import Parser, get_parser
from l1_doc_processing.parsers.docx import DocxParser
from l1_doc_processing.parsers.pdf import PDFParser


def test_get_parser_returns_pdf_parser_class_for_pdf_doc_type():
    """
    Input: doc_type "pdf".
    Output: get_parser's return value.

    Checks that the registered parser class for "pdf" is PDFParser
    itself (a class, not an instance).
    """
    parser_cls = get_parser("pdf")

    assert parser_cls is PDFParser



def test_get_parser_returns_docx_parser_class_for_docx_doc_type():
    """
    Input: doc_type "docx".
    Output: get_parser's return value.
    """
    parser_cls = get_parser("docx")

    assert parser_cls is DocxParser


def test_get_parser_raises_value_error_for_unknown_doc_type():
    """
    Input: a doc_type with no registered parser.
    Output: none; get_parser is expected to raise.

    Checks that an unregistered doc_type fails loudly instead of
    silently returning nothing.
    """
    with pytest.raises(ValueError):
        get_parser("html")


def test_parser_cannot_be_instantiated_directly():
    """
    Input: none.
    Output: none; Parser() is expected to raise.

    Checks that Parser is an abstract base class: it declares parse as
    abstract, so it cannot be instantiated on its own.
    """
    with pytest.raises(TypeError):
        Parser()


def test_make_content_hash_is_the_sha256_of_the_text_and_depends_only_on_it():
    """
    Input: the same text hashed twice, and a different text.
    Output: make_content_hash's return values.
    """
    parser = PDFParser()

    first = parser.make_content_hash("hello world")
    second = parser.make_content_hash("hello world")
    other = parser.make_content_hash("hello there")

    assert first == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert first == second
    assert first != other
