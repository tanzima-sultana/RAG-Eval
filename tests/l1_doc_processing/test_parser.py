"""
test_parser.py

Tests for l1_doc_processing.parser: the Parser interface and the
get_parser registry lookup.
"""

import pytest

from l1_doc_processing.parser import Parser, get_parser
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


def test_get_parser_raises_value_error_for_unknown_doc_type():
    """
    Input: a doc_type with no registered parser.
    Output: none; get_parser is expected to raise.

    Checks that an unregistered doc_type fails loudly instead of
    silently returning nothing.
    """
    with pytest.raises(ValueError):
        get_parser("docx")


def test_parser_cannot_be_instantiated_directly():
    """
    Input: none.
    Output: none; Parser() is expected to raise.

    Checks that Parser is an abstract base class: it declares parse as
    abstract, so it cannot be instantiated on its own.
    """
    with pytest.raises(TypeError):
        Parser()
