"""
docx.py

Implements Parser for Word .docx documents. A .docx file is a zip of
XML files, so it is read with the standard library only (zipfile and
xml.etree): body text from word/document.xml, title and author from
docProps/core.xml, and page count from docProps/app.xml. Headers,
footers, footnotes and comments are not extracted.
"""

import datetime
import io
import xml.etree.ElementTree as ElementTree
import zipfile

from common.identifiers import make_content_hash
from l1_doc_processing.metadata import extract_metadata
from l1_doc_processing.parser import Parser
from l1_doc_processing.schema import Document
from l1_doc_processing.text_cleanup import clean_text


_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MARKUP_COMPATIBILITY_NAMESPACE = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_DUBLIN_CORE_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"
_EXTENDED_PROPERTIES_NAMESPACE = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"

_PARAGRAPH_TAG = _WORD_NAMESPACE + "p"
_TEXT_TAG = _WORD_NAMESPACE + "t"
_TAB_TAG = _WORD_NAMESPACE + "tab"
_BREAK_TAG = _WORD_NAMESPACE + "br"
_CARRIAGE_RETURN_TAG = _WORD_NAMESPACE + "cr"
# Word stores some content (e.g. text boxes) twice: a Choice version and
# a Fallback copy for older readers. The Fallback copy is skipped so
# that text is not extracted twice.
_FALLBACK_TAG = _MARKUP_COMPATIBILITY_NAMESPACE + "Fallback"


def _append_run_text(element: ElementTree.Element, text_parts: list[str]) -> None:
	"""
	element: an XML element inside a paragraph.
	text_parts: list the found text pieces are appended to.

	Walks element's children in document order, appending text (w:t),
	tabs (w:tab) and line breaks (w:br, w:cr). A paragraph nested inside
	(e.g. in a text box) starts on a new line.

	Returns nothing.
	"""
	for child in element:
		if child.tag == _FALLBACK_TAG:
			continue
		if child.tag == _TEXT_TAG:
			if child.text is not None:
				text_parts.append(child.text)
		elif child.tag == _TAB_TAG:
			text_parts.append("\t")
		elif child.tag == _BREAK_TAG or child.tag == _CARRIAGE_RETURN_TAG:
			text_parts.append("\n")
		elif child.tag == _PARAGRAPH_TAG:
			text_parts.append("\n")
			_append_run_text(child, text_parts)
		else:
			_append_run_text(child, text_parts)


def _collect_paragraph_texts(element: ElementTree.Element, paragraph_texts: list[str]) -> None:
	"""
	element: an XML element of word/document.xml.
	paragraph_texts: list each paragraph's text is appended to.

	Finds every top-level paragraph under element in document order,
	including paragraphs inside table cells, and appends its text.

	Returns nothing.
	"""
	for child in element:
		if child.tag == _FALLBACK_TAG:
			continue
		if child.tag == _PARAGRAPH_TAG:
			text_parts = []
			_append_run_text(child, text_parts)
			paragraph_texts.append("".join(text_parts))
		else:
			_collect_paragraph_texts(child, paragraph_texts)


def _extract_text(docx_zip: zipfile.ZipFile) -> str:
	"""
	docx_zip: the .docx file opened as a zip.

	Returns the body text, one line per paragraph.
	"""
	document_xml = docx_zip.read("word/document.xml")
	root = ElementTree.fromstring(document_xml)

	paragraph_texts = []
	_collect_paragraph_texts(root, paragraph_texts)
	return "\n".join(paragraph_texts)


def _read_optional_xml(docx_zip: zipfile.ZipFile, member_name: str) -> ElementTree.Element | None:
	"""
	docx_zip: the .docx file opened as a zip.
	member_name: path of an XML file inside the zip.

	Returns the parsed XML root, or None if the file is not in the zip.
	"""
	if member_name not in docx_zip.namelist():
		return None
	return ElementTree.fromstring(docx_zip.read(member_name))


def _find_non_blank_text(root: ElementTree.Element | None, tag: str) -> str | None:
	"""
	root: a parsed XML root, or None.
	tag: the tag of the element to read.

	Returns the element's stripped text, or None if root is None, the
	element is missing, or its text is blank.
	"""
	if root is None:
		return None
	element = root.find(tag)
	if element is None:
		return None
	if element.text is None:
		return None
	text = element.text.strip()
	if text == "":
		return None
	return text


def _extract_title(docx_zip: zipfile.ZipFile) -> str | None:
	"""
	docx_zip: the .docx file opened as a zip.

	Returns the title from docProps/core.xml, or None if not set.
	"""
	core_root = _read_optional_xml(docx_zip, "docProps/core.xml")
	return _find_non_blank_text(core_root, _DUBLIN_CORE_NAMESPACE + "title")


def _extract_docx_metadata(docx_zip: zipfile.ZipFile) -> dict:
	"""
	docx_zip: the .docx file opened as a zip.

	Reads author from docProps/core.xml and page count from
	docProps/app.xml. The page count is the value Word saved last time,
	so it can be missing or out of date; missing values are None.

	Returns a dict with "page_count" and "author".
	"""
	core_root = _read_optional_xml(docx_zip, "docProps/core.xml")
	author = _find_non_blank_text(core_root, _DUBLIN_CORE_NAMESPACE + "creator")

	app_root = _read_optional_xml(docx_zip, "docProps/app.xml")
	page_count_text = _find_non_blank_text(app_root, _EXTENDED_PROPERTIES_NAMESPACE + "Pages")
	page_count = None
	if page_count_text is not None:
		if page_count_text.isdigit():
			page_count = int(page_count_text)

	return {"page_count": page_count, "author": author}


class DocxParser(Parser):
	def parse(self, doc_id: str, raw_bytes: bytes, data_path: str) -> Document:
		"""
		doc_id: this document's identifier, already computed by the caller.
		raw_bytes: the .docx file's raw bytes.
		data_path: the original raw data path of the document.

		Extracts and cleans the body text, and reads title and metadata.
		Raises zipfile.BadZipFile or KeyError if raw_bytes is not a
		valid .docx file.

		Returns a fully-populated Document.
		"""
		docx_zip = zipfile.ZipFile(io.BytesIO(raw_bytes))

		with docx_zip:
			text = _extract_text(docx_zip)

			# Word keeps headers/footers outside the body text, so there are
			# no repeated page headers to strip. Stripping repeated lines
			# here would only delete real content, such as repeated table
			# cells ("Yes", "N/A") in forms.
			text = clean_text(text, strip_headers_footers=False)

			title = _extract_title(docx_zip)

			# Metadata: generic fields (from raw_bytes), then docx-specific fields on top
			metadata = extract_metadata(raw_bytes, "docx")
			metadata.update(_extract_docx_metadata(docx_zip))

		content_hash = make_content_hash(text)

		timestamp = datetime.datetime.now(datetime.timezone.utc)

		return Document(
			id=doc_id,
			text=text,
			title=title,
			path=data_path,
			format="docx",
			metadata=metadata,
			content_hash=content_hash,
			timestamp=timestamp,
		)
