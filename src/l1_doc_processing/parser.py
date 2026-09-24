"""
parser.py

Defines the Parser interface every format implementation must follow,
including the content hash every parser computes, plus the registry
mapping format name -> implementation and the lookup function used by
doc_processing.py.

Concrete parsers (e.g. PDFParser) subclass Parser, so they import this
module. Importing them back at the top of this module would create a
circular import, since this module would then depend on them too — so
the registry is built lazily inside get_parser instead, once Parser has
already finished being defined.
"""

import abc
import hashlib
import time 

from l1_doc_processing.schema import Document


class Parser(abc.ABC):
	def make_content_hash(self, text: str) -> str:
		"""
		text: a document's extracted text content.

		Hashes text with SHA-256, so two documents can be compared for
		exact content equality (e.g. for deduplication) independent of
		doc_id, file name or source location.

		Returns the hex-encoded SHA-256 digest.
		"""
		digest = hashlib.sha256(text.encode("utf-8"))
		return digest.hexdigest()

	@abc.abstractmethod
	def parse(self, doc_id: str, raw_bytes: bytes, data_path: str) -> Document:
		"""
		doc_id: this document's identifier, already computed from the raw
			 bytes by RawSource.make_doc_id. Implementations must set
			 Document.id to this value directly rather than recomputing it.
		raw_bytes: the document's raw, undecoded content.
		data_path: the original raw data path of the document.

		Returns a fully-populated Document.
		"""
		raise NotImplementedError


def get_parser(doc_type: str) -> type[Parser]:
	# imported here, not at module level, to avoid the circular import
	# described above
	from l1_doc_processing.parsers.pdf import PDFParser
	from l1_doc_processing.parsers.docx import DocxParser

	parser_registry: dict[str, type[Parser]] = {
		"pdf": PDFParser,
		"docx": DocxParser,
	}

	if doc_type not in parser_registry:
		raise ValueError(
			f"Unknown doc_type '{doc_type}'. Registered types: {list(parser_registry.keys())}"
		)
	return parser_registry[doc_type]

def parse_document_from_file(doc_type: str, doc_id: str, source_path: str) -> tuple:
	"""
	doc_type: document type (e.g. "docx").
	doc_id: document identifier.
	source_path: path of the raw file.

	Reads and parses the raw file in a parse worker process, and times
	the parse. Parse errors are returned, not raised.

	Returns (doc_id, document, parse_seconds, error_message). On
	failure, document and parse_seconds are None.
	"""
	parser_cls = get_parser(doc_type)
	parser = parser_cls()

	with open(source_path, "rb") as f:
		raw_bytes = f.read()

	parse_start = time.perf_counter()
	try:
		document = parser.parse(doc_id, raw_bytes, source_path)
	except Exception as error:
		error_message = f"{type(error).__name__}: {error}"
		return (doc_id, None, None, error_message)
	parse_seconds = time.perf_counter() - parse_start

	return (doc_id, document, parse_seconds, None)