"""
parser.py

Defines the Parser interface every format implementation must follow,
plus the registry mapping format name -> implementation and the lookup
function used by doc_processing.py.

Concrete parsers (e.g. PDFParser) subclass Parser, so they import this
module. Importing them back at the top of this module would create a
circular import, since this module would then depend on them too — so
the registry is built lazily inside get_parser instead, once Parser has
already finished being defined.
"""

import abc

from l1_doc_processing.schema import Document


class Parser(abc.ABC):
	@abc.abstractmethod
	def parse(self, doc_id: str, raw_bytes: bytes, data_path: str) -> Document:
		"""
		doc_id: this document's identifier, already computed by the
			 caller (doc_processing.py) via
			 common.identifiers.make_doc_id(doc_type, raw_bytes) before
			 any cache check. Implementations must set Document.id to
			 this value directly rather than recomputing it.
		raw_bytes: the document's raw, undecoded content.
		data_path: the original raw data path of the document.

		Returns a fully-populated Document.
		"""
		raise NotImplementedError


def get_parser(doc_type: str) -> type[Parser]:
	# imported here, not at module level, to avoid the circular import
	# described above
	from l1_doc_processing.parsers.pdf import PDFParser

	parser_registry: dict[str, type[Parser]] = {
		"pdf": PDFParser,
	}

	if doc_type not in parser_registry:
		raise ValueError(
			f"Unknown doc_type '{doc_type}'. Registered types: {list(parser_registry.keys())}"
		)
	return parser_registry[doc_type]