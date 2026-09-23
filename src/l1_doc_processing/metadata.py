"""
metadata.py

Builds the generic part of Document.metadata directly from a
document's raw bytes. Format-agnostic: works the same way for any
doc_type, since it only looks at the raw bytes themselves, never at
parsed content or any dataset-specific row shape.
"""


def extract_metadata(raw_bytes: bytes, doc_type: str) -> dict:
	"""
	raw_bytes: the document's raw, undecoded content.
	doc_type: the document format being processed (e.g. "pdf"), used to
		record which parser this document was handled by.

	Builds the metadata fields derivable from raw_bytes alone, with no
	knowledge of any particular format: doc_type as "format", and the
	raw content's size in bytes.

	Returns the metadata dict.
	"""
	metadata = {}
	metadata["format"] = doc_type
	metadata["raw_bytes_size"] = len(raw_bytes)
	return metadata
