

import hashlib


def make_doc_id(doc_type: str, raw_bytes: bytes) -> str:
	"""
	doc_type: the document format being processed (e.g. "pdf"), used to
		prefix doc_id so it stays readable and distinguishable at a
		glance from a different format's identifiers.
	raw_bytes: a document's raw, unparsed content — e.g. a PDF's raw
		bytes decoded from base64, or a row's file read straight off
		disk — before any format-specific parsing happens.

	Combines doc_type with a truncated SHA-256 hash of raw_bytes into a
	single, filesystem-safe identifier: purely content-addressed, so
	the same raw content always produces the same doc_id regardless of
	which file name, row, or dataset it came from. The hash is
	truncated to 16 hex characters (64 bits) to keep doc_id and
	filenames built from it compact; raw caching and processed-document
	caching are already namespaced per (doc_type, dataset_name)
	directory (see common.paths), so collision risk within one such
	directory stays negligible at realistic dataset sizes.

	Returns the "{doc_type}_{16-hex-char hash}" doc_id string.
	"""
	digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
	return f"{doc_type}_{digest}"


def make_content_hash(text: str) -> str:
	"""
	text: a document's extracted text content.

	Hashes text with SHA-256, so two documents can be compared for
	exact content equality (e.g. for deduplication) independent of
	doc_id, file_name, or source location — only the text itself
	matters.

	Returns the hex-encoded SHA-256 digest.
	"""
	digest = hashlib.sha256(text.encode("utf-8"))
	return digest.hexdigest()


_SHINGLE_SIZE = 5


def make_shingles(text: str) -> set[str]:
	"""
	text: a document's extracted text content.

	Splits text into whitespace-separated words and builds the set of
	overlapping word n-grams (shingles) of length _SHINGLE_SIZE. Used
	for near-duplicate detection via Jaccard similarity (see
	l1_doc_processing.dedup), which two documents can share even when
	their content_hash values differ.

	Returns the set of shingle strings.
	"""
	words = text.split()

	shingles = set()
	index = 0
	while index + _SHINGLE_SIZE <= len(words):
		shingle_words = words[index:index + _SHINGLE_SIZE]
		shingle = " ".join(shingle_words)
		shingles.add(shingle)
		index += 1

	return shingles
