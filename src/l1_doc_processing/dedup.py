"""
dedup.py

Detects duplicate and near-duplicate documents. An exact duplicate has
the same content_hash as a document already seen. A near duplicate has
a shingle set (see common.identifiers.make_shingles) similar enough,
by Jaccard similarity, to one already seen, even when its content_hash
differs.

Near-duplicate detection uses MinHash + LSH (via the datasketch
library) instead of comparing shingle sets pairwise against every
previously-kept document. Each document's shingle set is summarized as
a fixed-size MinHash signature (see make_minhash), and an incrementally
built MinHashLSH index (see make_lsh_index) is queried for candidate
matches instead of scanning every previously-kept document one at a
time. This makes near-duplicate detection approximate/probabilistic
rather than exact: a pair at or above the index's threshold is very
likely, but not guaranteed, to be found, and the same holds in reverse
for a pair below it.
"""

import datasketch

NUM_PERM = 128


def make_minhash(shingles: set) -> datasketch.MinHash:
	"""
	shingles: a document's shingle set (see common.identifiers.make_shingles).

	Builds a MinHash signature summarizing shingles, using NUM_PERM hash
	permutations. A MinHash signature is a fixed-size approximation of a
	shingle set that preserves Jaccard similarity: comparing two
	signatures estimates the Jaccard similarity of the two original
	shingle sets without needing to store or scan either set directly.

	Returns the MinHash signature.
	"""
	minhash = datasketch.MinHash(num_perm=NUM_PERM)
	for shingle in shingles:
		shingle_bytes = shingle.encode("utf-8")
		minhash.update(shingle_bytes)
	return minhash


def make_lsh_index(threshold: float = 0.90) -> datasketch.MinHashLSH:
	"""
	threshold: the Jaccard similarity at or above which two documents
		are considered near-duplicates. Passed straight through to
		MinHashLSH, which picks banding parameters internally so that
		pairs at or above threshold are very likely to be found as
		candidates, and pairs below it are increasingly unlikely to be.

	Builds a new, empty MinHashLSH index configured for NUM_PERM-sized
	MinHash signatures and threshold. Callers insert each kept
	document's MinHash signature into this same index as it's kept, so
	later documents in the same run can be checked against everything
	kept so far.

	Returns the empty MinHashLSH index.
	"""
	return datasketch.MinHashLSH(threshold=threshold, num_perm=NUM_PERM)


def is_exact_duplicate(content_hash: str, seen_content_hashes: set[str]) -> bool:
	"""
	content_hash: the content_hash of the document being checked.
	seen_content_hashes: content_hash values of documents already kept.

	Returns True if content_hash was already seen (the text is an exact
	duplicate of a kept document), else False.
	"""
	return content_hash in seen_content_hashes


def is_near_duplicate(minhash: datasketch.MinHash, lsh_index: datasketch.MinHashLSH) -> bool:
	"""
	minhash: the MinHash signature (see make_minhash) of the document
		being checked.
	lsh_index: a MinHashLSH index (see make_lsh_index) already
		populated with every previously-kept document's MinHash
		signature in this run.

	Queries lsh_index for any previously-kept document whose MinHash
	signature LSH estimates to be at or above the index's configured
	similarity threshold. The caller is responsible for inserting
	minhash into lsh_index afterward, once the document is confirmed
	not to be a duplicate.

	Returns True if lsh_index returns any candidate match, else False.
	"""
	candidate_doc_ids = lsh_index.query(minhash)
	return len(candidate_doc_ids) > 0
