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
built MinHashLSH index (see Deduplicator.make_lsh_index) is queried for candidate
matches instead of scanning every previously-kept document one at a
time. This makes near-duplicate detection approximate/probabilistic
rather than exact: a pair at or above the index's threshold is very
likely, but not guaranteed, to be found, and the same holds in reverse
for a pair below it.
"""

import datasketch

from common.identifiers import make_shingles
from l1_doc_processing.schema import Document

NUM_PERM = 128


class Deduplicator:
	"""
	Holds the content hashes and MinHash LSH index of every document kept
	so far in a run, so each new document can be checked against all of
	them.
	"""

	def __init__(self, threshold: float = 0.90):
		"""
		threshold: Jaccard similarity at or above which two documents are
			near-duplicates (see make_lsh_index).

		Starts with no seen content hashes and an empty LSH index.

		Returns nothing.
		"""
		self.seen_content_hashes = set()
		self.lsh_index = self.make_lsh_index(threshold)

	def make_minhash(self, shingles: set) -> datasketch.MinHash:
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

	def make_lsh_index(self, threshold: float) -> datasketch.MinHashLSH:
		"""
		threshold: the Jaccard similarity at or above which two documents
			are considered near-duplicates. Passed straight through to
			MinHashLSH, which picks banding parameters internally so that
			pairs at or above threshold are very likely to be found as
			candidates, and pairs below it are increasingly unlikely to be.

		Returns a new, empty MinHashLSH index configured for NUM_PERM-sized
		MinHash signatures and threshold.
		"""
		return datasketch.MinHashLSH(threshold=threshold, num_perm=NUM_PERM)

	def is_exact_duplicate(self, content_hash: str) -> bool:
		"""
		content_hash: the content_hash of the document being checked.

		Returns True if content_hash was already seen (the text is an exact
		duplicate of a kept document), else False.
		"""
		return content_hash in self.seen_content_hashes

	def is_near_duplicate(self, minhash: datasketch.MinHash) -> bool:
		"""
		minhash: the MinHash signature (see make_minhash) of the document
			being checked.

		Queries the LSH index for any kept document whose MinHash signature
		LSH estimates to be at or above the configured similarity
		threshold. Does not insert minhash into the index.

		Returns True if the index returns any candidate match, else False.
		"""
		candidate_doc_ids = self.lsh_index.query(minhash)
		return len(candidate_doc_ids) > 0

	def add(self, doc_id: str, document: Document) -> None:
		"""
		doc_id: id of a document already known to be kept.
		document: that Document.

		Records the document's content hash and MinHash without checking
		it. Used for documents loaded from disk.

		Returns nothing.
		"""
		self.seen_content_hashes.add(document.content_hash)
		shingles = make_shingles(document.text)
		if len(shingles) > 0:
			self.lsh_index.insert(doc_id, self.make_minhash(shingles))

	def add_if_new(self, doc_id: str, document: Document) -> bool:
		"""
		doc_id: id of a newly parsed document.
		document: that Document.

		Checks the document against every kept document. If it is not an
		exact or near duplicate, records it as kept.

		Returns True if the document was new and recorded, or False if it
		is a duplicate.
		"""
		if self.is_exact_duplicate(document.content_hash):
			return False

		shingles = make_shingles(document.text)
		minhash = None
		if len(shingles) > 0:
			minhash = self.make_minhash(shingles)
			if self.is_near_duplicate(minhash):
				return False

		self.seen_content_hashes.add(document.content_hash)
		if minhash is not None:
			self.lsh_index.insert(doc_id, minhash)
		return True
