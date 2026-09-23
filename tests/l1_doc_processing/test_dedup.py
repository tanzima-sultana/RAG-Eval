"""
test_dedup.py

Tests for l1_doc_processing.dedup: exact-duplicate detection via
content_hash, and near-duplicate detection via MinHash + LSH.
"""

import datetime

from common.identifiers import make_content_hash
from l1_doc_processing.dedup import Deduplicator
from l1_doc_processing.schema import Document


def test_is_exact_duplicate_true_when_content_hash_already_seen():
    """
    Input: a content_hash that is already in deduplicator.seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks that a content_hash present among the already-seen values is
    reported as an exact duplicate.
    """
    deduplicator = Deduplicator()
    deduplicator.seen_content_hashes = {"hash-a", "hash-b"}

    result = deduplicator.is_exact_duplicate("hash-a")

    assert result is True


def test_is_exact_duplicate_false_when_content_hash_not_seen():
    """
    Input: a content_hash that is not in deduplicator.seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks that a content_hash absent from the already-seen values is
    not reported as an exact duplicate.
    """
    deduplicator = Deduplicator()
    deduplicator.seen_content_hashes = {"hash-a", "hash-b"}

    result = deduplicator.is_exact_duplicate("hash-c")

    assert result is False


def test_is_exact_duplicate_false_when_nothing_seen_yet():
    """
    Input: any content_hash, and an empty deduplicator.seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks the first-document case, where nothing has been seen yet.
    """
    deduplicator = Deduplicator()

    result = deduplicator.is_exact_duplicate("hash-a")

    assert result is False


def test_is_near_duplicate_true_for_identical_shingle_sets():
    """
    Input: two MinHash signatures built from the exact same shingle
    set, with the second inserted into an LSH index and the first
    queried against it.
    Output: is_near_duplicate's return value.

    Checks that two documents with identical shingle sets (Jaccard
    similarity 1.0) are found as near-duplicates, well above the
    default 0.90 threshold.
    """
    shingles = set()
    shingles.add("the quick brown fox jumps")
    shingles.add("quick brown fox jumps over")
    shingles.add("brown fox jumps over the")

    deduplicator = Deduplicator(threshold=0.90)
    deduplicator.lsh_index.insert("already-seen-doc", deduplicator.make_minhash(shingles))

    result = deduplicator.is_near_duplicate(deduplicator.make_minhash(shingles))

    assert result is True


def test_is_near_duplicate_false_for_disjoint_shingle_sets():
    """
    Input: two MinHash signatures built from completely disjoint
    shingle sets, with one inserted into an LSH index and the other
    queried against it.
    Output: is_near_duplicate's return value.

    Checks that two documents sharing no shingles (Jaccard similarity
    0.0) are not found as near-duplicates.
    """
    shingles_a = set()
    shingles_a.add("alpha beta gamma delta epsilon")

    shingles_b = set()
    shingles_b.add("zulu yankee xray whiskey victor")

    deduplicator = Deduplicator(threshold=0.90)
    deduplicator.lsh_index.insert("already-seen-doc", deduplicator.make_minhash(shingles_a))

    result = deduplicator.is_near_duplicate(deduplicator.make_minhash(shingles_b))

    assert result is False


def test_is_near_duplicate_false_when_index_is_empty():
    """
    Input: a MinHash signature queried against a freshly built, empty
    LSH index.
    Output: is_near_duplicate's return value.

    Checks the first-document case, where the index has nothing in it
    yet to match against.
    """
    shingles = set()
    shingles.add("some content that has never been seen before")

    deduplicator = Deduplicator(threshold=0.90)

    result = deduplicator.is_near_duplicate(deduplicator.make_minhash(shingles))

    assert result is False


# ---------- Deduplicator ----------


def _make_document(text):
    """
    Returns a Document whose text is text and content_hash is its hash.
    """
    return Document(
        id="unused",
        text=text,
        title=None,
        path="path",
        format="pdf",
        metadata={},
        content_hash=make_content_hash(text),
        timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


_LONG_TEXT = "the quick brown fox jumps over the lazy dog again and again every single day of the week"


def test_deduplicator_accepts_a_new_document_and_rejects_its_exact_copy():
    """
    Input: the same Document text added twice with add_if_new.
    Output: add_if_new's return values.
    """
    deduplicator = Deduplicator()

    assert deduplicator.add_if_new("doc-1", _make_document(_LONG_TEXT)) is True
    assert deduplicator.add_if_new("doc-2", _make_document(_LONG_TEXT)) is False


def _sixty_word_text(last_word):
    """
    Returns a 60-word text of distinct words that ends with last_word.
    """
    words = []
    for index in range(59):
        words.append(f"word{index}")
    words.append(last_word)
    return " ".join(words)


def test_deduplicator_rejects_a_near_duplicate():
    """
    Input: a 60-word Document, then one that differs only in its last
    word (about 0.96 shingle similarity, above the 0.90 threshold).
    Output: add_if_new's return value for the second one.
    """
    deduplicator = Deduplicator()
    deduplicator.add_if_new("doc-1", _make_document(_sixty_word_text("end")))

    assert deduplicator.add_if_new("doc-2", _make_document(_sixty_word_text("finish"))) is False


def test_deduplicator_accepts_different_documents():
    """
    Input: two Documents with unrelated text.
    Output: add_if_new's return values.
    """
    deduplicator = Deduplicator()

    assert deduplicator.add_if_new("doc-1", _make_document(_LONG_TEXT)) is True
    assert deduplicator.add_if_new("doc-2", _make_document("a completely different sentence about contracts and payment terms here")) is True


def test_deduplicator_add_records_a_document_for_later_checks():
    """
    Input: a Document recorded with add (as for cached documents), then
    the same text checked with add_if_new.
    Output: add_if_new's return value.
    """
    deduplicator = Deduplicator()
    deduplicator.add("doc-1", _make_document(_LONG_TEXT))

    assert deduplicator.add_if_new("doc-2", _make_document(_LONG_TEXT)) is False


def test_deduplicator_accepts_short_text_without_shingles():
    """
    Input: two different Documents too short to build shingles.
    Output: add_if_new's return values.
    """
    deduplicator = Deduplicator()

    assert deduplicator.add_if_new("doc-1", _make_document("alpha")) is True
    assert deduplicator.add_if_new("doc-2", _make_document("beta")) is True
