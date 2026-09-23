"""
test_dedup.py

Tests for l1_doc_processing.dedup: exact-duplicate detection via
content_hash, and near-duplicate detection via MinHash + LSH.
"""

from l1_doc_processing.dedup import (
    is_exact_duplicate,
    is_near_duplicate,
    make_lsh_index,
    make_minhash,
)


def test_is_exact_duplicate_true_when_content_hash_already_seen():
    """
    Input: a content_hash that is already in seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks that a content_hash present among the already-seen values is
    reported as an exact duplicate.
    """
    seen_content_hashes = {"hash-a", "hash-b"}

    result = is_exact_duplicate("hash-a", seen_content_hashes)

    assert result is True


def test_is_exact_duplicate_false_when_content_hash_not_seen():
    """
    Input: a content_hash that is not in seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks that a content_hash absent from the already-seen values is
    not reported as an exact duplicate.
    """
    seen_content_hashes = {"hash-a", "hash-b"}

    result = is_exact_duplicate("hash-c", seen_content_hashes)

    assert result is False


def test_is_exact_duplicate_false_when_nothing_seen_yet():
    """
    Input: any content_hash, and an empty seen_content_hashes.
    Output: is_exact_duplicate's return value.

    Checks the first-document case, where nothing has been seen yet.
    """
    result = is_exact_duplicate("hash-a", set())

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

    lsh_index = make_lsh_index(threshold=0.90)
    lsh_index.insert("already-seen-doc", make_minhash(shingles))

    result = is_near_duplicate(make_minhash(shingles), lsh_index)

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

    lsh_index = make_lsh_index(threshold=0.90)
    lsh_index.insert("already-seen-doc", make_minhash(shingles_a))

    result = is_near_duplicate(make_minhash(shingles_b), lsh_index)

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

    lsh_index = make_lsh_index(threshold=0.90)

    result = is_near_duplicate(make_minhash(shingles), lsh_index)

    assert result is False
