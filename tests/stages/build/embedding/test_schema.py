"""
test_schema.py

Tests for stages.build.embedding.schema.Embedded, the plain dataclass one
embedding vector is held in.
"""

import dataclasses

import numpy as np
import pytest

from stages.build.embedding.schema import Embedded


def test_embedded_holds_every_field_it_is_constructed_with():
    """
    Input: one value per Embedded field.
    Output: the constructed Embedded.

    Checks that Embedded is a plain data container: every field is
    stored as-is and readable back by attribute name.
    """
    embedding_vector = np.array([1.0, 2.0, 3.0])
    embedded = Embedded(chunk_id="doc1_0", embeddings=embedding_vector, normalized=True)

    assert embedded.chunk_id == "doc1_0"
    assert np.array_equal(embedded.embeddings, embedding_vector)
    assert embedded.normalized is True


def test_embedded_converts_to_a_plain_dict_via_dataclasses_asdict():
    """
    Input: a fully-populated Embedded.
    Output: dataclasses.asdict(embedded).

    Checks that Embedded round-trips through dataclasses.asdict as a
    plain dict, with embeddings still an ndarray (not a list), which
    Embedding.embedding_to_json_dict relies on before separately
    converting embeddings to a list for JSON.
    """
    embedding_vector = np.array([1.0, 2.0])
    embedded = Embedded(chunk_id="doc1_0", embeddings=embedding_vector, normalized=False)

    data = dataclasses.asdict(embedded)

    assert data["chunk_id"] == "doc1_0"
    assert np.array_equal(data["embeddings"], embedding_vector)
    assert data["normalized"] is False


def test_embedded_equality_raises_when_embeddings_holds_more_than_one_value():
    """
    Input: two Embedded instances with identical multi-value
    embeddings arrays.
    Output: none; == is expected to raise ValueError.

    dataclasses' generated __eq__ compares field tuples with ==, and
    numpy raises ValueError ("truth value of an array with more than
    one element is ambiguous") once a multi-value array ends up on
    either side of that comparison. This documents that Embedded
    instances must be compared field-by-field (chunk_id, an explicit
    np.array_equal/allclose on embeddings, normalized) rather than
    with a bare ==, whenever embeddings holds more than one value --
    exactly how the rest of this test suite compares them.
    """
    first = Embedded(chunk_id="doc1_0", embeddings=np.array([1.0, 2.0]), normalized=True)
    second = Embedded(chunk_id="doc1_0", embeddings=np.array([1.0, 2.0]), normalized=True)

    with pytest.raises(ValueError):
        first == second
