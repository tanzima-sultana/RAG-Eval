"""
test_schema.py

Tests for l2_chunking.schema.Chunk, the plain dataclass every
chunking strategy produces.
"""

import dataclasses

from l2_chunking.schema import Chunk


def test_chunk_holds_every_field_it_is_constructed_with():
    """
    Input: one value per Chunk field.
    Output: the constructed Chunk.

    Checks that Chunk is a plain data container: every field is
    stored as-is and readable back by attribute name.
    """
    chunk = Chunk(id="doc1_0", text="hello world", size=2, type="fixed")

    assert chunk.id == "doc1_0"
    assert chunk.text == "hello world"
    assert chunk.size == 2
    assert chunk.type == "fixed"


def test_chunk_converts_to_a_plain_dict_via_dataclasses_asdict():
    """
    Input: a fully-populated Chunk.
    Output: dataclasses.asdict(chunk).

    Checks that Chunk round-trips through dataclasses.asdict as a
    plain dict, which chunk_to_json_dict relies on.
    """
    chunk = Chunk(id="doc1_0", text="hello world", size=2, type="fixed")

    data = dataclasses.asdict(chunk)

    assert data == {"id": "doc1_0", "text": "hello world", "size": 2, "type": "fixed"}
