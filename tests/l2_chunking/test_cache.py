"""
test_cache.py

Tests for l2_chunking.cache: converting Chunks to and from JSON, and
saving/loading a document's chunk files.
"""

from l2_chunking.cache import _load_cached_chunks, _save_chunks, chunk_to_json_dict, json_dict_to_chunk
from l2_chunking.schema import Chunk


def test_chunk_to_json_dict_and_back_round_trips():
    """
    Input: a fully-populated Chunk.
    Output: chunk_to_json_dict then json_dict_to_chunk, applied in sequence.
    """
    chunk = Chunk(id="doc1_0", text="hello world", size=2, type="fixed")

    data = chunk_to_json_dict(chunk)
    restored = json_dict_to_chunk(data)

    assert restored == chunk


def test_save_and_load_chunks_round_trips(isolated_data_dir):
    """
    Input: two Chunks of the same doc_id, saved in one _save_chunks call.
    Output: _load_cached_chunks's return value for that doc_id.
    """
    chunk_path = str(isolated_data_dir / "chunks_out")
    chunk_a = Chunk(id="doc1_0", text="a", size=1, type="fixed")
    chunk_b = Chunk(id="doc1_1", text="b", size=1, type="fixed")
    _save_chunks(chunk_path, {"doc1_0": chunk_a, "doc1_1": chunk_b})

    loaded = _load_cached_chunks(chunk_path, "doc1")

    assert loaded == {"doc1_0": chunk_a, "doc1_1": chunk_b}


def test_load_cached_chunks_returns_none_when_chunk_path_does_not_exist(isolated_data_dir):
    """
    Input: a chunk_path that has never been created.
    Output: _load_cached_chunks's return value.
    """
    chunk_path = str(isolated_data_dir / "never_created")

    assert _load_cached_chunks(chunk_path, "doc1") is None


def test_load_cached_chunks_returns_none_when_doc_id_has_no_cached_files(isolated_data_dir):
    """
    Input: a chunk_path holding another doc_id's chunks only.
    Output: _load_cached_chunks's return value for a different doc_id.
    """
    chunk_path = str(isolated_data_dir / "chunks_out")
    _save_chunks(chunk_path, {"doc1_0": Chunk(id="doc1_0", text="a", size=1, type="fixed")})

    assert _load_cached_chunks(chunk_path, "doc2") is None
