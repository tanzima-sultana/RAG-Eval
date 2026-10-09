"""
test_cache.py

Tests for l2_chunking.cache: saving a document's chunks as one JSON file
and loading them back.
"""

import json
import os

from l2_chunking import cache
from l2_chunking.schema import Chunk


def make_chunks(doc_id, chunk_count):
    """
    Input: doc_id and chunk_count.
    Output: a dict of chunk_id to Chunk with chunk_count chunks for doc_id.
    What it does: builds chunks with ids "{doc_id}_0", "{doc_id}_1", ...
    """
    chunks = {}
    for chunk_index in range(chunk_count):
        chunk_id = f"{doc_id}_{chunk_index}"
        chunks[chunk_id] = Chunk(chunk_id, f"text {chunk_index}", 2, "fixed")
    return chunks


def test_chunk_json_dict_round_trip():
    """
    Input: a Chunk.
    Output: the Chunk after converting to a dict and back.
    What it does: checks no field is lost in the conversion.
    """
    chunk = Chunk("doc_0", "some text", 2, "fixed")

    round_tripped_chunk = cache.json_dict_to_chunk(cache.chunk_to_json_dict(chunk))

    assert round_tripped_chunk == chunk


def test_save_then_load_returns_the_same_chunks(tmp_path):
    """
    Input: three chunks for one document saved to a folder.
    Output: the chunks loaded back for that document.
    What it does: checks loading returns exactly what was saved.
    """
    chunk_path = str(tmp_path)
    chunks = make_chunks("doc", 3)

    cache._save_chunks(chunk_path, "doc", chunks)
    loaded_chunks = cache._load_cached_chunks(chunk_path, "doc")

    assert loaded_chunks == chunks


def test_load_keeps_saved_order_past_ten_chunks(tmp_path):
    """
    Input: twelve chunks for one document.
    Output: the chunk ids loaded back.
    What it does: checks "doc_10" and "doc_11" come after "doc_9", in the
    order the chunks were saved.
    """
    chunk_path = str(tmp_path)
    chunks = make_chunks("doc", 12)

    cache._save_chunks(chunk_path, "doc", chunks)
    loaded_chunks = cache._load_cached_chunks(chunk_path, "doc")

    assert list(loaded_chunks.keys()) == list(chunks.keys())


def test_save_writes_one_file_per_document_and_no_temporary_file(tmp_path):
    """
    Input: two chunks for one document, saved to a folder that does not exist yet.
    Output: the files in the folder.
    What it does: checks the folder is created and holds only "doc.json",
    with no leftover ".tmp" file.
    """
    chunk_path = os.path.join(str(tmp_path), "chunks", "fixed", "model", "2_6_2_0.5")

    cache._save_chunks(chunk_path, "doc", make_chunks("doc", 2))

    assert os.listdir(chunk_path) == ["doc.json"]


def test_save_and_load_document_with_no_chunks(tmp_path):
    """
    Input: an empty chunks dict saved for a document.
    Output: what loading that document returns.
    What it does: checks it returns an empty dict, not None, so the
    document still counts as cached.
    """
    chunk_path = str(tmp_path)

    cache._save_chunks(chunk_path, "doc", {})

    assert cache._load_cached_chunks(chunk_path, "doc") == {}


def test_load_returns_none_when_folder_is_missing(tmp_path):
    """
    Input: a chunk folder path that does not exist.
    Output: None.
    What it does: checks a missing folder means nothing is cached.
    """
    chunk_path = os.path.join(str(tmp_path), "missing")

    assert cache._load_cached_chunks(chunk_path, "doc") is None


def test_load_returns_none_when_document_has_no_file(tmp_path):
    """
    Input: a folder holding chunks for another document only.
    Output: None for the document with no file.
    What it does: checks another document's chunks are not returned.
    """
    chunk_path = str(tmp_path)
    cache._save_chunks(chunk_path, "other", make_chunks("other", 2))

    assert cache._load_cached_chunks(chunk_path, "doc") is None


def test_leftover_temporary_file_is_not_treated_as_cache(tmp_path):
    """
    Input: a folder with only a half-written "doc.json.tmp", as left by a
    crash during saving.
    Output: what loading "doc" returns.
    What it does: checks the temporary file is ignored and "doc" counts
    as not cached.
    """
    chunk_path = str(tmp_path)
    temporary_file = os.path.join(chunk_path, "doc.json.tmp")
    with open(temporary_file, "w", encoding="utf-8") as f:
        f.write('[{"id": "doc_0", "te')

    assert cache._load_cached_chunks(chunk_path, "doc") is None


def test_saved_file_is_a_json_list_of_chunks(tmp_path):
    """
    Input: two chunks saved for one document.
    Output: the raw content of "doc.json".
    What it does: checks the file holds a JSON list with one dict per chunk.
    """
    chunk_path = str(tmp_path)
    cache._save_chunks(chunk_path, "doc", make_chunks("doc", 2))

    with open(os.path.join(chunk_path, "doc.json"), encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data == [
        {"id": "doc_0", "text": "text 0", "size": 2, "type": "fixed"},
        {"id": "doc_1", "text": "text 1", "size": 2, "type": "fixed"},
    ]
