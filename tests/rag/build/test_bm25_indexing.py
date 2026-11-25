"""
test_bm25_indexing.py

Tests for rag.build.indexing.bm25_indexing. BM25 is real.
"""

import os

import pytest

from common import paths
from rag.build.indexing.bm25_indexing import Bm25Indexing
from rag.build.indexing.bm25_indexing import load_bm25_index_from_path
from rag.build.indexing.bm25_indexing import load_saved_bm25_index
from rag.build.indexing.bm25_indexing import lowercase_whitespace_tokenize
from schema.index import Bm25Index

CHUNKING_HASH = "testchunkinghash"


@pytest.fixture
def make_bm25_indexing(make_bm25_indexing_args):
    """
    Input: the make_bm25_indexing_args fixture.
    Output: a function that builds a Bm25Indexing from keyword overrides of the BM25 args.
    What it does: returns a builder that creates Bm25Indexing with a fixed chunking hash.
    """
    def build_bm25_indexing(**overrides):
        """
        Input: any Bm25IndexingArgs field as a keyword argument.
        Output: a Bm25Indexing with the default BM25 args plus the overrides.
        What it does: creates the Bm25Indexing.
        """
        return Bm25Indexing(make_bm25_indexing_args(**overrides), CHUNKING_HASH)

    return build_bm25_indexing


def test_lowercase_whitespace_tokenize():
    """
    Input: text with upper case, tabs and new lines.
    Output: the words.
    What it does: checks text is lowercased and split on whitespace.
    """
    assert lowercase_whitespace_tokenize("The Cat\tsat\non  MAT") == ["the", "cat", "sat", "on", "mat"]


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"bm25_k1": None}, "bm25_k1 is missing"),
    ({"bm25_b": None}, "bm25_b is missing"),
    ({"bm25_tokenizer": "nltk"}, "bm25_tokenizer"),
])
def test_init_raises_for_invalid_or_missing_setting(make_bm25_indexing, overrides, message):
    """
    Input: Bm25IndexingArgs with one setting missing or not supported.
    Output: ValueError.
    What it does: checks Bm25Indexing validates the settings when it is created, and names the setting.
    """
    with pytest.raises(ValueError, match=message):
        make_bm25_indexing(**overrides)


# ---------- build and save ----------


def test_bm25_index_builds_and_saves(make_bm25_indexing, make_bm25_indexing_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, with bm25_k1 1.2 and bm25_b 0.5.
    Output: the Bm25Index, its hash, its path and its saved file.
    What it does: checks every chunk is in the retriever, k1 and b are used, the hash and path are the ones get_bm25_hash
    and get_bm25_index_path give, and the retriever is saved in that folder.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    bm25_args = make_bm25_indexing_args(bm25_k1=1.2, bm25_b=0.5)

    bm25_index, bm25_hash, bm25_index_path = Bm25Indexing(bm25_args, CHUNKING_HASH).bm25_indexing(chunks, str(tmp_path))

    assert len(bm25_index.retriever.docs) == 10
    assert bm25_index.retriever.vectorizer.k1 == 1.2
    assert bm25_index.retriever.vectorizer.b == 0.5
    expected_hash = paths.get_bm25_hash(bm25_args, CHUNKING_HASH)
    expected_path = paths.get_bm25_index_path(str(tmp_path), expected_hash)
    assert bm25_hash == expected_hash
    assert bm25_index_path == expected_path
    assert bm25_index.index_path == os.path.join(expected_path, "bm25_retriever.pkl")
    assert os.path.exists(bm25_index.index_path)


def test_bm25_search_finds_the_chunk_with_the_word(make_bm25_indexing, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, searched with "WORD3".
    Output: the top hit.
    What it does: checks the query is lowercased like the chunks and the chunk with that word comes first.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    bm25_index, bm25_hash, bm25_index_path = make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))

    documents = bm25_index.retriever.invoke("WORD3")

    assert documents[0].metadata["chunk_id"] == "doc1_3"


# ---------- cache ----------


def test_bm25_index_loads_from_cache_on_second_run(make_bm25_indexing, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: the same chunks indexed twice.
    Output: what the second run printed.
    What it does: checks the second run loads the saved retriever and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))
    capsys.readouterr()

    make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))

    printed = capsys.readouterr().out
    assert "BM25 index loaded from cache : 10 chunks" in printed
    assert "built" not in printed


def test_bm25_index_is_rebuilt_when_chunks_change(make_bm25_indexing, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings and chunking hash.
    Output: what the second run printed and its index.
    What it does: checks a saved retriever with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    bm25_index, bm25_hash, bm25_index_path = make_bm25_indexing().bm25_indexing(more_chunks, str(tmp_path))

    assert "BM25 index built : 12 chunks" in capsys.readouterr().out
    assert len(bm25_index.retriever.docs) == 12


def test_bm25_index_raises_for_no_chunks(make_bm25_indexing, tmp_path):
    """
    Input: no chunks.
    Output: ValueError.
    What it does: checks an empty retriever is never built.
    """
    with pytest.raises(ValueError, match="chunks is empty"):
        make_bm25_indexing().bm25_indexing({}, str(tmp_path))


# ---------- load_bm25_index_from_path / load_saved_bm25_index ----------


def test_load_bm25_index_from_path_loads_a_saved_bm25_index(make_bm25_indexing, make_bm25_indexing_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded straight from its folder.
    Output: the loaded Bm25Index.
    What it does: checks every chunk comes back, and the index path is the same pickle path the build gives.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))

    loaded_index = load_bm25_index_from_path(built_path)

    assert isinstance(loaded_index, Bm25Index)
    assert len(loaded_index.retriever.docs) == 10
    assert loaded_index.index_path == built_index.index_path


def test_load_bm25_index_from_path_raises_when_nothing_is_saved(tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks serving is told to run the build first.
    """
    with pytest.raises(FileNotFoundError, match="build"):
        load_bm25_index_from_path(str(tmp_path))


def test_load_saved_bm25_index_loads_the_index_with_the_same_chunks(make_bm25_indexing, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded with the same chunks from its folder.
    Output: the loaded Bm25Index.
    What it does: checks every chunk comes back, and the index path is the same pickle path the build gives.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))

    loaded_index = load_saved_bm25_index(chunks, os.path.dirname(built_index.index_path))

    assert isinstance(loaded_index, Bm25Index)
    assert len(loaded_index.retriever.docs) == 10
    assert loaded_index.index_path == built_index.index_path


def test_load_saved_bm25_index_raises_when_the_chunks_differ(make_bm25_indexing, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded with 12 chunks.
    Output: FileNotFoundError.
    What it does: checks an index holding other chunks is not used, and the build is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = make_bm25_indexing().bm25_indexing(chunks, str(tmp_path))
    more_chunks, more_embeddings = make_chunks_and_embeddings(12)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_bm25_index(more_chunks, os.path.dirname(built_index.index_path))


def test_load_saved_bm25_index_raises_when_nothing_is_saved(make_chunks_and_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks the build is named when no BM25 index is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_bm25_index(chunks, str(tmp_path))
