"""
test_embedding.py

Tests for langchain_rag.embedding. HuggingFaceEmbeddings is replaced with a fake that embeds a text by its length,
so no model is downloaded.
"""

import os

import numpy as np
import pytest

from common import paths_temp
from common.args_temp import EmbeddingArgs
from langchain_rag import embedding as embedding_module
from langchain_rag.embedding import Embedding
from langchain_rag.embedding import load_embeddings
from langchain_rag.embedding import save_embeddings
from schema.chunk import Chunk
from schema.embed import Embedded


@pytest.fixture
def patched_embeddings(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes Embedding load the fake instead of the real model.
    """
    monkeypatch.setattr(embedding_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return fake_huggingface_embeddings


def make_chunks(chunk_texts_by_doc_id):
    """
    Input: chunk_texts_by_doc_id, a dict of doc_id to that document's chunk texts.
    Output: a dict of chunk_id to Chunk, with ids "{doc_id}_{n}".
    What it does: builds the chunks Embedding takes.
    """
    chunks = {}
    for doc_id in chunk_texts_by_doc_id:
        chunk_texts = chunk_texts_by_doc_id[doc_id]
        for chunk_index in range(len(chunk_texts)):
            chunk_id = f"{doc_id}_{chunk_index}"
            chunks[chunk_id] = Chunk(chunk_id, chunk_texts[chunk_index], 1, "fixed")
    return chunks


# ---------- save_embeddings / load_embeddings ----------


def test_save_and_load_embeddings_round_trip(tmp_path):
    """
    Input: one document's embeddings saved and loaded back.
    Output: the loaded embeddings.
    What it does: checks chunk ids, vectors and normalized flags come back, and a missing document gives None.
    """
    embeddings = {}
    embeddings["doc_0"] = Embedded("doc_0", np.array([1.0, 2.0], dtype=np.float32), True)
    embeddings["doc_1"] = Embedded("doc_1", np.array([3.0, 4.0], dtype=np.float32), False)

    save_embeddings(str(tmp_path), "doc", embeddings)
    loaded_embeddings = load_embeddings(str(tmp_path), "doc")

    assert list(loaded_embeddings.keys()) == ["doc_0", "doc_1"]
    assert np.array_equal(loaded_embeddings["doc_1"].embeddings, [3.0, 4.0])
    assert loaded_embeddings["doc_0"].normalized is True
    assert loaded_embeddings["doc_1"].normalized is False
    assert load_embeddings(str(tmp_path), "other") is None


# ---------- compute_embeddings ----------


def test_compute_embeddings_embeds_every_chunk_and_saves_one_file_per_document(patched_embeddings, make_chunking_args, tmp_path):
    """
    Input: two documents' chunks.
    Output: the embeddings and the files in the embedding folder.
    What it does: checks each chunk gets its own vector, marked normalized, and one file is saved per document.
    """
    embedding_args = EmbeddingArgs(chunking_args=make_chunking_args())
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})

    embeddings = Embedding(embedding_args).compute_embeddings(chunks, str(tmp_path))

    assert list(embeddings.keys()) == ["d1_0", "d1_1", "d2_0"]
    assert embeddings["d1_1"].embeddings[0] == 3.0
    assert embeddings["d2_0"].embeddings[0] == 2.0
    assert embeddings["d1_0"].normalized is True
    embedding_path = paths_temp.get_embedding_path(str(tmp_path), embedding_args)
    assert sorted(os.listdir(embedding_path)) == ["d1.npz", "d2.npz"]


def test_compute_embeddings_sends_all_chunks_in_one_call(patched_embeddings, make_chunking_args, tmp_path):
    """
    Input: two documents' chunks.
    Output: the embed_documents calls.
    What it does: checks every chunk is embedded in a single embed_documents call.
    """
    embedding_args = EmbeddingArgs(chunking_args=make_chunking_args())
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})

    Embedding(embedding_args).compute_embeddings(chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["a", "bbb", "cc"]]


def test_compute_embeddings_loads_everything_from_cache_on_second_run(patched_embeddings, make_chunking_args, tmp_path, capsys):
    """
    Input: the same chunks embedded twice.
    Output: the second run's embeddings, the embed calls and what it printed.
    What it does: checks the second run embeds nothing and returns the same vectors.
    """
    embedding_args = EmbeddingArgs(chunking_args=make_chunking_args())
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})

    first_embeddings = Embedding(embedding_args).compute_embeddings(chunks, str(tmp_path))
    patched_embeddings.embed_documents_calls = []
    capsys.readouterr()
    second_embeddings = Embedding(embedding_args).compute_embeddings(chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == []
    assert "Embeddings : 3 (3 from cache, 0 new)" in capsys.readouterr().out
    for chunk_id in first_embeddings:
        assert np.array_equal(second_embeddings[chunk_id].embeddings, first_embeddings[chunk_id].embeddings)


def test_compute_embeddings_only_embeds_documents_not_cached(patched_embeddings, make_chunking_args, tmp_path, capsys):
    """
    Input: one document embedded, then that document and a new one embedded.
    Output: the embed calls, the embeddings and what it printed.
    What it does: checks only the new document's chunks are embedded and both are returned.
    """
    embedding_args = EmbeddingArgs(chunking_args=make_chunking_args())
    Embedding(embedding_args).compute_embeddings(make_chunks({"d1": ["a", "bbb"]}), str(tmp_path))
    patched_embeddings.embed_documents_calls = []
    capsys.readouterr()

    embeddings = Embedding(embedding_args).compute_embeddings(make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]}), str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["cc"]]
    assert sorted(embeddings.keys()) == ["d1_0", "d1_1", "d2_0"]
    assert "Embeddings : 3 (2 from cache, 1 new)" in capsys.readouterr().out
