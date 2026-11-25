"""
test_embedding.py

Tests for rag.build.embedding (embedding.py and cache.py). HuggingFaceEmbeddings is replaced with a fake that embeds a text
by its length, so no model is downloaded.
"""

import json
import os

import numpy as np
import pytest

from common import paths
from common.args import EmbeddingArgs
from rag.build.embedding import embedding as embedding_module
from rag.build.embedding.cache import load_document_embeddings
from rag.build.embedding.cache import load_embeddings
from rag.build.embedding.cache import save_embeddings
from rag.build.embedding.embedding import Embedding
from rag.build.embedding.embedding import make_embedding_model
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


def make_chunks(chunk_texts_by_document_id):
    """
    Input: chunk_texts_by_document_id, a dict of document id to that document's chunk texts.
    Output: a dict of chunk_id to Chunk, with ids "{document_id}_{n}".
    What it does: builds the chunks Embedding takes.
    """
    chunks = {}
    for document_id in chunk_texts_by_document_id:
        chunk_texts = chunk_texts_by_document_id[document_id]
        for chunk_index in range(len(chunk_texts)):
            chunk_id = f"{document_id}_{chunk_index}"
            chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                chunk_texts=chunk_texts[chunk_index],
                chunk_size=1,
                chunk_type="fixed",
                source="file.pdf",
                page_numbers=[0],
            )
    return chunks


def make_embedding(make_embedder_args, make_chunking_args, chunking_hash="chunkinghash", **embedder_overrides):
    """
    Input: the make_embedder_args and make_chunking_args fixtures; chunking_hash; any EmbedderArgs field as a keyword
    argument.
    Output: an Embedding with the default chunking args.
    What it does: builds the Embedding the way the build does.
    """
    embedding_args = EmbeddingArgs(embedder_args=make_embedder_args(**embedder_overrides), chunking_args=make_chunking_args())
    return Embedding(embedding_args, chunking_hash)


# ---------- save_embeddings / load_document_embeddings / load_embeddings ----------


def make_embedded(chunk_id, first_value, normalized):
    """
    Input: chunk_id; first_value, the first number of the vector; normalized.
    Output: an Embedded with a 2-number vector.
    What it does: builds a small fixed Embedded.
    """
    return Embedded(chunk_id, np.array([first_value, 1.0], dtype=np.float32), normalized)


def test_save_and_load_embeddings_round_trip(tmp_path):
    """
    Input: the embeddings of two documents saved under one embedding hash and loaded back.
    Output: the loaded embeddings.
    What it does: checks chunk ids, vectors, normalized flags and order come back, and an unknown embedding hash gives
    None.
    """
    embeddings = {}
    embeddings["d1_0"] = make_embedded("d1_0", 1.0, True)
    embeddings["d1_1"] = make_embedded("d1_1", 2.0, False)
    embeddings["d2_0"] = make_embedded("d2_0", 3.0, True)

    save_embeddings(str(tmp_path), embeddings, ["d1", "d2"], "embeddinghash")
    loaded_embeddings = load_embeddings(str(tmp_path), "embeddinghash")

    assert list(loaded_embeddings.keys()) == ["d1_0", "d1_1", "d2_0"]
    assert np.array_equal(loaded_embeddings["d1_1"].embeddings, [2.0, 1.0])
    assert loaded_embeddings["d1_0"].normalized is True
    assert loaded_embeddings["d1_1"].normalized is False
    assert load_embeddings(str(tmp_path), "otherhash") is None


def test_save_embeddings_saves_one_npz_per_new_document_and_lists_every_document(tmp_path):
    """
    Input: the embeddings of one new document saved with a list of two document ids, one of them saved before.
    Output: the files in the folder.
    What it does: checks only the new document's file is written, and the list holds both ids in the order given.
    """
    save_embeddings(str(tmp_path), {"d1_0": make_embedded("d1_0", 1.0, True)}, ["d1"], "firsthash")

    save_embeddings(str(tmp_path), {"d2_0": make_embedded("d2_0", 2.0, True)}, ["d1", "d2"], "secondhash")

    assert sorted(os.listdir(str(tmp_path))) == ["d1.npz", "d2.npz", "doc_ids_firsthash.json", "doc_ids_secondhash.json"]
    with open(os.path.join(str(tmp_path), "doc_ids_secondhash.json"), encoding="utf-8") as f:
        assert json.load(f) == ["d1", "d2"]


def test_load_document_embeddings_loads_one_document_by_its_id(tmp_path):
    """
    Input: the embeddings of one document saved.
    Output: that document's embeddings, and the result for an unknown document id.
    What it does: checks the document's embeddings come back, and an unknown document id gives None.
    """
    save_embeddings(str(tmp_path), {"d1_0": make_embedded("d1_0", 1.0, True)}, ["d1"], "embeddinghash")

    assert list(load_document_embeddings(str(tmp_path), "d1").keys()) == ["d1_0"]
    assert load_document_embeddings(str(tmp_path), "d2") is None


def test_load_embeddings_gives_none_when_a_listed_document_is_missing(tmp_path):
    """
    Input: a run list naming two documents, with one document's file removed.
    Output: the result of load_embeddings.
    What it does: checks load_embeddings gives None.
    """
    embeddings = {"d1_0": make_embedded("d1_0", 1.0, True), "d2_0": make_embedded("d2_0", 2.0, True)}
    save_embeddings(str(tmp_path), embeddings, ["d1", "d2"], "embeddinghash")
    os.remove(os.path.join(str(tmp_path), "d2.npz"))

    assert load_embeddings(str(tmp_path), "embeddinghash") is None


# ---------- compute_embeddings ----------


def test_compute_embeddings_embeds_every_chunk_and_saves_one_file_per_document(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path):
    """
    Input: two documents' chunks.
    Output: the embeddings, the embedding hash and the files in the embedding folder.
    What it does: checks each chunk gets its own vector, marked normalized, one file is saved per document, and the
    embedding hash is the one get_embedding_hash gives for the chunking hash.
    """
    embedding = make_embedding(make_embedder_args, make_chunking_args)
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})

    embeddings, embedding_hash, embedding_path = embedding.compute_embeddings(chunks, str(tmp_path))

    assert list(embeddings.keys()) == ["d1_0", "d1_1", "d2_0"]
    assert embeddings["d1_1"].embeddings[0] == 3.0
    assert embeddings["d2_0"].embeddings[0] == 2.0
    assert embeddings["d1_0"].normalized is True
    assert embedding_hash == paths.get_embedding_hash(make_embedder_args(), "chunkinghash")
    assert embedding_path == paths.get_embedding_path(str(tmp_path), make_embedder_args(), make_chunking_args())
    assert sorted(os.listdir(embedding_path)) == ["d1.npz", "d2.npz", f"doc_ids_{embedding_hash}.json"]


def test_compute_embeddings_sends_all_chunks_in_one_call(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path):
    """
    Input: two documents' chunks.
    Output: the embed_documents calls.
    What it does: checks every chunk is embedded in a single embed_documents call.
    """
    embedding = make_embedding(make_embedder_args, make_chunking_args)
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})

    embedding.compute_embeddings(chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["a", "bbb", "cc"]]


def test_compute_embeddings_loads_everything_from_cache_and_builds_no_model(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path, capsys):
    """
    Input: the same chunks embedded twice.
    Output: the second run's embeddings, the embed calls, its model and what it printed.
    What it does: checks the second run embeds nothing, builds no model, and returns the same vectors.
    """
    chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})
    first_embeddings, first_hash, first_path = make_embedding(make_embedder_args, make_chunking_args).compute_embeddings(chunks, str(tmp_path))
    patched_embeddings.embed_documents_calls = []
    capsys.readouterr()

    second_embedding = make_embedding(make_embedder_args, make_chunking_args)
    second_embeddings, second_hash, second_path = second_embedding.compute_embeddings(chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == []
    assert second_embedding.model is None
    assert second_hash == first_hash
    assert "Embeddings : 3 (3 from cache, 0 new)" in capsys.readouterr().out
    for chunk_id in first_embeddings:
        assert np.array_equal(second_embeddings[chunk_id].embeddings, first_embeddings[chunk_id].embeddings)


def test_compute_embeddings_only_embeds_documents_not_cached(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path, capsys):
    """
    Input: one document embedded, then that document and a new one embedded with another chunking hash.
    Output: the embed calls, the embeddings, the hashes and what it printed.
    What it does: checks the settings folder is shared, so only the new document's chunks are embedded, both documents'
    embeddings are returned in chunk order, and the embedding hash is another one.
    """
    first_embeddings, first_hash, first_path = make_embedding(make_embedder_args, make_chunking_args, "firstchunkinghash").compute_embeddings(make_chunks({"d1": ["a", "bbb"]}), str(tmp_path))
    patched_embeddings.embed_documents_calls = []
    capsys.readouterr()

    second_chunks = make_chunks({"d1": ["a", "bbb"], "d2": ["cc"]})
    embeddings, second_hash, second_path = make_embedding(make_embedder_args, make_chunking_args, "secondchunkinghash").compute_embeddings(second_chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["cc"]]
    assert list(embeddings.keys()) == ["d1_0", "d1_1", "d2_0"]
    assert second_hash != first_hash
    assert "Embeddings : 3 (2 from cache, 1 new)" in capsys.readouterr().out


def test_compute_embeddings_embeds_a_document_again_when_its_chunks_changed(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path):
    """
    Input: a document embedded with 2 chunks, then the same document id with 3 chunks.
    Output: the embed calls.
    What it does: checks the saved embeddings are not used when their chunk ids are not exactly the document's chunk ids.
    """
    make_embedding(make_embedder_args, make_chunking_args).compute_embeddings(make_chunks({"d1": ["a", "bbb"]}), str(tmp_path))
    patched_embeddings.embed_documents_calls = []

    embeddings, embedding_hash, embedding_path = make_embedding(make_embedder_args, make_chunking_args).compute_embeddings(make_chunks({"d1": ["a", "bbb", "cc"]}), str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["a", "bbb", "cc"]]
    assert list(embeddings.keys()) == ["d1_0", "d1_1", "d1_2"]


def test_compute_embeddings_embeds_a_document_again_when_normalize_embeddings_changed(patched_embeddings, make_embedder_args, make_chunking_args, tmp_path):
    """
    Input: a document embedded with normalize_embeddings True, then the same document with normalize_embeddings False.
    Output: the embed calls and the embeddings.
    What it does: checks the saved vectors are not used when their normalized flag differs from the config, and the new
    vectors are marked not normalized.
    """
    chunks = make_chunks({"d1": ["a", "bbb"]})
    make_embedding(make_embedder_args, make_chunking_args).compute_embeddings(chunks, str(tmp_path))
    patched_embeddings.embed_documents_calls = []

    embeddings, embedding_hash, embedding_path = make_embedding(make_embedder_args, make_chunking_args, normalize_embeddings=False).compute_embeddings(chunks, str(tmp_path))

    assert patched_embeddings.embed_documents_calls == [["a", "bbb"]]
    assert embeddings["d1_0"].normalized is False


# ---------- make_embedding_model ----------


def test_make_embedding_model_raises_for_unknown_provider(make_embedder_args):
    """
    Input: EmbedderArgs with embedding_provider "unknown".
    Output: ValueError.
    What it does: checks the factory rejects a provider it does not support and names the provider.
    """
    with pytest.raises(ValueError, match="unknown"):
        make_embedding_model(make_embedder_args(embedding_provider="unknown"))
