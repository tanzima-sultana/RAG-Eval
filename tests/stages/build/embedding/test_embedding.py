"""
test_embedding.py

Tests for stages.build.embedding.embedding.Embedding: the orchestrator that
loads each chunk's cached embedding, embeds every chunk that has none in one
encode() call, and saves the new embeddings as one file per document.

Every test uses fake_sentence_transformer_class, so no real embedding model is
downloaded or run, and writes only under pytest's tmp_path.
"""

import os

import numpy as np

from common import paths
from common.args import ChunkingArgs
from stages.build.chunking.schema import Chunk
from stages.build.embedding.embedding import Embedding
from stages.build.embedding.schema import Embedded


def make_chunking_args(framework=""):
    """
    Input: framework, "" for the built-in embedding or "langchain".
    Output: a ChunkingArgs with model_name "fake-model" and small chunk sizes.
    What it does: builds the settings the embedding folder path is made from.
    """
    chunking_args = ChunkingArgs(
        framework=framework,
        model_name="fake-model",
        chunk_type="fixed",
        min_chunk_size=1,
        max_chunk_size=10,
        fixed_chunk_overlap=0,
        semantic_threshold=0.0,
    )
    return chunking_args


def make_chunks(doc_id, texts):
    """
    Input: doc_id and texts, one text per chunk.
    Output: a dict of chunk_id to Chunk with ids "{doc_id}_0", "{doc_id}_1", ...
    What it does: builds the chunks of one document.
    """
    chunks = {}
    for chunk_index in range(len(texts)):
        chunk_id = f"{doc_id}_{chunk_index}"
        chunks[chunk_id] = Chunk(chunk_id, texts[chunk_index], 1, "fixed")
    return chunks


# ---------- __init__ ----------


def test_init_loads_the_model_named_in_chunking_args(fake_sentence_transformer_class):
    """
    Input: an Embedding built for the built-in framework.
    Output: its chunking_args and model.
    What it does: checks chunking_args is stored and the model is loaded with model_name.
    """
    chunking_args = make_chunking_args()

    embedding = Embedding(chunking_args)

    assert embedding.chunking_args is chunking_args
    assert embedding.model.model_name == "fake-model"


def test_init_does_not_load_a_model_for_langchain(fake_sentence_transformer_class):
    """
    Input: an Embedding built for the langchain framework.
    Output: its model.
    What it does: checks no model is loaded, because the Langchain class loads its own.
    """
    embedding = Embedding(make_chunking_args(framework="langchain"))

    assert embedding.model is None


# ---------- compute_embeddings ----------


def test_compute_embeddings_embeds_every_chunk_and_saves_one_file_per_document(fake_sentence_transformer_class, tmp_path):
    """
    Input: three chunks from two documents, nothing cached.
    Output: compute_embeddings' return value and the files in the embedding folder.
    What it does: checks every chunk gets its own vector and each document gets one {doc_id}.npz.
    """
    chunks = {}
    chunks.update(make_chunks("doc1", ["a", "bb"]))
    chunks.update(make_chunks("doc2", ["ccc"]))
    chunking_args = make_chunking_args()

    embeddings = Embedding(chunking_args).compute_embeddings(chunks, str(tmp_path))

    assert list(embeddings.keys()) == ["doc1_0", "doc1_1", "doc2_0"]
    assert type(embeddings["doc1_0"]) is Embedded
    assert embeddings["doc1_0"].normalized is True
    np.testing.assert_array_equal(embeddings["doc1_1"].embeddings, [2.0, 1.0])
    np.testing.assert_array_equal(embeddings["doc2_0"].embeddings, [3.0, 1.0])
    embedding_path = paths.get_embedding_path(str(tmp_path), chunking_args)
    assert sorted(os.listdir(embedding_path)) == ["doc1.npz", "doc2.npz"]


def test_compute_embeddings_sends_all_chunks_in_one_encode_call(fake_sentence_transformer_class, tmp_path):
    """
    Input: chunks from two documents, nothing cached.
    Output: the encode() calls the fake model received.
    What it does: checks every chunk text goes to the model in a single call.
    """
    chunks = {}
    chunks.update(make_chunks("doc1", ["a", "bb"]))
    chunks.update(make_chunks("doc2", ["ccc"]))
    embedding = Embedding(make_chunking_args())

    embedding.compute_embeddings(chunks, str(tmp_path))

    assert embedding.model.encode_calls == [["a", "bb", "ccc"]]


def test_compute_embeddings_loads_everything_from_cache_on_a_second_run(fake_sentence_transformer_class, tmp_path):
    """
    Input: the same chunks embedded twice.
    Output: the second run's encode() calls and return value.
    What it does: checks the second run never calls the model and returns the same vectors.
    """
    chunks = make_chunks("doc1", ["a", "bb"])
    first_embeddings = Embedding(make_chunking_args()).compute_embeddings(chunks, str(tmp_path))

    second_embedding = Embedding(make_chunking_args())
    second_embeddings = second_embedding.compute_embeddings(chunks, str(tmp_path))

    assert second_embedding.model.encode_calls == []
    assert list(second_embeddings.keys()) == list(first_embeddings.keys())
    np.testing.assert_array_equal(second_embeddings["doc1_1"].embeddings, first_embeddings["doc1_1"].embeddings)


def test_compute_embeddings_only_embeds_documents_that_are_not_cached(fake_sentence_transformer_class, tmp_path):
    """
    Input: doc1 embedded in a first run, then doc1 and doc2 in a second run.
    Output: the second run's encode() calls and return value.
    What it does: checks only doc2's chunks go to the model and both documents are returned.
    """
    Embedding(make_chunking_args()).compute_embeddings(make_chunks("doc1", ["a"]), str(tmp_path))

    chunks = {}
    chunks.update(make_chunks("doc1", ["a"]))
    chunks.update(make_chunks("doc2", ["bb", "ccc"]))
    embedding = Embedding(make_chunking_args())
    embeddings = embedding.compute_embeddings(chunks, str(tmp_path))

    assert embedding.model.encode_calls == [["bb", "ccc"]]
    assert sorted(embeddings.keys()) == ["doc1_0", "doc2_0", "doc2_1"]


def test_compute_embeddings_prints_cached_and_new_counts(fake_sentence_transformer_class, tmp_path, capsys):
    """
    Input: one document cached, one new.
    Output: what the second run printed.
    What it does: checks the counts line splits the embeddings into cached and new.
    """
    Embedding(make_chunking_args()).compute_embeddings(make_chunks("doc1", ["a"]), str(tmp_path))
    capsys.readouterr()

    chunks = {}
    chunks.update(make_chunks("doc1", ["a"]))
    chunks.update(make_chunks("doc2", ["bb"]))
    Embedding(make_chunking_args()).compute_embeddings(chunks, str(tmp_path))

    printed = capsys.readouterr().out
    assert "Embeddings : 2 (1 from cache, 1 new)" in printed
