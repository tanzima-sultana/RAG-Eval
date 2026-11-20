"""
test_indexing.py

Tests for rag.build.indexing. HuggingFaceEmbeddings is replaced with a fake, so no model is downloaded.
FAISS and BM25 are real.
"""

import os

import pytest

from common import paths_temp
from rag.build import indexing as indexing_module
from rag.build.indexing import Indexing
from rag.build.indexing import lowercase_whitespace_tokenize
from schema.index import Bm25Index
from schema.index import FaissIndex


@pytest.fixture
def patched_embeddings(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes Indexing load the fake instead of the real model.
    """
    monkeypatch.setattr(indexing_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return fake_huggingface_embeddings


def test_lowercase_whitespace_tokenize():
    """
    Input: text with upper case, tabs and new lines.
    Output: the words.
    What it does: checks text is lowercased and split on whitespace.
    """
    assert lowercase_whitespace_tokenize("The Cat\tsat\non  MAT") == ["the", "cat", "sat", "on", "mat"]


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"index_type": "qdrant"}, "index_type"),
    ({"ivf_nlist": 0}, "ivf_nlist"),
    ({"hnsw_m": 0}, "hnsw_m"),
    ({"hnsw_ef_construction": 0}, "hnsw_ef_construction"),
    ({"bm25_tokenizer": "nltk"}, "bm25_tokenizer"),
])
def test_init_raises_for_invalid_setting(patched_embeddings, make_index_args, overrides, message):
    """
    Input: IndexArgs with one invalid setting.
    Output: ValueError.
    What it does: checks Indexing validates the settings when it is created, and names the invalid one.
    """
    with pytest.raises(ValueError, match=message):
        Indexing(make_index_args(**overrides))


# ---------- compute_index ----------


@pytest.mark.parametrize("index_type, index_class", [
    ("faiss_flatip", FaissIndex),
    ("faiss_ivf", FaissIndex),
    ("faiss_hnsw", FaissIndex),
    ("bm25", Bm25Index),
])
def test_compute_index_returns_the_index_of_the_index_type(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, index_type, index_class):
    """
    Input: 10 chunks and each index type.
    Output: the index compute_index returns.
    What it does: checks each index type is built by its own method.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    index = Indexing(make_index_args(index_type=index_type)).compute_index(chunks, embeddings, str(tmp_path))

    assert isinstance(index, index_class)


# ---------- FAISS ----------


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_builds_and_saves(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, index_type):
    """
    Input: 10 chunks and each FAISS index type, with no saved index.
    Output: the FaissIndex and the files in its folder.
    What it does: checks every chunk is in the vector store with its text and chunk_id, and the store is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type=index_type)

    faiss_index = Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    assert faiss_index.index_type == index_type
    assert faiss_index.vector_store.index.ntotal == 10
    assert faiss_index.dimension == 4
    document = faiss_index.vector_store.docstore.search("doc0_2")
    assert document.page_content == chunks["doc0_2"].text
    assert document.metadata["chunk_id"] == "doc0_2"

    faiss_index_path = paths_temp.get_faiss_index_path(str(tmp_path), index_args)
    assert faiss_index.index_path == faiss_index_path
    assert os.path.exists(os.path.join(faiss_index_path, "index.faiss"))
    assert os.path.exists(os.path.join(faiss_index_path, "index.pkl"))


def test_faiss_search_with_a_chunk_vector_finds_that_chunk(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks in a flatip index, searched with chunk "doc1_3"'s vector.
    Output: the top hit and its score.
    What it does: checks vectors are stored at the right positions and the score is cosine similarity.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index = Indexing(make_index_args()).compute_index(chunks, embeddings, str(tmp_path))

    documents_and_scores = faiss_index.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc1_3"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert document.metadata["chunk_id"] == "doc1_3"
    assert score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_loads_from_cache_on_second_run(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, capsys, index_type):
    """
    Input: the same chunks indexed twice with each FAISS index type.
    Output: what the second run printed and its index.
    What it does: checks the second run loads the saved index and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type=index_type)
    Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    faiss_index = Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    printed = capsys.readouterr().out
    assert "loaded from cache : 10 vectors" in printed
    assert "built" not in printed
    assert faiss_index.vector_store.index.ntotal == 10


def test_faiss_index_is_rebuilt_when_chunks_change(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings.
    Output: what the second run printed and its index.
    What it does: checks a saved index with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    Indexing(make_index_args()).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    faiss_index = Indexing(make_index_args()).compute_index(more_chunks, more_embeddings, str(tmp_path))

    assert "FAISS flatip index built : 12 vectors" in capsys.readouterr().out
    assert faiss_index.vector_store.index.ntotal == 12


def test_faiss_ivf_raises_when_fewer_chunks_than_clusters(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 3 chunks with ivf_nlist 4.
    Output: ValueError.
    What it does: checks IVF needs at least ivf_nlist chunks.
    """
    chunks, embeddings = make_chunks_and_embeddings(3)
    indexing = Indexing(make_index_args(index_type="faiss_ivf", ivf_nlist=4))

    with pytest.raises(ValueError, match="ivf_nlist"):
        indexing.compute_index(chunks, embeddings, str(tmp_path))


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_raises_for_no_embeddings(patched_embeddings, make_index_args, tmp_path, index_type):
    """
    Input: no chunks and no embeddings.
    Output: ValueError.
    What it does: checks an empty index is never built.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        Indexing(make_index_args(index_type=index_type)).compute_index({}, {}, str(tmp_path))


# ---------- BM25 ----------


def test_bm25_index_builds_and_saves(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, with bm25_k1 1.2 and bm25_b 0.5.
    Output: the Bm25Index and its saved file.
    What it does: checks every chunk is in the retriever, k1 and b are used, and the retriever is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type="bm25", bm25_k1=1.2, bm25_b=0.5)

    bm25_index = Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    assert len(bm25_index.retriever.docs) == 10
    assert bm25_index.retriever.vectorizer.k1 == 1.2
    assert bm25_index.retriever.vectorizer.b == 0.5
    bm25_index_path = paths_temp.get_bm25_index_path(str(tmp_path), index_args)
    assert bm25_index.index_path == os.path.join(bm25_index_path, "bm25_retriever.pkl")
    assert os.path.exists(bm25_index.index_path)


def test_bm25_search_finds_the_chunk_with_the_word(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, searched with "WORD3".
    Output: the top hit.
    What it does: checks the query is lowercased like the chunks and the chunk with that word comes first.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    bm25_index = Indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))

    documents = bm25_index.retriever.invoke("WORD3")

    assert documents[0].metadata["chunk_id"] == "doc1_3"


def test_bm25_index_loads_from_cache_on_second_run(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: the same chunks indexed twice with bm25.
    Output: what the second run printed.
    What it does: checks the second run loads the saved retriever and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type="bm25")
    Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    Indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    printed = capsys.readouterr().out
    assert "BM25 index loaded from cache : 10 chunks" in printed
    assert "built" not in printed


def test_bm25_index_is_rebuilt_when_chunks_change(patched_embeddings, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings.
    Output: what the second run printed and its index.
    What it does: checks a saved retriever with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    Indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    bm25_index = Indexing(make_index_args(index_type="bm25")).compute_index(more_chunks, more_embeddings, str(tmp_path))

    assert "BM25 index built : 12 chunks" in capsys.readouterr().out
    assert len(bm25_index.retriever.docs) == 12


def test_bm25_index_raises_for_no_chunks(patched_embeddings, make_index_args, tmp_path):
    """
    Input: no chunks.
    Output: ValueError.
    What it does: checks an empty retriever is never built.
    """
    with pytest.raises(ValueError, match="chunks is empty"):
        Indexing(make_index_args(index_type="bm25")).compute_index({}, {}, str(tmp_path))
