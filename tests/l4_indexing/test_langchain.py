"""
test_langchain.py

Tests for l4_indexing.langchain.Langchain: building the LangChain FAISS,
BM25 and Qdrant indexes from the l2 chunks and l3 embeddings, and reusing
saved ones. HuggingFaceEmbeddings is faked and Qdrant runs in memory.
"""

import os

import pytest

from l4_indexing import langchain as langchain_module
from l4_indexing.langchain import Langchain
from l4_indexing.langchain import lowercase_whitespace_tokenize


def test_lowercase_whitespace_tokenize():
    """
    Input: text with upper case, tabs and new lines.
    Output: the words.
    What it does: checks text is split the same way as the built-in BM25 index.
    """
    assert lowercase_whitespace_tokenize("The Cat\tsat\non  MAT") == ["the", "cat", "sat", "on", "mat"]


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_get_faiss_index_builds_and_saves(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, index_type):
    """
    Input: 10 chunks and embeddings and each FAISS index type, with no saved index.
    Output: the LangchainFaissIndex and the files in the index folder.
    What it does: checks every chunk is in the vector store with its text and chunk_id, and the store is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index_path = str(tmp_path / "faiss")
    langchain = Langchain(make_index_args(index_type=index_type), make_chunking_args(framework="langchain"))

    faiss_index = langchain.get_faiss_index(faiss_index_path, chunks, embeddings)

    assert faiss_index.faiss_index_type == index_type
    assert faiss_index.vector_store.index.ntotal == 10
    assert faiss_index.dimension == 4
    document = faiss_index.vector_store.docstore.search("doc0_2")
    assert document.page_content == chunks["doc0_2"].text
    assert document.metadata["chunk_id"] == "doc0_2"
    assert os.path.exists(os.path.join(faiss_index_path, "index.faiss"))
    assert os.path.exists(os.path.join(faiss_index_path, "index.pkl"))


def test_faiss_search_with_a_chunk_vector_finds_that_chunk(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: 10 chunks, searched with chunk "doc1_3"'s l3 vector.
    Output: the top hit and its score.
    What it does: checks the l3 vectors are stored at the right positions and the score is cosine similarity.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    langchain = Langchain(make_index_args(), make_chunking_args(framework="langchain"))
    faiss_index = langchain.get_faiss_index(str(tmp_path), chunks, embeddings)

    documents_and_scores = faiss_index.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc1_3"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert document.metadata["chunk_id"] == "doc1_3"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_get_faiss_index_reuses_saved_index(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same chunks indexed twice.
    Output: the second LangchainFaissIndex.
    What it does: checks the second call loads the saved vector store and never builds.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    langchain = Langchain(make_index_args(), make_chunking_args(framework="langchain"))
    langchain.get_faiss_index(str(tmp_path), chunks, embeddings)

    def fail_build(vectors, dimension):
        """
        Input: the build arguments.
        Output: never returns.
        What it does: fails the test if a build happens.
        """
        raise AssertionError("the saved index should have been reused")

    monkeypatch.setattr(langchain_module, "build_flatip_index", fail_build)
    reloaded_index = langchain.get_faiss_index(str(tmp_path), chunks, embeddings)

    assert reloaded_index.vector_store.index.ntotal == 10


def test_get_bm25_index_builds_saves_and_ranks(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: 6 chunks "word{i} shared" and the query "WORD4".
    Output: the LangchainBm25Index, its file and its top document.
    What it does: checks the retriever holds every chunk, is saved, and lowercases the query.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    langchain = Langchain(make_index_args(index_type="bm25"), make_chunking_args(framework="langchain"))

    bm25_index = langchain.get_bm25_index(str(tmp_path), chunks)
    top_documents = bm25_index.retriever.invoke("WORD4")

    assert len(bm25_index.retriever.docs) == 6
    assert os.path.exists(os.path.join(str(tmp_path), "bm25_retriever.pkl"))
    assert top_documents[0].metadata["chunk_id"] == "doc0_4"


def test_get_bm25_index_reuses_saved_retriever(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same chunks indexed twice.
    Output: the second LangchainBm25Index.
    What it does: checks the second call loads the saved retriever and never builds.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    langchain = Langchain(make_index_args(index_type="bm25"), make_chunking_args(framework="langchain"))
    langchain.get_bm25_index(str(tmp_path), chunks)

    def fail_build(*args, **kwargs):
        """
        Input: the build arguments.
        Output: never returns.
        What it does: fails the test if a build happens.
        """
        raise AssertionError("the saved retriever should have been reused")

    monkeypatch.setattr(langchain_module.BM25Retriever, "from_texts", fail_build)
    reloaded_index = langchain.get_bm25_index(str(tmp_path), chunks)

    assert len(reloaded_index.retriever.docs) == 6


def test_get_bm25_index_unsupported_tokenizer_raises(tmp_path, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: bm25_tokenizer "nltk".
    Output: ValueError mentioning bm25_tokenizer.
    What it does: checks only the whitespace tokenizer is accepted.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    langchain = Langchain(make_index_args(index_type="bm25", bm25_tokenizer="nltk"), make_chunking_args(framework="langchain"))

    with pytest.raises(ValueError, match="bm25_tokenizer"):
        langchain.get_bm25_index(str(tmp_path), chunks)


def test_get_qdrant_index_uploads_langchain_payload(memory_qdrant_client, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: 5 chunks and embeddings, with no collection.
    Output: the LangchainQdrantIndex and a search with chunk "doc0_2"'s vector.
    What it does: checks the points have the payload QdrantVectorStore reads, so it returns the chunk text and id.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    langchain = Langchain(make_index_args(index_type="qdrant"), make_chunking_args(framework="langchain"))

    qdrant_index = langchain.get_qdrant_index("langchain_collection", chunks, embeddings)
    documents_and_scores = qdrant_index.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc0_2"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert qdrant_index.dimension == 4
    assert document.metadata["chunk_id"] == "doc0_2"
    assert document.page_content == chunks["doc0_2"].text


def test_get_qdrant_index_reuses_collection_with_same_chunks(memory_qdrant_client, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, monkeypatch):
    """
    Input: the same chunks indexed twice.
    Output: the second LangchainQdrantIndex.
    What it does: checks the second call uses the existing collection and uploads nothing.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    langchain = Langchain(make_index_args(index_type="qdrant"), make_chunking_args(framework="langchain"))
    langchain.get_qdrant_index("langchain_collection", chunks, embeddings)

    def fail_upsert(*args, **kwargs):
        """
        Input: the upsert arguments.
        Output: never returns.
        What it does: fails the test if anything is uploaded.
        """
        raise AssertionError("the existing collection should have been reused")

    monkeypatch.setattr(memory_qdrant_client, "upsert", fail_upsert)
    reused_index = langchain.get_qdrant_index("langchain_collection", chunks, embeddings)

    assert reused_index.collection_name == "langchain_collection"
