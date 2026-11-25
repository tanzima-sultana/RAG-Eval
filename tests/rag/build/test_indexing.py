"""
test_indexing.py

Tests for rag.build.indexing.indexing (the Indexing entry class). HuggingFaceEmbeddings is replaced with a fake, so no
model is downloaded. The indexing of each kind is tested in test_faiss_indexing.py, test_bm25_indexing.py and
test_indexing_vectordb.py.
"""

import pytest

from common import paths
from rag.build.embedding import embedding as embedding_module
from rag.build.indexing.bm25_indexing import Bm25Indexing
from rag.build.indexing.faiss_indexing import FaissIndexing
from rag.build.indexing.indexing import Indexing
from rag.build.indexing.vectordb.qdrant import IndexingVectorDb

CHUNKING_HASH = "testchunkinghash"
EMBEDDING_HASH = "testembeddinghash"


@pytest.fixture
def patched_embeddings(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes the embedding model factory, which Indexing uses, load the fake instead of the real model.
    """
    monkeypatch.setattr(embedding_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return fake_huggingface_embeddings


@pytest.fixture
def make_indexing(make_embedder_args, make_bm25_indexing_args, make_qdrant_args):
    """
    Input: the make_embedder_args, make_bm25_indexing_args and make_qdrant_args fixtures.
    Output: a function that builds an Indexing.
    What it does: returns a builder with the defaults faiss_types ["faiss_flatip"], no vector DB types and no Qdrant
    args, where any of them can be overridden.
    """
    def build_indexing(**overrides):
        """
        Input: embedder_args, bm25_args, faiss_types, vectordb_types or qdrant_args as a keyword argument.
        Output: an Indexing with the defaults plus the overrides.
        What it does: creates the Indexing.
        """
        field_values = {}
        field_values["embedder_args"] = make_embedder_args()
        field_values["bm25_args"] = make_bm25_indexing_args()
        field_values["faiss_types"] = ["faiss_flatip"]
        field_values["vectordb_types"] = []
        field_values["qdrant_args"] = None
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return Indexing(
            field_values["embedder_args"],
            field_values["bm25_args"],
            field_values["faiss_types"],
            field_values["vectordb_types"],
            field_values["qdrant_args"],
            CHUNKING_HASH,
            EMBEDDING_HASH
        )

    return build_indexing


# ---------- validate ----------


def test_init_raises_for_unknown_embedding_provider(patched_embeddings, make_indexing, make_embedder_args):
    """
    Input: EmbedderArgs with embedding_provider "unknown".
    Output: ValueError.
    What it does: checks Indexing validates the embedding provider when it is created, and names it.
    """
    with pytest.raises(ValueError, match="embedding_provider"):
        make_indexing(embedder_args=make_embedder_args(embedding_provider="unknown"))


def test_init_raises_when_faiss_types_and_vectordb_types_are_both_empty(patched_embeddings, make_indexing):
    """
    Input: faiss_types [] and vectordb_types [].
    Output: ValueError.
    What it does: checks at least one of FAISS or the vector DB is required, and the error names both settings.
    """
    with pytest.raises(ValueError, match="faiss_types and vectordb_types are both empty"):
        make_indexing(faiss_types=[], vectordb_types=[])


def test_init_raises_for_an_unsupported_faiss_type(patched_embeddings, make_indexing):
    """
    Input: faiss_types ["faiss_pq"].
    Output: ValueError.
    What it does: checks an unsupported FAISS type is rejected and named.
    """
    with pytest.raises(ValueError, match="faiss_type"):
        make_indexing(faiss_types=["faiss_pq"])


def test_init_raises_for_an_unsupported_vectordb_type(patched_embeddings, make_indexing, make_qdrant_args):
    """
    Input: vectordb_types ["chroma"].
    Output: ValueError.
    What it does: checks an unsupported vector DB type is rejected and named.
    """
    with pytest.raises(ValueError, match="vectordb_type"):
        make_indexing(vectordb_types=["chroma"], qdrant_args=make_qdrant_args())


def test_init_raises_when_vectordb_types_are_given_without_qdrant_args(patched_embeddings, make_indexing):
    """
    Input: vectordb_types ["qdrant"] with qdrant_args None.
    Output: ValueError.
    What it does: checks the missing Qdrant settings are named.
    """
    with pytest.raises(ValueError, match="qdrant_args"):
        make_indexing(faiss_types=[], vectordb_types=["qdrant"], qdrant_args=None)


def test_init_raises_for_an_invalid_bm25_setting(patched_embeddings, make_indexing, make_bm25_indexing_args):
    """
    Input: Bm25IndexingArgs with bm25_tokenizer "nltk".
    Output: ValueError.
    What it does: checks the BM25 settings are validated when Indexing is created, by Bm25Indexing.
    """
    with pytest.raises(ValueError, match="bm25_tokenizer"):
        make_indexing(bm25_args=make_bm25_indexing_args(bm25_tokenizer="nltk"))


def test_init_raises_for_an_invalid_qdrant_setting(patched_embeddings, make_indexing, make_qdrant_args):
    """
    Input: QdrantArgs with qdrant_batch_size 0.
    Output: ValueError.
    What it does: checks the Qdrant settings are validated when Indexing is created, by IndexingVectorDb.
    """
    with pytest.raises(ValueError, match="qdrant_batch_size"):
        make_indexing(faiss_types=[], vectordb_types=["qdrant"], qdrant_args=make_qdrant_args(qdrant_batch_size=0))


# ---------- what is set up ----------


def test_faiss_only_sets_up_bm25_and_faiss(patched_embeddings, make_indexing):
    """
    Input: faiss_types ["faiss_flatip"] and no vector DB types.
    Output: the Indexing.
    What it does: checks BM25 and FAISS are set up, the vector DB is not, and the embedding model is loaded.
    """
    indexing = make_indexing(faiss_types=["faiss_flatip"], vectordb_types=[])

    assert isinstance(indexing.bm25_indexing, Bm25Indexing)
    assert isinstance(indexing.faiss_indexing, FaissIndexing)
    assert indexing.vectordb_indexing is None
    assert indexing.embedding_model is not None


def test_vectordb_only_sets_up_bm25_and_the_vector_db(patched_embeddings, make_indexing, make_qdrant_args):
    """
    Input: no FAISS types and vectordb_types ["qdrant"].
    Output: the Indexing.
    What it does: checks BM25 and the vector DB are set up, FAISS is not, and the embedding model is loaded.
    """
    indexing = make_indexing(faiss_types=[], vectordb_types=["qdrant"], qdrant_args=make_qdrant_args())

    assert isinstance(indexing.bm25_indexing, Bm25Indexing)
    assert indexing.faiss_indexing is None
    assert isinstance(indexing.vectordb_indexing, IndexingVectorDb)
    assert indexing.embedding_model is not None


def test_both_sets_up_bm25_faiss_and_the_vector_db(patched_embeddings, make_indexing, make_qdrant_args):
    """
    Input: faiss_types ["faiss_hnsw"] and vectordb_types ["qdrant"].
    Output: the Indexing.
    What it does: checks all three kinds are set up and share one embedding model.
    """
    indexing = make_indexing(faiss_types=["faiss_hnsw"], vectordb_types=["qdrant"], qdrant_args=make_qdrant_args())

    assert isinstance(indexing.bm25_indexing, Bm25Indexing)
    assert isinstance(indexing.faiss_indexing, FaissIndexing)
    assert isinstance(indexing.vectordb_indexing, IndexingVectorDb)
    assert indexing.faiss_indexing.embedding_model is indexing.embedding_model
    assert indexing.vectordb_indexing.embedding_model is indexing.embedding_model


def test_bm25_is_always_built_with_its_hash_from_the_chunking_hash(patched_embeddings, make_indexing, make_bm25_indexing_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks indexed with BM25 through Indexing.
    Output: the BM25 index, its hash and its path.
    What it does: checks BM25 is built, and its hash is the one get_bm25_hash gives for the chunking hash.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    bm25_index, bm25_hash, bm25_index_path = make_indexing().bm25_indexing.bm25_indexing(chunks, str(tmp_path))

    assert len(bm25_index.retriever.docs) == 10
    assert bm25_hash == paths.get_bm25_hash(make_bm25_indexing_args(), CHUNKING_HASH)
