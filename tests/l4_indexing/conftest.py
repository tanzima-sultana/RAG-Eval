"""
conftest.py

Shared fakes and fixtures for the l4_indexing test suite.

Chunk i of a test set has the text "word{i} shared", so a BM25 search for
"word3" finds chunk 3 first. Its embedding is a fixed random unit vector,
so a FAISS or Qdrant search with chunk i's vector finds chunk i first.
Qdrant runs in memory and LangChain's HuggingFaceEmbeddings is replaced by
a fake, so no server is needed and no model is downloaded.
"""

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from common.args import ChunkingArgs
from common.args import IndexArgs
from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded
from l4_indexing import cache as cache_module
from l4_indexing import langchain as langchain_module

DIMENSION = 4


class FakeHuggingFaceEmbeddings(Embeddings):
    """
    Stand-in for langchain_huggingface.HuggingFaceEmbeddings: every text is
    embedded as the same unit vector of length DIMENSION.
    """

    def __init__(self, model_name=None, model_kwargs=None, encode_kwargs=None):
        """
        Input: the arguments the real class takes (ignored).
        Output: nothing.
        What it does: nothing to set up.
        """
        self.model_name = model_name

    def embed_documents(self, texts):
        """
        Input: texts, a list of strings.
        Output: one vector per text.
        What it does: returns the same unit vector for every text.
        """
        vectors = []
        for text in texts:
            vectors.append(self.embed_query(text))
        return vectors

    def embed_query(self, text):
        """
        Input: text, one string.
        Output: a unit vector of length DIMENSION.
        What it does: returns [1, 0, 0, 0].
        """
        vector = [0.0] * DIMENSION
        vector[0] = 1.0
        return vector


@pytest.fixture
def make_index_args():
    """
    Input: none.
    Output: a function that builds an IndexArgs.
    What it does: returns a builder with small defaults (ivf_nlist 2, hnsw_m 4)
    where any field can be overridden by keyword.
    """
    def build_index_args(**overrides):
        """
        Input: any IndexArgs field as a keyword argument.
        Output: an IndexArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["index_type"] = "faiss_flatip"
        field_values["ivf_nlist"] = 2
        field_values["hnsw_m"] = 4
        field_values["hnsw_ef_construction"] = 8
        field_values["bm25_k1"] = 1.5
        field_values["bm25_b"] = 0.75
        field_values["bm25_tokenizer"] = "whitespace"
        field_values["qdrant_batch_size"] = 256
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return IndexArgs(**field_values)

    return build_index_args


@pytest.fixture
def make_chunking_args():
    """
    Input: none.
    Output: a function that builds a ChunkingArgs.
    What it does: returns a builder with fixed chunking defaults where any
    field can be overridden by keyword.
    """
    def build_chunking_args(**overrides):
        """
        Input: any ChunkingArgs field as a keyword argument.
        Output: a ChunkingArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["framework"] = ""
        field_values["model_name"] = "fake-model"
        field_values["chunk_type"] = "fixed"
        field_values["min_chunk_size"] = 2
        field_values["max_chunk_size"] = 6
        field_values["fixed_chunk_overlap"] = 2
        field_values["semantic_threshold"] = 0.5
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return ChunkingArgs(**field_values)

    return build_chunking_args


@pytest.fixture
def make_chunks_and_embeddings():
    """
    Input: none.
    Output: a function that builds matching chunks and embeddings.
    What it does: returns a builder that makes chunk_count chunks spread
    over doc_count documents, with ids "doc{d}_{n}".
    """
    def build_chunks_and_embeddings(chunk_count, doc_count=2):
        """
        Input: chunk_count, the number of chunks; doc_count, the number of documents.
        Output: (chunks, embeddings), each a dict keyed by chunk_id, in the same order.
        What it does: chunk i gets the text "word{i} shared" and a fixed random unit vector.
        """
        random_generator = np.random.default_rng(0)
        chunks = {}
        embeddings = {}
        for chunk_index in range(chunk_count):
            doc_index = chunk_index % doc_count
            chunk_id = f"doc{doc_index}_{chunk_index}"
            chunk_text = f"word{chunk_index} shared"
            chunks[chunk_id] = Chunk(chunk_id, chunk_text, 2, "fixed")

            vector = random_generator.normal(size=DIMENSION).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            embeddings[chunk_id] = Embedded(chunk_id, vector, True)
        return chunks, embeddings

    return build_chunks_and_embeddings


@pytest.fixture
def memory_qdrant_client(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the in-memory QdrantClient every l4 call connects to.
    What it does: makes cache._connect_qdrant_client return one in-memory
    client, so no Qdrant server is needed.
    """
    qdrant_client = QdrantClient(":memory:")

    def connect_memory_qdrant_client():
        """
        Input: none.
        Output: the in-memory QdrantClient.
        What it does: stands in for cache._connect_qdrant_client.
        """
        return qdrant_client

    monkeypatch.setattr(cache_module, "_connect_qdrant_client", connect_memory_qdrant_client)
    return qdrant_client


@pytest.fixture
def fake_huggingface_embeddings(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes l4 LangChain building and loading use the fake
    instead of downloading the real model.
    """
    monkeypatch.setattr(langchain_module, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(cache_module, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    return FakeHuggingFaceEmbeddings
