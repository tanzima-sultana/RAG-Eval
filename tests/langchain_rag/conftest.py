"""
conftest.py

Shared fakes and fixtures for the langchain_rag test suite.
The real models are replaced with fakes, so no model is downloaded.
"""

import numpy as np
import pytest
import sentence_transformers
from langchain_core.embeddings import Embeddings

from common.args_temp import ChunkingArgs
from common.args_temp import IndexArgs
from common.args_temp import VectorDbArgs
from schema.chunk import Chunk
from schema.document import Document
from schema.document import DocumentType
from schema.document import Metadata
from schema.embed import Embedded

DIMENSION = 4


class FakeLangchainTokenizer:
    """
    Stand-in for a Hugging Face tokenizer: one token per word, with "[CLS]" and "[SEP]" around the text
    unless add_special_tokens is False.
    """

    def encode(self, text, max_length=None, truncation=None, add_special_tokens=True):
        """
        Input: text; max_length and truncation (ignored); add_special_tokens.
        Output: the list of words, wrapped in "[CLS]" and "[SEP]" when add_special_tokens is True.
        What it does: splits text on whitespace.
        """
        tokens = []
        if add_special_tokens:
            tokens.append("[CLS]")
        for word in text.split():
            tokens.append(word)
        if add_special_tokens:
            tokens.append("[SEP]")
        return tokens

    def decode(self, tokens):
        """
        Input: tokens, a list of words.
        Output: the words joined with single spaces.
        What it does: turns tokens back into text.
        """
        return " ".join(tokens)


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer. A sentence with "cat" is embedded as [1, 0],
    any other as [0, 1]. Every loaded model name is recorded in loaded_model_names.
    """

    loaded_model_names = []

    def __init__(self, model_name=None, device=None):
        """
        Input: model_name and device, as the real class takes them.
        Output: nothing.
        What it does: stores both, creates the fake tokenizer and records model_name.
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeLangchainTokenizer()
        self.max_seq_length = 256
        FakeSentenceTransformer.loaded_model_names.append(model_name)

    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        """
        Input: sentences, a list of strings; batch_size and show_progress_bar (ignored).
        Output: a numpy array with one 2-number embedding per sentence.
        What it does: embeds each sentence by the cat rule.
        """
        vectors = []
        for sentence in sentences:
            if "cat" in sentence.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors)


class FakeHuggingFaceEmbeddings(Embeddings):
    """
    Stand-in for langchain_huggingface.HuggingFaceEmbeddings. A text is embedded as
    [len(text), 1, 0, 0]. Every embed_documents call is recorded in embed_documents_calls.
    """

    embed_documents_calls = []

    def __init__(self, model_name=None, model_kwargs=None, encode_kwargs=None, show_progress=False):
        """
        Input: the arguments the real class takes.
        Output: nothing.
        What it does: stores model_name.
        """
        self.model_name = model_name

    def embed_documents(self, texts):
        """
        Input: texts, a list of strings.
        Output: one vector of length DIMENSION per text.
        What it does: records the call and embeds each text by its length.
        """
        FakeHuggingFaceEmbeddings.embed_documents_calls.append(list(texts))
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), 1.0, 0.0, 0.0])
        return vectors

    def embed_query(self, text):
        """
        Input: text, one string.
        Output: its vector.
        What it does: embeds text with embed_documents rules, without recording the call.
        """
        return [float(len(text)), 1.0, 0.0, 0.0]


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeSentenceTransformer class.
    What it does: makes LangChain and chunking load FakeSentenceTransformer instead of the real model.
    """
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    FakeSentenceTransformer.loaded_model_names = []
    return FakeSentenceTransformer


@pytest.fixture
def fake_huggingface_embeddings():
    """
    Input: none.
    Output: the FakeHuggingFaceEmbeddings class, with its recorded calls cleared.
    What it does: gives each test a clean FakeHuggingFaceEmbeddings. Each test file patches it into its module.
    """
    FakeHuggingFaceEmbeddings.embed_documents_calls = []
    return FakeHuggingFaceEmbeddings


@pytest.fixture
def make_chunking_args():
    """
    Input: none.
    Output: a function that builds a ChunkingArgs.
    What it does: returns a builder with small defaults where any field can be overridden by keyword.
    """
    def build_chunking_args(**overrides):
        """
        Input: any ChunkingArgs field as a keyword argument.
        Output: a ChunkingArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
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
def make_index_args(make_chunking_args):
    """
    Input: the make_chunking_args fixture.
    Output: a function that builds an IndexArgs.
    What it does: returns a builder with small defaults (ivf_nlist 2, hnsw_m 4) where any field can be overridden.
    """
    def build_index_args(**overrides):
        """
        Input: any IndexArgs field as a keyword argument.
        Output: an IndexArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["chunking_args"] = make_chunking_args()
        field_values["index_type"] = "faiss_flatip"
        field_values["ivf_nlist"] = 2
        field_values["hnsw_m"] = 4
        field_values["hnsw_ef_construction"] = 8
        field_values["bm25_k1"] = 1.5
        field_values["bm25_b"] = 0.75
        field_values["bm25_tokenizer"] = "whitespace"
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return IndexArgs(**field_values)

    return build_index_args


@pytest.fixture
def make_vectordb_args(make_chunking_args):
    """
    Input: the make_chunking_args fixture.
    Output: a function that builds a VectorDbArgs.
    What it does: returns a builder with small defaults (batch size 3) where any field can be overridden.
    """
    def build_vectordb_args(**overrides):
        """
        Input: any VectorDbArgs field as a keyword argument.
        Output: a VectorDbArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden.
        """
        field_values = {}
        field_values["chunking_args"] = make_chunking_args()
        field_values["vectordb_type"] = "qdrant"
        field_values["qdrant_url"] = "http://localhost:6333"
        field_values["qdrant_batch_size"] = 3
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return VectorDbArgs(**field_values)

    return build_vectordb_args


@pytest.fixture
def make_document():
    """
    Input: none.
    Output: a function that builds a Document from an id and text.
    What it does: returns a builder that fills in minimal valid metadata.
    """
    def build_document(doc_id, text):
        """
        Input: doc_id and text.
        Output: a pdf Document with that id and text.
        What it does: builds the Document with placeholder metadata.
        """
        metadata = Metadata(source="file.pdf", file_size_bytes=1, content_hash="hash")
        return Document(id=doc_id, type=DocumentType.PDF, text=text, metadata=metadata)

    return build_document


@pytest.fixture
def make_chunks_and_embeddings():
    """
    Input: none.
    Output: a function that builds matching chunks and embeddings.
    What it does: returns a builder of chunk_count chunks over doc_count documents, with ids "doc{d}_{n}".
    """
    def build_chunks_and_embeddings(chunk_count, doc_count=2):
        """
        Input: chunk_count, the number of chunks; doc_count, the number of documents.
        Output: (chunks, embeddings), each a dict keyed by chunk_id.
        What it does: chunk i gets the text "word{i} shared" and a fixed random unit vector.
        """
        random_generator = np.random.default_rng(0)
        chunks = {}
        embeddings = {}
        for chunk_index in range(chunk_count):
            doc_index = chunk_index % doc_count
            chunk_id = f"doc{doc_index}_{chunk_index}"
            chunks[chunk_id] = Chunk(chunk_id, f"word{chunk_index} shared", 2, "fixed")

            vector = random_generator.normal(size=DIMENSION).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            embeddings[chunk_id] = Embedded(chunk_id, vector, True)
        return chunks, embeddings

    return build_chunks_and_embeddings
