"""
conftest.py

Shared pytest fixtures for the l2_chunking test suite.
"""

import datetime
import types

import numpy as np
import pytest

from l1_doc_processing.schema import Document


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """
    tmp_path: pytest's built-in temporary directory for this test.
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Changes the current working directory to tmp_path for the
    duration of the test, so any code under test that writes to
    relative paths (data/processed/, chunks/, json/global.json -- see
    common.paths and common.global_values) writes into an isolated
    temporary directory instead of this repo's real files.

    Returns tmp_path, the isolated working directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeTokenizer:
    """
    Stand-in for a SentenceTransformer model's tokenizer: encodes text
    by splitting on whitespace (one token per word) and decodes by
    joining tokens back with a space, so token counts and chunk text
    stay simple and predictable in tests.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


@pytest.fixture
def fake_tokenizer():
    """
    Returns a fresh FakeTokenizer instance.
    """
    return FakeTokenizer()


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer: exposes
    only .tokenizer and .encode(), the two attributes/methods
    Chunking actually uses, so tests never download or run a real
    embedding model. encode() returns the same fixed vector for every
    sentence, so sentences never look dissimilar to each other unless
    a test overrides all_embeddings itself afterward.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeTokenizer()

    def encode(self, sentences, batch_size=256, show_progress_bar=False):
        return np.ones((len(sentences), 2))


@pytest.fixture
def fake_sentence_transformer_class(monkeypatch):
    """
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Replaces l2_chunking.chunking.SentenceTransformer with
    FakeSentenceTransformer, so constructing a Chunking instance never
    downloads or loads a real embedding model.

    Returns the FakeSentenceTransformer class.
    """
    monkeypatch.setattr("l2_chunking.chunking.SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


@pytest.fixture
def make_fake_chunking(fake_tokenizer):
    """
    fake_tokenizer: the FakeTokenizer this factory's chunking
        stand-ins default to.

    Returns a factory that builds a lightweight stand-in for a
    Chunking instance, exposing exactly the attributes
    Chunker.__init__ copies from it (max_chunk_size,
    fixed_chunk_overlap, semantic_threshold, model, tokenizer,
    all_embeddings, embeddings_offset), each with a sensible default
    so a test only has to override what it actually cares about. Used
    to unit-test a concrete Chunker subclass without constructing a
    real Chunking (which would load an embedding model).
    """

    def _make(
        max_chunk_size=10,
        fixed_chunk_overlap=0,
        semantic_threshold=0.5,
        model=None,
        tokenizer=None,
        all_embeddings=None,
        embeddings_offset=None,
    ):
        return types.SimpleNamespace(
            max_chunk_size=max_chunk_size,
            fixed_chunk_overlap=fixed_chunk_overlap,
            semantic_threshold=semantic_threshold,
            model=model,
            tokenizer=tokenizer if tokenizer is not None else fake_tokenizer,
            all_embeddings=all_embeddings,
            embeddings_offset=embeddings_offset,
        )

    return _make


@pytest.fixture
def document_factory():
    """
    Returns a factory that builds a minimally-populated Document for
    chunking tests, where only id and text actually matter -- every
    other field gets a placeholder value.
    """

    def _make(doc_id, text):
        return Document(
            id=doc_id,
            text=text,
            title=None,
            path="irrelevant",
            format="fake",
            metadata={},
            content_hash="irrelevant",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )

    return _make
