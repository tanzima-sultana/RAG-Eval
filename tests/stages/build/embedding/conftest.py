"""
conftest.py

Shared pytest fixtures for the stages.build.embedding test suite.
"""

import numpy as np
import pytest


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """
    tmp_path: pytest's built-in temporary directory for this test.
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Changes the current working directory to tmp_path for the
    duration of the test, so any code under test that writes to
    relative paths (a dataset's data/processed/, chunks/, and
    embeddings/ -- see common.paths) writes into an isolated temporary
    directory instead of this repo's real files.

    Returns tmp_path, the isolated working directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer: exposes
    only .encode(), the one method Embedding actually uses, so tests
    never download or run a real embedding model.

    Every call to encode() is recorded in encode_calls, as a plain
    list of the sentences it was given, so tests can check that
    compute_embeddings batches every pending chunk into a single call
    instead of calling encode() once per chunk.

    The vector returned for a sentence is [float(len(sentence)), 1.0]
    -- deterministic and distinct per sentence length, so tests can
    check each chunk_id ends up mapped back to the right vector.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []

    def encode(self, sentences, batch_size=256, normalize_embeddings=False, show_progress_bar=False):
        self.encode_calls.append(list(sentences))
        vectors = []
        for sentence in sentences:
            vectors.append([float(len(sentence)), 1.0])
        return np.array(vectors)


@pytest.fixture
def fake_sentence_transformer_class(monkeypatch):
    """
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Replaces stages.build.embedding.embedding.SentenceTransformer with
    FakeSentenceTransformer, so constructing an Embedding instance
    never downloads or loads a real embedding model.

    Returns the FakeSentenceTransformer class.
    """
    monkeypatch.setattr("stages.build.embedding.embedding.SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer
