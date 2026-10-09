"""
conftest.py

Shared fakes and fixtures for the l2_chunking test suite.

FakeTokenizer counts one token per whitespace-separated word, so chunk
sizes are easy to work out by hand. FakeSentenceTransformer replaces the
real model so no model is downloaded or run.
"""

import types

import numpy as np
import pytest

from common.args import ChunkingArgs
from l1_ingestion.schema import Document
from l1_ingestion.schema import DocumentType
from l1_ingestion.schema import Metadata
from l2_chunking import chunking as chunking_module


class FakeTokenizer:
    """
    Stand-in for a SentenceTransformer tokenizer: one token per word.
    """

    def encode(self, text, add_special_tokens=False):
        """
        Input: text, and add_special_tokens (ignored).
        Output: the list of words in text.
        What it does: splits text on whitespace.
        """
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        """
        Input: tokens, a list of words, and skip_special_tokens (ignored).
        Output: the words joined with single spaces.
        What it does: turns tokens back into text.
        """
        return " ".join(tokens)


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer. A sentence
    containing "cat" is embedded as [1, 0] and any other sentence as
    [0, 1], so cat sentences and other sentences are fully dissimilar.
    Every encoded sentence is recorded in encoded_sentences.
    """

    def __init__(self, model_name=None, device=None):
        """
        Input: model_name and device, as the real model takes them.
        Output: nothing.
        What it does: stores both and creates a FakeTokenizer.
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeTokenizer()
        self.encoded_sentences = []

    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        """
        Input: sentences, a list of strings; batch_size and
        show_progress_bar (ignored).
        Output: a numpy array with one 2-number embedding per sentence.
        What it does: records each sentence and embeds it by the cat rule.
        """
        vectors = []
        for sentence in sentences:
            self.encoded_sentences.append(sentence)
            if "cat" in sentence.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors)


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeSentenceTransformer class.
    What it does: makes Chunking load FakeSentenceTransformer instead of
    the real model for the duration of the test.
    """
    monkeypatch.setattr(chunking_module, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


@pytest.fixture
def make_chunking_args():
    """
    Input: none.
    Output: a function that builds a ChunkingArgs.
    What it does: returns a builder with small default sizes (min 2, max 6,
    overlap 2) where any field can be overridden by keyword.
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
def make_chunker():
    """
    Input: none.
    Output: a function that builds a Chunker subclass instance directly,
    without going through Chunking.
    What it does: returns a builder that wraps chunking_args, a
    FakeTokenizer and optional semantic embeddings in a stand-in for the
    Chunking object a Chunker copies its settings from.
    """
    def build_chunker(chunker_class, chunking_args, all_embeddings=None, embeddings_offset=None):
        """
        Input: chunker_class, the Chunker subclass to build; chunking_args;
        all_embeddings and embeddings_offset for semantic chunking.
        Output: an instance of chunker_class.
        What it does: builds a stand-in Chunking and passes it to the class.
        """
        fake_model = FakeSentenceTransformer()
        chunking = types.SimpleNamespace(
            chunking_args=chunking_args,
            model=fake_model,
            tokenizer=fake_model.tokenizer,
            all_embeddings=all_embeddings,
            embeddings_offset=embeddings_offset,
        )
        return chunker_class(chunking)

    return build_chunker
