"""
test_langchain.py

Tests for l2_chunking.langchain.Langchain, the LangChain-based chunker.
The SentenceTransformer that LangChain loads is replaced with a fake whose
tokenizer gives one token per word, so no model is downloaded.
"""

import os

import numpy as np
import pytest
import sentence_transformers

from common.paths import get_chunk_path
from l2_chunking.chunkers.sentence import SentenceChunking
from l2_chunking.chunking import Chunking
from l2_chunking.langchain import Langchain
from l2_chunking.langchain import SentenceTransformerEmbeddings


class FakeLangchainTokenizer:
    """
    Stand-in for a Hugging Face tokenizer as LangChain calls it: one token
    per word, with "[CLS]" and "[SEP]" added around the text unless
    add_special_tokens is False.
    """

    def encode(self, text, max_length=None, truncation=None, add_special_tokens=True):
        """
        Input: text; max_length and truncation (ignored); add_special_tokens.
        Output: the list of words, wrapped in "[CLS]" and "[SEP]" when
        add_special_tokens is True.
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


class FakeLangchainModel:
    """
    Stand-in for sentence_transformers.SentenceTransformer as LangChain
    uses it: a tokenizer, a max_seq_length of 256, and an encode method
    that embeds a sentence containing "cat" as [1, 0] and any other
    sentence as [0, 1]. Every model name loaded is recorded in
    loaded_model_names.
    """

    loaded_model_names = []

    def __init__(self, model_name=None, device=None):
        """
        Input: model_name and device, as LangChain passes them.
        Output: nothing.
        What it does: stores both and creates the fake tokenizer.
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeLangchainTokenizer()
        self.max_seq_length = 256
        FakeLangchainModel.loaded_model_names.append(model_name)

    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        """
        Input: sentences, a list of strings; batch_size and
        show_progress_bar (ignored).
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


@pytest.fixture
def fake_langchain_model(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeLangchainModel class.
    What it does: makes LangChain load FakeLangchainModel instead of the
    real SentenceTransformer for the duration of the test.
    """
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeLangchainModel)
    FakeLangchainModel.loaded_model_names = []
    return FakeLangchainModel


def get_chunk_texts(chunks):
    """
    Input: chunks, a dict of chunk_id to Chunk.
    Output: the list of chunk texts in order.
    What it does: collects each chunk's text.
    """
    chunk_texts = []
    for chunk_id in chunks:
        chunk_texts.append(chunks[chunk_id].text)
    return chunk_texts



def make_documents(make_document, text):
    """
    Input: the make_document fixture and one text.
    Output: a dict with one Document, "doc", holding that text.
    What it does: builds the documents dict the chunking methods take.
    """
    documents = {}
    documents["doc"] = make_document("doc", text)
    return documents


# ---------- compute_chunks ----------


def test_compute_chunks_saves_to_langchain_folder_and_loads_from_cache(fake_langchain_model, make_chunking_args, make_document, tmp_path, capsys):
    """
    Input: two documents chunked twice with LangChain fixed chunking.
    Output: the chunk files, the second run's chunks and what it printed.
    What it does: checks chunks are saved under chunks/langchain/, and the
    second run loads every document from that cache without loading the
    model again.
    """
    chunking_args = make_chunking_args(framework="langchain")
    langchain_chunker = Langchain(chunking_args)
    output_path = str(tmp_path)
    documents = {}
    documents["d1"] = make_document("d1", "one two three four five six seven eight")
    documents["d2"] = make_document("d2", "nine ten")

    first_chunks = langchain_chunker.compute_chunks(documents, output_path)
    capsys.readouterr()
    FakeLangchainModel.loaded_model_names = []
    second_chunks = langchain_chunker.compute_chunks(documents, output_path)

    chunk_path = get_chunk_path(output_path, chunking_args)
    assert "/chunks/langchain/fixed/" in chunk_path
    assert sorted(os.listdir(chunk_path)) == ["d1.json", "d2.json"]
    assert second_chunks == first_chunks
    assert "Documents : 2 (2 from cache, 0 newly chunked)" in capsys.readouterr().out
    assert FakeLangchainModel.loaded_model_names == []


def test_compute_chunks_raises_for_unknown_chunk_type(fake_langchain_model, make_chunking_args, make_document, tmp_path):
    """
    Input: langchain ChunkingArgs with chunk_type "paragraph".
    Output: NotImplementedError.
    What it does: checks only fixed, sentence and semantic are supported.
    """
    langchain_chunker = Langchain(make_chunking_args(framework="langchain", chunk_type="paragraph"))

    with pytest.raises(NotImplementedError, match="paragraph"):
        langchain_chunker.compute_chunks(make_documents(make_document, "some text"), str(tmp_path))


def test_chunking_routes_langchain_framework_to_langchain(fake_langchain_model, fake_sentence_transformer, make_chunking_args, make_document, tmp_path):
    """
    Input: Chunking with framework "langchain" and one document.
    Output: the chunks and the chunk folder.
    What it does: checks Chunking.compute_chunks hands the work to
    Langchain, which saves under chunks/langchain/.
    """
    chunking_args = make_chunking_args(framework="langchain")
    output_path = str(tmp_path)

    chunks = Chunking(chunking_args).compute_chunks(make_documents(make_document, "one two three"), output_path)

    assert get_chunk_texts(chunks) == ["one two three"]
    assert os.path.exists(os.path.join(output_path, "chunks", "langchain", "fixed"))


def test_chunking_does_not_load_model_for_langchain(fake_langchain_model, fake_sentence_transformer, make_chunking_args):
    """
    Input: Chunking with framework "langchain".
    Output: the Chunking object's model and tokenizer.
    What it does: checks Chunking does not load its own model, since
    Langchain loads one itself.
    """
    chunking = Chunking(make_chunking_args(framework="langchain"))

    assert chunking.model is None
    assert chunking.tokenizer is None


# ---------- fixed_chunking ----------


def test_fixed_splits_into_overlapping_windows(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: a 10-word document with max_chunk_size 4, overlap 1 and model "fake-model".
    Output: the chunks, and the model names loaded.
    What it does: checks the windows, ids, sizes and type, that LangChain
    stops once a window reaches the end, and that the model is loaded
    once by name.
    """
    chunking_args = make_chunking_args(framework="langchain", model_name="fake-model", max_chunk_size=4, fixed_chunk_overlap=1, min_chunk_size=1)
    langchain_chunker = Langchain(chunking_args)

    chunks_by_document = langchain_chunker.fixed_chunking(make_documents(make_document, "a b c d e f g h i j"))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["a b c d", "d e f g", "g h i j"]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2"]
    assert chunks["doc_0"].size == 4
    assert chunks["doc_0"].type == "fixed"
    assert FakeLangchainModel.loaded_model_names == ["fake-model"]


def test_fixed_empty_document_gives_no_chunks(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks no chunks are made.
    """
    langchain_chunker = Langchain(make_chunking_args(framework="langchain"))

    chunks_by_document = langchain_chunker.fixed_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}


# ---------- sentence_chunking ----------


def test_sentence_packs_whole_sentences_up_to_max_size(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: sentences of 3, 3 and 1 words with max_chunk_size 6.
    Output: the chunks, and the model names loaded.
    What it does: checks the first two sentences fill one chunk, the
    third starts a new one, and the model is loaded once by name.
    """
    langchain_chunker = Langchain(make_chunking_args(framework="langchain", chunk_type="sentence", model_name="fake-model", max_chunk_size=6))

    chunks_by_document = langchain_chunker.sentence_chunking(make_documents(make_document, "One two three. Four five six. Seven."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["One two three. Four five six.", "Seven."]
    assert chunks["doc_0"].size == 6
    assert chunks["doc_0"].type == "sentence"
    assert FakeLangchainModel.loaded_model_names == ["fake-model"]


def test_sentence_cuts_a_too_long_sentence_with_overlapping_windows(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: a 2-word sentence, a 10-word sentence and a 1-word sentence, with
    max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks the long sentence is cut into overlapping windows
    between the chunks before and after it, with continued ids.
    """
    chunking_args = make_chunking_args(framework="langchain", chunk_type="sentence", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    langchain_chunker = Langchain(chunking_args)

    chunks_by_document = langchain_chunker.sentence_chunking(make_documents(make_document, "Short one. a b c d e f g h i j. End."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["Short one.", "a b c d", "d e f g", "g h i j.", "End."]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]


def test_sentence_empty_document_gives_no_chunks(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks no chunks are made.
    """
    langchain_chunker = Langchain(make_chunking_args(framework="langchain", chunk_type="sentence"))

    chunks_by_document = langchain_chunker.sentence_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}


def test_sentence_matches_built_in_sentence_chunking(fake_langchain_model, make_chunker, make_chunking_args, make_document):
    """
    Input: one document with short sentences, a sentence too long for
    max_chunk_size, and more short sentences, chunked by both LangChain and
    the built-in SentenceChunking with the same settings.
    Output: both sets of chunks.
    What it does: checks LangChain sentence chunking gives the same chunks
    as the built-in one when both count one token per word.
    """
    text = (
        "The first sentence is here. A second one follows. "
        "This sentence is far too long to fit into a single chunk of the size we allow. "
        "Short again. Another short one. And the last sentence of the document."
    )
    documents = make_documents(make_document, text)
    built_in_chunker = make_chunker(SentenceChunking, make_chunking_args(chunk_type="sentence", min_chunk_size=2, max_chunk_size=8, fixed_chunk_overlap=2))
    langchain_chunker = Langchain(make_chunking_args(framework="langchain", chunk_type="sentence", min_chunk_size=2, max_chunk_size=8, fixed_chunk_overlap=2))

    built_in_chunks = built_in_chunker.chunk(documents["doc"])
    langchain_chunks = langchain_chunker.sentence_chunking(documents)["doc"]

    assert langchain_chunks == built_in_chunks


# ---------- semantic_chunking ----------


def test_embeddings_wrapper_returns_lists_of_floats():
    """
    Input: a SentenceTransformerEmbeddings around the fake model, and two texts.
    Output: embed_documents and embed_query results.
    What it does: checks embeddings come back as plain lists, one per text.
    """
    embeddings = SentenceTransformerEmbeddings(FakeLangchainModel())

    assert embeddings.embed_documents(["a cat", "a dog"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert embeddings.embed_query("a cat") == [1.0, 0.0]


def test_semantic_starts_new_chunk_at_topic_change(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: two cat sentences then two dog sentences, all fitting in max 20,
    with model "fake-model".
    Output: the chunks, and the model names loaded.
    What it does: checks the chunk is split where the topic changes, and
    the model is loaded once by name for both embedding and tokenizing.
    """
    chunking_args = make_chunking_args(framework="langchain", chunk_type="semantic", model_name="fake-model", min_chunk_size=1, max_chunk_size=20)
    langchain_chunker = Langchain(chunking_args)

    chunks_by_document = langchain_chunker.semantic_chunking(make_documents(make_document, "The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["The cat sat. The cat ran.", "Dogs bark loudly. Dogs run fast."]
    assert chunks["doc_0"].type == "semantic"
    assert FakeLangchainModel.loaded_model_names == ["fake-model"]


def test_semantic_cuts_a_too_long_chunk_with_overlapping_windows(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: two dog sentences of 7 words in total, with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks a semantic chunk longer than max_chunk_size is cut
    into overlapping token windows.
    """
    chunking_args = make_chunking_args(framework="langchain", chunk_type="semantic", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    langchain_chunker = Langchain(chunking_args)

    chunks_by_document = langchain_chunker.semantic_chunking(make_documents(make_document, "Dogs bark very loudly. Dogs run."))

    assert get_chunk_texts(chunks_by_document["doc"]) == ["Dogs bark very loudly.", "loudly. Dogs run."]


def test_semantic_empty_document_gives_no_chunks(fake_langchain_model, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks the empty piece SemanticChunker returns is dropped.
    """
    langchain_chunker = Langchain(make_chunking_args(framework="langchain", chunk_type="semantic"))

    chunks_by_document = langchain_chunker.semantic_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}
