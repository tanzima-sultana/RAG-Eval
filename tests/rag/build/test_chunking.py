"""
test_chunking.py

Tests for the chunking package: rag.build.chunking.chunking.Chunking (the flow), rag.build.chunking.chunkers.Chunker
(the chunkers) and rag.build.chunking.cache (save and load). The SentenceTransformer that LangChain and the chunkers
load is replaced with a fake whose tokenizer gives one token per word, so no model is downloaded.
"""

import os

import pytest

from common import paths
from rag.build.chunking.cache import load_chunks
from rag.build.chunking.cache import save_chunks
from rag.build.chunking.chunkers import Chunker
from rag.build.chunking.chunkers import SentenceTransformerEmbeddings
from rag.build.chunking.chunking import Chunking
from schema.chunk import Chunk


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
    Output: a dict with one File, "doc", holding that text.
    What it does: builds the documents dict the chunking methods take.
    """
    documents = {}
    documents["doc"] = make_document("doc", text)
    return documents


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"chunk_type": "paragraph"}, "chunk_type"),
    ({"min_chunk_size": 0}, "min_chunk_size"),
    ({"min_chunk_size": 5, "max_chunk_size": 4}, "max_chunk_size"),
    ({"fixed_chunk_overlap": -1}, "fixed_chunk_overlap"),
    ({"fixed_chunk_overlap": 6, "max_chunk_size": 6}, "fixed_chunk_overlap"),
])
def test_validate_raises_for_invalid_chunking_setting(make_embedder_args, make_chunking_args, overrides, message):
    """
    Input: ChunkingArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    chunking = Chunking(make_embedder_args(), make_chunking_args(**overrides))

    with pytest.raises(ValueError, match=message):
        chunking.validate()


def test_validate_raises_for_empty_model_name(make_embedder_args, make_chunking_args):
    """
    Input: EmbedderArgs with an empty model_name.
    Output: ValueError.
    What it does: checks validate names model_name.
    """
    chunking = Chunking(make_embedder_args(model_name=""), make_chunking_args())

    with pytest.raises(ValueError, match="model_name"):
        chunking.validate()


# ---------- save_chunks / load_chunks ----------


def test_save_and_load_chunks_round_trip(tmp_path):
    """
    Input: one document's chunks saved and loaded back.
    Output: the loaded chunks.
    What it does: checks the chunks come back the same, and a missing document gives None.
    """
    chunks = {}
    chunks["doc_0"] = Chunk("doc_0", "one two", 2, "fixed")
    chunks["doc_1"] = Chunk("doc_1", "three", 1, "fixed")

    save_chunks(str(tmp_path), "doc", chunks)

    assert load_chunks(str(tmp_path), "doc") == chunks
    assert load_chunks(str(tmp_path), "other") is None


# ---------- compute_chunks ----------


def test_compute_chunks_saves_one_file_per_document_and_loads_from_cache(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document, tmp_path, capsys):
    """
    Input: two documents chunked twice.
    Output: the chunk files, the second run's chunks, what it printed and the models loaded.
    What it does: checks one file per document, and the second run loads all from cache without a model.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    output_path = str(tmp_path)
    documents = {}
    documents["d1"] = make_document("d1", "one two three four five six seven eight")
    documents["d2"] = make_document("d2", "nine ten")

    first_chunks = Chunking(embedder_args, chunking_args).compute_chunks(documents, output_path)
    capsys.readouterr()
    fake_sentence_transformer.loaded_model_names = []
    second_chunks = Chunking(embedder_args, chunking_args).compute_chunks(documents, output_path)

    chunk_path = paths.get_chunk_path(output_path, embedder_args, chunking_args)
    assert sorted(os.listdir(chunk_path)) == ["d1.json", "d2.json"]
    assert second_chunks == first_chunks
    assert "Documents : 2 (2 from cache, 0 newly chunked)" in capsys.readouterr().out
    assert fake_sentence_transformer.loaded_model_names == []


def test_compute_chunks_only_chunks_documents_not_cached(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document, tmp_path, capsys):
    """
    Input: one document chunked, then that document and a new one chunked.
    Output: the second run's chunks and what it printed.
    What it does: checks only the new document is chunked and both documents' chunks are returned.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    output_path = str(tmp_path)
    first_documents = {}
    first_documents["d1"] = make_document("d1", "one two three")
    Chunking(embedder_args, chunking_args).compute_chunks(first_documents, output_path)
    capsys.readouterr()

    second_documents = {}
    second_documents["d1"] = make_document("d1", "one two three")
    second_documents["d2"] = make_document("d2", "four five")
    chunks = Chunking(embedder_args, chunking_args).compute_chunks(second_documents, output_path)

    assert list(chunks.keys()) == ["d1_0", "d2_0"]
    assert "Documents : 2 (1 from cache, 1 newly chunked)" in capsys.readouterr().out


def test_compute_chunks_raises_for_invalid_settings(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document, tmp_path):
    """
    Input: ChunkingArgs with chunk_type "paragraph".
    Output: ValueError.
    What it does: checks compute_chunks validates the settings before chunking.
    """
    chunking = Chunking(make_embedder_args(), make_chunking_args(chunk_type="paragraph"))

    with pytest.raises(ValueError, match="paragraph"):
        chunking.compute_chunks(make_documents(make_document, "some text"), str(tmp_path))


@pytest.mark.parametrize("chunk_type", ["fixed", "sentence", "semantic"])
def test_compute_chunks_calls_the_chunker_of_the_chunk_type(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document, tmp_path, chunk_type):
    """
    Input: one document chunked with each chunk type.
    Output: the chunks.
    What it does: checks compute_chunks runs the chunker of its chunk type, so each chunk has that type.
    """
    chunking = Chunking(make_embedder_args(), make_chunking_args(chunk_type=chunk_type))

    chunks = chunking.compute_chunks(make_documents(make_document, "One two three. Four five six."), str(tmp_path))

    for chunk_id in chunks:
        assert chunks[chunk_id].type == chunk_type


# ---------- fixed_chunking ----------


def test_fixed_splits_into_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a 10-word document with max_chunk_size 4 and overlap 1.
    Output: the chunks and the models loaded.
    What it does: checks the windows, ids, sizes and type, and the model is loaded once by name.
    """
    chunking_args = make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks_by_document = chunker.fixed_chunking(make_documents(make_document, "a b c d e f g h i j"))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["a b c d", "d e f g", "g h i j"]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2"]
    assert chunks["doc_0"].size == 4
    assert chunks["doc_0"].type == "fixed"
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_fixed_keeps_a_short_last_window_that_cannot_merge(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a 7-word document with max_chunk_size 6, overlap 0 and min_chunk_size 2.
    Output: the chunks.
    What it does: checks the 1-word last window cannot join the 6-word window before it without going over 6, so
    it is kept as it is.
    """
    chunking_args = make_chunking_args(min_chunk_size=2, max_chunk_size=6, fixed_chunk_overlap=0)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks_by_document = chunker.fixed_chunking(make_documents(make_document, "a b c d e f g"))

    assert get_chunk_texts(chunks_by_document["doc"]) == ["a b c d e f", "g"]


def test_fixed_empty_document_gives_no_chunks(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks no chunks are made.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args())

    chunks_by_document = chunker.fixed_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}


# ---------- sentence_chunking ----------


def test_sentence_packs_whole_sentences_up_to_max_size(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: sentences of 3, 3 and 1 words with max_chunk_size 6.
    Output: the chunks.
    What it does: checks the first two sentences fill one chunk and the third starts a new one.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="sentence", max_chunk_size=6))

    chunks_by_document = chunker.sentence_chunking(make_documents(make_document, "One two three. Four five six. Seven."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["One two three. Four five six.", "Seven."]
    assert chunks["doc_0"].size == 6
    assert chunks["doc_0"].type == "sentence"


def test_sentence_cuts_a_too_long_sentence_with_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a 2-word, a 10-word and a 1-word sentence, with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks the long sentence is cut into overlapping windows, with continued ids.
    """
    chunking_args = make_chunking_args(chunk_type="sentence", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks_by_document = chunker.sentence_chunking(make_documents(make_document, "Short one. a b c d e f g h i j. End."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["Short one.", "a b c d", "d e f g", "g h i j.", "End."]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]


def test_sentence_empty_document_gives_no_chunks(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks no chunks are made.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="sentence"))

    chunks_by_document = chunker.sentence_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}


# ---------- semantic_chunking ----------


def test_embeddings_wrapper_returns_lists_of_floats(fake_sentence_transformer):
    """
    Input: a SentenceTransformerEmbeddings around the fake model, and two texts.
    Output: embed_documents and embed_query results.
    What it does: checks embeddings come back as plain lists, one per text.
    """
    embeddings = SentenceTransformerEmbeddings(fake_sentence_transformer())

    assert embeddings.embed_documents(["a cat", "a dog"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert embeddings.embed_query("a cat") == [1.0, 0.0]


def test_semantic_starts_new_chunk_at_topic_change(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: two cat sentences then two dog sentences, all fitting in max 20.
    Output: the chunks and the models loaded.
    What it does: checks the chunk is split where the topic changes, and the model is loaded once by name.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=20))

    chunks_by_document = chunker.semantic_chunking(make_documents(make_document, "The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."))

    chunks = chunks_by_document["doc"]
    assert get_chunk_texts(chunks) == ["The cat sat. The cat ran.", "Dogs bark loudly. Dogs run fast."]
    assert chunks["doc_0"].type == "semantic"
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_semantic_merges_short_topics_into_one_chunk(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: the same cat and dog sentences as the topic-change test, with min_chunk_size 7 and max_chunk_size 20.
    Output: the chunks.
    What it does: checks the 6-word cat chunk is shorter than 7, so it is merged with the dog chunk into one chunk.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic", min_chunk_size=7, max_chunk_size=20))

    chunks_by_document = chunker.semantic_chunking(make_documents(make_document, "The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."))

    assert get_chunk_texts(chunks_by_document["doc"]) == ["The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."]


def test_semantic_cuts_a_too_long_chunk_with_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: two dog sentences of 7 words in total, with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks a semantic chunk longer than max_chunk_size is cut into overlapping windows.
    """
    chunking_args = make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks_by_document = chunker.semantic_chunking(make_documents(make_document, "Dogs bark very loudly. Dogs run."))

    assert get_chunk_texts(chunks_by_document["doc"]) == ["Dogs bark very loudly.", "loudly. Dogs run."]


def test_semantic_empty_document_gives_no_chunks(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_document):
    """
    Input: a document with empty text.
    Output: the document's chunks.
    What it does: checks the empty piece SemanticChunker returns is dropped.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic"))

    chunks_by_document = chunker.semantic_chunking(make_documents(make_document, ""))

    assert chunks_by_document["doc"] == {}


# ---------- merge_short_pieces ----------


@pytest.mark.parametrize("piece_texts, expected_texts", [
    # A 1-word piece joins the next piece
    (["a", "b c d"], ["a b c d"]),
    # Short pieces keep joining until the joined text reaches min_chunk_size
    (["a", "b", "c d e"], ["a b c d e"]),
    # A short last piece joins the previous piece
    (["a b c d", "e"], ["a b c d e"]),
    # A short piece that would go over max_chunk_size when joined is kept as it is
    (["a", "b c d e f g"], ["a", "b c d e f g"]),
    # A short last piece that would go over max_chunk_size when joined is kept as it is
    (["a b c d e f", "g"], ["a b c d e f", "g"]),
    # Pieces at or above min_chunk_size are not joined
    (["a b c", "d e f"], ["a b c", "d e f"]),
    # No pieces gives no pieces
    ([], []),
])
def test_merge_short_pieces(fake_sentence_transformer, make_embedder_args, make_chunking_args, piece_texts, expected_texts):
    """
    Input: pieces, with min_chunk_size 3 and max_chunk_size 6.
    Output: the merged pieces.
    What it does: checks each merge rule of merge_short_pieces.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(min_chunk_size=3, max_chunk_size=6))
    chunker.tokenizer = fake_sentence_transformer().tokenizer

    assert chunker.merge_short_pieces(piece_texts) == expected_texts
