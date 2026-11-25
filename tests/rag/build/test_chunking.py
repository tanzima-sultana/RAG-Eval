"""
test_chunking.py

Tests for the chunking package: rag.build.chunking.chunking.Chunking (the flow), rag.build.chunking.chunkers.Chunker
(the chunkers) and rag.build.chunking.cache (save and load). The SentenceTransformer is replaced with a fake whose
tokenizer gives one token per word, with its position in the text, so no model is downloaded.
"""

import json
import os

import pytest

from common import paths
from rag.build.chunking.cache import load_chunks
from rag.build.chunking.cache import load_document_chunks
from rag.build.chunking.cache import save_chunks
from rag.build.chunking.chunkers import Chunker
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
        chunk_texts.append(chunks[chunk_id].chunk_texts)
    return chunk_texts


def get_page_numbers(chunks):
    """
    Input: chunks, a dict of chunk_id to Chunk.
    Output: the list of each chunk's page numbers, in order.
    What it does: collects each chunk's page_numbers.
    """
    page_numbers = []
    for chunk_id in chunks:
        page_numbers.append(chunks[chunk_id].page_numbers)
    return page_numbers


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
    chunking = Chunking(make_embedder_args(), make_chunking_args(**overrides), "ingestionhash")

    with pytest.raises(ValueError, match=message):
        chunking.validate()


def test_validate_raises_for_empty_model_name(make_embedder_args, make_chunking_args):
    """
    Input: EmbedderArgs with an empty model_name.
    Output: ValueError.
    What it does: checks validate names model_name.
    """
    chunking = Chunking(make_embedder_args(model_name=""), make_chunking_args(), "ingestionhash")

    with pytest.raises(ValueError, match="model_name"):
        chunking.validate()


# ---------- save_chunks / load_chunks ----------


def make_chunk(document_id, chunk_index, page_numbers):
    """
    Input: document_id; chunk_index; page_numbers.
    Output: a Chunk with id {document_id}_{chunk_index}.
    What it does: builds a small fixed Chunk.
    """
    return Chunk(
        chunk_id=f"{document_id}_{chunk_index}",
        chunk_texts="one two",
        chunk_size=2,
        chunk_type="fixed",
        source="file.pdf",
        page_numbers=page_numbers,
    )


def test_save_and_load_chunks_round_trip(tmp_path):
    """
    Input: the chunks of two documents saved under one chunking hash and loaded back.
    Output: the loaded chunks.
    What it does: checks the chunks come back the same, in the same order, and an unknown chunking hash gives None.
    """
    chunks = {}
    for chunk in [make_chunk("d1", 0, [0]), make_chunk("d1", 1, [0, 1]), make_chunk("d2", 0, [0])]:
        chunks[chunk.chunk_id] = chunk

    save_chunks(str(tmp_path), chunks, ["d1", "d2"], "chunkinghash")

    assert load_chunks(str(tmp_path), "chunkinghash") == chunks
    assert load_chunks(str(tmp_path), "otherhash") is None


def test_save_chunks_saves_one_json_per_new_document_and_lists_every_document(tmp_path):
    """
    Input: the chunks of one new document saved with a list of two document ids, one of them saved before.
    Output: the files in the folder.
    What it does: checks only the new document's file is written, and the list holds both ids in the order given.
    """
    old_chunk = make_chunk("d1", 0, [0])
    new_chunk = make_chunk("d2", 0, [0])
    save_chunks(str(tmp_path), {old_chunk.chunk_id: old_chunk}, ["d1"], "firsthash")

    save_chunks(str(tmp_path), {new_chunk.chunk_id: new_chunk}, ["d1", "d2"], "secondhash")

    assert sorted(os.listdir(str(tmp_path))) == ["d1.json", "d2.json", "doc_ids_firsthash.json", "doc_ids_secondhash.json"]
    with open(os.path.join(str(tmp_path), "doc_ids_secondhash.json"), encoding="utf-8") as f:
        assert json.load(f) == ["d1", "d2"]


def test_load_document_chunks_loads_one_document_by_its_id(tmp_path):
    """
    Input: the chunks of one document saved.
    Output: that document's chunks, and the result for an unknown document id.
    What it does: checks the document's chunks come back the same, and an unknown document id gives None.
    """
    chunks = {}
    for chunk in [make_chunk("d1", 0, [0]), make_chunk("d1", 1, [1])]:
        chunks[chunk.chunk_id] = chunk
    save_chunks(str(tmp_path), chunks, ["d1"], "chunkinghash")

    assert load_document_chunks(str(tmp_path), "d1") == chunks
    assert load_document_chunks(str(tmp_path), "d2") is None


def test_load_chunks_gives_none_when_a_listed_document_is_missing(tmp_path):
    """
    Input: a run list naming two documents, with one document's file removed.
    Output: the result of load_chunks.
    What it does: checks load_chunks gives None.
    """
    chunks = {}
    for chunk in [make_chunk("d1", 0, [0]), make_chunk("d2", 0, [0])]:
        chunks[chunk.chunk_id] = chunk
    save_chunks(str(tmp_path), chunks, ["d1", "d2"], "chunkinghash")
    os.remove(os.path.join(str(tmp_path), "d2.json"))

    assert load_chunks(str(tmp_path), "chunkinghash") is None


# ---------- compute_chunks ----------


def test_compute_chunks_saves_the_chunks_and_loads_them_from_cache(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path, capsys):
    """
    Input: two files chunked twice.
    Output: the chunk files, the second run's chunks, what it printed and the models loaded.
    What it does: checks one file per document and the run list, and the second run loads all from cache without a
    model.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    output_path = str(tmp_path)
    files = [make_file("d1", ["one two three four five six seven eight"]), make_file("d2", ["nine ten"])]

    first_chunks, first_hash, first_path = Chunking(embedder_args, chunking_args, "ingestionhash").compute_chunks(files, output_path)
    capsys.readouterr()
    fake_sentence_transformer.loaded_model_names = []
    second_chunks, second_hash, second_path = Chunking(embedder_args, chunking_args, "ingestionhash").compute_chunks(files, output_path)

    chunk_path = paths.get_chunk_path(output_path, embedder_args, chunking_args)
    assert sorted(os.listdir(chunk_path)) == ["d1.json", "d2.json", f"doc_ids_{first_hash}.json"]
    assert second_chunks == first_chunks
    assert second_hash == first_hash
    assert "Files : 2 (2 from cache, 0 newly chunked)" in capsys.readouterr().out
    assert fake_sentence_transformer.loaded_model_names == []


def test_compute_chunks_returns_the_chunking_hash_and_the_chunk_path_of_its_run(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path):
    """
    Input: one file chunked.
    Output: the chunking hash and the chunk path it returns.
    What it does: checks the hash is the one get_chunking_hash gives for the embedder, chunk settings and ingestion hash,
    and the path is the one get_chunk_path gives.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    chunks, chunking_hash, chunking_path = Chunking(embedder_args, chunking_args, "ingestionhash").compute_chunks([make_file("d1", ["one two"])], str(tmp_path))

    assert chunking_hash == paths.get_chunking_hash(embedder_args, chunking_args, "ingestionhash")
    assert chunking_path == paths.get_chunk_path(str(tmp_path), embedder_args, chunking_args)


def test_compute_chunks_chunks_only_the_files_without_saved_chunks(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path, capsys):
    """
    Input: one file chunked, then that file and a new one chunked with another ingestion hash.
    Output: the second run's chunks, what it printed and the models loaded.
    What it does: checks the settings folder is shared, so the first file's chunks are loaded, only the new file is
    chunked, both files' chunks are returned in file order, and the chunking hash is another one.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    first_file = make_file("d1", ["one two three"])
    first_chunks, first_hash, first_path = Chunking(embedder_args, chunking_args, "firstingestionhash").compute_chunks([first_file], str(tmp_path))
    fake_sentence_transformer.loaded_model_names = []
    capsys.readouterr()

    second_chunks, second_hash, second_path = Chunking(embedder_args, chunking_args, "secondingestionhash").compute_chunks([first_file, make_file("d2", ["four five"])], str(tmp_path))

    assert list(second_chunks.keys()) == ["d1_0", "d2_0"]
    assert second_chunks["d1_0"] == first_chunks["d1_0"]
    assert second_hash != first_hash
    assert "Files : 2 (1 from cache, 1 newly chunked)" in capsys.readouterr().out
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_compute_chunks_chunks_again_a_file_whose_content_changed(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path):
    """
    Input: a file chunked, then a file with another document id and other text chunked.
    Output: the second run's chunks and the models loaded.
    What it does: checks the changed file has no saved chunks, so it is chunked and gets chunk ids of its new document id.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    Chunking(embedder_args, chunking_args, "firstingestionhash").compute_chunks([make_file("old", ["one two three"])], str(tmp_path))
    fake_sentence_transformer.loaded_model_names = []

    chunks, chunking_hash, chunking_path = Chunking(embedder_args, chunking_args, "secondingestionhash").compute_chunks([make_file("new", ["one two three edited"])], str(tmp_path))

    assert list(chunks.keys()) == ["new_0"]
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_compute_chunks_raises_for_invalid_settings(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path):
    """
    Input: ChunkingArgs with chunk_type "paragraph".
    Output: ValueError.
    What it does: checks compute_chunks validates the settings before chunking.
    """
    chunking = Chunking(make_embedder_args(), make_chunking_args(chunk_type="paragraph"), "ingestionhash")

    with pytest.raises(ValueError, match="paragraph"):
        chunking.compute_chunks([make_file("d1", ["some text"])], str(tmp_path))


@pytest.mark.parametrize("chunk_type", ["fixed", "sentence", "semantic"])
def test_compute_chunks_calls_the_chunker_of_the_chunk_type(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file, tmp_path, chunk_type):
    """
    Input: one file chunked with each chunk type.
    Output: the chunks.
    What it does: checks compute_chunks runs the chunker of its chunk type, so each chunk has that type.
    """
    chunking = Chunking(make_embedder_args(), make_chunking_args(chunk_type=chunk_type), "ingestionhash")

    chunks, chunking_hash, chunking_path = chunking.compute_chunks([make_file("d1", ["One two three. Four five six."])], str(tmp_path))

    assert len(chunks) > 0
    for chunk_id in chunks:
        assert chunks[chunk_id].chunk_type == chunk_type


# ---------- fixed_chunking ----------


def test_fixed_splits_into_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a 10-word file with max_chunk_size 4 and overlap 1.
    Output: the chunks and the models loaded.
    What it does: checks the windows, ids, sizes, type and source, and the model is loaded once by name.
    """
    chunking_args = make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.fixed_chunking([make_file("doc", ["a b c d e f g h i j"])])

    assert get_chunk_texts(chunks) == ["a b c d", "d e f g", "g h i j"]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2"]
    assert chunks["doc_0"].chunk_id == "doc_0"
    assert chunks["doc_0"].chunk_size == 4
    assert chunks["doc_0"].chunk_type == "fixed"
    assert chunks["doc_0"].source == "file.pdf"
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_fixed_chunk_text_is_an_exact_slice_of_document_texts(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a file whose words are separated by double spaces and a newline.
    Output: the chunks.
    What it does: checks the chunk text keeps the original spacing, as an exact slice of document_texts.
    """
    chunking_args = make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.fixed_chunking([make_file("doc", ["a  b\nc d  e f"])])

    assert get_chunk_texts(chunks) == ["a  b\nc d", "e f"]


def test_fixed_chunk_across_a_page_break_has_both_page_numbers(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a file of two pages, "a b c" and "d e f", with max_chunk_size 4 and no overlap.
    Output: the chunks.
    What it does: checks the first chunk runs from page 0 into page 1 and has both page numbers, the second is on page 1.
    """
    chunking_args = make_chunking_args(min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=0)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.fixed_chunking([make_file("doc", ["a b c", "d e f"])])

    assert get_chunk_texts(chunks) == ["a b c\n\fd", "e f"]
    assert get_page_numbers(chunks) == [[0, 1], [1]]


def test_fixed_chunk_ids_keep_running_over_the_pages_of_a_document(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a file of two pages with max_chunk_size 2 and no overlap.
    Output: the chunk ids.
    What it does: checks the numbering is {document_id}_{n} over the whole document, not restarting on each page.
    """
    chunking_args = make_chunking_args(min_chunk_size=1, max_chunk_size=2, fixed_chunk_overlap=0)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.fixed_chunking([make_file("doc", ["a b", "c d"])])

    assert list(chunks.keys()) == ["doc_0", "doc_1"]
    assert get_page_numbers(chunks) == [[0], [1]]


def test_fixed_keeps_a_short_last_window_that_cannot_merge(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a 7-word file with max_chunk_size 6, overlap 0 and min_chunk_size 2.
    Output: the chunks.
    What it does: checks the 1-word last window cannot join the 6-word window before it without going over 6, so it
    is kept as it is.
    """
    chunking_args = make_chunking_args(min_chunk_size=2, max_chunk_size=6, fixed_chunk_overlap=0)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.fixed_chunking([make_file("doc", ["a b c d e f g"])])

    assert get_chunk_texts(chunks) == ["a b c d e f", "g"]


# ---------- sentence_chunking ----------


def test_sentence_packs_whole_sentences_up_to_max_size(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: sentences of 3, 3 and 1 words with max_chunk_size 6.
    Output: the chunks.
    What it does: checks the first two sentences fill one chunk, as an exact slice, and the third starts a new one.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="sentence", max_chunk_size=6))

    chunks = chunker.sentence_chunking([make_file("doc", ["One two three. Four five six. Seven."])])

    assert get_chunk_texts(chunks) == ["One two three. Four five six.", "Seven."]
    assert chunks["doc_0"].chunk_size == 6
    assert chunks["doc_0"].chunk_type == "sentence"


def test_sentence_cuts_a_too_long_sentence_with_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a 2-word, a 10-word and a 1-word sentence, with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks the long sentence is cut into overlapping windows, with continued ids.
    """
    chunking_args = make_chunking_args(chunk_type="sentence", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.sentence_chunking([make_file("doc", ["Short one. a b c d e f g h i j. End."])])

    assert get_chunk_texts(chunks) == ["Short one.", "a b c d", "d e f g", "g h i j.", "End."]
    assert list(chunks.keys()) == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]


def test_sentence_across_a_page_break_stays_in_one_chunk_with_both_page_numbers(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: one sentence that starts on page 0 and ends on page 1.
    Output: the chunks.
    What it does: checks the sentence is not cut at the page break and its chunk has both page numbers.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="sentence", min_chunk_size=1, max_chunk_size=6))

    chunks = chunker.sentence_chunking([make_file("doc", ["Alpha beta gamma", "delta epsilon."])])

    assert get_chunk_texts(chunks) == ["Alpha beta gamma\n\fdelta epsilon."]
    assert get_page_numbers(chunks) == [[0, 1]]


# ---------- semantic_chunking ----------


def test_semantic_starts_new_chunk_at_topic_change(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: two cat sentences then two dog sentences, all fitting in max 20.
    Output: the chunks and the models loaded.
    What it does: checks the chunk is split where the topic changes, and the model is loaded once by name.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=20))

    chunks = chunker.semantic_chunking([make_file("doc", ["The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."])])

    assert get_chunk_texts(chunks) == ["The cat sat. The cat ran.", "Dogs bark loudly. Dogs run fast."]
    assert chunks["doc_0"].chunk_type == "semantic"
    assert fake_sentence_transformer.loaded_model_names == ["fake-model"]


def test_semantic_merges_short_topics_into_one_chunk(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: the same cat and dog sentences as the topic-change test, with min_chunk_size 7 and max_chunk_size 20.
    Output: the chunks.
    What it does: checks the 6-word cat chunk is shorter than 7, so it is merged with the dog chunk into one chunk.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic", min_chunk_size=7, max_chunk_size=20))

    chunks = chunker.semantic_chunking([make_file("doc", ["The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."])])

    assert get_chunk_texts(chunks) == ["The cat sat. The cat ran. Dogs bark loudly. Dogs run fast."]


def test_semantic_cuts_a_too_long_chunk_with_overlapping_windows(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: two dog sentences of 6 words in total, with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks a semantic chunk longer than max_chunk_size is cut into overlapping windows.
    """
    chunking_args = make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1)
    chunker = Chunker(make_embedder_args(), chunking_args)

    chunks = chunker.semantic_chunking([make_file("doc", ["Dogs bark very loudly. Dogs run."])])

    assert get_chunk_texts(chunks) == ["Dogs bark very loudly.", "loudly. Dogs run."]


def test_semantic_chunk_across_a_page_break_has_both_page_numbers(fake_sentence_transformer, make_embedder_args, make_chunking_args, make_file):
    """
    Input: a cat topic that runs from page 0 into page 1, then a dog topic on page 1.
    Output: the chunks.
    What it does: checks the cat chunk, which spans the page break, has both page numbers and the dog chunk has page 1.
    """
    chunker = Chunker(make_embedder_args(), make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=20))

    chunks = chunker.semantic_chunking([make_file("doc", ["The cat sat. The cat", "ran. Dogs bark loudly. Dogs run fast."])])

    assert get_chunk_texts(chunks) == ["The cat sat. The cat\n\fran.", "Dogs bark loudly. Dogs run fast."]
    assert get_page_numbers(chunks) == [[0, 1], [1]]
