"""
test_chunking.py

Tests for l2_chunking.chunking.Chunking, the l2 orchestrator. The real
SentenceTransformer is replaced by the conftest FakeSentenceTransformer,
and chunks are written under pytest's tmp_path.
"""

import json
import os

import pytest

from common.paths import get_chunk_path
from l2_chunking.chunking import Chunking


def make_documents(make_document):
    """
    Input: the make_document fixture.
    Output: a dict of doc_id to Document with two documents.
    What it does: builds a cat document and a dog document.
    """
    documents = {}
    documents["d1"] = make_document("d1", "The cat sat. The cat ran. The cat slept.")
    documents["d2"] = make_document("d2", "Dogs bark loudly. Dogs run fast.")
    return documents


def count_json_files(folder):
    """
    Input: folder, a folder path.
    Output: the number of .json files in it.
    What it does: counts the saved document cache files.
    """
    json_file_count = 0
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            json_file_count += 1
    return json_file_count


def test_init_loads_model_by_name_and_uses_its_tokenizer(fake_sentence_transformer, make_chunking_args):
    """
    Input: ChunkingArgs with model_name "fake-model".
    Output: the Chunking object's model and tokenizer.
    What it does: checks the model is loaded with model_name and its
    tokenizer is kept.
    """
    chunking = Chunking(make_chunking_args(model_name="fake-model"))

    assert chunking.model.model_name == "fake-model"
    assert chunking.tokenizer is chunking.model.tokenizer


def test_compute_chunks_raises_for_invalid_args_and_writes_nothing(fake_sentence_transformer, make_chunking_args, make_document, tmp_path):
    """
    Input: ChunkingArgs with min_chunk_size larger than max_chunk_size.
    Output: ValueError, and the output folder after the call.
    What it does: checks validation runs before any chunk is written.
    """
    chunking = Chunking(make_chunking_args(min_chunk_size=10, max_chunk_size=6))
    output_path = str(tmp_path)

    with pytest.raises(ValueError):
        chunking.compute_chunks(make_documents(make_document), output_path)

    assert os.listdir(output_path) == []


@pytest.mark.parametrize("chunk_type", ["fixed", "sentence", "semantic"])
def test_compute_chunks_chunks_every_document_and_saves_them(fake_sentence_transformer, make_chunking_args, make_document, tmp_path, chunk_type):
    """
    Input: two documents, for each chunk type.
    Output: the returned chunks and the saved chunk files.
    What it does: checks both documents have chunks of the right type and
    each document is saved as one file in the chunk folder.
    """
    chunking_args = make_chunking_args(chunk_type=chunk_type)
    chunking = Chunking(chunking_args)
    output_path = str(tmp_path)

    chunks = chunking.compute_chunks(make_documents(make_document), output_path)

    assert "d1_0" in chunks
    assert "d2_0" in chunks
    for chunk_id in chunks:
        assert chunks[chunk_id].type == chunk_type
    chunk_path = get_chunk_path(output_path, chunking_args)
    assert count_json_files(chunk_path) == 2


def test_compute_chunks_loads_saved_chunks_on_second_run(fake_sentence_transformer, make_chunking_args, make_document, tmp_path):
    """
    Input: two runs over the same documents, with one saved document file
    edited between the runs.
    Output: the chunks from the second run.
    What it does: checks the second run returns the saved file's content
    instead of chunking the document again.
    """
    chunking_args = make_chunking_args()
    chunking = Chunking(chunking_args)
    output_path = str(tmp_path)
    documents = make_documents(make_document)
    chunking.compute_chunks(documents, output_path)

    saved_file_path = os.path.join(get_chunk_path(output_path, chunking_args), "d1.json")
    with open(saved_file_path, encoding="utf-8") as saved_file:
        saved_chunks = json.load(saved_file)
    saved_chunks[0]["text"] = "EDITED IN CACHE"
    with open(saved_file_path, "w", encoding="utf-8") as saved_file:
        json.dump(saved_chunks, saved_file)

    chunks = chunking.compute_chunks(documents, output_path)

    assert chunks["d1_0"].text == "EDITED IN CACHE"


def test_compute_chunks_only_chunks_documents_not_in_cache(fake_sentence_transformer, make_chunking_args, make_document, tmp_path, capsys):
    """
    Input: a first run with d1 only, then a second run with d1 and d2.
    Output: what the second run printed.
    What it does: checks d1 comes from the cache and only d2 is newly chunked.
    """
    chunking = Chunking(make_chunking_args())
    output_path = str(tmp_path)
    documents = make_documents(make_document)
    first_run_documents = {}
    first_run_documents["d1"] = documents["d1"]
    chunking.compute_chunks(first_run_documents, output_path)
    capsys.readouterr()

    chunking.compute_chunks(documents, output_path)

    printed = capsys.readouterr().out
    assert "Documents : 2 (1 from cache, 1 newly chunked)" in printed


def test_compute_chunks_semantic_embeds_only_uncached_documents(fake_sentence_transformer, make_chunking_args, make_document, tmp_path):
    """
    Input: a semantic first run with d1 only, then a run with d1 and d2.
    Output: the sentences the model embedded in the second run.
    What it does: checks d1's sentences are not embedded again once its
    chunks are cached.
    """
    chunking = Chunking(make_chunking_args(chunk_type="semantic"))
    output_path = str(tmp_path)
    documents = make_documents(make_document)
    first_run_documents = {}
    first_run_documents["d1"] = documents["d1"]
    chunking.compute_chunks(first_run_documents, output_path)
    chunking.model.encoded_sentences = []

    chunking.compute_chunks(documents, output_path)

    assert chunking.model.encoded_sentences == ["Dogs bark loudly.", "Dogs run fast."]


def test_compute_chunks_with_no_documents_returns_empty_dict(fake_sentence_transformer, make_chunking_args, tmp_path):
    """
    Input: an empty documents dict.
    Output: compute_chunks's return value.
    What it does: checks an empty dict is returned without errors.
    """
    chunking = Chunking(make_chunking_args())

    chunks = chunking.compute_chunks({}, str(tmp_path))

    assert chunks == {}
