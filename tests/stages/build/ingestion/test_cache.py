"""
test_cache.py

Tests for stages.build.ingestion.cache: saving ingested Documents as one
{doc_id}.json each plus an ordered doc_ids.json in the processed data
folder, and loading the first dataset_size of them back.
"""

import json
import os

import pytest

from stages.build.ingestion.cache import load_documents
from stages.build.ingestion.cache import save_documents
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def make_document(doc_id, text="Some text."):
    """
    Input: doc_id and text.
    Output: a Document with every metadata field filled in.
    What it does: builds a Document to save and load.
    """
    metadata = Metadata(
        source="owner/dataset#1",
        file_size_bytes=1234,
        content_hash="abc123",
        minhash_signature=[1, 2, 3],
        page_count=7,
        parser="pypdfium2",
        author="An Author",
        title="A Title",
        creation_date="2020-01-01",
    )
    document = Document(id=doc_id, type=DocumentType.PDF, text=text, metadata=metadata)
    return document


def make_documents(doc_ids):
    """
    Input: doc_ids, a list of document ids.
    Output: a dict of doc_id to Document, in the order of doc_ids.
    What it does: builds one Document per id with a text naming the id.
    """
    documents = {}
    for doc_id in doc_ids:
        documents[doc_id] = make_document(doc_id, text=f"Text of {doc_id}.")
    return documents


def test_save_documents_writes_one_file_per_document_and_doc_ids(tmp_path):
    """
    Input: three documents saved under tmp_path.
    Output: the files in tmp_path/data/processed/.
    What it does: checks each {doc_id}.json exists and doc_ids.json holds
    the ids in the saved order, with no temporary files left behind.
    """
    save_documents(str(tmp_path), make_documents(["c", "a", "b"]))

    processed_data_path = tmp_path / "data" / "processed"
    file_names = sorted(os.listdir(processed_data_path))
    assert file_names == ["a.json", "b.json", "c.json", "doc_ids.json"]
    with open(processed_data_path / "doc_ids.json", encoding="utf-8") as f:
        assert json.load(f) == ["c", "a", "b"]


def test_load_documents_returns_every_field_unchanged(tmp_path):
    """
    Input: one saved document.
    Output: the loaded document.
    What it does: checks every field, including metadata, is the same as
    the saved document.
    """
    documents = make_documents(["doc1"])
    save_documents(str(tmp_path), documents)

    loaded_documents = load_documents(str(tmp_path), 1)

    assert loaded_documents["doc1"] == documents["doc1"]
    assert type(loaded_documents["doc1"].type) is DocumentType


def test_load_documents_returns_the_first_dataset_size_in_saved_order(tmp_path):
    """
    Input: four saved documents, loaded with dataset_size 2.
    Output: the loaded documents.
    What it does: checks only the first two ids of doc_ids.json are
    loaded, in that order.
    """
    save_documents(str(tmp_path), make_documents(["d", "b", "a", "c"]))

    loaded_documents = load_documents(str(tmp_path), 2)

    assert list(loaded_documents.keys()) == ["d", "b"]


def test_load_documents_returns_none_when_too_few_are_cached(tmp_path):
    """
    Input: two saved documents, loaded with dataset_size 3.
    Output: load_documents' return value.
    What it does: checks None is returned, so ingestion calls the library.
    """
    save_documents(str(tmp_path), make_documents(["a", "b"]))

    assert load_documents(str(tmp_path), 3) is None


def test_load_documents_returns_none_when_nothing_is_cached(tmp_path):
    """
    Input: an empty output folder.
    Output: load_documents' return value.
    What it does: checks None is returned when there is no doc_ids.json.
    """
    assert load_documents(str(tmp_path), 1) is None


def test_load_documents_raises_when_a_listed_file_is_missing(tmp_path):
    """
    Input: two saved documents, then one document file deleted.
    Output: the raised error.
    What it does: checks a FileNotFoundError names the missing document.
    """
    save_documents(str(tmp_path), make_documents(["a", "b"]))
    os.remove(tmp_path / "data" / "processed" / "b.json")

    with pytest.raises(FileNotFoundError, match="b"):
        load_documents(str(tmp_path), 2)


def test_load_documents_reads_files_saved_in_the_library_layout(tmp_path):
    """
    Input: a {doc_id}.json written by hand with the same fields the
    doc_ingest library writes, and a doc_ids.json listing it.
    Output: the loaded document.
    What it does: checks files moved over from the library's old processed
    folder load as they are, without conversion.
    """
    processed_data_path = tmp_path / "data" / "processed"
    os.makedirs(processed_data_path)
    document_dict = {
        "id": "doc1",
        "type": "pdf",
        "text": "Some text.",
        "metadata": {
            "source": "owner/dataset#1",
            "file_size_bytes": 1234,
            "content_hash": "abc123",
            "minhash_signature": None,
            "page_count": 7,
            "parser": "pypdfium2",
            "author": None,
            "title": None,
            "creation_date": None,
        },
    }
    with open(processed_data_path / "doc1.json", "w", encoding="utf-8") as f:
        json.dump(document_dict, f)
    with open(processed_data_path / "doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(["doc1"], f)

    loaded_documents = load_documents(str(tmp_path), 1)

    assert loaded_documents["doc1"].metadata.source == "owner/dataset#1"
    assert loaded_documents["doc1"].type == DocumentType.PDF
