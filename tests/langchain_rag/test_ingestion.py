"""
test_ingestion.py

Tests for langchain_rag.ingestion. PyPDFium2Loader is replaced with a fake that reads each file as plain text,
so the test files do not need to be real PDFs.
"""

import hashlib
import json
import os

import pytest
from langchain_core.documents import Document as LangchainDocument

from common.args_temp import IngestionArgs
from langchain_rag import ingestion as ingestion_module
from langchain_rag.ingestion import convert_langchain_document
from langchain_rag.ingestion import ingestion
from langchain_rag.ingestion import validate


class FakePyPDFium2Loader:
    """
    Stand-in for PyPDFium2Loader: reads the file as text. A file whose text is "FAIL" raises an error.
    Every loaded file path is recorded in loaded_files.
    """

    loaded_files = []

    def __init__(self, file_path, mode="page"):
        """
        Input: file_path and mode, as the real class takes them.
        Output: nothing.
        What it does: stores both.
        """
        self.file_path = file_path
        self.mode = mode

    def load(self):
        """
        Input: none.
        Output: a list with one LangChain Document holding the file's text.
        What it does: records the file and reads it, or raises if its text is "FAIL".
        """
        FakePyPDFium2Loader.loaded_files.append(self.file_path)
        with open(self.file_path, encoding="utf-8") as f:
            text = f.read()
        if text == "FAIL":
            raise RuntimeError("broken file")

        metadata = {"source": self.file_path, "total_pages": 1, "author": "", "title": "A Title"}
        return [LangchainDocument(page_content=text, metadata=metadata)]


@pytest.fixture
def fake_loader(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakePyPDFium2Loader class.
    What it does: makes ingestion use the fake loader.
    """
    monkeypatch.setattr(ingestion_module, "PyPDFium2Loader", FakePyPDFium2Loader)
    FakePyPDFium2Loader.loaded_files = []
    return FakePyPDFium2Loader


def make_input_folder(tmp_path, file_texts):
    """
    Input: tmp_path; file_texts, a dict of file name to text.
    Output: the input folder path as a string.
    What it does: writes each file into tmp_path/raw.
    """
    input_path = tmp_path / "raw"
    input_path.mkdir()
    for file_name in file_texts:
        (input_path / file_name).write_text(file_texts[file_name], encoding="utf-8")
    return str(input_path)


def make_ingestion_args(input_path, output_path, **overrides):
    """
    Input: input_path; output_path; any IngestionArgs field as a keyword argument.
    Output: an IngestionArgs with defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["dataset_size"] = 10
    field_values["dataset_name"] = "dataset"
    field_values["input_path"] = input_path
    field_values["output_path"] = output_path
    field_values["document_type"] = "pdf"
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return IngestionArgs(**field_values)


def get_documents_path(output_path):
    """
    Input: output_path, the root output folder.
    Output: the folder ingestion saves documents to.
    What it does: joins the langchain framework-first layout.
    """
    return os.path.join(output_path, "langchain", "dataset", "documents")


# ---------- validate ----------


@pytest.mark.parametrize("field_name, field_value, message", [
    ("dataset_size", 0, "dataset_size"),
    ("dataset_name", "", "dataset_name"),
    ("input_path", "missing/folder", "input_path"),
    ("output_path", "", "output_path"),
    ("document_type", "docx", "document_type"),
])
def test_validate_raises_for_invalid_setting(tmp_path, field_name, field_value, message):
    """
    Input: IngestionArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    ingestion_args = make_ingestion_args(str(tmp_path), str(tmp_path / "out"))
    setattr(ingestion_args, field_name, field_value)

    with pytest.raises(ValueError, match=message):
        validate(ingestion_args)


# ---------- convert_langchain_document ----------


def test_convert_langchain_document_fills_id_and_metadata(tmp_path):
    """
    Input: a LangChain Document of a file, with an empty author.
    Output: our Document.
    What it does: checks the id is the first 16 characters of the file hash and the metadata is copied.
    """
    document_file = tmp_path / "a.pdf"
    document_file.write_bytes(b"some bytes")
    langchain_metadata = {"source": str(document_file), "total_pages": 3, "author": "", "title": "A Title"}
    langchain_document = LangchainDocument(page_content="Some text.", metadata=langchain_metadata)

    document = convert_langchain_document(langchain_document, "pdf")

    content_hash = hashlib.sha256(b"some bytes").hexdigest()
    assert document.id == content_hash[:16]
    assert document.text == "Some text."
    assert document.type.value == "pdf"
    assert document.metadata.content_hash == content_hash
    assert document.metadata.file_size_bytes == 10
    assert document.metadata.page_count == 3
    assert document.metadata.parser == "langchain_pypdfium2"
    assert document.metadata.author is None
    assert document.metadata.title == "A Title"
    assert document.metadata.creation_date is None


# ---------- ingestion ----------


def test_ingestion_reads_only_document_type_files_up_to_dataset_size(tmp_path, fake_loader):
    """
    Input: three pdf files and one txt file, with dataset_size 2.
    Output: the documents and the loaded files.
    What it does: checks only the first two pdf files in name order are loaded.
    """
    input_path = make_input_folder(tmp_path, {"c.pdf": "text c", "a.pdf": "text a", "b.pdf": "text b", "d.txt": "text d"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)

    documents = ingestion(ingestion_args)

    loaded_file_names = []
    for loaded_file in FakePyPDFium2Loader.loaded_files:
        loaded_file_names.append(os.path.basename(loaded_file))
    assert loaded_file_names == ["a.pdf", "b.pdf"]

    document_texts = []
    for doc_id in documents:
        document_texts.append(documents[doc_id].text)
    assert document_texts == ["text a", "text b"]


def test_ingestion_skips_empty_and_failed_files(tmp_path, fake_loader, capsys):
    """
    Input: one good file, one file with only whitespace and one file the loader fails on.
    Output: the documents and what was printed.
    What it does: checks only the good file becomes a document and the counts are printed.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "  \n ", "c.pdf": "FAIL"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    documents = ingestion(ingestion_args)

    assert len(documents) == 1
    assert "Files read : 3, loaded : 1, failed : 1, empty : 1" in capsys.readouterr().out


def test_ingestion_saves_documents_and_doc_ids(tmp_path, fake_loader):
    """
    Input: two pdf files with dataset_size 2.
    Output: the files in the documents folder.
    What it does: checks one {doc_id}.json per document and doc_ids_2.json listing them.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    output_path = str(tmp_path / "out")

    documents = ingestion(make_ingestion_args(input_path, output_path, dataset_size=2))

    documents_path = get_documents_path(output_path)
    expected_file_names = ["doc_ids_2.json"]
    for doc_id in documents:
        expected_file_names.append(f"{doc_id}.json")
    assert sorted(os.listdir(documents_path)) == sorted(expected_file_names)

    with open(os.path.join(documents_path, "doc_ids_2.json"), encoding="utf-8") as f:
        saved_doc_ids = json.load(f)
    assert saved_doc_ids == list(documents.keys())


def test_ingestion_loads_from_cache_on_second_run(tmp_path, fake_loader, capsys):
    """
    Input: the same ingestion run twice.
    Output: the second run's documents and the loaded files.
    What it does: checks the second run loads every document from the cache and never calls the loader.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)

    first_documents = ingestion(ingestion_args)
    FakePyPDFium2Loader.loaded_files = []
    capsys.readouterr()
    second_documents = ingestion(ingestion_args)

    assert second_documents == first_documents
    assert FakePyPDFium2Loader.loaded_files == []
    assert "Documents : 2 (all from cache)" in capsys.readouterr().out


def test_ingestion_dataset_sizes_share_document_files(tmp_path, fake_loader):
    """
    Input: ingestion with dataset_size 1, then dataset_size 2.
    Output: the files in the documents folder.
    What it does: checks each size gets its own doc_ids list, and the shared document is saved once.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    output_path = str(tmp_path / "out")

    ingestion(make_ingestion_args(input_path, output_path, dataset_size=1))
    documents = ingestion(make_ingestion_args(input_path, output_path, dataset_size=2))

    expected_file_names = ["doc_ids_1.json", "doc_ids_2.json"]
    for doc_id in documents:
        expected_file_names.append(f"{doc_id}.json")
    assert sorted(os.listdir(get_documents_path(output_path))) == sorted(expected_file_names)
