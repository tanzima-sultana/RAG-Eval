"""
test_ingestion.py

Tests for rag.build.ingestion (ingestion.py and cache.py). PyPDFium2Loader is replaced with a fake that reads each
file as plain text and returns one page per "\\f"-separated part, so the test files do not need to be real PDFs.
"""

import hashlib
import json
import os

import pytest
from langchain_core.documents import Document as LangchainDocument

from common.args import IngestionArgs
from common.paths import get_output_path
from rag.build.ingestion import ingestion as ingestion_module
from rag.build.ingestion.cache import load_file
from rag.build.ingestion.cache import load_files
from rag.build.ingestion.cache import save_files
from rag.build.ingestion.ingestion import compute_ingestion_hash
from rag.build.ingestion.ingestion import convert_langchain_pages
from rag.build.ingestion.ingestion import ingestion
from rag.build.ingestion.ingestion import list_input_files
from rag.build.ingestion.ingestion import validate


class FakePyPDFium2Loader:
    """
    Stand-in for PyPDFium2Loader: reads the file as text and returns one page per "\\f"-separated part. A file whose
    text is "FAIL" raises an error. Every loaded file path is recorded in loaded_files.
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
        Output: a list with one LangChain Document per page of the file.
        What it does: records the file and reads it, or raises if its text is "FAIL".
        """
        FakePyPDFium2Loader.loaded_files.append(self.file_path)
        with open(self.file_path, encoding="utf-8") as f:
            text = f.read()
        if text == "FAIL":
            raise RuntimeError("broken file")

        page_texts = text.split("\f")
        langchain_pages = []
        for page_number in range(len(page_texts)):
            metadata = {"source": self.file_path, "total_pages": len(page_texts), "page": page_number, "author": "", "title": "A Title"}
            langchain_pages.append(LangchainDocument(page_content=page_texts[page_number], metadata=metadata))
        return langchain_pages


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


def get_dataset_output_path(ingestion_args):
    """
    Input: ingestion_args.
    Output: the dataset's output folder, {output_path}/{dataset_name}, the folder ingestion takes.
    What it does: builds the folder the same way run_build does.
    """
    return get_output_path(ingestion_args.output_path, ingestion_args.dataset_name)


def get_documents_path(output_path):
    """
    Input: output_path, the root output folder.
    Output: the folder ingestion saves documents to.
    What it does: joins the dataset output layout.
    """
    return os.path.join(output_path, "dataset", "documents")


def document_id_of(file_text):
    """
    Input: file_text, the text a test file was written with.
    Output: the document id of that file: the first 16 characters of the SHA-256 of its bytes.
    What it does: computes the expected document id.
    """
    return hashlib.sha256(file_text.encode("utf-8")).hexdigest()[:16]


def get_document_texts(files):
    """
    Input: files, a list of File.
    Output: the list of document texts in order.
    What it does: collects each file's document_texts.
    """
    document_texts = []
    for file in files:
        document_texts.append(file.document_texts)
    return document_texts


def get_page_ids(file):
    """
    Input: file, one File.
    Output: the list of the page ids of its pages, in the order of its pages dict.
    What it does: collects each page's id.
    """
    page_ids = []
    for page_number in file.pages:
        page_ids.append(file.pages[page_number].page_id)
    return page_ids


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


# ---------- convert_langchain_pages ----------


def test_convert_langchain_pages_fills_the_file_pages_and_metadata():
    """
    Input: pages 0 and 2 of a 3-page file as LangChain Documents, with an empty author.
    Output: our File.
    What it does: checks the document id is the first 16 characters of the file hash, the pages are keyed by their page
    number with page ids made of the document id and the page number, the document text is the pages joined, and the
    file-level values and metadata are filled.
    """
    content_hash = hashlib.sha256(b"some bytes").hexdigest()
    langchain_metadata = {"source": "a.pdf", "total_pages": 3, "author": "", "title": "A Title"}
    langchain_pages = [
        LangchainDocument(page_content="Page zero.", metadata={**langchain_metadata, "page": 0}),
        LangchainDocument(page_content="Page two.", metadata={**langchain_metadata, "page": 2}),
    ]

    file = convert_langchain_pages(langchain_pages, "pdf", content_hash, 10)

    assert file.document_id == content_hash[:16]
    assert file.document_type.value == "pdf"
    assert file.source == "a.pdf"
    assert file.document_texts == "Page zero.\n\fPage two."
    assert list(file.pages.keys()) == [0, 2]
    assert file.pages[2].page_id == f"{content_hash[:16]}_2"
    assert file.pages[2].page_number == 2
    assert file.pages[2].page_texts == "Page two."
    assert file.metadata.content_hash == content_hash
    assert file.metadata.file_size_bytes == 10
    assert file.metadata.page_count == 3
    assert file.metadata.parser == "langchain_pypdfium2"
    assert file.metadata.author is None
    assert file.metadata.title == "A Title"
    assert file.metadata.creation_date is None


# ---------- ingestion ----------


def test_ingestion_reads_only_document_type_files_up_to_dataset_size(tmp_path, fake_loader):
    """
    Input: three pdf files and one txt file, with dataset_size 2.
    Output: the files and the loaded files.
    What it does: checks only the first two pdf files in name order are loaded.
    """
    input_path = make_input_folder(tmp_path, {"c.pdf": "text c", "a.pdf": "text a", "b.pdf": "text b", "d.txt": "text d"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    loaded_file_names = []
    for loaded_file in FakePyPDFium2Loader.loaded_files:
        loaded_file_names.append(os.path.basename(loaded_file))
    assert loaded_file_names == ["a.pdf", "b.pdf"]
    assert get_document_texts(files) == ["text a", "text b"]


def test_ingestion_returns_one_file_per_input_file_with_its_pages(tmp_path, fake_loader):
    """
    Input: two files of three pages each.
    Output: the files.
    What it does: checks 2 files come back in file name order, each with its 3 pages keyed by 0-based page number, the
    page ids made of the document id and the page number, the document text joined, and the source.
    """
    text_a = "a0\fa1\fa2"
    text_b = "b0\fb1\fb2"
    input_path = make_input_folder(tmp_path, {"a.pdf": text_a, "b.pdf": text_b})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert len(files) == 2
    assert files[0].document_id == document_id_of(text_a)
    assert files[1].document_id == document_id_of(text_b)
    assert files[0].source == os.path.join(input_path, "a.pdf")
    assert files[0].document_texts == "a0\n\fa1\n\fa2"
    assert list(files[0].pages.keys()) == [0, 1, 2]
    assert get_page_ids(files[0]) == [f"{document_id_of(text_a)}_0", f"{document_id_of(text_a)}_1", f"{document_id_of(text_a)}_2"]
    assert files[1].pages[2].page_texts == "b2"
    assert files[0].metadata.page_count == 3


def test_ingestion_returns_the_ingestion_hash(tmp_path, fake_loader):
    """
    Input: one ingestion run, then the same run again.
    Output: the ingestion hash it returns.
    What it does: checks the returned hash is the one compute_ingestion_hash gives, on a first run and from the cache.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    first_files, first_hash, first_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))
    second_files, second_hash, second_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert first_hash == compute_ingestion_hash(ingestion_args)
    assert second_hash == first_hash


def test_ingestion_skips_blank_pages_and_keeps_the_other_page_numbers(tmp_path, fake_loader, capsys):
    """
    Input: one file with three pages, the middle one blank.
    Output: the files and what was printed.
    What it does: checks the blank page is skipped and counted, the other pages keep their page numbers, and the page
    count of the metadata is still 3.
    """
    text_a = "page 0\f  \n \fpage 2"
    input_path = make_input_folder(tmp_path, {"a.pdf": text_a})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert list(files[0].pages.keys()) == [0, 2]
    assert get_page_ids(files[0]) == [f"{document_id_of(text_a)}_0", f"{document_id_of(text_a)}_2"]
    assert files[0].document_texts == "page 0\n\fpage 2"
    assert files[0].metadata.page_count == 3
    assert "Files read : 1, from cache : 0, parsed : 1, failed : 0, duplicate : 0, empty : 0, files loaded : 1, pages loaded : 2, empty pages : 1" in capsys.readouterr().out


def test_ingestion_skips_a_file_with_only_blank_pages(tmp_path, fake_loader, capsys):
    """
    Input: one good file and one file with only whitespace.
    Output: the files and what was printed.
    What it does: checks the blank file gives no File and is counted as empty.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "  \n "})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert get_document_texts(files) == ["text a"]
    assert "Files read : 2, from cache : 0, parsed : 1, failed : 0, duplicate : 0, empty : 1, files loaded : 1, pages loaded : 1, empty pages : 1" in capsys.readouterr().out


def test_ingestion_skips_failed_and_duplicate_files(tmp_path, fake_loader, capsys):
    """
    Input: one good file, one file the loader fails on and one file with the same content as the good file.
    Output: the files and what was printed.
    What it does: checks the failed file gives no File, the duplicate file is skipped, and both are counted.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a\ftext a 1", "b.pdf": "FAIL", "c.pdf": "text a\ftext a 1"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"))

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert len(files) == 1
    assert files[0].source == os.path.join(input_path, "a.pdf")
    assert "Files read : 3, from cache : 0, parsed : 1, failed : 1, duplicate : 1, empty : 0, files loaded : 1, pages loaded : 2, empty pages : 0" in capsys.readouterr().out


def test_ingestion_saves_files_and_document_ids(tmp_path, fake_loader):
    """
    Input: two pdf files of two pages each, with dataset_size 2.
    Output: the files in the documents folder, and the path ingestion returns.
    What it does: checks the returned path is the documents folder, one {document_id}.json per file holding its pages, and doc_ids_{ingestion_hash}.json listing
    the document ids.
    """
    text_a = "a0\fa1"
    text_b = "b0\fb1"
    input_path = make_input_folder(tmp_path, {"a.pdf": text_a, "b.pdf": text_b})
    output_path = str(tmp_path / "out")
    ingestion_args = make_ingestion_args(input_path, output_path, dataset_size=2)

    files, ingestion_hash, ingestion_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    documents_path = get_documents_path(output_path)
    assert ingestion_path == documents_path
    document_ids = [document_id_of(text_a), document_id_of(text_b)]
    expected_file_names = [f"doc_ids_{ingestion_hash}.json", f"{document_ids[0]}.json", f"{document_ids[1]}.json"]
    assert sorted(os.listdir(documents_path)) == sorted(expected_file_names)

    with open(os.path.join(documents_path, f"doc_ids_{ingestion_hash}.json"), encoding="utf-8") as f:
        assert json.load(f) == document_ids
    with open(os.path.join(documents_path, f"{document_ids[0]}.json"), encoding="utf-8") as f:
        saved_file_dict = json.load(f)
    assert saved_file_dict["document_id"] == document_ids[0]
    assert list(saved_file_dict["pages"].keys()) == ["0", "1"]


def test_ingestion_loads_from_cache_on_second_run(tmp_path, fake_loader, capsys):
    """
    Input: the same ingestion run twice.
    Output: the second run's files and the loaded files.
    What it does: checks the second run loads every file from the cache and never calls the loader.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "a0\fa1", "b.pdf": "b0\fb1"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)

    first_files, first_hash, first_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))
    FakePyPDFium2Loader.loaded_files = []
    capsys.readouterr()
    second_files, second_hash, second_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert second_files == first_files
    assert FakePyPDFium2Loader.loaded_files == []
    assert "Files : 2, pages : 4 (all from cache)" in capsys.readouterr().out


def test_ingestion_input_sets_share_document_files(tmp_path, fake_loader):
    """
    Input: ingestion with dataset_size 1, then dataset_size 2.
    Output: the files in the documents folder.
    What it does: checks each input set gets its own list of document ids, and the shared document is saved once.
    """
    text_a = "text a"
    text_b = "text b"
    input_path = make_input_folder(tmp_path, {"a.pdf": text_a, "b.pdf": text_b})
    output_path = str(tmp_path / "out")
    one_file_args = make_ingestion_args(input_path, output_path, dataset_size=1)
    two_file_args = make_ingestion_args(input_path, output_path, dataset_size=2)

    ingestion(one_file_args, get_dataset_output_path(one_file_args))
    ingestion(two_file_args, get_dataset_output_path(two_file_args))

    expected_file_names = [
        f"doc_ids_{compute_ingestion_hash(one_file_args)}.json",
        f"doc_ids_{compute_ingestion_hash(two_file_args)}.json",
        f"{document_id_of(text_a)}.json",
        f"{document_id_of(text_b)}.json",
    ]
    assert sorted(os.listdir(get_documents_path(output_path))) == sorted(expected_file_names)


def test_ingestion_reads_the_files_again_when_an_input_file_changes(tmp_path, fake_loader):
    """
    Input: two pdf files ingested, then one file edited.
    Output: the files and the loaded files.
    What it does: checks the ingestion hash changes, so the run list is missed, only the edited file is parsed again and
    it gets a new File.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)
    first_files, first_hash, first_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    with open(os.path.join(input_path, "b.pdf"), "w", encoding="utf-8") as f:
        f.write("text b edited")
    FakePyPDFium2Loader.loaded_files = []
    second_files, second_hash, second_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    assert len(FakePyPDFium2Loader.loaded_files) == 1
    assert second_hash != first_hash
    assert second_files != first_files


def test_ingestion_parses_only_the_file_that_changed(tmp_path, fake_loader, capsys):
    """
    Input: two pdf files ingested, then one file edited and ingested again.
    Output: the files, the loaded files and what was printed.
    What it does: checks only the edited file is parsed again, the other file is loaded from its saved file by its
    document id, and both files are returned in file name order.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)
    first_files, first_hash, first_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    with open(os.path.join(input_path, "b.pdf"), "w", encoding="utf-8") as f:
        f.write("text b edited")
    FakePyPDFium2Loader.loaded_files = []
    capsys.readouterr()
    second_files, second_hash, second_path = ingestion(ingestion_args, get_dataset_output_path(ingestion_args))

    loaded_file_names = []
    for loaded_file in FakePyPDFium2Loader.loaded_files:
        loaded_file_names.append(os.path.basename(loaded_file))
    assert loaded_file_names == ["b.pdf"]
    assert second_files[0] == first_files[0]
    assert get_document_texts(second_files) == ["text a", "text b edited"]
    assert "from cache : 1, parsed : 1" in capsys.readouterr().out


def test_ingestion_parses_only_the_new_files_when_dataset_size_grows(tmp_path, fake_loader):
    """
    Input: ingestion with dataset_size 1, then dataset_size 2.
    Output: the loaded files.
    What it does: checks the second run parses only the second file, since the first file's document is already saved.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    output_path = str(tmp_path / "out")
    one_file_args = make_ingestion_args(input_path, output_path, dataset_size=1)
    two_file_args = make_ingestion_args(input_path, output_path, dataset_size=2)
    ingestion(one_file_args, get_dataset_output_path(one_file_args))

    FakePyPDFium2Loader.loaded_files = []
    files, ingestion_hash, ingestion_path = ingestion(two_file_args, get_dataset_output_path(two_file_args))

    loaded_file_names = []
    for loaded_file in FakePyPDFium2Loader.loaded_files:
        loaded_file_names.append(os.path.basename(loaded_file))
    assert loaded_file_names == ["b.pdf"]
    assert get_document_texts(files) == ["text a", "text b"]


def test_ingestion_raises_for_invalid_settings_before_reading(tmp_path, fake_loader):
    """
    Input: ingestion args with document_type "docx".
    Output: ValueError, and the loaded files.
    What it does: checks ingestion validates its own args first, so no file is read.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), document_type="docx")

    with pytest.raises(ValueError, match="document_type"):
        ingestion(ingestion_args, get_dataset_output_path(ingestion_args))
    assert FakePyPDFium2Loader.loaded_files == []


# ---------- cache: save_files / load_file / load_files ----------


def test_save_and_load_files_round_trip(tmp_path, make_file):
    """
    Input: two files (two pages and one page) saved under one ingestion hash and loaded back.
    Output: the loaded files.
    What it does: checks the files come back the same, in the same order, with their pages still keyed by int page
    numbers, and an unknown ingestion hash gives None.
    """
    files = [make_file("d1", ["text 0", "text 1"]), make_file("d2", ["text 2"])]

    save_files(str(tmp_path), files, ["d1", "d2"], "ingestionhash")

    loaded_files = load_files(str(tmp_path), "ingestionhash")
    assert loaded_files == files
    assert loaded_files[0].pages[1].page_id == "d1_1"
    assert load_files(str(tmp_path), "otherhash") is None


def test_save_files_saves_only_the_new_files_and_lists_every_document(tmp_path, make_file):
    """
    Input: one new file saved with a list of two document ids, one of them saved before.
    Output: the files in the folder.
    What it does: checks only the new file is written, and the list holds both ids in the order given.
    """
    old_file = make_file("d1", ["text 0"])
    new_file = make_file("d2", ["text 1"])
    save_files(str(tmp_path), [old_file], ["d1"], "firsthash")

    save_files(str(tmp_path), [new_file], ["d1", "d2"], "secondhash")

    assert sorted(os.listdir(str(tmp_path))) == ["d1.json", "d2.json", "doc_ids_firsthash.json", "doc_ids_secondhash.json"]
    with open(os.path.join(str(tmp_path), "doc_ids_secondhash.json"), encoding="utf-8") as f:
        assert json.load(f) == ["d1", "d2"]


def test_load_file_loads_one_saved_file_by_its_document_id(tmp_path, make_file):
    """
    Input: one file saved.
    Output: the file loaded by its document id, and the result for an unknown document id.
    What it does: checks the saved file comes back the same, and an unknown document id gives None.
    """
    file = make_file("d1", ["text 0", "text 1"])
    save_files(str(tmp_path), [file], ["d1"], "ingestionhash")

    assert load_file(str(tmp_path), "d1") == file
    assert load_file(str(tmp_path), "d2") is None


def test_load_files_gives_none_when_a_listed_file_is_missing(tmp_path, make_file):
    """
    Input: a run list naming two documents, with one document's file removed.
    Output: the result of load_files.
    What it does: checks load_files gives None, so ingestion goes on to load or parse each file itself.
    """
    files = [make_file("d1", ["text 0"]), make_file("d2", ["text 1"])]
    save_files(str(tmp_path), files, ["d1", "d2"], "ingestionhash")
    os.remove(os.path.join(str(tmp_path), "d2.json"))

    assert load_files(str(tmp_path), "ingestionhash") is None


# ---------- list_input_files / compute_ingestion_hash ----------


def test_list_input_files_keeps_the_first_dataset_size_files_of_document_type(tmp_path):
    """
    Input: three pdf files and one txt file, with dataset_size 2.
    Output: the listed file names.
    What it does: checks the files are the first two pdf files in name order.
    """
    input_path = make_input_folder(tmp_path, {"c.pdf": "c", "a.pdf": "a", "b.pdf": "b", "a.txt": "t"})

    assert list_input_files(make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)) == ["a.pdf", "b.pdf"]


def test_compute_ingestion_hash_is_12_hex_characters_and_the_same_for_the_same_files(tmp_path):
    """
    Input: the same input folder hashed twice.
    Output: the two ingestion hashes.
    What it does: checks the hash is 12 hex characters and stable.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)

    ingestion_hash = compute_ingestion_hash(ingestion_args)

    assert len(ingestion_hash) == 12
    int(ingestion_hash, 16)
    assert compute_ingestion_hash(ingestion_args) == ingestion_hash


@pytest.mark.parametrize("change", ["edit", "add_inside", "remove"])
def test_compute_ingestion_hash_changes_when_a_read_file_changes(tmp_path, change):
    """
    Input: three pdf files with dataset_size 3, then one file edited, a file added that sorts inside the first 3, or
    a file removed.
    Output: the ingestion hash before and after.
    What it does: checks any change to the files ingestion reads gives a new ingestion hash.
    """
    input_path = make_input_folder(tmp_path, {"b.pdf": "text b", "c.pdf": "text c", "d.pdf": "text d"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=3)
    ingestion_hash_before = compute_ingestion_hash(ingestion_args)

    if change == "edit":
        with open(os.path.join(input_path, "c.pdf"), "w", encoding="utf-8") as f:
            f.write("text c edited")
    elif change == "add_inside":
        with open(os.path.join(input_path, "a.pdf"), "w", encoding="utf-8") as f:
            f.write("text a")
    else:
        os.remove(os.path.join(input_path, "d.pdf"))

    assert compute_ingestion_hash(ingestion_args) != ingestion_hash_before


def test_compute_ingestion_hash_ignores_files_ingestion_does_not_read(tmp_path):
    """
    Input: two pdf files with dataset_size 2, then a pdf file added that sorts after them and a txt file added.
    Output: the ingestion hash before and after.
    What it does: checks files outside the first dataset_size files of document_type do not change the ingestion hash.
    """
    input_path = make_input_folder(tmp_path, {"a.pdf": "text a", "b.pdf": "text b"})
    ingestion_args = make_ingestion_args(input_path, str(tmp_path / "out"), dataset_size=2)
    ingestion_hash_before = compute_ingestion_hash(ingestion_args)

    with open(os.path.join(input_path, "c.pdf"), "w", encoding="utf-8") as f:
        f.write("text c")
    with open(os.path.join(input_path, "a.txt"), "w", encoding="utf-8") as f:
        f.write("text")

    assert compute_ingestion_hash(ingestion_args) == ingestion_hash_before
