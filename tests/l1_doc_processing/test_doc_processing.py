"""
test_doc_processing.py

Tests for l1_doc_processing.doc_processing.DocProcessing.

Every test uses a local dataset (a folder of small text files) and a
fake parser, so no network access and no real PDFs are needed. The
working directory is a temporary folder (isolated_data_dir), so all
output under {dataset_name}/data/ stays out of the repo.
"""

import datetime
import json
import os

import pytest

from common import dataset_config as dataset_config_module
from common.dataset_config import DatasetConfig
from common.identifiers import make_content_hash, make_doc_id
from l1_doc_processing import doc_processing as doc_processing_module
from l1_doc_processing.doc_processing import DocProcessing
from l1_doc_processing.schema import Document, Progress


DATASET_NAME = "myset"


class _FakeParser:
    """
    Stand-in for a real Parser. Decodes raw_bytes as UTF-8 text and
    wraps it in a Document. Raises ValueError when the text is "BAD".

    parsed_doc_ids records every doc_id it was asked to parse, in call
    order, so a test can check whether parsing happened.
    """

    parsed_doc_ids = []

    def parse(self, doc_id, raw_bytes, data_path):
        """
        doc_id, raw_bytes, data_path: same as Parser.parse.

        Returns a Document built from raw_bytes decoded as text.
        Raises ValueError if the text is "BAD".
        """
        _FakeParser.parsed_doc_ids.append(doc_id)
        text = raw_bytes.decode("utf-8")
        if text == "BAD":
            raise ValueError("fake parse failure")
        document = Document(
            id=doc_id,
            text=text,
            title=None,
            path=data_path,
            format="pdf",
            metadata={},
            content_hash=make_content_hash(text),
            timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        )
        return document


def _fake_get_parser(doc_type):
    """
    doc_type: ignored.

    Returns the _FakeParser class, in place of get_parser.
    """
    return _FakeParser


def _write_source_files(source_dir, file_texts):
    """
    source_dir: folder to write the local dataset files into.
    file_texts: list of (file_name, text) pairs.

    Writes each text into source_dir/file_name.

    Returns nothing.
    """
    os.makedirs(source_dir, exist_ok=True)
    for file_name, text in file_texts:
        file_path = os.path.join(source_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)


def _doc_id_for(text):
    """
    text: a source file's text.

    Returns the doc_id DocProcessing computes for that file.
    """
    return make_doc_id("pdf", text.encode("utf-8"))


@pytest.fixture
def local_dataset(isolated_data_dir, monkeypatch):
    """
    Registers DATASET_NAME as a local dataset whose identifier is a
    fresh source folder, and swaps in the fake parser.

    Returns the source folder path. Tests write their files into it.
    """
    source_dir = str(isolated_data_dir / "source_docs")
    os.makedirs(source_dir)

    config = DatasetConfig(
        identifier=source_dir,
        doc_type="pdf",
        extension="pdf",
        content_field="",
        content_encoding="bytes",
        file_name_field="",
        split="train",
    )
    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, DATASET_NAME, config)
    monkeypatch.setattr(doc_processing_module, "get_parser", _fake_get_parser)
    _FakeParser.parsed_doc_ids = []
    return source_dir


def _make_processor(source_dir, dataset_size):
    """
    source_dir: the local dataset folder.
    dataset_size: number of processed documents wanted.

    Returns a DocProcessing for the registered local dataset.
    """
    return DocProcessing("pdf", DATASET_NAME, dataset_size, None, source_dir)


def _read_progress_file():
    """
    Returns the saved progress.json of DATASET_NAME as a dict.
    """
    with open(f"{DATASET_NAME}/data/progress.json", encoding="utf-8") as f:
        return json.load(f)


# ---------- progress ----------


def test_load_progress_returns_empty_progress_on_first_run(local_dataset):
    """
    Input: no progress.json on disk.
    Output: load_progress's return value.
    """
    processor = _make_processor(local_dataset, 1)

    progress = processor.load_progress()

    assert progress.dataset_identifier == local_dataset
    assert progress.split == "train"
    assert progress.doc_type == "pdf"
    assert progress.rows_consumed == 0
    assert progress.processed_doc_ids == []
    assert progress.skipped_doc_ids == []


def test_save_and_load_progress_round_trips_and_leaves_no_temp_file(local_dataset):
    """
    Input: a Progress saved with save_progress.
    Output: the Progress read back with load_progress, and the files
    left in the data folder.
    """
    processor = _make_processor(local_dataset, 1)
    saved = Progress(local_dataset, "train", "pdf", 7, ["pdf_a", "pdf_b"], ["pdf_c"])

    processor.save_progress(saved)
    loaded = processor.load_progress()

    assert loaded == saved
    assert os.listdir(f"{DATASET_NAME}/data") == ["progress.json"]


def test_make_progress_uses_the_dataset_config(local_dataset):
    """
    Input: counts and id collections for make_progress.
    Output: the returned Progress.
    """
    processor = _make_processor(local_dataset, 1)

    progress = processor.make_progress(3, ["pdf_a"], {"pdf_b"})

    assert progress == Progress(local_dataset, "train", "pdf", 3, ["pdf_a"], ["pdf_b"])


# ---------- documents on disk ----------


def test_document_to_json_dict_and_back_round_trips(local_dataset):
    """
    Input: a Document converted to a JSON dict and back.
    Output: the rebuilt Document.
    """
    processor = _make_processor(local_dataset, 1)
    document = _FakeParser().parse("pdf_x", b"hello", "some/path")

    data = processor.document_to_json_dict(document)
    rebuilt = processor.json_dict_to_document(data)

    assert isinstance(data["timestamp"], str)
    assert rebuilt == document


def test_save_and_load_processed_document_round_trips(local_dataset):
    """
    Input: a Document saved with save_processed_document.
    Output: load_processed_document's return value.
    """
    processor = _make_processor(local_dataset, 1)
    document = _FakeParser().parse("pdf_x", b"hello", "some/path")

    processor.save_processed_document("pdf_x", document)
    loaded = processor.load_processed_document("pdf_x")

    assert loaded == document
    assert os.path.isfile(f"{DATASET_NAME}/data/processed/pdf_x.json")


def test_load_processed_document_returns_none_when_missing(local_dataset):
    """
    Input: a doc_id with no processed file.
    Output: load_processed_document's return value.
    """
    processor = _make_processor(local_dataset, 1)

    assert processor.load_processed_document("pdf_missing") is None


def test_save_raw_bytes_writes_file_and_returns_its_path(local_dataset):
    """
    Input: raw bytes for one doc_id.
    Output: save_raw_bytes's return value and the written file.
    """
    processor = _make_processor(local_dataset, 1)

    file_path = processor.save_raw_bytes("pdf_x", b"raw content")

    assert file_path == f"{DATASET_NAME}/data/raw/pdf_x.pdf"
    with open(file_path, "rb") as f:
        assert f.read() == b"raw content"


# ---------- source rows ----------


def test_get_iterator_for_local_dataset_skips_rows_consumed(local_dataset):
    """
    Input: three source files and rows_consumed of 1.
    Output: the file names get_iterator yields.
    """
    _write_source_files(local_dataset, [("c.txt", "c"), ("a.txt", "a"), ("b.txt", "b")])
    processor = _make_processor(local_dataset, 1)

    file_names = []
    for file_name in processor.get_iterator(1):
        file_names.append(file_name)

    assert file_names == ["b.txt", "c.txt"]


def test_decode_raw_bytes_reads_local_file(local_dataset):
    """
    Input: a local file name.
    Output: decode_raw_bytes's return value.
    """
    _write_source_files(local_dataset, [("a.txt", "hello")])
    processor = _make_processor(local_dataset, 1)

    assert processor.decode_raw_bytes("a.txt") == b"hello"


@pytest.mark.parametrize(
    "content_encoding, raw_value, expected_bytes",
    [
        ("base64", "aGVsbG8=", b"hello"),
        ("bytes", b"hello", b"hello"),
        ("text", "hello", b"hello"),
    ],
)
def test_decode_raw_bytes_decodes_web_row_per_content_encoding(
    local_dataset, content_encoding, raw_value, expected_bytes
):
    """
    Input: a web dataset row whose content field uses content_encoding.
    Output: decode_raw_bytes's return value.
    """
    processor = _make_processor(local_dataset, 1)
    processor.dataset_url = "someone/some-dataset"
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = content_encoding

    assert processor.decode_raw_bytes({"content": raw_value}) == expected_bytes


def test_decode_raw_bytes_raises_on_unknown_content_encoding(local_dataset):
    """
    Input: a web dataset row with an unknown content_encoding.
    Output: none; ValueError is expected.
    """
    processor = _make_processor(local_dataset, 1)
    processor.dataset_url = "someone/some-dataset"
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = "zip"

    with pytest.raises(ValueError):
        processor.decode_raw_bytes({"content": "x"})


# ---------- process ----------


def test_process_first_run_processes_dataset_size_documents_in_order(local_dataset):
    """
    Input: four source files, dataset_size 3, no earlier progress.
    Output: process's return value, the saved progress, and the files
    on disk.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma"), ("d.txt", "delta")])
    processor = _make_processor(local_dataset, 3)

    documents = processor.process()

    expected_ids = [_doc_id_for("alpha"), _doc_id_for("beta"), _doc_id_for("gamma")]
    assert list(documents.keys()) == expected_ids
    progress = _read_progress_file()
    assert progress["rows_consumed"] == 3
    assert progress["processed_doc_ids"] == expected_ids
    assert progress["skipped_doc_ids"] == []
    assert len(os.listdir(f"{DATASET_NAME}/data/processed")) == 3
    assert len(os.listdir(f"{DATASET_NAME}/data/raw")) == 3


def test_process_returns_cached_documents_without_parsing_when_enough_exist(local_dataset):
    """
    Input: a first run with dataset_size 2, then a second run with the
    same dataset_size.
    Output: the second run's return value and the parse calls it made.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")])
    _make_processor(local_dataset, 2).process()
    _FakeParser.parsed_doc_ids = []

    documents = _make_processor(local_dataset, 2).process()

    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("beta")]
    assert _FakeParser.parsed_doc_ids == []


def test_process_smaller_request_keeps_all_processed_ids_in_progress(local_dataset):
    """
    Input: a first run with dataset_size 3, then a second run with
    dataset_size 1.
    Output: the second run's return value and the saved progress.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")])
    _make_processor(local_dataset, 3).process()

    documents = _make_processor(local_dataset, 1).process()

    assert list(documents.keys()) == [_doc_id_for("alpha")]
    progress = _read_progress_file()
    assert progress["processed_doc_ids"] == [_doc_id_for("alpha"), _doc_id_for("beta"), _doc_id_for("gamma")]


def test_process_resumes_from_rows_consumed_for_a_bigger_request(local_dataset):
    """
    Input: a first run with dataset_size 2, then a second run with
    dataset_size 4.
    Output: the second run's return value, its parse calls, and the
    saved progress.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma"), ("d.txt", "delta")])
    _make_processor(local_dataset, 2).process()
    _FakeParser.parsed_doc_ids = []

    documents = _make_processor(local_dataset, 4).process()

    assert len(documents) == 4
    assert _FakeParser.parsed_doc_ids == [_doc_id_for("gamma"), _doc_id_for("delta")]
    progress = _read_progress_file()
    assert progress["rows_consumed"] == 4
    assert progress["processed_doc_ids"] == [
        _doc_id_for("alpha"),
        _doc_id_for("beta"),
        _doc_id_for("gamma"),
        _doc_id_for("delta"),
    ]


def test_process_skips_parse_failures_and_reads_more_rows(local_dataset):
    """
    Input: a file that fails to parse between good files, dataset_size 2.
    Output: process's return value and the saved progress.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "BAD"), ("c.txt", "gamma")])
    processor = _make_processor(local_dataset, 2)

    documents = processor.process()

    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("gamma")]
    progress = _read_progress_file()
    assert progress["rows_consumed"] == 3
    assert progress["skipped_doc_ids"] == [_doc_id_for("BAD")]


def test_process_does_not_parse_a_previously_skipped_document_again(local_dataset):
    """
    Input: a run where one file fails to parse, then a second run over
    a new copy of the same failing content.
    Output: the parse calls made by the second run.
    """
    _write_source_files(local_dataset, [("a.txt", "BAD"), ("b.txt", "alpha")])
    _make_processor(local_dataset, 1).process()
    _write_source_files(local_dataset, [("c.txt", "BAD"), ("d.txt", "beta")])
    _FakeParser.parsed_doc_ids = []

    _make_processor(local_dataset, 2).process()

    assert _FakeParser.parsed_doc_ids == [_doc_id_for("beta")]


def test_process_skips_exact_duplicate_text(local_dataset):
    """
    Input: two files with the same text, then a different one,
    dataset_size 2.
    Output: process's return value.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "alpha"), ("c.txt", "gamma")])
    processor = _make_processor(local_dataset, 2)

    documents = processor.process()

    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("gamma")]
    assert _read_progress_file()["rows_consumed"] == 3


def _skip_validate_dataset(doc_type, dataset_name, dataset_size, dataset_url, dataset_path):
    """
    Same arguments as validate.validate_dataset. Does nothing.
    """
    return None


def test_process_returns_fewer_documents_when_the_source_runs_out(local_dataset, monkeypatch):
    """
    Input: two source files, dataset_size 5.
    Output: process's return value and the saved progress.

    validate_dataset would reject 5 > 2 files up front (that case is
    covered in test_validate.py), so it is switched off here to test
    process's end-of-source handling, e.g. when rows are lost to
    duplicates or parse failures.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta")])
    monkeypatch.setattr(doc_processing_module.validate, "validate_dataset", _skip_validate_dataset)
    processor = _make_processor(local_dataset, 5)

    documents = processor.process()

    assert len(documents) == 2
    progress = _read_progress_file()
    assert progress["rows_consumed"] == 2
    assert len(progress["processed_doc_ids"]) == 2


def test_process_marks_a_missing_processed_file_as_skipped(local_dataset):
    """
    Input: a first run with dataset_size 2, one processed file deleted,
    then a second run with dataset_size 2.
    Output: the second run's return value and the saved progress.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "beta"), ("c.txt", "gamma")])
    _make_processor(local_dataset, 2).process()
    os.remove(f"{DATASET_NAME}/data/processed/{_doc_id_for('alpha')}.json")

    documents = _make_processor(local_dataset, 2).process()

    assert list(documents.keys()) == [_doc_id_for("beta"), _doc_id_for("gamma")]
    progress = _read_progress_file()
    assert progress["processed_doc_ids"] == [_doc_id_for("beta"), _doc_id_for("gamma")]
    assert progress["skipped_doc_ids"] == [_doc_id_for("alpha")]


def test_process_raises_before_writing_anything_when_doc_type_mismatches(local_dataset):
    """
    Input: a DocProcessing whose doc_type differs from the registered one.
    Output: none; ValueError is expected and nothing is written.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha")])
    processor = DocProcessing("html", DATASET_NAME, 1, None, local_dataset)

    with pytest.raises(ValueError):
        processor.process()

    assert not os.path.exists(DATASET_NAME)


def test_process_raises_when_saved_progress_is_from_another_source(local_dataset):
    """
    Input: a progress.json saved for a different dataset identifier.
    Output: none; ValueError is expected.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha")])
    processor = _make_processor(local_dataset, 1)
    processor.save_progress(Progress("someone/other-dataset", "train", "pdf", 0, [], []))

    with pytest.raises(ValueError):
        processor.process()
