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
from l1_doc_processing import raw_source as raw_source_module
from l1_doc_processing.doc_processing import DocProcessing
from l1_doc_processing.progress import Progress, save_progress
from l1_doc_processing.schema import Document


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
    save_progress(DATASET_NAME, Progress("someone/other-dataset", "train", "pdf", 0, [], []))

    with pytest.raises(ValueError):
        processor.process()


# ---------- process with web rows, batches and parse times ----------


class _FakeUrlResponse:
    """
    Stand-in for the response urllib.request.urlopen returns: a context
    manager whose read() returns fixed bytes.
    """

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


def _make_web_processor(source_dir, dataset_size, content_encoding, row_filter):
    """
    source_dir: the registered local_dataset folder (used only as the
        config identifier).
    dataset_size: number of processed documents wanted.
    content_encoding: content_encoding to set on the config.
    row_filter: row_filter to set on the config.

    Returns a DocProcessing that behaves like a web dataset whose rows
    keep their content in the "content" column.
    """
    processor = _make_processor(source_dir, dataset_size)
    processor.dataset_url = "someone/web-dataset"
    processor.raw_source.dataset_url = "someone/web-dataset"
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = content_encoding
    processor.dataset_config.row_filter = row_filter
    return processor


def _fake_rows_iterator(rows):
    """
    rows: the web rows to return.

    Returns a get_iterator replacement that yields rows after skipping
    rows_consumed of them, like the real streamed dataset.
    """
    def fake_get_iterator(rows_consumed):
        return iter(rows[rows_consumed:])
    return fake_get_iterator


def test_process_skips_rows_not_matching_the_row_filter(local_dataset, monkeypatch):
    """
    Input: web rows in English and French, row_filter {"language": "en"},
    dataset_size 2.
    Output: process's return value, the parse calls, and saved progress.
    """
    rows = [
        {"content": "alpha", "language": "en"},
        {"content": "bonjour", "language": "fr"},
        {"content": "beta", "language": "en"},
        {"content": "gamma", "language": "en"},
    ]
    monkeypatch.setattr(doc_processing_module.validate, "validate_dataset", _skip_validate_dataset)
    processor = _make_web_processor(local_dataset, 2, "text", {"language": "en"})
    monkeypatch.setattr(processor.raw_source, "get_iterator", _fake_rows_iterator(rows))

    documents = processor.process()

    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("beta")]
    assert _FakeParser.parsed_doc_ids == [_doc_id_for("alpha"), _doc_id_for("beta")]
    assert _read_progress_file()["rows_consumed"] == 3


def test_process_skips_rows_whose_download_fails(local_dataset, monkeypatch):
    """
    Input: three url rows where the second download fails, dataset_size 2.
    Output: process's return value and saved progress.
    """
    downloads = {
        "https://example.com/1": b"alpha",
        "https://example.com/3": b"gamma",
    }

    def fake_urlopen(request, timeout):
        if request.full_url not in downloads:
            raise raw_source_module.urllib.error.URLError("not found")
        return _FakeUrlResponse(downloads[request.full_url])

    rows = [
        {"content": "https://example.com/1"},
        {"content": "https://example.com/2"},
        {"content": "https://example.com/3"},
    ]
    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(doc_processing_module.validate, "validate_dataset", _skip_validate_dataset)
    processor = _make_web_processor(local_dataset, 2, "url", None)
    monkeypatch.setattr(processor.raw_source, "get_iterator", _fake_rows_iterator(rows))

    documents = processor.process()

    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("gamma")]
    assert _read_progress_file()["rows_consumed"] == 3


def test_process_prints_parse_time_for_successfully_parsed_documents_only(local_dataset, capsys):
    """
    Input: two good files and one that fails to parse, dataset_size 2.
    Output: the printed parse time summary.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "BAD"), ("c.txt", "gamma")])
    processor = _make_processor(local_dataset, 2)

    processor.process()

    output = capsys.readouterr().out
    assert "----- Document parse time -----" in output
    assert "Documents parsed : 2" in output


def test_process_fetches_another_batch_for_the_shortfall(local_dataset, capsys):
    """
    Input: four files where the first two fail to parse, dataset_size 2.
    Output: the printed batch lines and process's return value.
    """
    _write_source_files(local_dataset, [("a.txt", "BAD"), ("b.txt", "alpha"), ("c.txt", "gamma"), ("d.txt", "delta")])
    processor = _make_processor(local_dataset, 2)

    documents = processor.process()

    output = capsys.readouterr().out
    assert "Batch 1: fetching 2 row(s)." in output
    assert "Batch 2: fetching 1 row(s)." in output
    assert list(documents.keys()) == [_doc_id_for("alpha"), _doc_id_for("gamma")]
    assert _read_progress_file()["rows_consumed"] == 3


def test_process_saves_progress_after_each_batch(local_dataset, monkeypatch):
    """
    Input: a first batch with one parse failure, then a crash while
    fetching the second batch.
    Output: the progress saved before the crash.
    """
    _write_source_files(local_dataset, [("a.txt", "alpha"), ("b.txt", "BAD"), ("c.txt", "gamma")])
    processor = _make_processor(local_dataset, 2)
    real_fetch_raw_batch = processor.raw_source.fetch_raw_batch
    fetch_calls = [0]

    def fetch_then_crash(row_iterator, row_count, run_counts):
        fetch_calls[0] += 1
        if fetch_calls[0] == 2:
            raise RuntimeError("simulated crash")
        return real_fetch_raw_batch(row_iterator, row_count, run_counts)

    monkeypatch.setattr(processor.raw_source, "fetch_raw_batch", fetch_then_crash)

    with pytest.raises(RuntimeError):
        processor.process()

    progress = _read_progress_file()
    assert progress["rows_consumed"] == 2
    assert progress["processed_doc_ids"] == [_doc_id_for("alpha")]
    assert progress["skipped_doc_ids"] == [_doc_id_for("BAD")]
