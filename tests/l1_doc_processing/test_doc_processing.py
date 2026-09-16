"""
test_doc_processing.py

Tests for l1_doc_processing.doc_processing.DocProcessing.

All tests run against the local dataset_path branch (never dataset_url)
and a fake parser, so they need no network access and no real PDF
bytes. Filesystem effects are isolated to a temporary directory via the
isolated_data_dir fixture (see conftest.py), and a local DatasetConfig
is registered per test via the registered_local_dataset_config fixture
below.

Two tests are direct regressions for bugs found and fixed earlier on
this dataset_path/DocProcessing conversion:
  - test_compute_raw_data_returns_a_plain_dict_when_cache_already_covers_dataset_size
    guards against compute_raw_data returning a (dict, int) tuple on
    the cache-satisfied path instead of a plain dict.
  - test_process_raises_before_touching_the_filesystem_when_doc_type_mismatches
    guards against process() forgetting to call validate().
"""

import base64
import datetime
import os

import pytest

from common.dataset_config import DatasetConfig
from common import dataset_config as dataset_config_module
from common.identifiers import make_content_hash
from l1_doc_processing.doc_processing import DocProcessing
from l1_doc_processing.schema import Document


class _FakeSplitInfo:
    """
    Stand-in for a datasets.SplitInfo: exposes only num_examples, the
    single attribute get_actual_dataset_size reads.
    """

    def __init__(self, num_examples):
        self.num_examples = num_examples


class _FakeDatasetInfo:
    """
    Stand-in for a datasets.DatasetInfo: exposes only splits, which is
    either a dict of split name to _FakeSplitInfo, or None for a
    dataset (e.g. a WebDataset-format one) that publishes no
    split-size metadata at all.
    """

    def __init__(self, splits):
        self.splits = splits


class _FakeBuilder:
    """
    Stand-in for a datasets.DatasetBuilder: exposes only info, the
    single attribute get_actual_dataset_size reads.
    """

    def __init__(self, splits):
        self.info = _FakeDatasetInfo(splits)


class _FakeParser:
    """
    Stand-in for a real l1_doc_processing.parser.Parser implementation,
    used so DocProcessing tests do not depend on pypdf or real PDF
    bytes. Decodes raw_bytes as UTF-8 text and wraps it in a Document.
    Raises ValueError when raw_bytes equals the sentinel value b"BAD",
    to exercise process_raw_data's parse-failure handling.

    parsed_doc_ids records, in call order, every doc_id this parser was
    asked to parse (including ones that then raised), so a test can
    confirm whether parsing did or did not happen for a given doc_id.
    """

    parsed_doc_ids = []

    def parse(self, doc_id, raw_bytes, data_path):
        _FakeParser.parsed_doc_ids.append(doc_id)
        if raw_bytes == b"BAD":
            raise ValueError("simulated parse failure")
        text = raw_bytes.decode("utf-8")
        return Document(
            id=doc_id,
            text=text,
            title=None,
            path=data_path,
            format="fake",
            metadata={},
            content_hash=make_content_hash(text),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )


@pytest.fixture
def fake_parser_class(monkeypatch):
    """
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Replaces l1_doc_processing.doc_processing.get_parser with a stub
    that always returns _FakeParser, regardless of doc_type, and
    resets _FakeParser.parsed_doc_ids to an empty list.

    Returns the _FakeParser class, so a test can inspect
    _FakeParser.parsed_doc_ids after calling DocProcessing methods.
    """
    _FakeParser.parsed_doc_ids = []

    def fake_get_parser(doc_type):
        return _FakeParser

    monkeypatch.setattr("l1_doc_processing.doc_processing.get_parser", fake_get_parser)
    return _FakeParser


@pytest.fixture
def registered_local_dataset_config(isolated_data_dir, monkeypatch):
    """
    isolated_data_dir: ensures this runs with an isolated cwd, so the
        registered dataset_path lives under the same temporary
        directory DocProcessing will write data/raw and data/processed
        into.
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Registers a DatasetConfig with doc_type "pdf" under
    common.dataset_config._DATASET_CONFIGS, keyed by a local
    dataset_path directory, so a DocProcessing constructed against
    that dataset_path does not need a real registered dataset or
    network access. Creates the dataset_path directory on disk. Uses
    monkeypatch.setitem so the registration is undone after the test.

    Returns the dataset_path string this config is registered under.
    """
    dataset_path = str(isolated_data_dir / "source_dataset")
    os.makedirs(dataset_path, exist_ok=True)

    dataset_config = DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="",
        content_encoding="bytes",
        file_name_field="",
        split="",
    )

    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, dataset_path, dataset_config)

    return dataset_path


def _write_bytes(path, content):
    """
    path: file path to write.
    content: bytes to write.

    Writes content to path in binary mode, as a small explicit helper
    so individual tests don't repeat the open/write boilerplate.

    Returns nothing.
    """
    with open(path, "wb") as f:
        f.write(content)


# ---------- __init__ ----------


def test_init_computes_raw_and_processed_paths_from_doc_type_and_dataset_name(
    registered_local_dataset_config,
):
    """
    Input: a DocProcessing constructed with doc_type "pdf" and
    dataset_name "myset".
    Output: processor.raw_path and processor.processed_path.

    Checks that the cache directories follow common.paths' naming
    convention.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)

    assert processor.raw_path == "data/raw/pdf_myset/"
    assert processor.processed_path == "data/processed/pdf_myset/docs/"


# ---------- validate ----------


def test_validate_passes_when_doc_type_matches_and_dataset_size_fits(
    registered_local_dataset_config,
):
    """
    Input: a DocProcessing whose doc_type matches its dataset_config's
    doc_type, with dataset_size no larger than what's actually in
    dataset_path.
    Output: none; validate() is expected not to raise.
    """
    dataset_path = registered_local_dataset_config
    _write_bytes(os.path.join(dataset_path, "doc1.pdf"), b"first document")
    processor = DocProcessing("pdf", "myset", 1, None, dataset_path)

    processor.validate()


def test_validate_raises_value_error_when_doc_type_does_not_match(
    registered_local_dataset_config,
):
    """
    Input: a DocProcessing constructed with a doc_type that does not
    match its dataset_config's registered doc_type ("pdf"), and
    dataset_size 0 so the doc_type check is what's under test.
    Output: none; validate() is expected to raise.
    """
    processor = DocProcessing(
        "not-pdf", "myset", 0, None, registered_local_dataset_config
    )

    with pytest.raises(ValueError):
        processor.validate()


def test_validate_raises_value_error_when_dataset_size_exceeds_actual_size(
    registered_local_dataset_config,
):
    """
    Input: a DocProcessing whose doc_type matches, but whose
    dataset_size (5) is larger than the number of files actually in
    dataset_path (1).
    Output: none; validate() is expected to raise.

    Regression coverage for the CUAD run that silently processed only
    508 of the 5000 requested documents: validate() must now catch
    this upfront instead of letting the pipeline proceed.
    """
    dataset_path = registered_local_dataset_config
    _write_bytes(os.path.join(dataset_path, "doc1.pdf"), b"first document")
    processor = DocProcessing("pdf", "myset", 5, None, dataset_path)

    with pytest.raises(ValueError):
        processor.validate()


# ---------- get_actual_dataset_size ----------


def test_get_actual_dataset_size_counts_files_for_local_dataset(
    registered_local_dataset_config,
):
    """
    Input: three files sitting in dataset_path.
    Output: processor.get_actual_dataset_size()'s return value.

    Checks the local-dataset branch: the actual size is just a file
    count, with no network access involved.
    """
    dataset_path = registered_local_dataset_config
    _write_bytes(os.path.join(dataset_path, "doc1.pdf"), b"one")
    _write_bytes(os.path.join(dataset_path, "doc2.pdf"), b"two")
    _write_bytes(os.path.join(dataset_path, "doc3.pdf"), b"three")
    processor = DocProcessing("pdf", "myset", 1, None, dataset_path)

    assert processor.get_actual_dataset_size() == 3


def test_get_actual_dataset_size_reads_split_num_examples_for_url_dataset(
    isolated_data_dir, monkeypatch
):
    """
    Input: a DocProcessing constructed against a dataset_url registered
    with a fake DatasetConfig, with datasets.load_dataset_builder
    stubbed out to avoid any real network access.
    Output: processor.get_actual_dataset_size()'s return value.

    Checks the web-dataset branch: the actual size comes from the
    builder's declared split size, not from downloading/streaming any
    document content.
    """

    def fake_load_dataset_builder(dataset_url):
        return _FakeBuilder({"train": _FakeSplitInfo(num_examples=42)})

    monkeypatch.setattr(
        "l1_doc_processing.doc_processing.load_dataset_builder",
        fake_load_dataset_builder,
    )

    dataset_url = "some-org/some-dataset"
    dataset_config = DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="content",
        content_encoding="bytes",
        file_name_field="file_name",
        split="train",
    )
    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, dataset_url, dataset_config)

    processor = DocProcessing("pdf", "myset", 1, dataset_url, None)

    assert processor.get_actual_dataset_size() == 42


def test_get_actual_dataset_size_is_none_when_builder_publishes_no_splits(
    isolated_data_dir, monkeypatch
):
    """
    Input: a DocProcessing constructed against a dataset_url whose
    builder.info.splits is None, matching WebDataset-format datasets
    (e.g. pixparse/pdfa-eng-wds) that publish no split-size metadata.
    Output: processor.get_actual_dataset_size()'s return value.

    Checks that a dataset with no declared split-size metadata
    reports its actual size as None (unknown) rather than raising.
    """

    def fake_load_dataset_builder(dataset_url):
        return _FakeBuilder(None)

    monkeypatch.setattr(
        "l1_doc_processing.doc_processing.load_dataset_builder",
        fake_load_dataset_builder,
    )

    dataset_url = "some-org/webdataset-without-split-sizes"
    dataset_config = DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="content",
        content_encoding="bytes",
        file_name_field="file_name",
        split="train",
    )
    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, dataset_url, dataset_config)

    processor = DocProcessing("pdf", "myset", 1, dataset_url, None)

    assert processor.get_actual_dataset_size() is None


def test_validate_proceeds_with_warning_when_dataset_size_is_unknown(
    isolated_data_dir, monkeypatch, capsys
):
    """
    Input: a DocProcessing constructed against a dataset_url whose
    builder.info.splits is None, with any requested dataset_size.
    Output: none; validate() is expected not to raise, but to print a
    warning that the size could not be checked.

    Checks that an undeterminable actual size does not block the
    pipeline (unlike an actual size that's too small, which does).
    """

    def fake_load_dataset_builder(dataset_url):
        return _FakeBuilder(None)

    monkeypatch.setattr(
        "l1_doc_processing.doc_processing.load_dataset_builder",
        fake_load_dataset_builder,
    )

    dataset_url = "some-org/webdataset-without-split-sizes"
    dataset_config = DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="content",
        content_encoding="bytes",
        file_name_field="file_name",
        split="train",
    )
    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, dataset_url, dataset_config)

    processor = DocProcessing("pdf", "myset", 10, dataset_url, None)

    processor.validate()

    captured = capsys.readouterr()
    assert "cannot be validated upfront" in captured.out


# ---------- load_cache ----------


def test_load_cache_reads_every_file_keyed_by_doc_id(registered_local_dataset_config):
    """
    Input: two raw cache files already written under processor.raw_path.
    Output: processor.load_cache()'s return value.

    Checks that load_cache returns every cached file's bytes, keyed by
    its doc_id (the filename with the extension stripped).
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    os.makedirs(processor.raw_path, exist_ok=True)
    _write_bytes(processor.raw_path + "pdf_aaa.pdf", b"content a")
    _write_bytes(processor.raw_path + "pdf_bbb.pdf", b"content b")

    cached = processor.load_cache()

    assert cached == {"pdf_aaa": b"content a", "pdf_bbb": b"content b"}


def test_load_cache_returns_empty_dict_when_raw_path_is_empty(
    registered_local_dataset_config,
):
    """
    Input: an existing, empty processor.raw_path directory.
    Output: processor.load_cache()'s return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    os.makedirs(processor.raw_path, exist_ok=True)

    assert processor.load_cache() == {}


# ---------- decode_raw_bytes ----------


def test_decode_raw_bytes_base64(registered_local_dataset_config):
    """
    Input: a row whose content field holds base64-encoded text, with
    dataset_config.content_encoding set to "base64".
    Output: processor.decode_raw_bytes's return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = "base64"
    row = {"content": base64.b64encode(b"hello").decode("ascii")}

    assert processor.decode_raw_bytes(row) == b"hello"


def test_decode_raw_bytes_bytes(registered_local_dataset_config):
    """
    Input: a row whose content field already holds raw bytes, with
    dataset_config.content_encoding set to "bytes".
    Output: processor.decode_raw_bytes's return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = "bytes"
    row = {"content": b"already bytes"}

    assert processor.decode_raw_bytes(row) == b"already bytes"


def test_decode_raw_bytes_text(registered_local_dataset_config):
    """
    Input: a row whose content field holds plain text, with
    dataset_config.content_encoding set to "text".
    Output: processor.decode_raw_bytes's return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = "text"
    row = {"content": "plain text"}

    assert processor.decode_raw_bytes(row) == "plain text".encode("utf-8")


def test_decode_raw_bytes_raises_on_unrecognized_encoding(
    registered_local_dataset_config,
):
    """
    Input: a row, with dataset_config.content_encoding set to a value
    that is none of "base64", "bytes", or "text".
    Output: none; decode_raw_bytes is expected to raise.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    processor.dataset_config.content_field = "content"
    processor.dataset_config.content_encoding = "zip"
    row = {"content": "irrelevant"}

    with pytest.raises(ValueError):
        processor.decode_raw_bytes(row)


# ---------- compute_raw_data ----------


def test_compute_raw_data_returns_a_plain_dict_when_cache_already_covers_dataset_size(
    registered_local_dataset_config,
):
    """
    Input: processor.raw_path already holding three cached raw files,
    and dataset_size set to 2 (less than what's cached).
    Output: processor.compute_raw_data()'s return value.

    Regression test: compute_raw_data must return a plain dict on
    every path, not a (dict, int) tuple. It previously returned a
    tuple specifically on this cache-already-satisfied path, which
    broke process_raw_data downstream.
    """
    processor = DocProcessing("pdf", "myset", 2, None, registered_local_dataset_config)
    os.makedirs(processor.raw_path, exist_ok=True)
    _write_bytes(processor.raw_path + "pdf_aaa.pdf", b"content a")
    _write_bytes(processor.raw_path + "pdf_bbb.pdf", b"content b")
    _write_bytes(processor.raw_path + "pdf_ccc.pdf", b"content c")

    result = processor.compute_raw_data()

    assert type(result) is dict
    assert len(result) == 2
    assert result == {"pdf_aaa": b"content a", "pdf_bbb": b"content b"}


def test_compute_raw_data_copies_new_files_from_local_dataset_path(
    registered_local_dataset_config,
):
    """
    Input: an empty raw cache, and two files sitting in dataset_path.
    Output: processor.compute_raw_data()'s return value, and the files
    written under processor.raw_path.

    Checks that the local (non-dataset_url) branch reads files
    straight from dataset_path and caches them under raw_path.
    """
    dataset_path = registered_local_dataset_config
    _write_bytes(os.path.join(dataset_path, "doc1.pdf"), b"first document")
    _write_bytes(os.path.join(dataset_path, "doc2.pdf"), b"second document")
    processor = DocProcessing("pdf", "myset", 2, None, dataset_path)

    result = processor.compute_raw_data()

    assert type(result) is dict
    assert len(result) == 2
    cached_file_names = sorted(os.listdir(processor.raw_path))
    assert len(cached_file_names) == 2


def test_compute_raw_data_skips_unreadable_local_entry_and_keeps_the_rest(
    registered_local_dataset_config,
):
    """
    Input: dataset_path containing one unreadable entry (a directory,
    which sorts before the real file) and one real file.
    Output: processor.compute_raw_data()'s return value.

    Checks that a per-file failure in the local branch is logged and
    skipped rather than aborting the whole batch.
    """
    dataset_path = registered_local_dataset_config
    os.makedirs(os.path.join(dataset_path, "bad_dir"), exist_ok=True)
    _write_bytes(os.path.join(dataset_path, "good.pdf"), b"good content")
    processor = DocProcessing("pdf", "myset", 2, None, dataset_path)

    result = processor.compute_raw_data()

    assert len(result) == 1
    doc_id = list(result.keys())[0]
    assert result[doc_id] == b"good content"


# ---------- process_raw_data ----------


def test_process_raw_data_reuses_cached_processed_document_without_reparsing(
    registered_local_dataset_config, fake_parser_class
):
    """
    Input: a processed Document already saved under processor.processed_path
    for doc_id "pdf_cached".
    Output: processor.process_raw_data's return value.

    Checks that an already-processed doc_id is loaded from cache and
    never handed to the parser again.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    os.makedirs(processor.processed_path, exist_ok=True)
    doc_id = "pdf_cached"
    cached_text = "cached text with plenty of words to build a shingle set from"
    cached_document = Document(
        id=doc_id,
        text=cached_text,
        title=None,
        path="irrelevant",
        format="fake",
        metadata={},
        content_hash=make_content_hash(cached_text),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    processor.save_processed_document(doc_id, cached_document)

    result = processor.process_raw_data({doc_id: b"irrelevant raw bytes"})

    assert doc_id in result
    assert result[doc_id].text == cached_text
    assert fake_parser_class.parsed_doc_ids == []


def test_process_raw_data_parses_and_saves_a_new_document(
    registered_local_dataset_config, fake_parser_class
):
    """
    Input: one doc_id with no cached processed Document yet.
    Output: processor.process_raw_data's return value, and the saved
    processed JSON file.

    Checks that a new document is parsed exactly once and persisted to
    processed_path.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    doc_id = "pdf_new"

    result = processor.process_raw_data({doc_id: b"brand new content words here"})

    assert doc_id in result
    assert result[doc_id].text == "brand new content words here"
    assert fake_parser_class.parsed_doc_ids == [doc_id]
    saved = processor.load_processed_document(doc_id)
    assert saved is not None
    assert saved.text == "brand new content words here"


def test_process_raw_data_skips_documents_that_fail_to_parse(
    registered_local_dataset_config, fake_parser_class
):
    """
    Input: two doc_ids, one whose raw_bytes trigger _FakeParser's
    simulated parse failure (b"BAD") and one that parses fine.
    Output: processor.process_raw_data's return value.

    Checks that a parse failure is skipped, not saved, and does not
    stop the rest of the batch from being processed.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    documents_raw_bytes = {"pdf_bad": b"BAD", "pdf_good": b"fine content words"}

    result = processor.process_raw_data(documents_raw_bytes)

    assert "pdf_bad" not in result
    assert "pdf_good" in result
    assert processor.load_processed_document("pdf_bad") is None


def test_process_raw_data_skips_exact_duplicate_content(
    registered_local_dataset_config, fake_parser_class
):
    """
    Input: two different doc_ids whose raw_bytes decode to the exact
    same text.
    Output: processor.process_raw_data's return value.

    Checks that the second document to arrive with identical text
    (identical content_hash) is skipped and never saved, since dict
    iteration order in CPython follows insertion order.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    same_text_bytes = b"identical duplicate content shared across two documents"
    documents_raw_bytes = {"pdf_first": same_text_bytes, "pdf_second": same_text_bytes}

    result = processor.process_raw_data(documents_raw_bytes)

    assert "pdf_first" in result
    assert "pdf_second" not in result
    assert processor.load_processed_document("pdf_second") is None


# ---------- process ----------


def test_process_raises_before_touching_the_filesystem_when_doc_type_mismatches(
    registered_local_dataset_config,
):
    """
    Input: a DocProcessing whose doc_type does not match its
    dataset_config's registered doc_type.
    Output: none; process() is expected to raise.

    Regression test: process() must call validate() before doing any
    work. It previously never called validate() at all, silently
    letting a doc_type/dataset mismatch through.
    """
    processor = DocProcessing(
        "not-pdf", "myset", 5, None, registered_local_dataset_config
    )

    with pytest.raises(ValueError):
        processor.process()

    assert not os.path.exists(processor.raw_path)


def test_process_end_to_end_with_local_dataset(
    registered_local_dataset_config, fake_parser_class
):
    """
    Input: two files in dataset_path, dataset_size 2, no dataset_url.
    Output: processor.process()'s return value.

    Checks the full process() pipeline for a local dataset: both files
    are copied into the raw cache, parsed, and saved as processed
    Documents.
    """
    dataset_path = registered_local_dataset_config
    _write_bytes(os.path.join(dataset_path, "doc1.pdf"), b"first document body")
    _write_bytes(os.path.join(dataset_path, "doc2.pdf"), b"second document body")
    processor = DocProcessing("pdf", "myset", 2, None, dataset_path)

    documents = processor.process()

    assert len(documents) == 2
    assert set(fake_parser_class.parsed_doc_ids) == set(documents.keys())
    assert len(os.listdir(processor.processed_path)) == 2


# ---------- document_to_json_dict / json_dict_to_document ----------


def test_document_to_json_dict_and_back_round_trips(registered_local_dataset_config):
    """
    Input: a fully-populated Document.
    Output: processor.document_to_json_dict then
    processor.json_dict_to_document, applied in sequence.

    Checks that the JSON-dict conversion is lossless, including the
    timestamp's ISO-8601 round trip.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    original = Document(
        id="pdf_x",
        text="hello",
        title="A Title",
        path="p",
        format="pdf",
        metadata={"a": 1},
        content_hash="hash",
        timestamp=timestamp,
    )

    data = processor.document_to_json_dict(original)

    assert data["timestamp"] == "2024-01-01T12:00:00+00:00"
    restored = processor.json_dict_to_document(data)
    assert restored == original


# ---------- save_processed_document / load_processed_document ----------


def test_save_and_load_processed_document_round_trips(registered_local_dataset_config):
    """
    Input: a Document saved via processor.save_processed_document.
    Output: processor.load_processed_document's return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    os.makedirs(processor.processed_path, exist_ok=True)
    document = Document(
        id="pdf_x",
        text="hi",
        title=None,
        path="p",
        format="pdf",
        metadata={},
        content_hash="h",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    processor.save_processed_document("pdf_x", document)
    loaded = processor.load_processed_document("pdf_x")

    assert loaded == document


def test_load_processed_document_returns_none_when_missing(
    registered_local_dataset_config,
):
    """
    Input: a processed_path with no file for the requested doc_id.
    Output: processor.load_processed_document's return value.
    """
    processor = DocProcessing("pdf", "myset", 5, None, registered_local_dataset_config)
    os.makedirs(processor.processed_path, exist_ok=True)

    assert processor.load_processed_document("does-not-exist") is None
