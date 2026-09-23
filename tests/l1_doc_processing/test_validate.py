"""
test_validate.py

Tests for common.validate: validate_dataset, validate_progress and
validate_processed_dataset. Web datasets use a fake
load_dataset_builder, so no network access is needed.
"""

import os

import pytest
from datasets.exceptions import DatasetNotFoundError

from common import dataset_config as dataset_config_module
from common import validate
from common.dataset_config import DatasetConfig
from l1_doc_processing.progress import Progress


class _FakeSplitInfo:
    """
    Stand-in for datasets.SplitInfo, exposing only num_examples.
    """

    def __init__(self, num_examples):
        self.num_examples = num_examples


class _FakeInfo:
    """
    Stand-in for datasets.DatasetInfo, exposing only splits.
    """

    def __init__(self, splits):
        self.splits = splits


class _FakeBuilder:
    """
    Stand-in for a datasets builder, exposing only info.
    """

    def __init__(self, splits):
        self.info = _FakeInfo(splits)


def _register(monkeypatch, dataset_name, identifier):
    """
    Registers dataset_name with the given identifier for this test.

    Returns the registered DatasetConfig.
    """
    config = DatasetConfig(
        identifier=identifier,
        doc_type="pdf",
        extension="pdf",
        content_field="content",
        content_encoding="bytes",
        file_name_field="name",
        split="train",
    )
    monkeypatch.setitem(dataset_config_module._DATASET_CONFIGS, dataset_name, config)
    return config


def _make_local_dataset(folder, file_count):
    """
    Creates folder with file_count small files.

    Returns the folder path as a string.
    """
    os.makedirs(folder)
    for index in range(file_count):
        with open(os.path.join(folder, f"{index}.txt"), "w", encoding="utf-8") as f:
            f.write("x")
    return str(folder)


# ---------- validate_dataset: local ----------


def test_validate_dataset_passes_for_a_valid_local_dataset(tmp_path, monkeypatch):
    """
    Input: a registered local dataset with 3 files, dataset_size 3.
    Output: none; nothing should be raised.
    """
    folder = _make_local_dataset(tmp_path / "docs", 3)
    _register(monkeypatch, "myset", folder)

    validate.validate_dataset("pdf", "myset", 3, None, folder)


def test_validate_dataset_raises_when_dataset_name_is_not_registered(tmp_path):
    """
    Input: a dataset_name missing from the registry.
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        validate.validate_dataset("pdf", "not-registered", 1, None, str(tmp_path))


def test_validate_dataset_raises_when_path_does_not_match_identifier(tmp_path, monkeypatch):
    """
    Input: a dataset_path different from the registered identifier.
    Output: none; ValueError is expected.
    """
    folder = _make_local_dataset(tmp_path / "docs", 1)
    _register(monkeypatch, "myset", folder)

    with pytest.raises(ValueError):
        validate.validate_dataset("pdf", "myset", 1, None, str(tmp_path / "other"))


def test_validate_dataset_raises_when_url_does_not_match_identifier(monkeypatch):
    """
    Input: a dataset_url different from the registered identifier.
    Output: none; ValueError is expected.
    """
    _register(monkeypatch, "myset", "someone/right-dataset")

    with pytest.raises(ValueError):
        validate.validate_dataset("pdf", "myset", 1, "someone/wrong-dataset", None)


def test_validate_dataset_raises_when_doc_type_does_not_match(tmp_path, monkeypatch):
    """
    Input: doc_type "html" for a dataset registered as "pdf".
    Output: none; ValueError is expected.
    """
    folder = _make_local_dataset(tmp_path / "docs", 1)
    _register(monkeypatch, "myset", folder)

    with pytest.raises(ValueError):
        validate.validate_dataset("html", "myset", 1, None, folder)


def test_validate_dataset_raises_when_local_folder_does_not_exist(tmp_path, monkeypatch):
    """
    Input: a registered local dataset whose folder does not exist.
    Output: none; FileNotFoundError is expected.
    """
    missing_folder = str(tmp_path / "missing")
    _register(monkeypatch, "myset", missing_folder)

    with pytest.raises(FileNotFoundError):
        validate.validate_dataset("pdf", "myset", 1, None, missing_folder)


def test_validate_dataset_raises_when_local_dataset_is_too_small(tmp_path, monkeypatch):
    """
    Input: a local dataset with 2 files, dataset_size 3.
    Output: none; ValueError is expected.
    """
    folder = _make_local_dataset(tmp_path / "docs", 2)
    _register(monkeypatch, "myset", folder)

    with pytest.raises(ValueError):
        validate.validate_dataset("pdf", "myset", 3, None, folder)


# ---------- validate_dataset: web ----------


def test_validate_dataset_raises_file_not_found_when_url_dataset_is_missing(monkeypatch):
    """
    Input: a registered web dataset that the hub reports as not found.
    Output: none; FileNotFoundError is expected.
    """
    _register(monkeypatch, "myset", "someone/missing-dataset")

    def fake_builder(dataset_url):
        raise DatasetNotFoundError("not found")

    monkeypatch.setattr(validate, "load_dataset_builder", fake_builder)

    with pytest.raises(FileNotFoundError):
        validate.validate_dataset("pdf", "myset", 1, "someone/missing-dataset", None)


def test_validate_dataset_raises_when_url_dataset_is_too_small(monkeypatch):
    """
    Input: a web dataset whose train split has 5 rows, dataset_size 10.
    Output: none; ValueError is expected.
    """
    _register(monkeypatch, "myset", "someone/small-dataset")

    def fake_builder(dataset_url):
        return _FakeBuilder({"train": _FakeSplitInfo(5)})

    monkeypatch.setattr(validate, "load_dataset_builder", fake_builder)

    with pytest.raises(ValueError):
        validate.validate_dataset("pdf", "myset", 10, "someone/small-dataset", None)


def test_validate_dataset_warns_and_passes_when_url_dataset_has_no_count(monkeypatch, capsys):
    """
    Input: a web dataset that publishes no split sizes.
    Output: printed output; nothing should be raised.
    """
    _register(monkeypatch, "myset", "someone/uncounted-dataset")

    def fake_builder(dataset_url):
        return _FakeBuilder(None)

    monkeypatch.setattr(validate, "load_dataset_builder", fake_builder)

    validate.validate_dataset("pdf", "myset", 10, "someone/uncounted-dataset", None)

    assert "Warning" in capsys.readouterr().out


# ---------- validate_progress ----------


def test_validate_progress_passes_when_everything_matches(monkeypatch):
    """
    Input: a Progress with the same identifier, split and doc_type.
    Output: none; nothing should be raised.
    """
    config = _register(monkeypatch, "myset", "someone/dataset")
    progress = Progress("someone/dataset", "train", "pdf", 5, ["pdf_a"], [])

    validate.validate_progress(progress, config, "pdf")


def test_validate_progress_lists_every_mismatched_field(monkeypatch):
    """
    Input: a Progress whose identifier and doc_type differ.
    Output: the raised ValueError's message.
    """
    config = _register(monkeypatch, "myset", "someone/dataset")
    progress = Progress("someone/old-dataset", "train", "html", 5, [], [])

    with pytest.raises(ValueError) as error_info:
        validate.validate_progress(progress, config, "pdf")

    message = str(error_info.value)
    assert "dataset_identifier" in message
    assert "doc_type" in message
    assert "split:" not in message


# ---------- validate_processed_dataset ----------


def test_validate_processed_dataset_passes_when_enough_documents_exist(tmp_path):
    """
    Input: a folder with 3 files, dataset_size 3.
    Output: none; nothing should be raised.
    """
    folder = _make_local_dataset(tmp_path / "processed", 3)

    validate.validate_processed_dataset(folder, 3)


def test_validate_processed_dataset_raises_when_folder_is_missing(tmp_path):
    """
    Input: a folder path that does not exist.
    Output: none; FileNotFoundError is expected.
    """
    with pytest.raises(FileNotFoundError):
        validate.validate_processed_dataset(str(tmp_path / "missing"), 1)


def test_validate_processed_dataset_raises_when_folder_is_empty(tmp_path):
    """
    Input: an empty folder.
    Output: none; ValueError is expected.
    """
    folder = _make_local_dataset(tmp_path / "processed", 0)

    with pytest.raises(ValueError):
        validate.validate_processed_dataset(folder, 1)


def test_validate_processed_dataset_raises_when_too_few_documents(tmp_path):
    """
    Input: a folder with 2 files, dataset_size 3.
    Output: none; ValueError is expected.
    """
    folder = _make_local_dataset(tmp_path / "processed", 2)

    with pytest.raises(ValueError):
        validate.validate_processed_dataset(folder, 3)
