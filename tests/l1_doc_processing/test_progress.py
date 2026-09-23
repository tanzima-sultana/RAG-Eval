"""
test_progress.py

Tests for l1_doc_processing.progress: the Progress dataclass,
make_progress, load_progress and save_progress. The working directory is a temporary folder
(isolated_data_dir), so progress.json stays out of the repo.
"""

import dataclasses
import os

from common.dataset_config import DatasetConfig
from l1_doc_processing.progress import Progress, load_progress, make_progress, save_progress


DATASET_NAME = "myset"

CONFIG = DatasetConfig(
    identifier="someone/dataset",
    doc_type="pdf",
    extension="pdf",
    content_field="content",
    content_encoding="bytes",
    file_name_field="name",
    split="train",
)


def test_make_progress_uses_the_dataset_config():
    """
    Input: counts and id collections for make_progress.
    Output: the returned Progress.
    """
    progress = make_progress(CONFIG, "pdf", 3, ["pdf_a"], {"pdf_b"})

    assert progress == Progress("someone/dataset", "train", "pdf", 3, ["pdf_a"], ["pdf_b"])


def test_load_progress_returns_empty_progress_on_first_run(isolated_data_dir):
    """
    Input: no progress.json on disk.
    Output: load_progress's return value.
    """
    progress = load_progress(DATASET_NAME, CONFIG, "pdf")

    assert progress == Progress("someone/dataset", "train", "pdf", 0, [], [])


def test_save_and_load_progress_round_trips_and_leaves_no_temp_file(isolated_data_dir):
    """
    Input: a Progress saved with save_progress.
    Output: the Progress read back with load_progress, and the files
    left in the data folder.
    """
    saved = Progress("someone/dataset", "train", "pdf", 7, ["pdf_a", "pdf_b"], ["pdf_c"])

    save_progress(DATASET_NAME, saved)
    loaded = load_progress(DATASET_NAME, CONFIG, "pdf")

    assert loaded == saved
    assert os.listdir(f"{DATASET_NAME}/data") == ["progress.json"]


def test_save_progress_overwrites_the_previous_progress(isolated_data_dir):
    """
    Input: two Progress values saved one after the other.
    Output: the Progress read back with load_progress.
    """
    save_progress(DATASET_NAME, Progress("someone/dataset", "train", "pdf", 1, ["pdf_a"], []))
    newer = Progress("someone/dataset", "train", "pdf", 2, ["pdf_a", "pdf_b"], [])

    save_progress(DATASET_NAME, newer)

    assert load_progress(DATASET_NAME, CONFIG, "pdf") == newer


def test_progress_holds_every_field_and_converts_to_a_plain_dict():
    """
    Input: a fully-populated Progress.
    Output: its fields and dataclasses.asdict(progress).

    save_progress writes dataclasses.asdict(progress) as JSON, so every
    field must be plain JSON data.
    """
    progress = Progress(
        dataset_identifier="someone/dataset",
        split="train",
        doc_type="pdf",
        rows_consumed=12,
        processed_doc_ids=["pdf_a", "pdf_b"],
        skipped_doc_ids=["pdf_c"],
    )

    data = dataclasses.asdict(progress)

    assert data == {
        "dataset_identifier": "someone/dataset",
        "split": "train",
        "doc_type": "pdf",
        "rows_consumed": 12,
        "processed_doc_ids": ["pdf_a", "pdf_b"],
        "skipped_doc_ids": ["pdf_c"],
    }
