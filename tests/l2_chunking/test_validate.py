"""
test_validate.py

Tests for the chunking-related checks in common.validate:
validate_processed_dataset and validate_chunk_config. The working
directory is a temporary folder (isolated_data_dir), so files written
under {dataset_name}/data/ stay out of the repo.
"""

import os

import pytest

from common import paths
from common import validate
from common.chunk_config import ChunkConfig
from l1_doc_processing.progress import Progress, save_progress


DATASET_NAME = "myset"


def _write_processed_files(file_count):
    """
    file_count: number of processed document files to create.

    Returns nothing.
    """
    processed_path = paths.get_processed_dataset_path(DATASET_NAME)
    os.makedirs(processed_path, exist_ok=True)
    for index in range(file_count):
        with open(os.path.join(processed_path, f"doc{index}.json"), "w", encoding="utf-8") as f:
            f.write("{}")


def _write_progress(processed_count):
    """
    processed_count: number of doc_ids to list as processed in progress.json.

    Returns nothing.
    """
    processed_doc_ids = []
    for index in range(processed_count):
        processed_doc_ids.append(f"doc{index}")
    save_progress(DATASET_NAME, Progress("someone/dataset", "train", "pdf", processed_count, processed_doc_ids, []))


def _make_chunk_config(chunk_type="fixed", min_chunk_size=10, max_chunk_size=100, fixed_chunk_overlap=0, semantic_threshold=0.0):
    """
    Returns a ChunkConfig with the given values.
    """
    return ChunkConfig(
        dataset_name=DATASET_NAME,
        chunk_type=chunk_type,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        fixed_chunk_overlap=fixed_chunk_overlap,
        semantic_threshold=semantic_threshold,
    )


# ---------- validate_processed_dataset ----------


def test_validate_processed_dataset_passes_when_enough_documents_are_processed(isolated_data_dir):
    """
    Input: 3 processed files and a progress.json listing 3 doc_ids, dataset_size 3.
    Output: none; nothing should be raised.
    """
    _write_processed_files(3)
    _write_progress(3)

    validate.validate_processed_dataset(DATASET_NAME, 3)


def test_validate_processed_dataset_raises_when_processed_folder_is_missing(isolated_data_dir):
    """
    Input: no processed folder at all.
    Output: none; FileNotFoundError is expected.
    """
    with pytest.raises(FileNotFoundError):
        validate.validate_processed_dataset(DATASET_NAME, 1)


def test_validate_processed_dataset_raises_when_processed_folder_is_empty(isolated_data_dir):
    """
    Input: an empty processed folder.
    Output: none; ValueError is expected.
    """
    _write_processed_files(0)

    with pytest.raises(ValueError):
        validate.validate_processed_dataset(DATASET_NAME, 1)


def test_validate_processed_dataset_raises_when_too_few_processed_files(isolated_data_dir):
    """
    Input: 2 processed files, dataset_size 3.
    Output: none; ValueError is expected.
    """
    _write_processed_files(2)
    _write_progress(2)

    with pytest.raises(ValueError):
        validate.validate_processed_dataset(DATASET_NAME, 3)


def test_validate_processed_dataset_raises_when_progress_json_is_missing(isolated_data_dir):
    """
    Input: 3 processed files but no progress.json.
    Output: none; FileNotFoundError is expected.
    """
    _write_processed_files(3)

    with pytest.raises(FileNotFoundError):
        validate.validate_processed_dataset(DATASET_NAME, 3)


def test_validate_processed_dataset_raises_when_progress_lists_too_few_doc_ids(isolated_data_dir):
    """
    Input: 3 processed files but a progress.json listing only 2 doc_ids,
    dataset_size 3.
    Output: none; ValueError is expected.
    """
    _write_processed_files(3)
    _write_progress(2)

    with pytest.raises(ValueError):
        validate.validate_processed_dataset(DATASET_NAME, 3)


# ---------- validate_chunk_config ----------


@pytest.mark.parametrize(
    "chunk_type, semantic_threshold",
    [
        ("fixed", 0.0),
        ("sentence", 0.0),
        ("semantic", 0.5),
    ],
)
def test_validate_chunk_config_passes_for_each_supported_chunk_type(chunk_type, semantic_threshold):
    """
    Input: a valid ChunkConfig for each supported chunk_type.
    Output: none; nothing should be raised.
    """
    validate.validate_chunk_config(_make_chunk_config(chunk_type=chunk_type, semantic_threshold=semantic_threshold))


def test_validate_chunk_config_raises_for_an_unknown_chunk_type():
    """
    Input: chunk_type "paragraph".
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        validate.validate_chunk_config(_make_chunk_config(chunk_type="paragraph"))


@pytest.mark.parametrize("min_chunk_size", [100, 150])
def test_validate_chunk_config_raises_when_min_chunk_size_is_not_smaller_than_max(min_chunk_size):
    """
    Input: min_chunk_size equal to or larger than max_chunk_size (100).
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        validate.validate_chunk_config(_make_chunk_config(min_chunk_size=min_chunk_size, max_chunk_size=100))


@pytest.mark.parametrize("fixed_chunk_overlap", [100, 150])
def test_validate_chunk_config_raises_when_overlap_is_not_smaller_than_max(fixed_chunk_overlap):
    """
    Input: fixed_chunk_overlap equal to or larger than max_chunk_size (100).
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        validate.validate_chunk_config(_make_chunk_config(fixed_chunk_overlap=fixed_chunk_overlap, max_chunk_size=100))


def test_validate_chunk_config_raises_for_semantic_without_a_threshold():
    """
    Input: chunk_type "semantic" with semantic_threshold 0.0.
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        validate.validate_chunk_config(_make_chunk_config(chunk_type="semantic", semantic_threshold=0.0))
