import os

import pandas as pd
import pytest

import src.dataset as dataset


def _make_parquet_files(tmp_path, row_counts):
    """Create one parquet file per entry in row_counts and return their paths."""
    paths = []
    for i, n in enumerate(row_counts):
        df = pd.DataFrame({"id": range(n), "text": [f"row-{i}-{j}" for j in range(n)]})
        path = tmp_path / f"part-{i}.parquet"
        df.to_parquet(path)
        paths.append(str(path))
    return paths


def test_load_dataset_trims_to_requested_size(tmp_path, monkeypatch):
    files = _make_parquet_files(tmp_path, row_counts=[5, 5])
    monkeypatch.setattr(dataset, "LOCAL_DATASET", files)
    output_path = str(tmp_path / "processed")
    monkeypatch.setattr(dataset, "PROCESSED_DATA_PATH", output_path)

    result_path = dataset.load_dataset(dataset_size=7)

    assert result_path == f"{output_path}.parquet"
    result_df = pd.read_parquet(result_path)
    assert len(result_df) == 7


def test_load_dataset_stops_reading_once_enough_rows_collected(tmp_path, monkeypatch):
    files = _make_parquet_files(tmp_path, row_counts=[5, 5, 5])
    monkeypatch.setattr(dataset, "LOCAL_DATASET", files)
    monkeypatch.setattr(dataset, "PROCESSED_DATA_PATH", str(tmp_path / "processed"))

    read_calls = []
    original_read_parquet = pd.read_parquet

    def tracking_read_parquet(path, *args, **kwargs):
        read_calls.append(path)
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(dataset.pd, "read_parquet", tracking_read_parquet)

    dataset.load_dataset(dataset_size=6)

    # First file (5 rows) isn't enough, so a second file must be read;
    # 5 + 5 = 10 >= 6, so the third file should never be touched.
    assert read_calls == files[:2]


def test_load_dataset_creates_output_directory(tmp_path, monkeypatch):
    files = _make_parquet_files(tmp_path, row_counts=[3])
    monkeypatch.setattr(dataset, "LOCAL_DATASET", files)
    nested_output = tmp_path / "nested" / "dir" / "processed"
    monkeypatch.setattr(dataset, "PROCESSED_DATA_PATH", str(nested_output))

    result_path = dataset.load_dataset(dataset_size=3)

    assert os.path.exists(result_path)


def test_load_dataset_raises_when_no_local_files(monkeypatch):
    monkeypatch.setattr(dataset, "LOCAL_DATASET", [])

    with pytest.raises(FileNotFoundError):
        dataset.load_dataset(dataset_size=10)
