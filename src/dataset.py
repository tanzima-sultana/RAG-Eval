import os

import pandas as pd

from config import LOCAL_DATASET
from constants import PROCESSED_DATA_PATH


def load_dataset(dataset_size: int) -> str:
    """Read `dataset_size` rows from LOCAL_DATASET, save them to PROCESSED_DATA_PATH,
    and return the path of the saved file."""
    if not LOCAL_DATASET:
        raise FileNotFoundError("No parquet files found in LOCAL_DATASET")

    frames = []
    rows_collected = 0
    for file_path in LOCAL_DATASET:
        if rows_collected >= dataset_size:
            break
        df = pd.read_parquet(file_path)
        frames.append(df)
        rows_collected += len(df)

    dataset = pd.concat(frames, ignore_index=True).head(dataset_size)

    output_path = f"{PROCESSED_DATA_PATH}.parquet"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dataset.to_parquet(output_path, index=False)

    return output_path
