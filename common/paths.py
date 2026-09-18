
import os

from common import global_values
from common.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_DIR


# ------ Dataset
def dataset_raw_path() -> str:
    doc_type = global_values.get("doc_type")
    dataset_name = global_values.get("dataset_name")

    return RAW_DATA_DIR + f"{doc_type}_{dataset_name}/"

def dataset_processed_path() -> str:
    doc_type = global_values.get("doc_type")
    dataset_name = global_values.get("dataset_name")
    return PROCESSED_DATA_DIR + f"{doc_type}_{dataset_name}/docs/"

def get_file_path(path: str, file_name: str, extension: str) -> str:
    return os.path.join(path, f"{file_name}.{extension}")

# ------ Chunking
def get_chunk_path(chunk_type:str) -> str:

    doc_type = global_values.get("doc_type")
    dataset_name = global_values.get("dataset_name")

    max_chunk_size = global_values.get("max_chunk_size")
    fixed_chunk_overlap = global_values.get("fixed_chunk_overlap")
    semantic_threshold = global_values.get("semantic_threshold")

    return CHUNK_DIR + f"{chunk_type}/{doc_type}_{dataset_name}/{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}"