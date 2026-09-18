
import os

from common import global_values

# ------ Dataset
def dataset_raw_path() -> str:

    dataset_name = global_values.get("dataset_name")
    return f"{dataset_name}/raw"

def dataset_processed_path() -> str:
    dataset_name = global_values.get("dataset_name")
    return f"{dataset_name}/processed"

def get_file_path(path: str, file_name: str, extension: str) -> str:
    return os.path.join(path, f"{file_name}.{extension}")

# ------ Chunking
def get_chunk_path(chunk_type:str) -> str:

    dataset_name = global_values.get("dataset_name")

    max_chunk_size = global_values.get("max_chunk_size")
    fixed_chunk_overlap = global_values.get("fixed_chunk_overlap")
    semantic_threshold = global_values.get("semantic_threshold")

    return f"{dataset_name}/chunks/{chunk_type}/{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}/"

# ------ Embedding
def get_embeddings_path(chunk_type:str) -> str:

    dataset_name = global_values.get("dataset_name")

    max_chunk_size = global_values.get("max_chunk_size")
    fixed_chunk_overlap = global_values.get("fixed_chunk_overlap")
    semantic_threshold = global_values.get("semantic_threshold")

    return f"{dataset_name}/embeddings/{chunk_type}/{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}/"