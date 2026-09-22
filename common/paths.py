
import os

# ------ Dataset
def get_raw_dataset_path(dataset_name:str) -> str:

    return f"{dataset_name}/data/raw"

def get_processed_dataset_path(dataset_name:str) -> str:
    return f"{dataset_name}/data/processed"

def get_file_path(path: str, file_name: str, extension: str) -> str:
    return os.path.join(path, f"{file_name}.{extension}")

# ------ Chunking
def get_chunk_path(dataset_name:str, chunk_type:str, min_chunk_size:int, max_chunk_size:int, fixed_chunk_overlap:int, semantic_threshold:float) -> str:

    return f"{dataset_name}/chunks/{chunk_type}/{min_chunk_size}_{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}/"

# ------ Embedding
def get_embeddings_path(dataset_name:str, chunk_type:str, min_chunk_size:int, max_chunk_size:int, fixed_chunk_overlap:int, semantic_threshold:float) -> str:

    return f"{dataset_name}/embeddings/{chunk_type}/{min_chunk_size}_{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}/"