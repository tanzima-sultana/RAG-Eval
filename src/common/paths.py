
import os

from common.chunk_config import ChunkConfig

# ------ Dataset
def get_raw_dataset_path(dataset_name:str) -> str:

	return f"{dataset_name}/data/raw"

def get_processed_dataset_path(dataset_name:str) -> str:
	return f"{dataset_name}/data/processed"

def get_progress_path(dataset_name:str) -> str:
	return f"{dataset_name}/data/"

def get_file_path(path: str, file_name: str, extension: str) -> str:
	return os.path.join(path, f"{file_name}.{extension}")

# ------ Chunking
def get_chunk_path(chunk_config: ChunkConfig) -> str:

	return (
		f"{chunk_config.dataset_name}/chunks/{chunk_config.chunk_type}/"
		f"{chunk_config.min_chunk_size}_{chunk_config.max_chunk_size}_"
		f"{chunk_config.fixed_chunk_overlap}_{chunk_config.semantic_threshold}/"
	)

# ------ Embedding
def get_embeddings_path(chunk_config: ChunkConfig) -> str:

	return (
		f"{chunk_config.dataset_name}/embeddings/{chunk_config.chunk_type}/"
		f"{chunk_config.min_chunk_size}_{chunk_config.max_chunk_size}_"
		f"{chunk_config.fixed_chunk_overlap}_{chunk_config.semantic_threshold}/"
	)

# ------ Indexing
def get_index_path(index_type:str, chunk_config: ChunkConfig) -> str:

	return (
		f"{chunk_config.dataset_name}/index_{index_type}/{chunk_config.chunk_type}/"
		f"{chunk_config.min_chunk_size}_{chunk_config.max_chunk_size}_"
		f"{chunk_config.fixed_chunk_overlap}_{chunk_config.semantic_threshold}/"
	)