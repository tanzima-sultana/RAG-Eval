import os

from common.args_temp import ChunkingArgs
from common.args_temp import EmbeddingArgs


# ------ Dataset
def get_output_path(output_path: str, framework: str, dataset_name: str) -> str:
	"""
	Input: output_path, the root output folder; framework; dataset_name.
	Output: {output_path}/{framework}/{dataset_name}
	What it does: builds the dataset's output folder for one framework.
	"""
	return os.path.join(output_path, framework, dataset_name)


# ------ Ingestion
def get_documents_path(output_path: str) -> str:
	"""
	Input: output_path, the dataset's output folder.
	Output: {output_path}/documents
	What it does: builds the folder that holds the ingested documents.
	"""
	return os.path.join(output_path, "documents")


# ------ Chunking
def get_chunk_path(output_path: str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args.
	Output: {output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}
	What it does: builds the folder that holds one chunking run's chunks.
	"""
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return os.path.join(output_path, "chunks", chunking_args.chunk_type, chunking_args.model_name, settings_folder)


# ------ Embedding
def get_embedding_path(output_path: str, embedding_args: EmbeddingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; embedding_args.
	Output: {output_path}/embeddings/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}
	What it does: builds the folder that holds one chunking run's embeddings.
	"""
	chunking_args = embedding_args.chunking_args
	settings_folder = (
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}"
	)
	return os.path.join(output_path, "embeddings", chunking_args.chunk_type, chunking_args.model_name, settings_folder)
