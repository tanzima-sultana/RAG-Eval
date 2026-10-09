
import os

from common.args import ChunkingArgs
from common.constants import NO_FRAMEWORK

# ------ Chunking
def get_chunk_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the chunk folder path as a string, ending with "/".
	What it does: builds the folder that holds one chunking run's chunks:
	{output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	for the built-in chunkers, and the same with a {framework}/ folder after
	chunks/ when a framework such as langchain is used, so the two never
	share a cache.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/chunks/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)

# ------ Embedding
def get_embedding_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the chunk folder path as a string, ending with "/".
	What it does: builds the folder that holds one chunking run's chunks:
	{output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	for the built-in chunkers, and the same with a {framework}/ folder after
	chunks/ when a framework such as langchain is used, so the two never
	share a cache.
	"""

	if chunking_args.framework == NO_FRAMEWORK:
		framework_folder = ""
	else:
		framework_folder = f"{chunking_args.framework}/"

	return (
		f"{output_path}/embeddings/{framework_folder}{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)