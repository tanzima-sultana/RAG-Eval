
import os

from common.args import ChunkingArgs

# ------ Chunking
def get_chunk_path(output_path:str, chunking_args: ChunkingArgs) -> str:
	"""
	Input: output_path, the dataset's output folder; chunking_args, the
	chunking parameters.
	Output: the chunk folder path as a string, ending with "/".
	What it does: builds the folder that holds one chunking run's chunks:
	{output_path}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/
	"""

	return (
		f"{output_path}/chunks/{chunking_args.chunk_type}/{chunking_args.model_name}/"
		f"{chunking_args.min_chunk_size}_{chunking_args.max_chunk_size}_"
		f"{chunking_args.fixed_chunk_overlap}_{chunking_args.semantic_threshold}/"
	)