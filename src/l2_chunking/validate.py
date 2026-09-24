import os 
import json
from common import paths

from common.constants import FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING
from common.chunk_config import ChunkConfig

def validate_processed_dataset(dataset_name:str, dataset_size:int):
	"""
	dataset_name: name of the dataset.
	dataset_size: number of processed documents required.

	Checks that the dataset's processed folder exists and holds at least
	dataset_size documents, and that its progress.json exists and lists at
	least dataset_size processed doc_ids.

	Raises FileNotFoundError if the processed folder or progress.json is
	missing, and ValueError if there are too few processed documents.

	Returns nothing.
	"""

	# 1. Processed data path does not exists
	processed_path = paths.get_processed_dataset_path(dataset_name)
	if not os.path.exists(processed_path):
		raise FileNotFoundError(
			f"Processed dataset path '{processed_path}' does not exist."
		)
	
	# 2. No files exists in the path
	sorted_file_names = sorted(os.listdir(processed_path))
	if len(sorted_file_names) == 0:
		raise ValueError(
			f"'{processed_path}' exists but holds no processed documents."
			f"before chunking it."
		)

	# 3. dataset size is bigger than already processed documents
	if dataset_size > len(sorted_file_names):
		raise ValueError(
			f"Requested dataset_size {dataset_size} exceeds the "
			f"{len(sorted_file_names)} document(s) already processed "
			f"under '{processed_path}'."
		)
	
	# 4. Progress path does not exists
	progress_path = paths.get_progress_path(dataset_name)
	file_path = paths.get_file_path(progress_path, "progress", "json")
	if not os.path.exists(file_path):
		raise FileNotFoundError(
				f"Progress path '{file_path}' does not exist."
			)

	with open(file_path, encoding="utf-8") as f:
		data = json.load(f)

	# 5. Check processed doc_ids saved in Progress
	processed_doc_ids=data["processed_doc_ids"]
	if len(processed_doc_ids) < dataset_size:
		raise ValueError(
			f"Requested dataset_size {dataset_size} exceeds the "
			f"{len(sorted_file_names)} document(s) already processed."
		)

def validate_chunk_config(chunk_config: ChunkConfig):
	"""
	chunk_config: the chunking parameters to check.

	Checks that chunk_type is supported, min_chunk_size is smaller than
	max_chunk_size, fixed_chunk_overlap is smaller than max_chunk_size, and
	semantic chunking has a non-zero semantic_threshold.

	Raises ValueError for the first check that fails.

	Returns nothing.
	"""

	# 1. Check chunk_type
	if chunk_config.chunk_type not in (FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING):
		raise ValueError(
			f"chunk_type ({chunk_config.chunk_type}) is not supported yet. Provide a valid chunk_type."
		)
	
	# 2. Check min and max chunk size
	if chunk_config.min_chunk_size >= chunk_config.max_chunk_size:
		raise ValueError(
			f"min_chunk_size ({chunk_config.min_chunk_size}) must be smaller than max_chunk_size ({chunk_config.max_chunk_size})"
		)
	# 3. Check fixed chunk overlap
	if chunk_config.fixed_chunk_overlap >= chunk_config.max_chunk_size:
		raise ValueError(
			f"fixed_chunk_overlap ({chunk_config.fixed_chunk_overlap}) must be smaller than "
			f"max_chunk_size ({chunk_config.max_chunk_size}); otherwise the sliding window "
			"never advances."
		)
	# 4. Check semnatic threshold
	if chunk_config.chunk_type == SEMANTIC_CHUNKING and chunk_config.semantic_threshold == 0.0:
		raise ValueError(
			f"provide valid semantic threshold for semantic chunking"
		)