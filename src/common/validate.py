import os 

# ------------ 1. DocProcessing
import json
from common import paths
from datasets import load_dataset_builder
from datasets.exceptions import DatasetNotFoundError
from common.dataset_config import DatasetConfig, get_dataset_config
from l1_doc_processing.progress import Progress

from common.constants import FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING
from common.chunk_config import ChunkConfig

def validate_dataset(doc_type: str, dataset_name: str, dataset_size: int, dataset_url: str, dataset_path: str):
	"""
	doc_type: document type the user asked to process (e.g. "pdf").
	dataset_name: name of the dataset, used to look up its registered
		config in common/dataset_config.py.
	dataset_size: number of processed documents the user wants.
	dataset_url: Hugging Face dataset id, for a web dataset.
	dataset_path: local directory, for a local dataset.

	Runs five checks before any download or processing:
	1. dataset_name is registered in common/dataset_config.py.
	2. The given dataset_url or dataset_path matches the registered
	   identifier.
	3. doc_type matches the registered doc_type.
	4. The dataset exists at dataset_url or dataset_path.
	5. The dataset's raw document count is at least dataset_size.
	   If the dataset does not publish a count, prints a warning and
	   leaves this check to doc processing.

	Raises ValueError if dataset_name is not registered, the given
	dataset_url or dataset_path does not match the registered
	identifier, doc_type does not match, or the known raw count is
	smaller than dataset_size.
	Raises FileNotFoundError if the dataset does not exist.

	Returns nothing. If all checks pass, nothing is raised.
	"""

	# 1. dataset is registered or not
	dataset_config = get_dataset_config(dataset_name)

	# 2. given url or path matches identifier or not
	if dataset_url:
		if dataset_url != dataset_config.identifier:
			raise ValueError(
				f"Given dataset_url '{dataset_url}' does not match the registered dataset_url '{dataset_config.identifier}'"
			)
	
	if dataset_path:
		if dataset_path != dataset_config.identifier:
			raise ValueError(
				f"Given dataset_path '{dataset_path}' does not match the registered dataset_path '{dataset_config.identifier}'"
			)

	# 3. given doc_type matches dataset_config.doc_type
	if doc_type != dataset_config.doc_type:
		raise ValueError(
			f"doc_type '{doc_type}' does not match the registered doc_type '{dataset_config.doc_type}'"
		)
	
	builder = None
	# 4. Dataset actually exists in the given url or path
	if dataset_url:
		try:
			builder = load_dataset_builder(dataset_url)
		except DatasetNotFoundError:
			raise FileNotFoundError(
				f"Dataset '{dataset_url}' does not exist or is not accessible."
			)
	else:
		if not os.path.isdir(dataset_path):
			raise FileNotFoundError(
				f"Dataset path '{dataset_path}' does not exist."
			)
	
	# 5. given dataset_size is less than or equal to actual dataset size
	actual_dataset_size = None
	if dataset_url:
		if builder.info.splits is None:
			actual_dataset_size = None
		else:
			split_info = builder.info.splits.get(dataset_config.split)
			if split_info is None:
				actual_dataset_size = None
			else:
				actual_dataset_size =  split_info.num_examples
	else:
		actual_dataset_size = len(os.listdir(dataset_path))
	
	if actual_dataset_size is None:
		print(
			f"Warning: dataset '{dataset_url or dataset_path}' does not "
			f"publish a document count in its metadata, so dataset_size "
			f"{dataset_size} cannot be validated upfront. It will be "
			f"checked during processing."
		)
	elif actual_dataset_size < dataset_size:
		raise ValueError(
			f"Requested dataset_size {dataset_size} exceeds the actual "
			f"dataset size {actual_dataset_size} for dataset "
			f"'{dataset_url or dataset_path}'."
		)


def validate_progress(progress: Progress, dataset_config: DatasetConfig, doc_type: str):
	"""
	progress: Progress loaded from the dataset's progress.json.
	dataset_config: registered DatasetConfig of the current run. Its
		identifier and split are compared with the saved ones.
	doc_type: document type of the current run (e.g. "pdf").

	Checks that the saved progress was built from the same dataset
	identifier, split and doc_type as the current run, so a run never resumes on
	top of progress from a different source.

	Raises ValueError listing every field that does not match, with
	its saved and current value.

	Returns nothing. If all fields match, nothing is raised.
	"""
	mismatches = []

	if progress.dataset_identifier != dataset_config.identifier:
		mismatches.append(
			f"dataset_identifier: saved '{progress.dataset_identifier}', current '{dataset_config.identifier}'"
		)

	if progress.split != dataset_config.split:
		mismatches.append(
			f"split: saved '{progress.split}', current '{dataset_config.split}'"
		)

	if progress.doc_type != doc_type:
		mismatches.append(
			f"doc_type: saved '{progress.doc_type}', current '{doc_type}'"
		)

	if len(mismatches) > 0:
		mismatch_lines = "\n  ".join(mismatches)
		raise ValueError(
			f"Saved progress does not match the current run:\n  {mismatch_lines}\n"
			f"Use a different dataset_name, or remove the dataset folder. "
			f"If only the URL changed and the data is the same, update "
			f"dataset_identifier in progress.json."
		)

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

# ------------ 2. Chunking
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