"""
progress.py

Defines the Progress dataclass and builds, saves and loads the progress
of doc processing for one dataset ({dataset_name}/data/progress.json),
so a run can resume where the last one stopped.
"""

import dataclasses
import json
import os

from common import paths
from common.dataset_config import DatasetConfig


@dataclasses.dataclass
class Progress:
	"""
	Tracks how far doc processing has got for one dataset, so a run
	can resume where the last one stopped.

	dataset_identifier: dataset_url or dataset_path this progress was
		built from, taken from the registered DatasetConfig.identifier.
	split: dataset split the rows were read from (e.g. "train").
	doc_type: document type the rows were processed as (e.g. "pdf").
	rows_consumed: number of source rows read so far, kept or skipped.
		The next run starts reading after these rows.
	processed_doc_ids: doc_ids processed and saved, in processing order.
	skipped_doc_ids: doc_ids that failed parsing or whose processed
		file went missing. They are not parsed again.
	"""
	dataset_identifier: str
	split: str
	doc_type: str
	rows_consumed: int
	processed_doc_ids: list[str]
	skipped_doc_ids: list[str]


def make_progress(dataset_config: DatasetConfig, doc_type: str, rows_consumed: int, processed_doc_ids: list[str], skipped_doc_ids: set[str]) -> Progress:
	"""
	dataset_config: registered DatasetConfig of the dataset.
	doc_type: document type being processed (e.g. "pdf").
	rows_consumed: source rows read so far.
	processed_doc_ids: processed doc_ids, in processing order.
	skipped_doc_ids: doc_ids that failed parsing or went missing.

	Returns a Progress for this dataset with the given values.
	"""
	progress = Progress(
		dataset_identifier=dataset_config.identifier,
		split=dataset_config.split,
		doc_type=doc_type,
		rows_consumed=rows_consumed,
		processed_doc_ids=processed_doc_ids,
		skipped_doc_ids=list(skipped_doc_ids),
	)
	return progress


def load_progress(dataset_name: str, dataset_config: DatasetConfig, doc_type: str) -> Progress:
	"""
	dataset_name: name of the dataset; its progress is read from
		{dataset_name}/data/progress.json.
	dataset_config: registered DatasetConfig, used for an empty Progress.
	doc_type: document type being processed, used for an empty Progress.

	Returns the saved Progress, or, on a first run, an empty Progress
	(rows_consumed 0, empty lists) for the current dataset.
	"""
	progress_path = paths.get_progress_path(dataset_name)
	file_path = paths.get_file_path(progress_path, "progress", "json")

	if not os.path.exists(file_path):
		empty_progress = make_progress(dataset_config, doc_type, 0, [], set())
		return empty_progress

	with open(file_path, encoding="utf-8") as f:
		data = json.load(f)

	progress = Progress(
		dataset_identifier=data["dataset_identifier"],
		split=data["split"],
		doc_type=data["doc_type"],
		rows_consumed=data["rows_consumed"],
		processed_doc_ids=data["processed_doc_ids"],
		skipped_doc_ids=data["skipped_doc_ids"]
	)
	return progress


def save_progress(dataset_name: str, progress: Progress) -> None:
	"""
	dataset_name: name of the dataset.
	progress: the Progress to save.

	Writes progress to {dataset_name}/data/progress.json through a
	temporary file, so a crash never leaves a half-written file.

	Returns nothing.
	"""
	progress_path = paths.get_progress_path(dataset_name)
	file_path = paths.get_file_path(progress_path, "progress", "json")
	temporary_file_path = file_path + ".tmp"

	os.makedirs(progress_path, exist_ok=True)

	with open(temporary_file_path, "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(progress), f)

	os.replace(temporary_file_path, file_path)
