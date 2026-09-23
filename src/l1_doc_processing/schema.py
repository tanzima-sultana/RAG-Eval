
import dataclasses
import datetime

@dataclasses.dataclass
class Document:
	id: str
	text: str
	title: str | None
	path: str
	format: str
	metadata: dict[str, object]
	content_hash: str
	timestamp: datetime.datetime


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