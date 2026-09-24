"""
cache.py

Saves and loads processed Documents as JSON files under
{dataset_name}/data/processed/{doc_id}.json, and loads the first
processed documents of a dataset in the order listed in progress.json.
"""

import dataclasses
import datetime
import json
import os

from common import paths
from l1_doc_processing.schema import Document


def document_to_json_dict(document: Document) -> dict:
	"""
	document: the Document to convert.

	Returns a JSON-serializable dict, with timestamp as an ISO-8601 string.
	"""
	data = dataclasses.asdict(document)
	data["timestamp"] = document.timestamp.isoformat()
	return data


def json_dict_to_document(data: dict) -> Document:
	"""
	data: a dict produced by document_to_json_dict.

	Returns the Document, with timestamp parsed back into a datetime.
	"""
	data = dict(data)
	data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
	return Document(**data)


def load_processed_document(dataset_name: str, doc_id: str) -> Document | None:
	"""
	dataset_name: name of the dataset.
	doc_id: the processed document to load.

	Returns the Document from {dataset_name}/data/processed/{doc_id}.json,
	or None if the file does not exist.
	"""
	processed_path = paths.get_processed_dataset_path(dataset_name)
	file_path = paths.get_file_path(processed_path, doc_id, "json")
	if not os.path.exists(file_path):
		return None
	with open(file_path, encoding="utf-8") as f:
		return json_dict_to_document(json.load(f))


def save_processed_document(dataset_name: str, doc_id: str, document: Document) -> None:
	"""
	dataset_name: name of the dataset.
	doc_id: the document's id, used as the file name.
	document: the Document to save.

	Writes document to {dataset_name}/data/processed/{doc_id}.json.

	Returns nothing.
	"""
	processed_path = paths.get_processed_dataset_path(dataset_name)
	file_path = paths.get_file_path(processed_path, doc_id, "json")

	os.makedirs(processed_path, exist_ok=True)

	with open(file_path, "w", encoding="utf-8") as f:
		json.dump(document_to_json_dict(document), f)


def load_document_ids(dataset_name: str, dataset_size:int) -> list:
	"""
	dataset_name: name of the dataset.
	dataset_size: number of doc_ids to return.

	Returns the first dataset_size processed doc_ids from
	{dataset_name}/data/progress.json, in processing order.
	"""
	progress_path = paths.get_progress_path(dataset_name)
	file_path = paths.get_file_path(progress_path, "progress", "json")

	with open(file_path, encoding="utf-8") as f:
		data = json.load(f)

	processed_doc_ids=data["processed_doc_ids"]	
	return processed_doc_ids[:dataset_size]

def load_documents(dataset_name: str, dataset_size:int) -> dict[str, Document]:
	"""
	dataset_name: name of the dataset.
	dataset_size: number of documents to load.

	Raises FileNotFoundError if a doc_id listed in progress.json has no
	processed file.

	Returns a dict of doc_id to Document for the first dataset_size
	doc_ids in progress.json, in processing order.
	"""

	processed_doc_ids = load_document_ids(dataset_name, dataset_size)

	documents = {}
	for doc_id in processed_doc_ids:
		document = load_processed_document(dataset_name, doc_id)
		if document is None:
				raise FileNotFoundError(
						f"Processed document '{doc_id}' is listed in progress.json but its file is missing."
				)
		documents[doc_id] = document

	return documents
