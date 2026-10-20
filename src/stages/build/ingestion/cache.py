"""
cache.py

Saves and loads ingested Documents as JSON files, one {doc_id}.json per
document, in the processed data folder of a dataset (see
common.paths.get_processed_data_path). A doc_ids.json file in the same
folder keeps the document ids in the order ingestion returned them, so
the first dataset_size documents are always the same ones.

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file that a later
run would mistake for a complete cache.
"""

import json
import os

from common import paths
from stages.build.ingestion.schema import Document

def save_documents(output_path: str, documents: dict[str, Document]) -> None:
	"""
	Input: output_path, the dataset's output folder; documents, a dict of
	doc_id to Document in the order ingestion returned them.
	Output: nothing.
	What it does: writes each document to {doc_id}.json in the processed
	data folder, then writes the ordered doc ids to doc_ids.json. The ids
	are written last, so they only list documents whose files are complete.
	"""
	processed_data_path = paths.get_processed_data_path(output_path)
	os.makedirs(processed_data_path, exist_ok=True)

	doc_ids = []
	for doc_id in documents:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		temporary_file = document_file + ".tmp"
		with open(temporary_file, "w", encoding="utf-8") as f:
			json.dump(documents[doc_id].model_dump(mode="json"), f)
		os.replace(temporary_file, document_file)
		doc_ids.append(doc_id)

	doc_ids_file = os.path.join(processed_data_path, "doc_ids.json")
	temporary_file = doc_ids_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(doc_ids, f)
	os.replace(temporary_file, doc_ids_file)

def load_documents(output_path: str, dataset_size: int) -> dict[str, Document] | None:
	"""
	Input: output_path, the dataset's output folder; dataset_size, the
	number of documents wanted.
	Output: a dict of doc_id to Document with the first dataset_size
	documents of doc_ids.json, in that order, or None if fewer than
	dataset_size documents are cached.
	What it does: reads doc_ids.json from the processed data folder and, if
	it lists enough ids, reads each of the first dataset_size {doc_id}.json
	files into a Document. Raises FileNotFoundError if an id in
	doc_ids.json has no file.
	"""
	processed_data_path = paths.get_processed_data_path(output_path)
	doc_ids_file = os.path.join(processed_data_path, "doc_ids.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		cached_doc_ids = json.load(f)

	if len(cached_doc_ids) < dataset_size:
		return None

	documents = {}
	for doc_id in cached_doc_ids[:dataset_size]:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		if not os.path.exists(document_file):
			raise FileNotFoundError(
				f"No document file for {doc_id} in {processed_data_path}, but it is listed in doc_ids.json."
			)
		with open(document_file, encoding="utf-8") as f:
			document_dict = json.load(f)
		documents[doc_id] = Document.model_validate(document_dict)

	return documents
