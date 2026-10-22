"""
cache.py

Saves and loads ingested Documents as JSON files, one {doc_id}.json per
document, in the processed data folder of a dataset (see
common.paths.get_processed_data_path). The built-in ingestion and each
framework, such as langchain, have their own processed data folder. A doc_ids.json file in the same
folder keeps the document ids in the order ingestion returned them, so
the first dataset_size documents are always the same ones.

LangChain ingestion can return fewer documents than the PDFs it read, so
its folder also holds read_files.json: every raw PDF it has read, in
order, with the id of the document it gave, or None if it failed to load.
A request for N documents then means the documents of the first N PDFs.

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file that a later
run would mistake for a complete cache.
"""

import json
import os

from common import paths
from stages.build.ingestion.schema import Document

def save_documents(output_path: str, framework: str, documents: dict[str, Document]) -> None:
	"""
	Input: output_path, the dataset's output folder; framework, "" for the
	built-in ingestion or a framework such as langchain; documents, a dict
	of doc_id to Document in the order ingestion returned them.
	Output: nothing.
	What it does: writes each document to {doc_id}.json in the processed
	data folder, then writes the ordered doc ids to doc_ids.json. The ids
	are written last, so they only list documents whose files are complete.
	"""
	processed_data_path = paths.get_processed_data_path(output_path, framework)
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

def load_documents(output_path: str, framework: str, dataset_size: int) -> dict[str, Document] | None:
	"""
	Input: output_path, the dataset's output folder; framework, "" for the
	built-in ingestion or a framework such as langchain; dataset_size, the
	number of documents wanted.
	Output: a dict of doc_id to Document with the first dataset_size
	documents of doc_ids.json, in that order, or None if fewer than
	dataset_size documents are cached.
	What it does: reads doc_ids.json from the processed data folder and, if
	it lists enough ids, reads each of the first dataset_size {doc_id}.json
	files into a Document. Raises FileNotFoundError if an id in
	doc_ids.json has no file.
	"""
	processed_data_path = paths.get_processed_data_path(output_path, framework)
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

def save_read_files(output_path: str, framework: str, read_files: list[dict]) -> None:
	"""
	Input: output_path, the dataset's output folder; framework, the
	framework of the cache; read_files, one {"file_name", "doc_id"} per PDF
	read, in read order, with doc_id None for a PDF that failed to load.
	Output: nothing.
	What it does: writes read_files to read_files.json in the processed
	data folder, through a temporary file.
	"""
	processed_data_path = paths.get_processed_data_path(output_path, framework)
	os.makedirs(processed_data_path, exist_ok=True)
	read_files_file = os.path.join(processed_data_path, "read_files.json")
	temporary_file = read_files_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(read_files, f)
	os.replace(temporary_file, read_files_file)

def load_read_files(output_path: str, framework: str) -> list[dict]:
	"""
	Input: output_path, the dataset's output folder; framework, the
	framework of the cache.
	Output: the {"file_name", "doc_id"} list from read_files.json, or an
	empty list if the file does not exist.
	What it does: reads read_files.json from the processed data folder.
	"""
	processed_data_path = paths.get_processed_data_path(output_path, framework)
	read_files_file = os.path.join(processed_data_path, "read_files.json")
	if not os.path.exists(read_files_file):
		return []

	with open(read_files_file, encoding="utf-8") as f:
		read_files = json.load(f)
	return read_files

def load_documents_by_ids(output_path: str, framework: str, doc_ids: list[str]) -> dict[str, Document]:
	"""
	Input: output_path, the dataset's output folder; framework, the
	framework of the cache; doc_ids, the ids of the documents to load.
	Output: a dict of doc_id to Document, in the order of doc_ids.
	What it does: reads each {doc_id}.json from the processed data folder.
	Raises FileNotFoundError if a document has no file.
	"""
	processed_data_path = paths.get_processed_data_path(output_path, framework)

	documents = {}
	for doc_id in doc_ids:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		if not os.path.exists(document_file):
			raise FileNotFoundError(
				f"No document file for {doc_id} in {processed_data_path}, but it is listed in read_files.json."
			)
		with open(document_file, encoding="utf-8") as f:
			document_dict = json.load(f)
		documents[doc_id] = Document.model_validate(document_dict)

	return documents
