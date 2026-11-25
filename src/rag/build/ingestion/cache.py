import json
import os

from schema.file import File


def save_files(processed_data_path: str, new_files: list[File], document_ids: list[str], ingestion_hash: str) -> None:
	"""
	Input: processed_data_path; new_files, the files this run parsed; document_ids, the document ids of every file of
	this run, parsed or loaded from the cache, in order; ingestion_hash, the hash of the input files of this run.
	Output: nothing.
	What it does: saves each new file to {document_id}.json, then saves the list of document ids to
	doc_ids_{ingestion_hash}.json. The files loaded from the cache are already saved.
	"""
	os.makedirs(processed_data_path, exist_ok=True)

	for file in new_files:
		document_file = os.path.join(processed_data_path, f"{file.document_id}.json")
		with open(document_file, "w", encoding="utf-8") as f:
			json.dump(file.model_dump(mode="json"), f)

	# Written last, so it only lists files whose documents are saved
	doc_ids_file = os.path.join(processed_data_path, f"doc_ids_{ingestion_hash}.json")
	with open(doc_ids_file, "w", encoding="utf-8") as f:
		json.dump(document_ids, f)


def load_file(processed_data_path: str, document_id: str) -> File | None:
	"""
	Input: processed_data_path; document_id, the first 16 characters of the SHA-256 of a file's bytes.
	Output: the File, or None if no file with this content is saved.
	What it does: loads the file from {document_id}.json.
	"""
	document_file = os.path.join(processed_data_path, f"{document_id}.json")
	if not os.path.exists(document_file):
		return None

	with open(document_file, encoding="utf-8") as f:
		file_dict = json.load(f)
	return File.model_validate(file_dict)


def load_files(processed_data_path: str, ingestion_hash: str) -> list[File] | None:
	"""
	Input: processed_data_path; ingestion_hash, the hash of the input files.
	Output: the list of File of this run, or None if the run is not saved or one of its files is missing.
	What it does: loads every file listed in doc_ids_{ingestion_hash}.json.
	"""
	doc_ids_file = os.path.join(processed_data_path, f"doc_ids_{ingestion_hash}.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		document_ids = json.load(f)

	files = []
	for document_id in document_ids:
		file = load_file(processed_data_path, document_id)
		if file is None:
			return None
		files.append(file)

	return files
