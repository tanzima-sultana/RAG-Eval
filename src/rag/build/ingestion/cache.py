import json
import os

from schema.file import File


def save_documents(processed_data_path: str, documents: dict[str, File], input_hash: str) -> None:
	"""
	Input: processed_data_path; documents; input_hash, the hash of the input files the documents were read from.
	Output: nothing.
	What it does: saves each document to {doc_id}.json, then doc_ids_{input_hash}.json.
	"""
	os.makedirs(processed_data_path, exist_ok=True)

	doc_ids = []
	for doc_id in documents:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		with open(document_file, "w", encoding="utf-8") as f:
			json.dump(documents[doc_id].model_dump(mode="json"), f)
		doc_ids.append(doc_id)

	# Written last, so it only lists documents whose files are saved
	doc_ids_file = os.path.join(processed_data_path, f"doc_ids_{input_hash}.json")
	with open(doc_ids_file, "w", encoding="utf-8") as f:
		json.dump(doc_ids, f)


def load_documents(processed_data_path: str, input_hash: str) -> dict[str, File] | None:
	"""
	Input: processed_data_path; input_hash, the hash of the input files.
	Output: a dict of doc_id to File, or None if the documents of these input files are not saved.
	What it does: loads every document listed in doc_ids_{input_hash}.json.
	"""
	doc_ids_file = os.path.join(processed_data_path, f"doc_ids_{input_hash}.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		doc_ids = json.load(f)

	documents = {}
	for doc_id in doc_ids:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		with open(document_file, encoding="utf-8") as f:
			document_dict = json.load(f)
		documents[doc_id] = File.model_validate(document_dict)

	return documents


def load_saved_documents(documents_path: str, dataset_size: int) -> dict[str, File]:
	"""
	Input: documents_path, the documents folder from the build manifest; dataset_size.
	Output: a dict of doc_id to File.
	What it does: loads the documents the build saved, and raises FileNotFoundError if they are not saved. It never
	reads the input files.
	"""
	documents = load_documents(documents_path, dataset_size)
	if documents is None:
		raise FileNotFoundError(
			f"No saved documents for dataset_size {dataset_size} in {documents_path}. Run scripts/build.sh first."
		)

	print(f"Documents : {len(documents)} (all from cache)")
	return documents
