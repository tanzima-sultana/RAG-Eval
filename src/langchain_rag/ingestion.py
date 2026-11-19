import hashlib
import json
import os

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document as LangchainDocument

from common import paths_temp
from common.constants import LANGCHAIN_FRAMEWORK
from common.args_temp import IngestionArgs
from schema.document import Document
from schema.document import DocumentType
from schema.document import Metadata


def convert_langchain_document(langchain_document: LangchainDocument, document_type: str) -> Document:
	"""
	Input: langchain_document, one loaded LangChain Document; document_type.
	Output: our own Document.
	What it does: converts the LangChain Document into our Document.
	"""
	langchain_metadata = langchain_document.metadata
	document_file = langchain_metadata["source"]

	with open(document_file, "rb") as f:
		raw_bytes = f.read()
	content_hash = hashlib.sha256(raw_bytes).hexdigest()

	author = langchain_metadata.get("author", "")
	if author == "":
		author = None

	title = langchain_metadata.get("title", "")
	if title == "":
		title = None

	creation_date = langchain_metadata.get("creationdate", "")
	if creation_date == "":
		creation_date = None

	metadata = Metadata(
		source=document_file,
		file_size_bytes=len(raw_bytes),
		content_hash=content_hash,
		minhash_signature=None,
		page_count=langchain_metadata["total_pages"],
		parser="langchain_pypdfium2",
		author=author,
		title=title,
		creation_date=creation_date,
	)

	document = Document(
		id=content_hash[:16],
		type=DocumentType(document_type),
		text=langchain_document.page_content,
		metadata=metadata,
	)
	return document


def save_documents(processed_data_path: str, documents: dict[str, Document]) -> None:
	"""
	Input: processed_data_path; documents.
	Output: nothing.
	What it does: saves each document to {doc_id}.json, then doc_ids.json.
	"""
	os.makedirs(processed_data_path, exist_ok=True)

	doc_ids = []
	for doc_id in documents:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		with open(document_file, "w", encoding="utf-8") as f:
			json.dump(documents[doc_id].model_dump(mode="json"), f)
		doc_ids.append(doc_id)

	# Written last, so it only lists documents whose files are saved
	doc_ids_file = os.path.join(processed_data_path, "doc_ids.json")
	with open(doc_ids_file, "w", encoding="utf-8") as f:
		json.dump(doc_ids, f)


def load_documents(processed_data_path: str) -> dict[str, Document] | None:
	"""
	Input: processed_data_path.
	Output: a dict of doc_id to Document, or None if nothing is saved.
	What it does: loads every document listed in doc_ids.json.
	"""
	doc_ids_file = os.path.join(processed_data_path, "doc_ids.json")
	if not os.path.exists(doc_ids_file):
		return None

	with open(doc_ids_file, encoding="utf-8") as f:
		doc_ids = json.load(f)

	documents = {}
	for doc_id in doc_ids:
		document_file = os.path.join(processed_data_path, f"{doc_id}.json")
		with open(document_file, encoding="utf-8") as f:
			document_dict = json.load(f)
		documents[doc_id] = Document.model_validate(document_dict)

	return documents


def validate(ingestion_args: IngestionArgs) -> None:
	"""
	Input: ingestion_args, the ingestion settings.
	Output: nothing.
	What it does: raises ValueError if a setting is not valid.
	"""
	if ingestion_args.dataset_size <= 0:
		raise ValueError(f"dataset_size ({ingestion_args.dataset_size}) must be greater than 0.")

	if ingestion_args.dataset_name == "":
		raise ValueError("dataset_name is empty.")

	if not os.path.isdir(ingestion_args.input_path):
		raise ValueError(f"input_path ({ingestion_args.input_path}) is not an existing folder.")

	if ingestion_args.output_path == "":
		raise ValueError("output_path is empty.")

	if ingestion_args.document_type != DocumentType.PDF:
		raise ValueError(
			f"document_type ({ingestion_args.document_type}) is not supported. Use \"{DocumentType.PDF.value}\"."
		)


def ingestion(ingestion_args: IngestionArgs) -> dict[str, Document]:
	"""
	Input: ingestion_args, the ingestion settings.
	Output: a dict of doc_id to Document.
	What it does: loads the documents from the cache, or loads the files with LangChain and saves them.
	"""
	# 1. Validate the settings
	validate(ingestion_args)

	# 2. Load from the cache of this dataset_size if it exists
	output_path = paths_temp.get_output_path(
		ingestion_args.output_path, LANGCHAIN_FRAMEWORK, ingestion_args.dataset_name
	)
	processed_data_path = paths_temp.get_documents_path(output_path, ingestion_args.dataset_size)
	print(f"Processed data folder : {processed_data_path}")
	cached_documents = load_documents(processed_data_path)
	if cached_documents is not None:
		print(f"Documents : {len(cached_documents)} (all from cache)")
		return cached_documents

	# 3. Up to dataset_size files of document_type in input_path, in file name order. Fewer is fine
	print(f"Input folder : {ingestion_args.input_path}")
	document_file_names = []
	for file_name in sorted(os.listdir(ingestion_args.input_path)):
		if file_name.endswith("." + ingestion_args.document_type):
			document_file_names.append(file_name)
	document_file_names_to_read = document_file_names[:ingestion_args.dataset_size]

	# 4. Load each file with LangChain as one Document, in "single" mode, and convert it
	documents = {}
	failed_count = 0
	empty_count = 0
	for file_name in document_file_names_to_read:
		document_file = os.path.join(ingestion_args.input_path, file_name)
		try:
			loader = PyPDFium2Loader(document_file, mode="single")
			loaded_documents = loader.load()
		except Exception as error:
			print(f"Failed to load {document_file} : {error}")
			failed_count += 1
			continue

		# A document with no text, or only spaces and newlines, gives nothing to chunk, so it is dropped
		langchain_document = loaded_documents[0]
		if langchain_document.page_content.strip() == "":
			print(f"Empty document skipped : {document_file}")
			empty_count += 1
			continue

		document = convert_langchain_document(langchain_document, ingestion_args.document_type)
		documents[document.id] = document

	print(
		f"Files read : {len(document_file_names_to_read)}, loaded : {len(documents)}, "
		f"failed : {failed_count}, empty : {empty_count}"
	)

	# 5. Save, so the next run loads them from the cache
	save_documents(processed_data_path, documents)

	return documents
