import hashlib
import json
import os

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document as LangchainDocument

from common import paths
from common.args import IngestArgs
from common.constants import NO_FRAMEWORK
from stages.build.ingestion import cache
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def load_langchain_documents(output_path: str, dataset_size: int) -> dict[str, LangchainDocument]:
	"""
	Input: output_path, the dataset's output folder; dataset_size, the
	number of documents wanted.
	Output: a dict of doc_id to LangChain Document, one per PDF, in the
	order of the built-in doc_ids.json.
	What it does: takes the first dataset_size ids from the built-in
	doc_ids.json, so LangChain reads the same documents as the built-in
	ingestion. Loads each raw {doc_id}.pdf with LangChain's PyPDFium2Loader
	in "single" mode, which gives the whole PDF as one Document. Raises
	ValueError if fewer than dataset_size ids are listed, and
	FileNotFoundError if doc_ids.json or a raw PDF is missing.
	"""
	# 1. The same document ids, in the same order, as the built-in ingestion
	processed_data_path = paths.get_processed_data_path(output_path, NO_FRAMEWORK)
	doc_ids_file = os.path.join(processed_data_path, "doc_ids.json")
	if not os.path.exists(doc_ids_file):
		raise FileNotFoundError(
			f"No doc_ids.json in {processed_data_path}. Run the built-in ingestion first."
		)

	with open(doc_ids_file, encoding="utf-8") as f:
		built_in_doc_ids = json.load(f)

	if len(built_in_doc_ids) < dataset_size:
		raise ValueError(
			f"doc_ids.json lists {len(built_in_doc_ids)} documents, fewer than dataset_size ({dataset_size})."
		)

	# 2. Load each raw PDF with LangChain
	raw_data_path = paths.get_raw_data_path(output_path)
	print(f"Raw data folder : {raw_data_path}")

	langchain_documents = {}
	for doc_id in built_in_doc_ids[:dataset_size]:
		pdf_file = os.path.join(raw_data_path, f"{doc_id}.pdf")
		if not os.path.exists(pdf_file):
			raise FileNotFoundError(
				f"No raw PDF for {doc_id} in {raw_data_path}, but it is listed in doc_ids.json."
			)

		loader = PyPDFium2Loader(pdf_file, mode="single")
		loaded_documents = loader.load()
		langchain_documents[doc_id] = loaded_documents[0]

	print(f"Documents : {len(langchain_documents)} (loaded with LangChain PyPDFium2Loader)")

	return langchain_documents


def convert_langchain_document(doc_id: str, langchain_document: LangchainDocument) -> Document:
	"""
	Input: doc_id, the document's id; langchain_document, the LangChain
	Document that PyPDFium2Loader returned for that document's raw PDF.
	Output: the same document as this project's stages.build.ingestion.schema.Document.
	What it does: keeps LangChain's text as it is, reads the raw PDF bytes
	for the file size and SHA-256 hash, and copies the page count, author,
	title and creation date from LangChain's metadata. LangChain gives an
	empty string for a missing author, title or creation date; those
	become None, like in the built-in Documents.
	"""
	langchain_metadata = langchain_document.metadata
	pdf_file = langchain_metadata["source"]

	with open(pdf_file, "rb") as f:
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
		source=pdf_file,
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
		id=doc_id,
		type=DocumentType.PDF,
		text=langchain_document.page_content,
		metadata=metadata,
	)
	return document


def langchain_ingestion(ingest_args: IngestArgs) -> dict[str, Document]:
	"""
	Input: ingest_args, an IngestArgs holding the ingestion settings, with
	framework "langchain".
	Output: a dict of doc_id to this project's Document, one per ingested document.
	What it does: loads the first dataset_size documents from the LangChain
	processed data folder if enough are cached. Otherwise loads the raw
	PDFs of the built-in documents with LangChain, converts each into our
	own Document, saves them in the LangChain processed data folder and
	returns them.
	"""
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	print(f"Processed data folder : {paths.get_processed_data_path(output_path, ingest_args.framework)}")

	cached_documents = cache.load_documents(output_path, ingest_args.framework, ingest_args.dataset_size)
	if cached_documents is not None:
		print(f"Documents : {len(cached_documents)} (all from cache)")
		return cached_documents

	langchain_documents = load_langchain_documents(output_path, ingest_args.dataset_size)

	documents = {}
	for doc_id in langchain_documents:
		document = convert_langchain_document(doc_id, langchain_documents[doc_id])
		documents[doc_id] = document

	cache.save_documents(output_path, ingest_args.framework, documents)
	print(f"Documents : {len(documents)} (all newly converted and saved)")

	return documents
