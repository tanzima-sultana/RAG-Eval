import hashlib
import os

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document as LangchainDocument

from common import paths
from common.args import IngestArgs
from stages.build.ingestion import cache
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def load_langchain_documents(raw_data_path: str, pdf_file_names: list[str]) -> dict[str, LangchainDocument]:
	"""
	Input: raw_data_path, the raw data folder; pdf_file_names, the PDF file
	names in it to load.
	Output: a dict of file name to LangChain Document, for each PDF that
	loaded.
	What it does: loads each PDF with LangChain's PyPDFium2Loader in
	"single" mode, which gives the whole PDF as one Document. A PDF that
	fails to load is skipped and printed.
	"""
	langchain_documents = {}
	failed_count = 0
	for file_name in pdf_file_names:
		pdf_file = os.path.join(raw_data_path, file_name)
		try:
			loader = PyPDFium2Loader(pdf_file, mode="single")
			loaded_documents = loader.load()
		except Exception as error:
			print(f"Failed to load {pdf_file} : {error}")
			failed_count += 1
			continue
		langchain_documents[file_name] = loaded_documents[0]

	print(f"PDFs read : {len(pdf_file_names)}, loaded : {len(langchain_documents)}, failed : {failed_count}")

	return langchain_documents


def convert_langchain_document(langchain_document: LangchainDocument) -> Document:
	"""
	Input: langchain_document, the LangChain Document that PyPDFium2Loader
	returned for one raw PDF.
	Output: the same document as this project's stages.build.ingestion.schema.Document.
	What it does: keeps LangChain's text as it is, reads the raw PDF bytes
	for the file size and SHA-256 hash, takes the first 16 characters of
	the hash as the document id, and copies the page count, author,
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
		id=content_hash[:16],
		type=DocumentType.PDF,
		text=langchain_document.page_content,
		metadata=metadata,
	)
	return document


def langchain_ingestion(ingest_args: IngestArgs) -> dict[str, Document]:
	"""
	Input: ingest_args, an IngestArgs holding the ingestion settings, with
	framework "langchain".
	Output: a dict of doc_id to this project's Document, one per PDF that
	LangChain loaded from the first dataset_size raw PDFs.
	What it does: takes the raw PDFs in file name order. The ones already
	read come from the cache (read_files.json and the document files); the
	ones up to dataset_size not read yet are loaded with LangChain,
	converted into our own Document and saved. Returns the documents of the
	first dataset_size PDFs, so the same dataset_size always gives the same
	documents, whatever was cached before.
	"""
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	print(f"Processed data folder : {paths.get_processed_data_path(output_path, ingest_args.framework)}")

	# 1. The PDFs already read, in order, with the document each one gave
	read_files = cache.load_read_files(output_path, ingest_args.framework)
	cached_read_count = len(read_files)

	# 2. Load the PDFs up to dataset_size that are not read yet
	new_documents = {}
	if cached_read_count < ingest_args.dataset_size:
		raw_data_path = paths.get_raw_data_path(output_path)
		print(f"Raw data folder : {raw_data_path}")

		pdf_file_names = []
		for file_name in sorted(os.listdir(raw_data_path)):
			if file_name.endswith(".pdf"):
				pdf_file_names.append(file_name)
		pdf_file_names_to_read = pdf_file_names[cached_read_count:ingest_args.dataset_size]

		langchain_documents = load_langchain_documents(raw_data_path, pdf_file_names_to_read)

		for file_name in pdf_file_names_to_read:
			if file_name in langchain_documents:
				document = convert_langchain_document(langchain_documents[file_name])
				new_documents[document.id] = document
				read_files.append({"file_name": file_name, "doc_id": document.id})
			else:
				read_files.append({"file_name": file_name, "doc_id": None})

	# 3. The documents of the first dataset_size PDFs: new ones from step 2, the rest from the cache
	first_read_files = read_files[:ingest_args.dataset_size]
	cached_doc_ids = []
	for read_file in first_read_files:
		if read_file["doc_id"] is not None and read_file["doc_id"] not in new_documents:
			cached_doc_ids.append(read_file["doc_id"])
	cached_documents = cache.load_documents_by_ids(output_path, ingest_args.framework, cached_doc_ids)

	documents = {}
	for read_file in first_read_files:
		doc_id = read_file["doc_id"]
		if doc_id is None:
			continue
		if doc_id in new_documents:
			documents[doc_id] = new_documents[doc_id]
		else:
			documents[doc_id] = cached_documents[doc_id]

	# 4. Save when new PDFs were read: the documents first, then read_files.json, so it only lists saved documents
	if len(read_files) > cached_read_count:
		cache.save_documents(output_path, ingest_args.framework, documents)
		cache.save_read_files(output_path, ingest_args.framework, read_files)

	print(
		f"PDFs : {len(first_read_files)} ({min(cached_read_count, len(first_read_files))} from cache, "
		f"{len(read_files) - cached_read_count} newly read)"
	)
	print(f"Documents : {len(documents)} ({len(cached_documents)} from cache, {len(new_documents)} new)")

	return documents
