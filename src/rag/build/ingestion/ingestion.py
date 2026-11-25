import hashlib
import json
import os

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document as LangchainDocument

from common import paths
from common.args import IngestionArgs
from rag.build.ingestion.cache import load_file
from rag.build.ingestion.cache import load_files
from rag.build.ingestion.cache import save_files
from schema.file import File
from schema.file import FileType
from schema.file import Metadata
from schema.file import Page


def convert_langchain_pages(langchain_pages: list[LangchainDocument], document_type: str, content_hash: str, file_size_bytes: int) -> File:
	"""
	Input: langchain_pages, the non-blank pages of one file as LangChain Documents, in page order, at least one;
	document_type; content_hash, the SHA-256 of the file's bytes; file_size_bytes, the size of the file.
	Output: our own File.
	What it does: converts the pages of one file into our File. The document id is the first 16 characters of
	content_hash. Each page is a Page keyed by its 0-based page number, with the page id {document_id}_{page_number}.
	The file's text is the pages' texts joined with "\n\f", the delimiter LangChain's single mode uses.
	"""
	# The file-level values are the same on every page, so the first page gives them
	first_langchain_metadata = langchain_pages[0].metadata
	document_id = content_hash[:16]

	author = first_langchain_metadata.get("author", "")
	if author == "":
		author = None

	title = first_langchain_metadata.get("title", "")
	if title == "":
		title = None

	creation_date = first_langchain_metadata.get("creationdate", "")
	if creation_date == "":
		creation_date = None

	pages = {}
	page_texts = []
	for langchain_page in langchain_pages:
		page_number = langchain_page.metadata["page"]
		pages[page_number] = Page(
			page_id=f"{document_id}_{page_number}",
			page_number=page_number,
			page_texts=langchain_page.page_content,
		)
		page_texts.append(langchain_page.page_content)

	metadata = Metadata(
		file_size_bytes=file_size_bytes,
		content_hash=content_hash,
		minhash_signature=None,
		page_count=first_langchain_metadata["total_pages"],
		parser="langchain_pypdfium2",
		author=author,
		title=title,
		creation_date=creation_date,
	)

	file = File(
		document_id=document_id,
		document_type=FileType(document_type),
		document_texts="\n\f".join(page_texts),
		source=first_langchain_metadata["source"],
		metadata=metadata,
		pages=pages,
	)
	return file


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

	if ingestion_args.document_type != FileType.PDF:
		raise ValueError(
			f"document_type ({ingestion_args.document_type}) is not supported. Use \"{FileType.PDF.value}\"."
		)


def list_input_files(ingestion_args: IngestionArgs) -> list[str]:
	"""
	Input: ingestion_args, the ingestion settings.
	Output: the file names ingestion reads.
	What it does: lists the files of document_type in input_path in file name order and keeps the first dataset_size.
	Fewer is fine.
	"""
	document_file_names = []
	for file_name in sorted(os.listdir(ingestion_args.input_path)):
		if file_name.endswith("." + ingestion_args.document_type):
			document_file_names.append(file_name)
	return document_file_names[:ingestion_args.dataset_size]


def compute_ingestion_hash(ingestion_args: IngestionArgs) -> str:
	"""
	Input: ingestion_args, the ingestion settings.
	Output: the first 12 hex characters of the SHA-256 of the input files ingestion reads.
	What it does: takes the SHA-256 of each file's content, and hashes the file names with their hashes as JSON with
	sorted keys, so editing, adding or removing any of these files gives a new ingestion hash.
	"""
	validate(ingestion_args)

	file_hashes = {}
	for file_name in list_input_files(ingestion_args):
		with open(os.path.join(ingestion_args.input_path, file_name), "rb") as f:
			file_hashes[file_name] = hashlib.sha256(f.read()).hexdigest()

	file_hashes_json = json.dumps(file_hashes, sort_keys=True)
	return hashlib.sha256(file_hashes_json.encode("utf-8")).hexdigest()[:12]


def ingestion(ingestion_args: IngestionArgs, output_path: str) -> tuple[list[File], str, str]:
	"""
	Input: ingestion_args, the ingestion settings; output_path, the dataset's output folder.
	Output: (files, ingestion_hash, documents_path): one File per input file, each holding its pages, the hash of the input
	files, and the folder the files are saved in.
	What it does: loads the files already saved, by the content hash of each input file, parses only the input files
	with no saved file, page by page with LangChain, and saves them. If one input file changes, only that file is parsed.
	"""
	# 1. Validate the settings, then compute the hash of the input files
	validate(ingestion_args)
	ingestion_hash = compute_ingestion_hash(ingestion_args)

	# 2. Load everything from the cache if this exact run was saved before
	processed_data_path = paths.get_documents_path(output_path)
	print(f"Processed data folder : {processed_data_path}")
	cached_files = load_files(processed_data_path, ingestion_hash)
	if cached_files is not None:
		cached_page_count = 0
		for cached_file in cached_files:
			cached_page_count += len(cached_file.pages)
		print(f"Files : {len(cached_files)}, pages : {cached_page_count} (all from cache)")
		return cached_files, ingestion_hash, processed_data_path

	# 3. The files to read: up to dataset_size files of document_type in input_path, in file name order
	print(f"Input folder : {ingestion_args.input_path}")
	document_file_names_to_read = list_input_files(ingestion_args)

	# 4. For each input file: load its saved file by its document id, or parse it page by page and convert its
	# non-blank pages into a File
	files = []
	new_files = []
	loaded_document_ids = set()
	from_cache_count = 0
	failed_count = 0
	duplicate_count = 0
	empty_file_count = 0
	page_count = 0
	empty_page_count = 0
	for file_name in document_file_names_to_read:
		document_file = os.path.join(ingestion_args.input_path, file_name)
		try:
			with open(document_file, "rb") as f:
				raw_bytes = f.read()
		except Exception as error:
			print(f"Failed to read {document_file} : {error}")
			failed_count += 1
			continue

		# The same file content twice gives the same document id, so the second one is skipped
		content_hash = hashlib.sha256(raw_bytes).hexdigest()
		document_id = content_hash[:16]
		if document_id in loaded_document_ids:
			print(f"Duplicate file skipped : {document_file}")
			duplicate_count += 1
			continue

		# A file with this content was parsed before, so it is not parsed again
		saved_file = load_file(processed_data_path, document_id)
		if saved_file is not None:
			files.append(saved_file)
			loaded_document_ids.add(document_id)
			from_cache_count += 1
			page_count += len(saved_file.pages)
			continue

		try:
			loader = PyPDFium2Loader(document_file, mode="page")
			langchain_pages = loader.load()
		except Exception as error:
			print(f"Failed to load {document_file} : {error}")
			failed_count += 1
			continue

		# A page with no text, or only spaces and newlines, gives nothing to chunk, so it is dropped
		non_blank_langchain_pages = []
		for langchain_page in langchain_pages:
			if langchain_page.page_content.strip() == "":
				empty_page_count += 1
				continue
			non_blank_langchain_pages.append(langchain_page)

		# A file with no non-blank page gives nothing to chunk, so it is dropped
		if len(non_blank_langchain_pages) == 0:
			print(f"Empty file skipped : {document_file}")
			empty_file_count += 1
			continue

		file = convert_langchain_pages(non_blank_langchain_pages, ingestion_args.document_type, content_hash, len(raw_bytes))
		files.append(file)
		new_files.append(file)
		loaded_document_ids.add(document_id)
		page_count += len(file.pages)

	print(
		f"Files read : {len(document_file_names_to_read)}, from cache : {from_cache_count}, parsed : {len(new_files)}, "
		f"failed : {failed_count}, duplicate : {duplicate_count}, empty : {empty_file_count}, "
		f"files loaded : {len(files)}, pages loaded : {page_count}, empty pages : {empty_page_count}"
	)

	# 5. Save the new files and the list of this run, so the next run loads them from the cache
	document_ids = []
	for file in files:
		document_ids.append(file.document_id)
	save_files(processed_data_path, new_files, document_ids, ingestion_hash)

	return files, ingestion_hash, processed_data_path
