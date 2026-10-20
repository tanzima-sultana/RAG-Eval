from doc_ingest import ingest_documents
from doc_ingest import settings

from common import paths
from common.args import IngestArgs
from stages.build.ingestion import cache
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def convert_document(library_document) -> Document:
	"""
	Input: library_document, one doc_ingest.schema.Document returned by
	ingest_documents.
	Output: the same document as this project's stages.build.ingestion.schema.Document.
	What it does: copies every field, including each metadata field, from
	the library's Document into a new Document of our own schema.
	"""
	library_metadata = library_document.metadata

	metadata = Metadata(
		source=library_metadata.source,
		file_size_bytes=library_metadata.file_size_bytes,
		content_hash=library_metadata.content_hash,
		minhash_signature=library_metadata.minhash_signature,
		page_count=library_metadata.page_count,
		parser=library_metadata.parser,
		author=library_metadata.author,
		title=library_metadata.title,
		creation_date=library_metadata.creation_date,
	)

	document_type = DocumentType(library_document.type.value)

	document = Document(
		id=library_document.id,
		type=document_type,
		text=library_document.text,
		metadata=metadata,
	)
	return document


def ingestion(ingest_args: IngestArgs) -> dict[str, Document]:
	"""
	Input: ingest_args, an IngestArgs holding the ingestion settings.
	Output: a dict of doc_id to this project's Document, one per ingested document.
	What it does: loads the first dataset_size documents from the dataset's
	processed data folder if enough are cached. Otherwise runs the doc_ingest
	library on ingest_args.input_path, prints the run summary, converts each
	library Document into our own Document, saves them and returns them.
	"""
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)
	print(f"Processed data folder : {paths.get_processed_data_path(output_path)}")

	cached_documents = cache.load_documents(output_path, ingest_args.dataset_size)
	if cached_documents is not None:
		print(f"Documents : {len(cached_documents)} (all from cache)")
		return cached_documents

	download_settings = settings.DownloadSettings(download_workers=ingest_args.download_workers)
	process_settings = settings.ProcessSettings(parse_workers=ingest_args.parse_workers)
	parser_settings = settings.ParserSettings(pdf_parser=ingest_args.pdf_parser)

	ingest_result = ingest_documents(
		size=ingest_args.dataset_size,
		input_path=ingest_args.input_path,
		download_settings=download_settings,
		process_settings=process_settings,
		parser_settings=parser_settings,
	)

	library_documents = ingest_result.documents
	summary = ingest_result.summary

	print(summary)

	documents = {}
	for library_document in library_documents:
		document = convert_document(library_document)
		documents[document.id] = document

	cache.save_documents(output_path, documents)
	print(f"Documents : {len(documents)} (all newly ingested and saved)")

	return documents
