from doc_ingest import ingest_documents
from doc_ingest import settings
from doc_ingest.schema import Document

from common.args import IngestArgs


def ingestion(ingest_args: IngestArgs) -> list[Document]:
	"""
	Input: ingest_args, an IngestArgs holding the ingestion settings.
	Output: a list of doc_ingest Documents, one per ingested document.
	What it does: runs the doc_ingest library on ingest_args.input_path,
	caching results under ingest_args.output_path, prints the run summary
	and returns the documents.
	"""
	download_settings = settings.DownloadSettings(download_workers=ingest_args.download_workers)
	process_settings = settings.ProcessSettings(parse_workers=ingest_args.parse_workers)
	parser_settings = settings.ParserSettings(pdf_parser=ingest_args.pdf_parser)

	ingest_result = ingest_documents(
		size=ingest_args.dataset_size,
		input_path=ingest_args.input_path,
		cache_dir=ingest_args.output_path,
		download_settings=download_settings,
		process_settings=process_settings,
		parser_settings=parser_settings,
	)

	documents = ingest_result.documents
	processed_dir = ingest_result.processed_dir
	summary = ingest_result.summary

	print(processed_dir)
	print(summary)

	return documents
