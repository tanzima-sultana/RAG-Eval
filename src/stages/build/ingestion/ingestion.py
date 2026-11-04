from doc_ingest import ingest_documents

from common import paths
from common.constants import NO_FRAMEWORK
from common.constants import LANGCHAIN_FRAMEWORK
from common.args import IngestArgs
from stages.build.ingestion import cache
from stages.build.ingestion.langchain import langchain_ingestion
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def convert_document(library_document) -> Document:
	"""
	Input: library_document, one doc_ingest Document returned by
	ingest_documents.
	Output: the same document as this project's stages.build.ingestion.schema.Document.
	What it does: copies id, type, text and source into a new Document of
	our own schema. The other metadata fields are left empty (None).
	"""
	metadata = Metadata(
		source=library_document.source,
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
	What it does: runs the built-in ingestion for framework "", or the
	LangChain ingestion for framework "langchain". Raises ValueError for
	any other framework.
	"""
	if ingest_args.framework == NO_FRAMEWORK:
		return built_in_ingestion(ingest_args)
	elif ingest_args.framework == LANGCHAIN_FRAMEWORK:
		return langchain_ingestion(ingest_args)
	else:
		raise ValueError(
			f"framework ({ingest_args.framework}) is not supported. "
			f"Use \"{LANGCHAIN_FRAMEWORK}\" or \"{NO_FRAMEWORK}\" for the built-in ingestion."
		)


def built_in_ingestion(ingest_args: IngestArgs) -> dict[str, Document]:
	"""
	Input: ingest_args, an IngestArgs holding the ingestion settings.
	Output: a dict of doc_id to this project's Document, one per ingested document.
	What it does: loads the first dataset_size documents from the dataset's
	processed data folder if enough are cached. Otherwise runs the doc_ingest
	library on ingest_args.input_path, prints the run summary, converts each
	library Document into our own Document, saves them and returns them.
	"""
	output_path = paths.get_output_path(ingest_args.dataset_name, ingest_args.output_path)
	print(f"Processed data folder : {paths.get_processed_data_path(output_path, NO_FRAMEWORK)}")

	cached_documents = cache.load_documents(output_path, NO_FRAMEWORK, ingest_args.dataset_size)
	if cached_documents is not None:
		print(f"Documents : {len(cached_documents)} (all from cache)")
		return cached_documents

	ingest_result = ingest_documents(
		path=ingest_args.input_path,
		size=ingest_args.dataset_size,
		document_type=ingest_args.document_type,
	)

	library_documents = ingest_result.documents
	summary = ingest_result.summary

	print(summary)

	documents = {}
	for library_document in library_documents:
		document = convert_document(library_document)
		documents[document.id] = document

	cache.save_documents(output_path, NO_FRAMEWORK, documents)
	print(f"Documents : {len(documents)} (all newly ingested and saved)")

	return documents
