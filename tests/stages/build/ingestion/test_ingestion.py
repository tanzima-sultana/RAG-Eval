"""
test_ingestion.py

Tests for stages.build.ingestion.ingestion: convert_document, which copies a
doc_ingest Document into this project's Document, and ingestion, which
loads the documents from the processed data folder when enough are cached,
and otherwise calls doc_ingest.ingest_documents with the IngestArgs values,
saves the converted documents and returns them keyed by id.

ingest_documents is replaced with a fake in these tests, so no dataset is
downloaded. Only RAG-Eval's wiring to the library is tested.
"""

import types

from doc_ingest import schema as library_schema

from common.args import IngestArgs
from stages.build.ingestion import ingestion as ingestion_module
from stages.build.ingestion.ingestion import convert_document
from stages.build.ingestion.ingestion import ingestion
from stages.build.ingestion.schema import Document
from stages.build.ingestion.schema import DocumentType
from stages.build.ingestion.schema import Metadata


def make_library_document(doc_id="doc1", doc_type="pdf", text="Some text."):
    """
    Input: doc_id, doc_type (a DocumentType value string) and text.
    Output: a doc_ingest.schema.Document with every metadata field filled in.
    What it does: builds a library Document to feed into convert_document.
    """
    library_metadata = library_schema.Metadata(
        source="https://example.com/file.pdf",
        file_size_bytes=1234,
        content_hash="abc123",
        minhash_signature=[1, 2, 3],
        page_count=7,
        parser="pypdfium2",
        author="An Author",
        title="A Title",
        creation_date="2020-01-01",
    )
    library_document = library_schema.Document(
        id=doc_id,
        type=library_schema.DocumentType(doc_type),
        text=text,
        metadata=library_metadata,
    )
    return library_document


def make_ingest_args(output_path, dataset_size=5):
    """
    Input: output_path, the base output folder (a pytest tmp_path in tests);
    dataset_size, the number of documents to ask for.
    Output: an IngestArgs with a distinct value in every other field.
    What it does: builds IngestArgs so tests can check which values reach
    ingest_documents.
    """
    ingest_args = IngestArgs(
        dataset_size=dataset_size,
        input_path="owner/dataset",
        output_path=str(output_path),
        download_workers=3,
        max_batch_rows=11,
        hugging_face_split="test",
        parse_workers=4,
        parse_timeout_seconds=9,
        dedup_threshold=0.5,
        pdf_parser="pypdfium2",
    )
    return ingest_args


class FakeIngestDocuments:
    """
    Stand-in for doc_ingest.ingest_documents. Records the arguments it was
    called with and returns a result object holding the given library
    documents, a processed_dir and a summary.
    """

    def __init__(self, library_documents):
        """
        Input: library_documents, the list of library Documents to return.
        Output: nothing.
        What it does: stores the documents and starts with no recorded call.
        """
        self.library_documents = library_documents
        self.called_kwargs = None

    def __call__(self, **kwargs):
        """
        Input: the keyword arguments ingestion passes to ingest_documents.
        Output: a result object with documents, processed_dir and summary.
        What it does: records kwargs and returns the fake result.
        """
        self.called_kwargs = kwargs
        result = types.SimpleNamespace(
            documents=self.library_documents,
            processed_dir="some_output/processed",
            summary="FAKE SUMMARY",
        )
        return result


# ---------- convert_document ----------


def test_convert_document_returns_this_projects_schema_types():
    """
    Input: a library Document.
    Output: the converted Document.
    What it does: checks the result and its metadata and type use
    stages.build.ingestion.schema classes, not doc_ingest's.
    """
    library_document = make_library_document()

    document = convert_document(library_document)

    assert type(document) is Document
    assert type(document.metadata) is Metadata
    assert type(document.type) is DocumentType


def test_convert_document_copies_every_field():
    """
    Input: a library Document with every field filled in.
    Output: the converted Document.
    What it does: checks every field, including every metadata field,
    has the same value as in the library Document.
    """
    library_document = make_library_document()

    document = convert_document(library_document)

    assert document.model_dump() == library_document.model_dump()


def test_convert_document_keeps_optional_metadata_fields_as_none():
    """
    Input: a library Document whose optional metadata fields are None.
    Output: the converted Document.
    What it does: checks the None values are kept, not replaced.
    """
    library_metadata = library_schema.Metadata(
        source="file.pdf",
        file_size_bytes=0,
        content_hash="hash",
    )
    library_document = library_schema.Document(
        id="doc1",
        type=library_schema.DocumentType.PDF,
        text="text",
        metadata=library_metadata,
    )

    document = convert_document(library_document)

    assert document.metadata.minhash_signature is None
    assert document.metadata.page_count is None
    assert document.metadata.parser is None
    assert document.metadata.author is None
    assert document.metadata.title is None
    assert document.metadata.creation_date is None


def test_convert_document_maps_each_document_type():
    """
    Input: one library Document per DocumentType value.
    Output: each converted Document's type.
    What it does: checks every library type maps to our type with the
    same value.
    """
    for library_type in library_schema.DocumentType:
        library_document = make_library_document(doc_type=library_type.value)

        document = convert_document(library_document)

        assert document.type == DocumentType(library_type.value)


# ---------- ingestion ----------


def test_ingestion_passes_ingest_args_to_ingest_documents(monkeypatch, tmp_path):
    """
    Input: IngestArgs with distinct values and a fake ingest_documents.
    Output: the keyword arguments the fake was called with.
    What it does: checks size, input_path and the settings values ingestion
    passes come from IngestArgs, and that no cache_dir is passed, so the
    library never decides where this project's data is saved.
    """
    fake_ingest_documents = FakeIngestDocuments([])
    monkeypatch.setattr(ingestion_module, "ingest_documents", fake_ingest_documents)
    ingest_args = make_ingest_args(tmp_path)

    ingestion(ingest_args)

    called_kwargs = fake_ingest_documents.called_kwargs
    assert called_kwargs["size"] == 5
    assert called_kwargs["input_path"] == "owner/dataset"
    assert "cache_dir" not in called_kwargs
    assert called_kwargs["download_settings"].download_workers == 3
    assert called_kwargs["process_settings"].parse_workers == 4
    assert called_kwargs["parser_settings"].pdf_parser == "pypdfium2"


def test_ingestion_returns_converted_documents_keyed_by_id(monkeypatch, tmp_path):
    """
    Input: a fake ingest_documents returning two library Documents.
    Output: ingestion's return value.
    What it does: checks the result is a dict of doc_id to our Document,
    in the same order the library returned them.
    """
    library_documents = []
    library_documents.append(make_library_document(doc_id="first", text="First text."))
    library_documents.append(make_library_document(doc_id="second", text="Second text."))
    fake_ingest_documents = FakeIngestDocuments(library_documents)
    monkeypatch.setattr(ingestion_module, "ingest_documents", fake_ingest_documents)

    documents = ingestion(make_ingest_args(tmp_path, dataset_size=2))

    assert list(documents.keys()) == ["first", "second"]
    assert type(documents["first"]) is Document
    assert documents["first"].text == "First text."
    assert documents["second"].text == "Second text."


def test_ingestion_returns_empty_dict_when_no_documents(monkeypatch, tmp_path):
    """
    Input: a fake ingest_documents returning no documents.
    Output: ingestion's return value.
    What it does: checks an empty dict is returned.
    """
    fake_ingest_documents = FakeIngestDocuments([])
    monkeypatch.setattr(ingestion_module, "ingest_documents", fake_ingest_documents)

    documents = ingestion(make_ingest_args(tmp_path))

    assert documents == {}


def test_ingestion_prints_processed_dir_and_summary(monkeypatch, capsys, tmp_path):
    """
    Input: a fake ingest_documents with a known processed_dir and summary.
    Output: what ingestion printed.
    What it does: checks both values are printed.
    """
    fake_ingest_documents = FakeIngestDocuments([])
    monkeypatch.setattr(ingestion_module, "ingest_documents", fake_ingest_documents)

    ingestion(make_ingest_args(tmp_path))

    printed = capsys.readouterr().out
    assert "some_output/processed" in printed
    assert "FAKE SUMMARY" in printed


def test_ingestion_saves_documents_to_the_processed_data_folder(monkeypatch, tmp_path):
    """
    Input: a fake ingest_documents returning two library Documents.
    Output: the files in {output_path}/dataset/data/processed/.
    What it does: checks each document and doc_ids.json are saved under the
    dataset's own output folder.
    """
    library_documents = []
    library_documents.append(make_library_document(doc_id="first"))
    library_documents.append(make_library_document(doc_id="second"))
    fake_ingest_documents = FakeIngestDocuments(library_documents)
    monkeypatch.setattr(ingestion_module, "ingest_documents", fake_ingest_documents)

    ingestion(make_ingest_args(tmp_path, dataset_size=2))

    processed_data_path = tmp_path / "dataset" / "data" / "processed"
    assert (processed_data_path / "first.json").exists()
    assert (processed_data_path / "second.json").exists()
    assert (processed_data_path / "doc_ids.json").exists()


def test_ingestion_loads_from_cache_without_calling_the_library(monkeypatch, tmp_path):
    """
    Input: a first ingestion run that saves two documents, then a second
    run with a fake ingest_documents that would return nothing.
    Output: the second run's return value and whether the fake was called.
    What it does: checks the second run returns the saved documents, in
    order, and never calls the library.
    """
    library_documents = []
    library_documents.append(make_library_document(doc_id="first", text="First text."))
    library_documents.append(make_library_document(doc_id="second", text="Second text."))
    monkeypatch.setattr(ingestion_module, "ingest_documents", FakeIngestDocuments(library_documents))
    ingestion(make_ingest_args(tmp_path, dataset_size=2))

    second_fake_ingest_documents = FakeIngestDocuments([])
    monkeypatch.setattr(ingestion_module, "ingest_documents", second_fake_ingest_documents)
    documents = ingestion(make_ingest_args(tmp_path, dataset_size=2))

    assert second_fake_ingest_documents.called_kwargs is None
    assert list(documents.keys()) == ["first", "second"]
    assert documents["second"].text == "Second text."


def test_ingestion_calls_the_library_when_too_few_documents_are_cached(monkeypatch, tmp_path):
    """
    Input: a first run that saves one document, then a second run asking
    for two documents.
    Output: the second run's return value and the fake's recorded call.
    What it does: checks the library is called for the full dataset_size
    and that both documents are returned and saved.
    """
    monkeypatch.setattr(ingestion_module, "ingest_documents", FakeIngestDocuments([make_library_document(doc_id="first")]))
    ingestion(make_ingest_args(tmp_path, dataset_size=1))

    library_documents = []
    library_documents.append(make_library_document(doc_id="first"))
    library_documents.append(make_library_document(doc_id="second"))
    second_fake_ingest_documents = FakeIngestDocuments(library_documents)
    monkeypatch.setattr(ingestion_module, "ingest_documents", second_fake_ingest_documents)
    documents = ingestion(make_ingest_args(tmp_path, dataset_size=2))

    assert second_fake_ingest_documents.called_kwargs["size"] == 2
    assert list(documents.keys()) == ["first", "second"]
    assert (tmp_path / "dataset" / "data" / "processed" / "second.json").exists()
