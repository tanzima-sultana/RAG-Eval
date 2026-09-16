"""
Entry point for the doc_processing layer.
"""

import base64
import dataclasses
import datetime
import itertools
import json
import os

from datasets import load_dataset

from common.dataset_config import DatasetConfig, get_dataset_config
from common.identifiers import make_doc_id, make_shingles
from common.paths import dataset_raw_path, dataset_processed_path, get_file_path
from l1_doc_processing.dedup import is_exact_duplicate, is_near_duplicate, make_lsh_index, make_minhash
from l1_doc_processing.parser import get_parser
from l1_doc_processing.schema import Document


def load_cache(raw_path: str) -> dict[str, bytes]:
    """
    raw_path: directory holding raw cache files already written for
        this dataset.

    Reads every file already in raw_path into memory, keyed by the
    doc_id encoded in its filename. Files are read in sorted filename
    order, so the result is deterministic across runs regardless of
    os.listdir's own (filesystem-dependent, unordered) order — the same
    ordering guarantee the local dataset branch already relies on.

    Returns a dict of doc_id to raw_bytes for every raw file in
    raw_path.
    """
    cached_raw_bytes = {}
    for file_name in sorted(os.listdir(raw_path)):
        doc_id, _ = os.path.splitext(file_name)
        file_path = os.path.join(raw_path, file_name)
        with open(file_path, "rb") as f:
            cached_raw_bytes[doc_id] = f.read()
    return cached_raw_bytes


def decode_raw_bytes(row: dict, dataset_config: DatasetConfig) -> bytes:
    """
    row: one dataset row.
    dataset_config: tells us which row field holds the row's raw
        content and how that content is encoded.

    Reads row's raw content field and decodes it into bytes per
    dataset_config's content_encoding ("base64", "bytes", or "text").

    Raises ValueError if content_encoding is none of those.

    Returns the row's raw content as bytes.
    """
    raw_value = row[dataset_config.content_field]

    if dataset_config.content_encoding == "base64":
        raw_bytes = base64.b64decode(raw_value)
    elif dataset_config.content_encoding == "bytes":
        raw_bytes = raw_value
    elif dataset_config.content_encoding == "text":
        raw_bytes = raw_value.encode("utf-8")
    else:
        raise ValueError(
            f"Unrecognized content_encoding '{dataset_config.content_encoding}'."
        )

    return raw_bytes


def compute_raw_data(
    dataset_url: str,
    dataset_path: str,
    dataset_size: int,
    raw_path: str,
    dataset_config: DatasetConfig,
    doc_type: str,
) -> tuple[dict[str, bytes], int]:
    """
    dataset_url: web dataset identifier to stream rows from, or empty
        if this dataset is local.
    dataset_path: local directory to read files from, or empty if this
        dataset is a web dataset.
    dataset_size: number of documents wanted in total.
    raw_path: directory to read cached raw files from and write newly
        fetched raw files into.
    dataset_config: this dataset's registered DatasetConfig (looked up
        by dataset_url for a web dataset, or dataset_path for a local
        one). For a web dataset, tells us which row field holds a row's
        raw content, how that content is encoded, and which split to
        stream rows from. In both cases, dataset_config.extension is
        the cache file's extension.
    doc_type: the document format being processed, combined with each
        document's raw_bytes to compute its doc_id.

    Loads cached raw_bytes from raw_path via load_cache. If there are
    already at least dataset_size of them, returns the first
    dataset_size of them, with skipped_raw_error_count 0.

    Otherwise, for a web dataset, streams the remaining rows needed
    from dataset_url. Each row's decode + doc_id step is wrapped in its
    own try/except: a row that fails (missing content field, bad
    encoding, etc.) is logged with its file name and the error, skipped,
    and counted in skipped_raw_error_count, rather than aborting the
    whole batch. A row that succeeds is written to disk as
    {doc_id}.{extension}. For a local dataset, reads the remaining files
    needed directly from dataset_path; each file's read + doc_id step is
    wrapped the same way, so an unreadable/unexpectedly-missing file is
    logged and skipped rather than aborting the batch, and a file that
    succeeds is copied into raw_path as {doc_id}.{extension}. Either
    way, combines the cached raw_bytes with the newly fetched raw_bytes.

    Returns (documents_raw_bytes, skipped_raw_error_count): a dict of
    doc_id to raw_bytes, and how many rows were skipped due to a raw
    ingestion error.
    """
    os.makedirs(raw_path, exist_ok=True)

    # ------- cache
    cached_raw_bytes = load_cache(raw_path)
    # If there is 10k cached and asked 5k, just return only the dataset_size
    if len(cached_raw_bytes) >= dataset_size:
        first_doc_ids = list(cached_raw_bytes.keys())[:dataset_size]
        first_raw_bytes = {}
        for doc_id in first_doc_ids:
            first_raw_bytes[doc_id] = cached_raw_bytes[doc_id]
        return first_raw_bytes, 0

    # ------- new
    extension = dataset_config.extension
    new_raw_bytes = {}
    skipped_raw_error_count = 0
    if dataset_url:
        # Stream and cache the rest
        dataset = load_dataset(dataset_url, split=dataset_config.split, streaming=True)
        new_rows_stream = itertools.islice(dataset, len(cached_raw_bytes), dataset_size)

        for row in new_rows_stream:

            # raw
            try:
                raw_bytes = decode_raw_bytes(row, dataset_config)
                doc_id = make_doc_id(doc_type, raw_bytes)
            except Exception as error:
                skipped_raw_error_count += 1
                row_file_name = row.get(dataset_config.file_name_field)
                print(
                    f"Skipping row (file_name={row_file_name!r}): "
                    f"{type(error).__name__}: {error}"
                )
                continue

            # cache
            cache_file_path = get_file_path(raw_path, doc_id, extension) #os.path.join(raw_path, f"{doc_id}.{extension}")
            with open(cache_file_path, "wb") as f:
                f.write(raw_bytes)

            # dict
            new_raw_bytes[doc_id] = raw_bytes
    else:
        # Copy the rest from the local dataset directory
        source_file_names = sorted(os.listdir(dataset_path))
        remaining_file_names = source_file_names[len(cached_raw_bytes):dataset_size]

        for source_file_name in remaining_file_names:

            # raw
            try:
                source_file_path = os.path.join(dataset_path, source_file_name)
                with open(source_file_path, "rb") as f:
                    raw_bytes = f.read()
                doc_id = make_doc_id(doc_type, raw_bytes)
            except Exception as error:
                skipped_raw_error_count += 1
                print(
                    f"Skipping file (file_name={source_file_name!r}): "
                    f"{type(error).__name__}: {error}"
                )
                continue

            # cache
            cache_file_path = get_file_path(raw_path, doc_id, extension) #os.path.join(raw_path, f"{doc_id}.{extension}")
            with open(cache_file_path, "wb") as f:
                f.write(raw_bytes)

            # dict
            new_raw_bytes[doc_id] = raw_bytes

    # Combine cached and newly-copied raw_bytes
    documents_raw_bytes = dict(cached_raw_bytes)
    for doc_id in new_raw_bytes:
        documents_raw_bytes[doc_id] = new_raw_bytes[doc_id]
    return documents_raw_bytes, skipped_raw_error_count


def document_to_json_dict(document: Document) -> dict:
    """
    document: a fully-populated Document.

    Converts document to a plain dict of JSON-serializable values,
    explicitly turning the timestamp field into an ISO-8601 string.

    Returns the JSON-serializable dict.
    """
    data = dataclasses.asdict(document)
    data["timestamp"] = document.timestamp.isoformat()
    return data


def json_dict_to_document(data: dict) -> Document:
    """
    data: a dict previously produced by document_to_json_dict, with
        timestamp stored as an ISO-8601 string.

    Reverses document_to_json_dict by parsing the timestamp string back
    into a datetime.datetime and the shingles list back into a set.

    Returns the reconstructed Document.
    """
    data = dict(data)
    data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
    return Document(**data)


def load_processed_document(processed_path: str, doc_id: str) -> Document | None:
    """
    processed_path: directory holding processed Document JSON files
        already written for this dataset.
    doc_id: identifies which processed Document to load.

    Checks whether doc_id has already been processed and cached under
    processed_path.

    Returns the cached Document if present, otherwise None.
    """
    file_path = get_file_path(processed_path, doc_id, "json") #os.path.join(processed_path, f"{doc_id}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, encoding="utf-8") as f:
        return json_dict_to_document(json.load(f))


def save_processed_document(processed_path: str, doc_id: str, document: Document) -> None:
    """
    processed_path: directory to write the processed Document JSON file
        into.
    doc_id: identifies the file to write.
    document: the Document to persist.

    Writes document to processed_path as {doc_id}.json.

    Returns nothing.
    """
    file_path = get_file_path(processed_path, doc_id, "json") #os.path.join(processed_path, f"{doc_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(document_to_json_dict(document), f)


def process_raw_data(
    documents_raw_bytes: dict[str, bytes],
    processed_path: str,
    doc_type: str,
    raw_path: str,
    dataset_config: DatasetConfig,
) -> dict[str, Document]:
    """
    documents_raw_bytes: dict of doc_id to raw_bytes, as returned by
        compute_raw_data.
    processed_path: directory to read cached processed Documents from
        and write processed Documents into.
    doc_type: the document format being processed, used to pick a
        Parser from the registry.
    raw_path: raw_path of the dataset.
    dataset_config: this dataset's registered DatasetConfig

    For each doc_id in documents_raw_bytes, first checks whether a
    processed Document already exists in processed_path for that
    doc_id; if so, loads and reuses it, skipping parsing entirely.
    Otherwise, parses raw_bytes into a new Document via the doc_type
    parser. A doc_id that fails to parse for any reason is skipped and
    counted, rather than stopping the run.

    Every Document that enters the result dict — cache-loaded or freshly
    parsed — has its content_hash recorded in content_hash_dict (keyed
    by doc_id) and, unless its shingle set is empty, its MinHash
    signature inserted into lsh_index, so later documents in this same
    run can be compared against everything kept so far. A document with
    an empty shingle set (too short to shingle) is never inserted into
    lsh_index and never queried against it either, since an empty
    shingle set carries no content to meaningfully compare. A
    freshly-parsed Document is checked, via dedup.py, against
    content_hash_dict and lsh_index: an exact-duplicate content_hash or
    a near-duplicate MinHash match (see l1_doc_processing.dedup) causes
    it to be skipped and counted separately from parse failures — not
    saved to processed_path, not added to the result dict, not recorded
    in content_hash_dict or lsh_index. A Document that passes both
    checks is saved to processed_path immediately.

    Returns a dict of doc_id to Document.
    """
    os.makedirs(processed_path, exist_ok=True)

    parser_cls = get_parser(doc_type)
    parser = parser_cls()

    documents = {}
    content_hash_dict = {}
    lsh_index = make_lsh_index(threshold=0.90)
    skipped_count = 0
    skipped_duplicate_count = 0

    extension = dataset_config.extension

    for doc_id in documents_raw_bytes:

        processed_document = load_processed_document(processed_path, doc_id)
        if processed_document is not None:
            documents[doc_id] = processed_document
            content_hash_dict[doc_id] = processed_document.content_hash
            cached_shingles = make_shingles(processed_document.text)
            if len(cached_shingles) > 0:
                cached_minhash = make_minhash(cached_shingles)
                lsh_index.insert(doc_id, cached_minhash)
            continue

        try:
            # Actual process
            data_path = get_file_path(raw_path, doc_id, extension)
            raw_bytes = documents_raw_bytes[doc_id]
            doc = parser.parse(doc_id, raw_bytes, data_path)
        except Exception:
            skipped_count += 1
            continue

        # Dedup check
        doc_shingles = make_shingles(doc.text)
        if len(doc_shingles) > 0:
            doc_minhash = make_minhash(doc_shingles)
        else:
            doc_minhash = None

        is_exact = is_exact_duplicate(doc.content_hash, content_hash_dict)
        if is_exact:
            is_near = False
        elif doc_minhash is not None:
            is_near = is_near_duplicate(doc_minhash, lsh_index)
        else:
            is_near = False

        if is_exact or is_near:
            skipped_duplicate_count += 1
            continue

        documents[doc_id] = doc
        content_hash_dict[doc_id] = doc.content_hash
        if doc_minhash is not None:
            lsh_index.insert(doc_id, doc_minhash)

        # Save
        save_processed_document(processed_path, doc_id, doc)

    print(f"Skipped {skipped_count} document(s) due to parsing errors.")
    print(f"Skipped {skipped_duplicate_count} document(s) due to duplicate detection.")

    return documents


def doc_processing(
    doc_type: str,
    dataset_name: str,
    dataset_size: int,
    dataset_url: str,
    dataset_path: str,
) -> dict[str, Document]:
    """
    doc_type: the document format being processed, e.g. "pdf".
    dataset_name: label for this dataset, used to build its cache
        directories.
    dataset_size: number of documents to load.
    dataset_url: web dataset identifier to load documents from.
    dataset_path: local dataset directory.

    Computes the raw and processed cache directories and looks up this
    dataset's registered DatasetConfig, by dataset_url for a web
    dataset or dataset_path for a local one. Validates that doc_type
    matches dataset_config.doc_type, since doc_type determines both the
    doc_id prefix and which parser runs — a mismatch here would produce
    disagreeing artifacts silently if left unchecked. Then loads
    dataset_size documents' raw bytes via compute_raw_data, and parses
    them into Documents via process_raw_data.

    Raises ValueError if doc_type does not match dataset_config.doc_type.

    Returns a dict of doc_id to Document.
    """
    raw_path = dataset_raw_path(doc_type, dataset_name)
    processed_path = dataset_processed_path(doc_type, dataset_name)

    # Match doc type
    dataset_config = get_dataset_config(dataset_url or dataset_path)
    if doc_type != dataset_config.doc_type:
        raise ValueError(
            f"doc_type '{doc_type}' does not match the registered "
            f"doc_type '{dataset_config.doc_type}' for dataset "
            f"'{dataset_url or dataset_path}'."
        )

    # Raw
    documents_raw_bytes, skipped_raw_error_count = compute_raw_data(
        dataset_url, dataset_path, dataset_size, raw_path, dataset_config, doc_type
    )

    print(f"Skipped {skipped_raw_error_count} document(s) due to raw ingestion errors.")

    # Process
    documents = process_raw_data(
        documents_raw_bytes, processed_path, doc_type, raw_path, dataset_config
    )

    return documents
