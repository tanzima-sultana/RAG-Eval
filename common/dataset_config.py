"""
dataset_config.py

Declares, per dataset (identified by its dataset_url for a web dataset
or its dataset_path for a local one), the format-specific details the
pipeline needs to read that dataset's rows/files: which key holds the
document's raw content, how that content is encoded/shaped, which key
holds the file name, which doc_type the dataset is meant to be
processed as, which extension its raw cache files are written with,
and which split of a web dataset to stream rows from. Centralizing
this here keeps per-dataset field-name knowledge out of the generic
l1_doc_processing pipeline code, and confines it in parsers (and the
raw-content hashing done before parsing, see
l1_doc_processing.run_doc_processing) to just looking these values up
rather than hardcoding a dataset's field names.

content_encoding tells a reader how to turn row[content_field] into
raw bytes for hashing/parsing:
  - "base64": row[content_field] is a base64-encoded string; decode it
    to get the raw bytes (e.g. a PDF's bytes embedded as text).
  - "bytes": row[content_field] is already a raw bytes value; use it
    as-is.
  - "text": row[content_field] is already plain text (not encoded
    document bytes); UTF-8 encode it to get raw bytes.
A dataset with no single raw-content field (a "structured" row, whose
content is spread across several fields rather than one) leaves
content_field empty; callers fall back to hashing/reading the row as a
whole in that case.

extension is kept separate from doc_type because they are not always
the same string: doc_type picks which Parser runs (e.g. a scanned_pdf
dataset is still parsed as a PDF), while extension is only the suffix
its raw cache files are written with (e.g. scanned_pdf's files are
still cached as {doc_id}.pdf).
"""

import dataclasses


@dataclasses.dataclass
class DatasetConfig:
    doc_type: str
    extension: str
    content_field: str
    content_encoding: str
    file_name_field: str
    split: str


_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "dvgodoy/CUAD_v1_Contract_Understanding_PDF": DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="pdf_bytes_base64",
        content_encoding="base64",
        file_name_field="file_name",
        split="train",
    ),
    "pixparse/pdfa-eng-wds": DatasetConfig(
        doc_type="pdf",
        extension="pdf",
        content_field="pdf",
        content_encoding="bytes",       
        file_name_field="__key__",
        split="train",
    ),
}


def get_dataset_config(dataset_identifier: str) -> DatasetConfig:
    """
    dataset_identifier: whichever of dataset_url or dataset_path
        identifies this dataset (the same string the CLI's
        --dataset_url or --dataset_path argument carries).

    Looks up dataset_identifier's registered configuration, so the
    pipeline and its parsers can read a dataset's rows/files without
    hardcoding that dataset's field names anywhere outside this file.
    Both web datasets (keyed by dataset_url) and local datasets (keyed
    by dataset_path) are registered in the same _DATASET_CONFIGS dict.

    Returns the DatasetConfig registered for dataset_identifier. Raises
    ValueError if dataset_identifier has no registered configuration.
    """
    if dataset_identifier not in _DATASET_CONFIGS:
        raise ValueError(
            f"Dataset identifier '{dataset_identifier}' is not "
            f"registered in common/dataset_config.py. Registered "
            f"identifiers: {list(_DATASET_CONFIGS.keys())}"
        )
    return _DATASET_CONFIGS[dataset_identifier]
