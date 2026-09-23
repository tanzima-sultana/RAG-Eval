"""
dataset_config.py

Registry of supported datasets, keyed by dataset_name (e.g. "cuad").
Each DatasetConfig says where the dataset lives (identifier: URL or
local path) and how to read its rows.

content_encoding: how to turn row[content_field] into raw bytes.
  - "base64": base64-decode it.
  - "bytes": use it as-is.
  - "text": UTF-8 encode it.
  - "url": it is a download link; download the file's bytes.

row_filter: optional {column: value} pairs a row must match to be
processed (e.g. {"language": "en"}). None processes every row.
"""

import dataclasses


@dataclasses.dataclass
class DatasetConfig:
	identifier: str
	doc_type: str
	extension: str
	content_field: str
	content_encoding: str
	file_name_field: str
	split: str
	row_filter: dict[str, str] | None = None


_DATASET_CONFIGS: dict[str, DatasetConfig] = {
	"cuad": DatasetConfig(
		identifier="dvgodoy/CUAD_v1_Contract_Understanding_PDF",
		doc_type="pdf",
		extension="pdf",
		content_field="pdf_bytes_base64",
		content_encoding="base64",
		file_name_field="file_name",
		split="train",
	),
	"pdfa-eng-wds": DatasetConfig(
		identifier="pixparse/pdfa-eng-wds",
		doc_type="pdf",
		extension="pdf",
		content_field="pdf",
		content_encoding="bytes",       
		file_name_field="__key__",
		split="train",
	),
	"docx-corpus": DatasetConfig(
		identifier="superdoc-dev/docx-corpus",
		doc_type="docx",
		extension="docx",
		content_field="url",
		content_encoding="url",
		file_name_field="filename",
		split="train",
		row_filter={"language": "en"},
	),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
	"""
	dataset_name: registered dataset name (e.g. "cuad"), exact case.

	Returns the DatasetConfig registered for dataset_name.
	Raises ValueError if dataset_name is not registered.
	"""
	if dataset_name not in _DATASET_CONFIGS:
		raise ValueError(
			f"Dataset '{dataset_name}' is not registered in common/dataset_config.py."
		)
	return _DATASET_CONFIGS[dataset_name]
