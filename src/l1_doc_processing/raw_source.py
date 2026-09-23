"""
raw_source.py

Reads a dataset's source rows and turns them into raw files under
{dataset_name}/data/raw/. Handles both web datasets (Hugging Face rows,
streamed) and local datasets (files in a folder), the row_filter, and
decoding or downloading each row's content. Rows are decoded/downloaded
in parallel, in batches.
"""

import base64
import concurrent.futures
import dataclasses
import os
import urllib.request
from typing import Iterator

from datasets import load_dataset

from common import identifiers
from common import paths
from common.constants import DOWNLOAD_WORKERS
from common.dataset_config import DatasetConfig
from l1_doc_processing.summary import RunCounts


@dataclasses.dataclass
class RawBatch:
	"""
	Result of RawSource.fetch_raw_batch.

	raw_items: (doc_id, raw file path) of every row decoded and saved
		to raw/, in source row order.
	rows_read: source rows read for this batch, including filtered
		rows and rows that failed to decode.
	source_exhausted: True if the source ran out of rows.
	"""
	raw_items: list[tuple[str, str]]
	rows_read: int
	source_exhausted: bool


class RawSource:
	def __init__(self, dataset_name: str, dataset_config: DatasetConfig, doc_type: str, dataset_url: str, dataset_path: str):
		"""
		dataset_name: name of the dataset; raw files go to
			{dataset_name}/data/raw/.
		dataset_config: registered DatasetConfig of the dataset.
		doc_type: document type, used as the doc_id prefix.
		dataset_url: Hugging Face dataset id, for a web dataset.
		dataset_path: local directory, for a local dataset.

		Returns nothing.
		"""
		self.dataset_name = dataset_name
		self.dataset_config = dataset_config
		self.doc_type = doc_type
		self.dataset_url = dataset_url
		self.dataset_path = dataset_path

	def get_iterator(self, rows_consumed: int) -> Iterator:
		"""
		rows_consumed: number of source rows already read in earlier runs.

		Returns an iterator over the source rows, starting after the first
		rows_consumed rows: Hugging Face rows (streamed) for dataset_url,
		or sorted file names for dataset_path.
		"""
		if self.dataset_url:
			dataset = load_dataset(self.dataset_url, split=self.dataset_config.split, streaming=True)
			dataset = dataset.skip(rows_consumed)
			row_iterator = iter(dataset)
		else:
			source_file_names = sorted(os.listdir(self.dataset_path))[rows_consumed:]
			row_iterator = iter(source_file_names)
		return row_iterator

	def row_matches_filter(self, row) -> bool:
		"""
		row: a Hugging Face row (dataset_url) or a file name (dataset_path).

		Returns True if the config has no row_filter, or if every
		{column: value} pair in row_filter matches the row. The filter
		only applies to web dataset rows; local files always match.
		"""
		row_filter = self.dataset_config.row_filter
		if row_filter is None:
			return True
		if not self.dataset_url:
			return True

		for column_name in row_filter:
			if row.get(column_name) != row_filter[column_name]:
				return False
		return True

	def download_url_bytes(self, url: str) -> bytes:
		"""
		url: direct download link of one document.

		Raises urllib.error.URLError (or a subclass) if the download
		fails or takes longer than 60 seconds.

		Returns the downloaded bytes.
		"""
		request = urllib.request.Request(url, headers={"User-Agent": "RAG-Eval/1.0"})
		with urllib.request.urlopen(request, timeout=60) as response:
			return response.read()

	def decode_raw_bytes(self, row) -> bytes | None:
		"""
		row: a Hugging Face row (dataset_url) or a file name (dataset_path).

		Decodes (or downloads, for "url") the row's content per the
		config's content_encoding for a web dataset, or reads the file for
		a local one. Safe to call from several threads at once.

		Returns the raw bytes, or None (after printing why) if decoding,
		downloading or reading failed, including an unknown
		content_encoding.
		"""
		try:
			if self.dataset_url:
				raw_value = row[self.dataset_config.content_field]

				if self.dataset_config.content_encoding == "base64":
					raw_bytes = base64.b64decode(raw_value)
				elif self.dataset_config.content_encoding == "bytes":
					raw_bytes = raw_value
				elif self.dataset_config.content_encoding == "text":
					raw_bytes = raw_value.encode("utf-8")
				elif self.dataset_config.content_encoding == "url":
					raw_bytes = self.download_url_bytes(raw_value)
				else:
					raise ValueError(
						f"Unrecognized content_encoding '{self.dataset_config.content_encoding}'."
					)
			else:
				source_file_path = os.path.join(self.dataset_path, row)
				with open(source_file_path, "rb") as f:
					raw_bytes = f.read()
		except Exception as error:
			print(f"Skipping row: decode/download failed: {type(error).__name__}: {error}")
			return None
		
		return raw_bytes

	def save_raw_bytes(self, doc_id: str, raw_bytes: bytes) -> str:
		"""
		doc_id: the document's id, used as the file name.
		raw_bytes: the document's raw content.

		Writes raw_bytes to {dataset_name}/data/raw/{doc_id}.{extension}.

		Returns the written file path.
		"""
		raw_path = paths.get_raw_dataset_path(self.dataset_name)
		file_path = paths.get_file_path(raw_path, doc_id, self.dataset_config.extension)

		os.makedirs(raw_path, exist_ok=True)

		with open(file_path, "wb") as f:
			f.write(raw_bytes)
		return file_path

	def fetch_raw_batch(self, row_iterator: Iterator, row_count: int, run_counts: RunCounts) -> RawBatch:
		"""
		row_iterator: the source rows, from get_iterator.
		row_count: number of rows matching the row_filter to fetch.
		run_counts: counts to add filtered rows and decode errors to.

		Reads rows until row_count of them match the row_filter or the
		source runs out. Decodes (or downloads) the matching rows in
		parallel with DOWNLOAD_WORKERS threads, then saves each one to
		raw/ in source row order.

		Returns a RawBatch.
		"""
		rows_to_decode = []
		rows_read = 0
		source_exhausted = False
		while len(rows_to_decode) < row_count:
			try:
				row = next(row_iterator)
			except StopIteration:
				source_exhausted = True
				break
			rows_read += 1

			if not self.row_matches_filter(row):
				run_counts.filtered += 1
				continue
			rows_to_decode.append(row)

		# executor.map returns results in the same order as rows_to_decode
		with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
			decoded_results = list(executor.map(self.decode_raw_bytes, rows_to_decode))

		raw_items = []
		for raw_bytes in decoded_results:
			if raw_bytes is None:
				run_counts.decode_error += 1
				continue
			doc_id = identifiers.make_doc_id(self.doc_type, raw_bytes)
			source_path = self.save_raw_bytes(doc_id, raw_bytes)
			raw_items.append((doc_id, source_path))

		return RawBatch(raw_items=raw_items, rows_read=rows_read, source_exhausted=source_exhausted)
