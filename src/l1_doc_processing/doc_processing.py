import dataclasses
import json
import os
import datetime
import base64
from typing import Iterator

from datasets import load_dataset

from common import paths
from common.dataset_config import get_dataset_config
from common import validate
from common import identifiers
from l1_doc_processing.dedup import is_exact_duplicate, is_near_duplicate, make_lsh_index, make_minhash
from l1_doc_processing.parser import get_parser
from l1_doc_processing.schema import Document, Progress


class DocProcessing:
	def __init__(self, doc_type: str, dataset_name: str, dataset_size: int, dataset_url: str, dataset_path: str):
		"""
		doc_type: document type to process (e.g. "pdf").
		dataset_name: name of the dataset.
		dataset_size: number of processed documents wanted at the end.
		dataset_url: Hugging Face dataset id, for a web dataset.
		dataset_path: local directory, for a local dataset.

		Stores the given arguments on the instance, and looks up the
		dataset's registered DatasetConfig by dataset_name. Raises
		ValueError if dataset_name is not registered.

		Returns nothing.
		"""
		self.doc_type = doc_type
		self.dataset_name = dataset_name
		self.dataset_size = dataset_size
		self.dataset_url = dataset_url
		self.dataset_path = dataset_path

		self.dataset_config = get_dataset_config(self.dataset_name)

	def load_progress(self) -> Progress:
		"""
		Reads {dataset_name}/data/progress.json.

		Returns the saved Progress, or, on a first run, an empty Progress
		(rows_consumed 0, empty lists) for the current dataset.
		"""
		progress_path = paths.get_progress_path(self.dataset_name)
		file_path = paths.get_file_path(progress_path, "progress", "json")

		if not os.path.exists(file_path):
			
			empty_progress = Progress(
				dataset_identifier=self.dataset_config.identifier,
				split=self.dataset_config.split,
				doc_type=self.doc_type,
				rows_consumed=0,
				processed_doc_ids=[],
				skipped_doc_ids=[],
			)
			return empty_progress

		with open(file_path, encoding="utf-8") as f:
			data = json.load(f)

		progress = Progress(
			dataset_identifier=data["dataset_identifier"],
			split=data["split"],
			doc_type=data["doc_type"],
			rows_consumed=data["rows_consumed"],
			processed_doc_ids=data["processed_doc_ids"],
			skipped_doc_ids=data["skipped_doc_ids"]
		)
		return progress

	def save_progress(self, progress: Progress) -> None:
		"""
		progress: the Progress to save.

		Writes progress to {dataset_name}/data/progress.json through a
		temporary file, so a crash never leaves a half-written file.

		Returns nothing.
		"""
		progress_path = paths.get_progress_path(self.dataset_name)
		file_path = paths.get_file_path(progress_path, "progress", "json")
		temporary_file_path = file_path + ".tmp"

		os.makedirs(progress_path, exist_ok=True)

		with open(temporary_file_path, "w", encoding="utf-8") as f:
			json.dump(dataclasses.asdict(progress), f)

		os.replace(temporary_file_path, file_path)

	def json_dict_to_document(self, data: dict) -> Document:
		"""
		data: a dict produced by document_to_json_dict.

		Returns the Document, with timestamp parsed back into a datetime.
		"""
		data = dict(data)
		data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
		return Document(**data)

	def document_to_json_dict(self, document: Document) -> dict:
		"""
		document: the Document to convert.

		Returns a JSON-serializable dict, with timestamp as an ISO-8601 string.
		"""
		data = dataclasses.asdict(document)
		data["timestamp"] = document.timestamp.isoformat()
		return data
	
	def load_processed_document(self, doc_id: str) -> Document | None:
		"""
		doc_id: the processed document to load.

		Returns the Document from {dataset_name}/data/processed/{doc_id}.json,
		or None if the file does not exist.
		"""
		processed_path = paths.get_processed_dataset_path(self.dataset_name)
		file_path = paths.get_file_path(processed_path, doc_id, "json")
		if not os.path.exists(file_path):
			return None
		with open(file_path, encoding="utf-8") as f:
			return self.json_dict_to_document(json.load(f))

	def save_processed_document(self, doc_id: str, document: Document) -> None:
		"""
		doc_id: the document's id, used as the file name.
		document: the Document to save.

		Writes document to {dataset_name}/data/processed/{doc_id}.json.

		Returns nothing.
		"""
		processed_path = paths.get_processed_dataset_path(self.dataset_name)
		file_path = paths.get_file_path(processed_path, doc_id, "json")

		os.makedirs(processed_path, exist_ok=True)

		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(self.document_to_json_dict(document), f)

	def get_iterator(self, rows_consumed:int)-> Iterator:
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

	def save_raw_bytes(self, doc_id:str, raw_bytes) -> str:
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

	def decode_raw_bytes(self, row: dict) -> bytes:
		"""
		row: a Hugging Face row (dataset_url) or a file name (dataset_path).

		Returns the row's raw content as bytes: decoded per the config's
		content_encoding for a web dataset, or the file's bytes for a
		local one. Raises ValueError for an unknown content_encoding.
		"""

		if self.dataset_url:
			raw_value = row[self.dataset_config.content_field]

			if self.dataset_config.content_encoding == "base64":
				raw_bytes = base64.b64decode(raw_value)
			elif self.dataset_config.content_encoding == "bytes":
				raw_bytes = raw_value
			elif self.dataset_config.content_encoding == "text":
				raw_bytes = raw_value.encode("utf-8")
			else:
				raise ValueError(
					f"Unrecognized content_encoding '{self.dataset_config.content_encoding}'."
				)
		else:
			source_file_path = os.path.join(self.dataset_path, row)
			with open(source_file_path, "rb") as f:
				raw_bytes = f.read()
		
		return raw_bytes

	def process(self) -> dict[str, Document]:
		"""
		Validates the dataset and saved progress, loads already processed
		documents, then streams, decodes, parses and dedups new rows until
		dataset_size documents are processed or the source runs out.
		Progress is saved before returning.

		Returns a dict of doc_id to Document, in processing order. It has
		fewer than dataset_size entries only if the source ran out.
		"""

		documents = {}

		# 1. Validate dataset
		validate.validate_dataset(
			self.doc_type,
			self.dataset_name,
			self.dataset_size,
			self.dataset_url,
			self.dataset_path,
		)

		# 2. Validate data download and process progress using progress.json
		progress = self.load_progress()
		validate.validate_progress(progress, self.dataset_config, self.doc_type)

		# 3. Check cached processed documents
		seen_content_hashes = set()
		lsh_index = make_lsh_index(threshold=0.90)

		# processed_doc_ids stays a list so the processing order is kept
		processed_doc_ids = []
		skipped_doc_ids = set(progress.skipped_doc_ids)
		rows_consumed = progress.rows_consumed

		cached_index = 0
		while cached_index < len(progress.processed_doc_ids):
			doc_id = progress.processed_doc_ids[cached_index]
			cached_index += 1

			cached_doc = self.load_processed_document(doc_id)
			# If a processed doc is physically removed somehow
			if cached_doc is None:
				print(f"Processed document {doc_id} is missing on disk, marking it as skipped.")
				skipped_doc_ids.add(doc_id)
				continue

			processed_doc_ids.append(doc_id)
			documents[doc_id] = cached_doc

			seen_content_hashes.add(cached_doc.content_hash)
			cached_shingles = identifiers.make_shingles(cached_doc.text)
			if len(cached_shingles) > 0:
				cached_minhash = make_minhash(cached_shingles)
				lsh_index.insert(doc_id, cached_minhash)

			# if there is dataset_size cached documents already exists
			if len(documents) == self.dataset_size:
				# Keep the not-yet-checked cached ids in progress too
				unchecked_doc_ids = progress.processed_doc_ids[cached_index:]
				all_processed_doc_ids = processed_doc_ids + unchecked_doc_ids
				self.save_progress(self.make_progress(rows_consumed, all_processed_doc_ids, skipped_doc_ids))
				print(f"Loaded {len(documents)} already processed document(s). Nothing new to process.")
				return documents

		# 4. Stream & process raw byte data
		row_iterator = self.get_iterator(rows_consumed)

		parser_cls = get_parser(self.doc_type)
		parser = parser_cls()

		cached_count = len(documents)
		decode_error_count = 0
		previously_skipped_count = 0
		parse_error_count = 0
		duplicate_count = 0

		while True:
			# Stream row
			try:
				row = next(row_iterator)
			except StopIteration:
				# At the end of raw data
				self.save_progress(self.make_progress(rows_consumed, processed_doc_ids, skipped_doc_ids))
				self.print_run_summary(cached_count, len(documents), decode_error_count, previously_skipped_count, parse_error_count, duplicate_count)
				print("Reached the end of the dataset.")
				return documents

			# Convert row to raw_bytes
			try:
				raw_bytes = self.decode_raw_bytes(row)
				rows_consumed += 1
			except Exception as error:
				print(f"Skipping row {rows_consumed}: {type(error).__name__}: {error}")
				rows_consumed += 1
				decode_error_count += 1
				continue

			# Save raw_bytes, just for reference
			doc_id = identifiers.make_doc_id(self.doc_type, raw_bytes)
			source_path = self.save_raw_bytes(doc_id, raw_bytes)

			if doc_id in skipped_doc_ids:
				previously_skipped_count += 1
				continue

			# Process raw_bytes
			try:
				# Actual process
				processed_doc = parser.parse(doc_id, raw_bytes, source_path)
			except Exception as error:
				print(f"Skipping document {doc_id}: parse failed: {type(error).__name__}: {error}")
				skipped_doc_ids.add(doc_id)
				parse_error_count += 1
				continue

			# Dedup check
			doc_shingles = identifiers.make_shingles(processed_doc.text)
			if len(doc_shingles) > 0:
				doc_minhash = make_minhash(doc_shingles)
			else:
				doc_minhash = None

			is_exact = is_exact_duplicate(processed_doc.content_hash, seen_content_hashes)
			if is_exact:
				is_near = False
			elif doc_minhash is not None:
				is_near = is_near_duplicate(doc_minhash, lsh_index)
			else:
				is_near = False

			if is_exact or is_near:
				duplicate_count += 1
				continue

			# Add the processed_doc in documents
			processed_doc_ids.append(doc_id)

			documents[doc_id] = processed_doc
			seen_content_hashes.add(processed_doc.content_hash)
			if doc_minhash is not None:
				lsh_index.insert(doc_id, doc_minhash)

			# Save processed document
			self.save_processed_document(doc_id, processed_doc)

			# if dataset_size already reached
			if len(documents) == self.dataset_size:
				self.save_progress(self.make_progress(rows_consumed, processed_doc_ids, skipped_doc_ids))
				self.print_run_summary(cached_count, len(documents), decode_error_count, previously_skipped_count, parse_error_count, duplicate_count)
				return documents

	def make_progress(self, rows_consumed: int, processed_doc_ids: list[str], skipped_doc_ids: set[str]) -> Progress:
		"""
		rows_consumed: source rows read so far.
		processed_doc_ids: processed doc_ids, in processing order.
		skipped_doc_ids: doc_ids that failed parsing or went missing.

		Returns a Progress for this dataset with the given values.
		"""
		progress = Progress(
			dataset_identifier=self.dataset_config.identifier,
			split=self.dataset_config.split,
			doc_type=self.doc_type,
			rows_consumed=rows_consumed,
			processed_doc_ids=processed_doc_ids,
			skipped_doc_ids=list(skipped_doc_ids),
		)
		return progress

	def print_run_summary(self, cached_count: int, total_count: int, decode_error_count: int, previously_skipped_count: int, parse_error_count: int, duplicate_count: int) -> None:
		"""
		Takes the counts collected in process().

		Prints how many documents were loaded from cache, newly
		processed, and skipped by reason.

		Returns nothing.
		"""
		print(f"Loaded {cached_count} already processed document(s).")
		print(f"Newly processed {total_count - cached_count} document(s). Total: {total_count}.")
		print(f"Skipped {decode_error_count} row(s) due to decode errors.")
		print(f"Skipped {previously_skipped_count} document(s) that failed in an earlier run.")
		print(f"Skipped {parse_error_count} document(s) due to parse errors.")
		print(f"Skipped {duplicate_count} document(s) due to duplicate detection.")
