
import base64
import dataclasses
import datetime
import itertools
import json
import os
import time


from datasets import load_dataset, load_dataset_builder

from common.paths import dataset_raw_path, dataset_processed_path, get_file_path
from common.dataset_config import DatasetConfig, get_dataset_config
from common.identifiers import make_doc_id, make_shingles
from common import timing
from common import global_values
from l1_doc_processing.dedup import is_exact_duplicate, is_near_duplicate, make_lsh_index, make_minhash
from l1_doc_processing.parser import get_parser
from l1_doc_processing.schema import Document

class DocProcessing:
	def __init__(self, doc_type: str, dataset_name: str, dataset_size: int, dataset_url: str, dataset_path: str):

		self.doc_type = doc_type 
		self.dataset_name = dataset_name 
		self.dataset_size = dataset_size
		self.dataset_url = dataset_url 
		self.dataset_path = dataset_path 
		self.dataset_config = get_dataset_config(self.dataset_url or self.dataset_path)
		
		# Update global values
		global_values.set("doc_type", self.doc_type)
		global_values.set("dataset_name", self.dataset_name)

		# Get paths
		self.raw_path = dataset_raw_path()
		self.processed_path = dataset_processed_path()
		

	def validate(self):
		if self.doc_type != self.dataset_config.doc_type:
			raise ValueError(
				f"doc_type '{self.doc_type}' does not match the registered "
				f"doc_type '{self.dataset_config.doc_type}' for dataset "
			)

		actual_dataset_size = self.get_actual_dataset_size()
		if actual_dataset_size is None:
			print(
				f"Warning: dataset '{self.dataset_url or self.dataset_path}' "
				f"does not publish a document count in its metadata, so "
				f"dataset_size {self.dataset_size} cannot be validated "
				f"upfront. Proceeding anyway."
			)
		elif actual_dataset_size < self.dataset_size:
			raise ValueError(
				f"Requested dataset_size {self.dataset_size} exceeds the "
				f"actual dataset size {actual_dataset_size} for dataset "
				f"'{self.dataset_url or self.dataset_path}'."
			)

	def get_actual_dataset_size(self) -> int | None:
		"""
		Determines how many documents this dataset actually has,
		without downloading or streaming any document content.

		For a web dataset (self.dataset_url set), reads the declared
		number of examples for dataset_config.split from the
		dataset's builder metadata. Some datasets (e.g. WebDataset-
		format ones like pixparse/pdfa-eng-wds) publish no split-size
		metadata at all, in which case the actual size cannot be
		determined without downloading the dataset; None is returned
		for this case rather than raising. For a local dataset
		(self.dataset_path set), counts the files directly in
		dataset_path.

		Returns the actual dataset size, or None if it cannot be
		determined from metadata alone.
		"""
		if self.dataset_url:
			builder = load_dataset_builder(self.dataset_url)
			if builder.info.splits is None:
				return None
			split_info = builder.info.splits.get(self.dataset_config.split)
			if split_info is None:
				return None
			return split_info.num_examples
		else:
			return len(os.listdir(self.dataset_path))

	def load_cache(self) -> dict[str, bytes]:
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
		for file_name in sorted(os.listdir(self.raw_path)):
			doc_id, _ = os.path.splitext(file_name)
			file_path = os.path.join(self.raw_path, file_name)
			with open(file_path, "rb") as f:
				cached_raw_bytes[doc_id] = f.read()
		return cached_raw_bytes
	
	def decode_raw_bytes(self, row: dict) -> bytes:
		"""
		row: one dataset row.
		dataset_config: tells us which row field holds the row's raw
			content and how that content is encoded.

		Reads row's raw content field and decodes it into bytes per
		dataset_config's content_encoding ("base64", "bytes", or "text").

		Raises ValueError if content_encoding is none of those.

		Returns the row's raw content as bytes.
		"""
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

		return raw_bytes

	def compute_raw_data(self)-> dict[str, bytes]:

		compute_raw_data_start = time.perf_counter()

		# ------- cache
		os.makedirs(self.raw_path, exist_ok=True)
		load_cache_start = time.perf_counter()
		cached_raw_bytes = self.load_cache()
		timing.record("load_cache", time.perf_counter() - load_cache_start)
		# If there is 10k cached and asked 5k, just return only the dataset_size
		if len(cached_raw_bytes) >= self.dataset_size:
			first_doc_ids = list(cached_raw_bytes.keys())[:self.dataset_size]
			first_raw_bytes = {}
			for doc_id in first_doc_ids:
				first_raw_bytes[doc_id] = cached_raw_bytes[doc_id]
			timing.record("compute_raw_data_total", time.perf_counter() - compute_raw_data_start)
			return first_raw_bytes

		# ------- new
		extension = self.dataset_config.extension
		new_raw_bytes = {}
		skipped_raw_error_count = 0
		if self.dataset_url:
			# Stream and cache the rest
			dataset = load_dataset(self.dataset_url, split=self.dataset_config.split, streaming=True)
			new_rows_stream = itertools.islice(dataset, len(cached_raw_bytes), self.dataset_size)
			new_rows_iterator = iter(new_rows_stream)

			while True:
				download_row_start = time.perf_counter()
				try:
					row = next(new_rows_iterator)
				except StopIteration:
					break
				timing.record("download_row", time.perf_counter() - download_row_start)

				# raw
				decode_row_start = time.perf_counter()
				try:
					raw_bytes = self.decode_raw_bytes(row)
					doc_id = make_doc_id(self.doc_type, raw_bytes)
				except Exception as error:
					timing.record("decode_raw_bytes", time.perf_counter() - decode_row_start)
					skipped_raw_error_count += 1
					row_file_name = row.get(self.dataset_config.file_name_field)
					print(
						f"Skipping row (file_name={row_file_name!r}): "
						f"{type(error).__name__}: {error}"
					)
					continue
				timing.record("decode_raw_bytes", time.perf_counter() - decode_row_start)

				# cache
				write_cache_start = time.perf_counter()
				cache_file_path = get_file_path(self.raw_path, doc_id, extension)
				with open(cache_file_path, "wb") as f:
					f.write(raw_bytes)
				timing.record("write_raw_cache_file", time.perf_counter() - write_cache_start)

				# dict
				new_raw_bytes[doc_id] = raw_bytes
		else:
			# Copy the rest from the local dataset directory
			source_file_names = sorted(os.listdir(self.dataset_path))
			remaining_file_names = source_file_names[len(cached_raw_bytes):self.dataset_size]

			for source_file_name in remaining_file_names:

				# raw
				read_local_file_start = time.perf_counter()
				try:
					source_file_path = os.path.join(self.dataset_path, source_file_name)
					with open(source_file_path, "rb") as f:
						raw_bytes = f.read()
					doc_id = make_doc_id(self.doc_type, raw_bytes)
				except Exception as error:
					timing.record("read_local_raw_file", time.perf_counter() - read_local_file_start)
					skipped_raw_error_count += 1
					print(
						f"Skipping file (file_name={source_file_name!r}): "
						f"{type(error).__name__}: {error}"
					)
					continue
				timing.record("read_local_raw_file", time.perf_counter() - read_local_file_start)

				# cache
				write_cache_start = time.perf_counter()
				cache_file_path = get_file_path(self.raw_path, doc_id, extension)
				with open(cache_file_path, "wb") as f:
					f.write(raw_bytes)
				timing.record("write_raw_cache_file", time.perf_counter() - write_cache_start)

				# dict
				new_raw_bytes[doc_id] = raw_bytes

		print(f"Skipped {skipped_raw_error_count} document(s) due to raw ingestion errors.")

		# Combine cached and newly-copied raw_bytes
		documents_raw_bytes = dict(cached_raw_bytes)
		for doc_id in new_raw_bytes:
			documents_raw_bytes[doc_id] = new_raw_bytes[doc_id]
		timing.record("compute_raw_data_total", time.perf_counter() - compute_raw_data_start)
		return documents_raw_bytes

	def document_to_json_dict(self, document: Document) -> dict:
		"""
		document: a fully-populated Document.

		Converts document to a plain dict of JSON-serializable values,
		explicitly turning the timestamp field into an ISO-8601 string.

		Returns the JSON-serializable dict.
		"""
		data = dataclasses.asdict(document)
		data["timestamp"] = document.timestamp.isoformat()
		return data


	def json_dict_to_document(self, data: dict) -> Document:
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


	def load_processed_document(self, doc_id: str) -> Document | None:
		"""
		processed_path: directory holding processed Document JSON files
			already written for this dataset.
		doc_id: identifies which processed Document to load.

		Checks whether doc_id has already been processed and cached under
		processed_path.

		Returns the cached Document if present, otherwise None.
		"""
		file_path = get_file_path(self.processed_path, doc_id, "json")
		if not os.path.exists(file_path):
			return None
		with open(file_path, encoding="utf-8") as f:
			return self.json_dict_to_document(json.load(f))


	def save_processed_document(self, doc_id: str, document: Document) -> None:
		"""
		processed_path: directory to write the processed Document JSON file
			into.
		doc_id: identifies the file to write.
		document: the Document to persist.

		Writes document to processed_path as {doc_id}.json.

		Returns nothing.
		"""
		file_path = get_file_path(self.processed_path, doc_id, "json") #os.path.join(processed_path, f"{doc_id}.json")
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(self.document_to_json_dict(document), f)

	def process_raw_data(self, documents_raw_bytes: dict[str, bytes]) -> dict[str, Document]:

		process_raw_data_start = time.perf_counter()

		os.makedirs(self.processed_path, exist_ok=True)

		parser_cls = get_parser(self.doc_type)
		parser = parser_cls()

		documents = {}
		content_hash_dict = {}
		lsh_index = make_lsh_index(threshold=0.90)
		skipped_count = 0
		skipped_duplicate_count = 0

		extension = self.dataset_config.extension

		for doc_id in documents_raw_bytes:

			load_processed_start = time.perf_counter()
			processed_document = self.load_processed_document(doc_id)
			timing.record("load_processed_document", time.perf_counter() - load_processed_start)
			if processed_document is not None:
				documents[doc_id] = processed_document
				content_hash_dict[doc_id] = processed_document.content_hash
				make_shingles_start = time.perf_counter()
				cached_shingles = make_shingles(processed_document.text)
				timing.record("make_shingles", time.perf_counter() - make_shingles_start)
				if len(cached_shingles) > 0:
					make_minhash_start = time.perf_counter()
					cached_minhash = make_minhash(cached_shingles)
					timing.record("make_minhash", time.perf_counter() - make_minhash_start)
					lsh_index.insert(doc_id, cached_minhash)
				continue

			try:
				# Actual process
				parse_start = time.perf_counter()
				data_path = get_file_path(self.raw_path, doc_id, extension)
				raw_bytes = documents_raw_bytes[doc_id]
				doc = parser.parse(doc_id, raw_bytes, data_path)
				timing.record("parse_document", time.perf_counter() - parse_start)
			except Exception:
				skipped_count += 1
				continue

			# Dedup check
			make_shingles_start = time.perf_counter()
			doc_shingles = make_shingles(doc.text)
			timing.record("make_shingles", time.perf_counter() - make_shingles_start)
			if len(doc_shingles) > 0:
				make_minhash_start = time.perf_counter()
				doc_minhash = make_minhash(doc_shingles)
				timing.record("make_minhash", time.perf_counter() - make_minhash_start)
			else:
				doc_minhash = None

			dedup_check_start = time.perf_counter()
			is_exact = is_exact_duplicate(doc.content_hash, content_hash_dict)
			if is_exact:
				is_near = False
			elif doc_minhash is not None:
				is_near = is_near_duplicate(doc_minhash, lsh_index)
			else:
				is_near = False
			timing.record("dedup_check", time.perf_counter() - dedup_check_start)

			if is_exact or is_near:
				skipped_duplicate_count += 1
				continue

			documents[doc_id] = doc
			content_hash_dict[doc_id] = doc.content_hash
			if doc_minhash is not None:
				lsh_index.insert(doc_id, doc_minhash)

			# Save
			save_start = time.perf_counter()
			self.save_processed_document(doc_id, doc)
			timing.record("save_processed_document", time.perf_counter() - save_start)

		print(f"Skipped {skipped_count} document(s) due to parsing errors.")
		print(f"Skipped {skipped_duplicate_count} document(s) due to duplicate detection.")

		timing.record("process_raw_data_total", time.perf_counter() - process_raw_data_start)

		return documents
		
	def process(self)-> dict[str, Document]:

		self.validate()

		# Raw
		documents_raw_bytes = self.compute_raw_data()

		# Process
		documents = self.process_raw_data(documents_raw_bytes)

		return documents
