import time

from common.dataset_config import get_dataset_config
from common import validate

from l1_doc_processing.schema import Document
from l1_doc_processing.dedup import Deduplicator
from l1_doc_processing.raw_source import RawSource

from l1_doc_processing.cache import load_processed_document, save_processed_document
from l1_doc_processing.parser import get_parser
from l1_doc_processing.progress import load_progress, make_progress, save_progress
from l1_doc_processing.summary import RunCounts, print_parse_time_summary, print_run_summary


class DocProcessing:
	def __init__(self, doc_type: str, dataset_name: str, dataset_size: int, dataset_url: str, dataset_path: str):
		"""
		doc_type: document type to process (e.g. "pdf").
		dataset_name: name of the dataset.
		dataset_size: number of processed documents wanted at the end.
		dataset_url: Hugging Face dataset id, for a web dataset.
		dataset_path: local directory, for a local dataset.

		Stores the given arguments on the instance, looks up the
		dataset's registered DatasetConfig by dataset_name, and creates
		the RawSource that reads its rows. Raises ValueError if
		dataset_name is not registered.

		Returns nothing.
		"""
		self.doc_type = doc_type
		self.dataset_name = dataset_name
		self.dataset_size = dataset_size
		self.dataset_url = dataset_url
		self.dataset_path = dataset_path

		self.dataset_config = get_dataset_config(self.dataset_name)
		self.raw_source = RawSource(self.dataset_name, self.dataset_config, self.doc_type, self.dataset_url, self.dataset_path)

	def process(self) -> dict[str, Document]:
		"""
		Validates the dataset and saved progress, and loads already
		processed documents. Then, in batches: fetches (downloading in
		parallel) exactly as many rows as documents are still missing,
		parses, dedups and saves them, and saves progress. Batches repeat
		until dataset_size documents are processed or the source runs out.

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
		progress = load_progress(self.dataset_name, self.dataset_config, self.doc_type)
		validate.validate_progress(progress, self.dataset_config, self.doc_type)

		# 3. Check cached processed documents
		deduplicator = Deduplicator(threshold=0.90)

		# processed_doc_ids stays a list so the processing order is kept
		processed_doc_ids = []
		skipped_doc_ids = set(progress.skipped_doc_ids)
		rows_consumed = progress.rows_consumed

		cached_index = 0
		while cached_index < len(progress.processed_doc_ids):
			doc_id = progress.processed_doc_ids[cached_index]
			cached_index += 1

			cached_doc = load_processed_document(self.dataset_name, doc_id)
			# If a processed doc is physically removed somehow
			if cached_doc is None:
				print(f"Processed document {doc_id} is missing on disk, marking it as skipped.")
				skipped_doc_ids.add(doc_id)
				continue

			processed_doc_ids.append(doc_id)
			documents[doc_id] = cached_doc
			deduplicator.add(doc_id, cached_doc)

			# if there is dataset_size cached documents already exists
			if len(documents) == self.dataset_size:
				# Keep the not-yet-checked cached ids in progress too
				unchecked_doc_ids = progress.processed_doc_ids[cached_index:]
				all_processed_doc_ids = processed_doc_ids + unchecked_doc_ids
				save_progress(self.dataset_name, make_progress(self.dataset_config, self.doc_type, rows_consumed, all_processed_doc_ids, skipped_doc_ids))
				print(f"Loaded {len(documents)} already processed document(s). Nothing new to process.")
				return documents

		# 4. Fetch raw data in batches, then process each batch
		row_iterator = self.raw_source.get_iterator(rows_consumed)

		parser_cls = get_parser(self.doc_type)
		parser = parser_cls()

		cached_count = len(documents)
		run_counts = RunCounts()
		# (doc_id, seconds) for every document parsed successfully in this run
		parse_times = []
		batch_number = 0
		source_exhausted = False

		while len(documents) < self.dataset_size and not source_exhausted:
			# 4a. Download/decode exactly as many rows as documents still missing
			batch_number += 1
			rows_needed = self.dataset_size - len(documents)
			print(f"Batch {batch_number}: fetching {rows_needed} row(s).")
			raw_batch = self.raw_source.fetch_raw_batch(row_iterator, rows_needed, run_counts)
			rows_consumed += raw_batch.rows_read
			source_exhausted = raw_batch.source_exhausted

			# 4b. Parse, dedup and save the batch, in row order
			for doc_id, source_path in raw_batch.raw_items:
				if doc_id in skipped_doc_ids:
					run_counts.previously_skipped += 1
					continue

				with open(source_path, "rb") as f:
					raw_bytes = f.read()

				parse_start = time.perf_counter()
				try:
					processed_doc = parser.parse(doc_id, raw_bytes, source_path)
					parse_seconds = time.perf_counter() - parse_start
					parse_times.append((doc_id, parse_seconds))
				except Exception as error:
					print(f"Skipping document {doc_id}: parse failed: {type(error).__name__}: {error}")
					skipped_doc_ids.add(doc_id)
					run_counts.parse_error += 1
					continue

				if not deduplicator.add_if_new(doc_id, processed_doc):
					run_counts.duplicate += 1
					continue

				processed_doc_ids.append(doc_id)
				documents[doc_id] = processed_doc
				save_processed_document(self.dataset_name, doc_id, processed_doc)

			# 4c. Save progress after every batch
			save_progress(self.dataset_name, make_progress(self.dataset_config, self.doc_type, rows_consumed, processed_doc_ids, skipped_doc_ids))

		print_run_summary(cached_count, len(documents), run_counts)
		print_parse_time_summary(parse_times)
		if len(documents) < self.dataset_size:
			print("Reached the end of the dataset.")
		return documents
