import time
import concurrent.futures
import multiprocessing

from common.dataset_config import get_dataset_config
from common.constants import PARSE_DOC_COUNT, PARSE_WORKERS

from l1_doc_processing.schema import Document
from l1_doc_processing.dedup import Deduplicator
from l1_doc_processing.raw_source import RawSource

from l1_doc_processing.validate import validate_dataset, validate_progress
from l1_doc_processing.cache import load_processed_document, save_processed_document
from l1_doc_processing.parser import parse_document_from_file
from l1_doc_processing.progress import load_progress, make_progress, save_progress
from l1_doc_processing.summary import SummaryCounts, print_parse_time_summary, print_run_summary


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
		validate_dataset(
			self.doc_type,
			self.dataset_name,
			self.dataset_size,
			self.dataset_url,
			self.dataset_path,
		)

		# 2. Validate data download and process progress using progress.json
		progress = load_progress(self.dataset_name, self.dataset_config, self.doc_type)
		validate_progress(progress, self.dataset_config, self.doc_type)

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

		# For result summary
		cached_count = len(documents)
		summary_counts = SummaryCounts()

		# (doc_id, seconds) for every document parsed successfully in this run
		parse_times = []
		batch_number = 0
		source_exhausted = False

		# "spawn" starts each worker as a fresh Python process instead of
		# forking this one, so workers never inherit threads or locks left
		# open by the Hugging Face streaming and download code
		spawn_context = multiprocessing.get_context("spawn")
		with concurrent.futures.ProcessPoolExecutor(max_workers=PARSE_WORKERS, mp_context=spawn_context) as parse_executor:

			while len(documents) < self.dataset_size and not source_exhausted:

				# 4a. Download/decode exactly as many rows as documents still missing
				batch_number += 1
				rows_needed = self.dataset_size - len(documents)
				print(f"Batch {batch_number}: fetching {rows_needed} row(s).")

				raw_batch = self.raw_source.fetch_raw_batch(row_iterator, rows_needed, summary_counts)
				rows_consumed += raw_batch.rows_read
				source_exhausted = raw_batch.source_exhausted

				# 4b. Collect the batch's documents that still need parsing
				# Need list to parallel process
				doc_type_to_parse = []
				doc_ids_to_parse = []
				source_paths_to_parse = []
				for doc_id, source_path in raw_batch.raw_items:
					if doc_id in skipped_doc_ids:
						summary_counts.skipped += 1
						continue
					doc_type_to_parse.append(self.doc_type)
					doc_ids_to_parse.append(doc_id)
					source_paths_to_parse.append(source_path)

				# 4c. Parse them in parallel, in PARSE_WORKERS processes.
				# executor.map returns results in the same order as doc_ids_to_parse
				parse_results = parse_executor.map(
						parse_document_from_file,
						doc_type_to_parse,
						doc_ids_to_parse,
						source_paths_to_parse,
						chunksize=PARSE_DOC_COUNT,
				)
			
				# 4d. Dedup and save the parsed documents one by one, in row order
				for doc_id, processed_doc, parse_seconds, error_message in parse_results:
					if error_message is not None:
						print(f"Skipping document {doc_id}: parse failed: {error_message}")
						skipped_doc_ids.add(doc_id)
						summary_counts.parse_error += 1
						continue

					parse_times.append((doc_id, parse_seconds))

					if not deduplicator.add_if_new(doc_id, processed_doc):
						summary_counts.duplicate += 1
						continue

					processed_doc_ids.append(doc_id)
					documents[doc_id] = processed_doc
					save_processed_document(self.dataset_name, doc_id, processed_doc)

				# 4e. Save progress after every batch
				save_progress(self.dataset_name, make_progress(self.dataset_config, self.doc_type, rows_consumed, processed_doc_ids, skipped_doc_ids))


		print_run_summary(cached_count, len(documents), summary_counts)
		print_parse_time_summary(parse_times)
		if len(documents) < self.dataset_size:
			print("Reached the end of the dataset.")

		return documents
