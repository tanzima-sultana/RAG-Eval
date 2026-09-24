"""
summary.py

Prints the end-of-run summary of chunking: documents chunked or taken
from the chunk cache, per-document chunk times, the sentence embedding
step of semantic chunking, and chunk size statistics.
"""

import os

from l2_chunking.schema import Chunk


def print_run_summary(document_count: int, cached_document_count: int, cached_chunk_count: int, new_chunk_count: int) -> None:
	"""
	document_count: documents loaded for this run.
	cached_document_count: documents whose chunks were loaded from the cache.
	cached_chunk_count: chunks loaded from the cache.
	new_chunk_count: chunks newly computed and saved.

	Prints how many documents and chunks came from the cache and how many
	were newly chunked.

	Returns nothing.
	"""
	print(f"Loaded {document_count} processed document(s).")
	print(f"Loaded cached chunks for {cached_document_count} document(s): {cached_chunk_count} chunk(s).")
	print(f"Newly chunked {document_count - cached_document_count} document(s): {new_chunk_count} chunk(s).")
	print(f"Total chunks: {cached_chunk_count + new_chunk_count}.")


def print_sentence_embedding_summary(sentence_count: int, seconds: float) -> None:
	"""
	sentence_count: sentences embedded for semantic chunking.
	seconds: time the embedding took.

	Prints the sentence embedding time of semantic chunking.

	Returns nothing.
	"""
	print("\n----- Sentence embedding time (semantic) -----")
	print(f"Sentences embedded : {sentence_count}")
	print(f"Total              : {seconds:.3f}s")


def print_chunk_time_summary(chunk_times: list[tuple[str, float]]) -> None:
	"""
	chunk_times: (doc_id, seconds) for every document chunked in this run.

	Prints how many documents were chunked and their total, average,
	fastest and slowest chunk time, with the fastest and slowest doc_id.

	Returns nothing.
	"""
	print("\n----- Document chunk time -----")
	if len(chunk_times) == 0:
		print("No documents were chunked in this run.")
		return

	total_seconds = 0.0
	fastest_doc_id, fastest_seconds = chunk_times[0]
	slowest_doc_id, slowest_seconds = chunk_times[0]
	for doc_id, seconds in chunk_times:
		total_seconds += seconds
		if seconds < fastest_seconds:
			fastest_doc_id = doc_id
			fastest_seconds = seconds
		if seconds > slowest_seconds:
			slowest_doc_id = doc_id
			slowest_seconds = seconds
	average_seconds = total_seconds / len(chunk_times)

	print(f"Documents chunked : {len(chunk_times)}")
	print(f"Total             : {total_seconds:.3f}s")
	print(f"Average           : {average_seconds:.3f}s")
	print(f"Fastest           : {fastest_seconds:.3f}s ({fastest_doc_id})")
	print(f"Slowest           : {slowest_seconds:.3f}s ({slowest_doc_id})")


def print_chunk_statistics(chunks: dict[str, Chunk], chunk_path: str) -> None:
	"""
	chunks: every chunk of this run, keyed by chunk_id.
	chunk_path: folder holding the saved chunk files.

	Prints the chunk count, average chunk size in tokens, and the total
	and average size of the chunk files in bytes.

	Returns nothing.
	"""
	total_chunk_count = len(chunks)

	total_token_size = 0
	for chunk_id in chunks:
		total_token_size += chunks[chunk_id].size
	if total_chunk_count > 0:
		average_chunk_token_size = total_token_size / total_chunk_count
	else:
		average_chunk_token_size = 0

	total_file_byte_size = 0
	chunk_file_count = 0
	if os.path.exists(chunk_path):
		chunk_file_names = os.listdir(chunk_path)
		chunk_file_count = len(chunk_file_names)
		for chunk_file_name in chunk_file_names:
			file_path = os.path.join(chunk_path, chunk_file_name)
			total_file_byte_size += os.path.getsize(file_path)
	if chunk_file_count > 0:
		average_file_byte_size = total_file_byte_size / chunk_file_count
	else:
		average_file_byte_size = 0

	print("\n----- Chunk statistics -----")
	print(f"Total chunks            : {total_chunk_count}")
	print(f"Average chunk size      : {average_chunk_token_size:.1f} tokens")
	print(f"Total chunk file size   : {total_file_byte_size} bytes")
	print(f"Average chunk file size : {average_file_byte_size:.1f} bytes")
