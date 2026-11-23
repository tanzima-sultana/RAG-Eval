import dataclasses
import json
import os

from schema.retrieved import Retrieved
from schema.setup import Setup


def save_retrieved_entry(retrieved_file: str, question: str, retrieved: Retrieved) -> None:
	"""
	Input: retrieved_file, the cache file of one setup and one question; question, the original question text;
	retrieved, its Retrieved.
	Output: nothing.
	What it does: saves the question and the Retrieved without its id as one JSON file, through a temporary file. The
	id is left out because the file is named by the question, not the id, so any eval set with this question reuses it.
	"""
	os.makedirs(os.path.dirname(retrieved_file), exist_ok=True)

	entry = {}
	entry["question"] = question
	entry["setup"] = dataclasses.asdict(retrieved.setup)
	entry["top_k"] = retrieved.top_k
	entry["rerank_top_k"] = retrieved.rerank_top_k
	entry["chunk_ids"] = retrieved.chunk_ids
	entry["chunk_scores"] = retrieved.chunk_scores
	entry["doc_ids"] = retrieved.doc_ids

	with open(retrieved_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(entry, f)
	os.replace(retrieved_file + ".tmp", retrieved_file)


def load_retrieved_entry(retrieved_file: str) -> Retrieved | None:
	"""
	Input: retrieved_file, the cache file of one setup and one question.
	Output: its Retrieved with an empty id, or None if the file does not exist. The caller sets the id.
	What it does: reads the JSON file and builds the Retrieved with its Setup.
	"""
	if not os.path.exists(retrieved_file):
		return None

	with open(retrieved_file, encoding="utf-8") as f:
		entry = json.load(f)

	return Retrieved(
		id="",
		setup=Setup(**entry["setup"]),
		top_k=entry["top_k"],
		rerank_top_k=entry["rerank_top_k"],
		chunk_ids=entry["chunk_ids"],
		chunk_scores=entry["chunk_scores"],
		doc_ids=entry["doc_ids"]
	)
