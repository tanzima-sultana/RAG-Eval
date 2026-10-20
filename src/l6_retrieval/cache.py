"""
cache.py

Saves and loads the Retrieved outputs of one retrieval as one JSON file, in the retrieved folder of its chunk
type and index (see common.paths.get_retrieved_path and get_hybrid_retrieved_path):

{retrieval_type}_top{top_k}.json:                                 the outputs, not reranked.
{retrieval_type}_top{top_k}_rerank{rerank_top_k}_{model}.json:    the same outputs reranked by the cross-encoder.

The folder name holds the chunk and index settings, the query split (tuning or held_out), the tuning size and the
query types, but not the queries or the split seed. Delete outputs/<dataset>/retrieved/ after regenerating the
queries or changing TUNING_SPLIT_SEED.

Each file is first written to a temporary file and then renamed into place, so a crash mid-write never leaves a
half-written file that a later run would mistake for a complete cache.
"""

import dataclasses
import json
import os

from l6_retrieval.schema import Retrieved

def load_cached_retrieved(retrieved_file: str) -> list[Retrieved] | None:
	"""
	Input: retrieved_file, the cache file of one retrieval.
	Output: the cached Retrieved list, in the order it was saved, or None if the file does not exist.
	What it does: reads the file and builds one Retrieved per entry.
	"""
	if not os.path.exists(retrieved_file):
		return None

	with open(retrieved_file, encoding="utf-8") as f:
		retrieved_dicts = json.load(f)

	retrieved_outputs = []
	for retrieved_dict in retrieved_dicts:
		retrieved_outputs.append(Retrieved(**retrieved_dict))

	return retrieved_outputs

def save_retrieved(retrieved_file: str, retrieved_outputs: list[Retrieved]) -> None:
	"""
	Input: retrieved_file, the cache file of one retrieval; retrieved_outputs, its Retrieved list.
	Output: nothing.
	What it does: writes the list as JSON to retrieved_file, creating its folder first if needed.
	"""
	os.makedirs(os.path.dirname(retrieved_file), exist_ok=True)

	retrieved_dicts = []
	for retrieved in retrieved_outputs:
		retrieved_dicts.append(dataclasses.asdict(retrieved))

	temporary_retrieved_file = retrieved_file + ".tmp"
	with open(temporary_retrieved_file, "w", encoding="utf-8") as f:
		json.dump(retrieved_dicts, f)
	os.replace(temporary_retrieved_file, retrieved_file)
