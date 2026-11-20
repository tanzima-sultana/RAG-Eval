import dataclasses
import json
import os

from schema.retrieved import Retrieved
from schema.setup import Setup


def save_retrieved(retrieved_file: str, retrieved_items: list[Retrieved]) -> None:
	"""
	Input: retrieved_file, the cache file of one retrieval setup; retrieved_items, its Retrieved list.
	Output: nothing.
	What it does: writes the list as JSON to a temporary file, then renames it to retrieved_file.
	"""
	os.makedirs(os.path.dirname(retrieved_file), exist_ok=True)

	retrieved_dicts = []
	for retrieved in retrieved_items:
		retrieved_dicts.append(dataclasses.asdict(retrieved))

	with open(retrieved_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(retrieved_dicts, f)
	os.replace(retrieved_file + ".tmp", retrieved_file)


def load_retrieved(retrieved_file: str) -> list[Retrieved] | None:
	"""
	Input: retrieved_file, the cache file of one retrieval setup.
	Output: the saved Retrieved list, or None if the file does not exist.
	What it does: reads the JSON file and builds one Retrieved per entry, with its Setup.
	"""
	if not os.path.exists(retrieved_file):
		return None

	with open(retrieved_file, encoding="utf-8") as f:
		retrieved_dicts = json.load(f)

	retrieved_items = []
	for retrieved_dict in retrieved_dicts:
		retrieved_dict["setup"] = Setup(**retrieved_dict["setup"])
		retrieved_items.append(Retrieved(**retrieved_dict))
	return retrieved_items
