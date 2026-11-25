import dataclasses
import json
import os

from schema.query import Query


def save_query(query_file_path: str, query: Query) -> None:
	"""
	Input: query_file_path, the file the query is saved to; query, one Query.
	Output: nothing.
	What it does: saves the query as JSON to a temporary file, then renames it to query_file_path, so a crash never
	leaves a half-written file. A file with the same name is overwritten.
	"""
	os.makedirs(os.path.dirname(query_file_path), exist_ok=True)

	temporary_file = query_file_path + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(query), f, indent=2)
	os.replace(temporary_file, query_file_path)


def load_query(query_file_path: str) -> Query | None:
	"""
	Input: query_file_path, the file a query was saved to.
	Output: the saved Query, or None if there is no file.
	What it does: reads the JSON file and builds the Query from it.
	"""
	if not os.path.exists(query_file_path):
		return None

	with open(query_file_path, encoding="utf-8") as f:
		query_dict = json.load(f)
	query = Query(**query_dict)
	return query
