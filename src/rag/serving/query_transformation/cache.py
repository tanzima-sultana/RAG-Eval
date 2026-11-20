import json
import os

import numpy as np

from schema.query import Ans
from schema.query import Query
from schema.query import Qus


def save_queries(query_path: str, queries: list[Query]) -> None:
	"""
	Input: query_path, the query folder; queries, the Query list. Every query holds the same query types.
	Output: nothing.
	What it does: saves the texts and tokens to queries.json and the vectors to embeddings.npz, each through a
	temporary file. A query type that is None is left out of both files. The multi vectors of all queries are stored
	one after another.
	"""
	os.makedirs(query_path, exist_ok=True)

	query_dicts = []
	query_ids = []
	original_vectors = []
	rewritten_vectors = []
	multi_vectors = []
	hyde_vectors = []
	for query in queries:
		query_dict = {"id": query.id}
		if query.original_qus is not None:
			query_dict["original_qus"] = {"qus": query.original_qus.qus, "tokens": query.original_qus.tokens}
			original_vectors.append(query.original_qus.embeddings)
		if query.rewritten_qus is not None:
			query_dict["rewritten_qus"] = {"qus": query.rewritten_qus.qus, "tokens": query.rewritten_qus.tokens}
			rewritten_vectors.append(query.rewritten_qus.embeddings)
		if query.multi_qus is not None:
			multi_qus_dicts = []
			for multi_qus in query.multi_qus:
				multi_qus_dicts.append({"qus": multi_qus.qus, "tokens": multi_qus.tokens})
				multi_vectors.append(multi_qus.embeddings)
			query_dict["multi_qus"] = multi_qus_dicts
		if query.hyde_ans is not None:
			query_dict["hyde_ans"] = {"ans": query.hyde_ans.ans}
			hyde_vectors.append(query.hyde_ans.embeddings)
		query_dict["normalized"] = query.normalized
		query_dict["embedding_model"] = query.embedding_model
		query_dict["llm"] = query.llm
		query_dict["llm_model"] = query.llm_model

		query_dicts.append(query_dict)
		query_ids.append(query.id)

	queries_file = os.path.join(query_path, "queries.json")
	with open(queries_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(query_dicts, f, indent=2)
	os.replace(queries_file + ".tmp", queries_file)

	# Only the query types the queries hold are saved
	saved_arrays = {"query_ids": np.array(query_ids)}
	if queries[0].original_qus is not None:
		saved_arrays["original"] = np.array(original_vectors, dtype=np.float32)
	if queries[0].rewritten_qus is not None:
		saved_arrays["rewritten"] = np.array(rewritten_vectors, dtype=np.float32)
	if queries[0].multi_qus is not None:
		saved_arrays["multi"] = np.array(multi_vectors, dtype=np.float32)
	if queries[0].hyde_ans is not None:
		saved_arrays["hyde"] = np.array(hyde_vectors, dtype=np.float32)

	# Written through an open file, because np.savez adds ".npz" to a path that does not already end with it
	embeddings_file = os.path.join(query_path, "embeddings.npz")
	with open(embeddings_file + ".tmp", "wb") as f:
		np.savez(f, **saved_arrays)
	os.replace(embeddings_file + ".tmp", embeddings_file)


def load_queries(query_path: str) -> list[Query] | None:
	"""
	Input: query_path, the query folder.
	Output: the saved list of Query, or None if queries.json or embeddings.npz is missing. A query type that was not
	saved is None.
	What it does: joins the texts and tokens of queries.json with the vectors of embeddings.npz. Raises ValueError if
	the two files are not in the same query order.
	"""
	queries_file = os.path.join(query_path, "queries.json")
	embeddings_file = os.path.join(query_path, "embeddings.npz")
	if not os.path.exists(queries_file) or not os.path.exists(embeddings_file):
		return None

	with open(queries_file, encoding="utf-8") as f:
		query_dicts = json.load(f)

	original_vectors = None
	rewritten_vectors = None
	multi_vectors = None
	hyde_vectors = None
	with np.load(embeddings_file) as saved_arrays:
		query_ids = saved_arrays["query_ids"]
		if "original" in saved_arrays.files:
			original_vectors = saved_arrays["original"]
		if "rewritten" in saved_arrays.files:
			rewritten_vectors = saved_arrays["rewritten"]
		if "multi" in saved_arrays.files:
			multi_vectors = saved_arrays["multi"]
		if "hyde" in saved_arrays.files:
			hyde_vectors = saved_arrays["hyde"]

	queries = []
	# Position of the query's first phrasing in multi_vectors
	multi_position = 0
	for index in range(len(query_dicts)):
		query_dict = query_dicts[index]
		if str(query_ids[index]) != query_dict["id"]:
			raise ValueError(
				f"queries.json and embeddings.npz in {query_path} are out of order at position {index}. "
				"Delete the folder and run again."
			)

		original_qus = None
		if "original_qus" in query_dict:
			original_qus = Qus(
				qus=query_dict["original_qus"]["qus"],
				tokens=query_dict["original_qus"]["tokens"],
				embeddings=original_vectors[index]
			)
		rewritten_qus = None
		if "rewritten_qus" in query_dict:
			rewritten_qus = Qus(
				qus=query_dict["rewritten_qus"]["qus"],
				tokens=query_dict["rewritten_qus"]["tokens"],
				embeddings=rewritten_vectors[index]
			)
		multi_qus = None
		if "multi_qus" in query_dict:
			multi_qus = []
			for multi_qus_dict in query_dict["multi_qus"]:
				multi_qus.append(Qus(
					qus=multi_qus_dict["qus"],
					tokens=multi_qus_dict["tokens"],
					embeddings=multi_vectors[multi_position]
				))
				multi_position = multi_position + 1
		hyde_ans = None
		if "hyde_ans" in query_dict:
			hyde_ans = Ans(
				ans=query_dict["hyde_ans"]["ans"],
				embeddings=hyde_vectors[index]
			)

		queries.append(Query(
			id=query_dict["id"],
			original_qus=original_qus,
			rewritten_qus=rewritten_qus,
			multi_qus=multi_qus,
			hyde_ans=hyde_ans,
			normalized=query_dict["normalized"],
			embedding_model=query_dict["embedding_model"],
			llm=query_dict["llm"],
			llm_model=query_dict["llm_model"]
		))

	return queries
