"""
cache.py

Saves and loads the Query list of one query run in its own folder (see common.paths.get_query_path):

queries.json:    one entry per query with its texts and BM25 tokens, readable to check what the LLM wrote.
embeddings.npz:  the float32 vectors, in the same query order: query_ids, original (one row per query),
                 rewritten (one row per query), multi (n_multi_qus rows per query) and hyde (one row per query).

Each file is first written to a temporary file and then renamed into place, so a crash mid-write never
leaves a half-written file.
"""

import json
import os

import numpy as np

from l5_query.schema import Ans
from l5_query.schema import Qus
from l5_query.schema import Query

def load_cached_queries(query_path: str) -> list[Query] | None:
	"""
	Input: query_path, the query cache folder.
	Output: the cached list of Query, or None if queries.json or embeddings.npz is missing.
	What it does: reads the texts and tokens from queries.json and the vectors from embeddings.npz,
	and joins them into one Query per entry.
	"""
	queries_file = os.path.join(query_path, "queries.json")
	embeddings_file = os.path.join(query_path, "embeddings.npz")
	if not os.path.exists(queries_file) or not os.path.exists(embeddings_file):
		return None

	with open(queries_file, encoding="utf-8") as f:
		query_dicts = json.load(f)

	with np.load(embeddings_file) as saved_arrays:
		query_ids = saved_arrays["query_ids"]
		original_vectors = saved_arrays["original"]
		rewritten_vectors = saved_arrays["rewritten"]
		multi_vectors = saved_arrays["multi"]
		hyde_vectors = saved_arrays["hyde"]

	queries = []
	for index in range(len(query_dicts)):
		query_dict = query_dicts[index]
		if str(query_ids[index]) != query_dict["id"]:
			raise ValueError(
				f"queries.json and embeddings.npz in {query_path} are out of order at position {index}. "
				"Delete the folder and run again."
			)

		original_qus = Qus(
			qus=query_dict["original_qus"]["qus"],
			tokens=query_dict["original_qus"]["tokens"],
			embeddings=original_vectors[index]
		)
		rewritten_qus = Qus(
			qus=query_dict["rewritten_qus"]["qus"],
			tokens=query_dict["rewritten_qus"]["tokens"],
			embeddings=rewritten_vectors[index]
		)
		multi_qus = []
		for multi_index in range(len(query_dict["multi_qus"])):
			multi_qus_dict = query_dict["multi_qus"][multi_index]
			multi_qus.append(Qus(
				qus=multi_qus_dict["qus"],
				tokens=multi_qus_dict["tokens"],
				embeddings=multi_vectors[index][multi_index]
			))
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
			llm_model=query_dict["llm_model"]
		))

	return queries

def save_queries(query_path: str, queries: list[Query]) -> None:
	"""
	Input: query_path, the query cache folder; queries, the Query list.
	Output: nothing.
	What it does: writes the texts and tokens to queries.json and the vectors to embeddings.npz in query_path,
	creating the folder first if needed. Every query must have the same number of multi_qus.
	"""
	os.makedirs(query_path, exist_ok=True)

	query_dicts = []
	query_ids = []
	original_vectors = []
	rewritten_vectors = []
	multi_vectors = []
	hyde_vectors = []
	for query in queries:
		multi_qus_dicts = []
		query_multi_vectors = []
		for multi_qus in query.multi_qus:
			multi_qus_dicts.append({"qus": multi_qus.qus, "tokens": multi_qus.tokens})
			query_multi_vectors.append(multi_qus.embeddings)

		query_dicts.append({
			"id": query.id,
			"original_qus": {"qus": query.original_qus.qus, "tokens": query.original_qus.tokens},
			"rewritten_qus": {"qus": query.rewritten_qus.qus, "tokens": query.rewritten_qus.tokens},
			"multi_qus": multi_qus_dicts,
			"hyde_ans": {"ans": query.hyde_ans.ans},
			"normalized": query.normalized,
			"embedding_model": query.embedding_model,
			"llm_model": query.llm_model
		})
		query_ids.append(query.id)
		original_vectors.append(query.original_qus.embeddings)
		rewritten_vectors.append(query.rewritten_qus.embeddings)
		multi_vectors.append(query_multi_vectors)
		hyde_vectors.append(query.hyde_ans.embeddings)

	queries_file = os.path.join(query_path, "queries.json")
	temporary_queries_file = queries_file + ".tmp"
	with open(temporary_queries_file, "w", encoding="utf-8") as f:
		json.dump(query_dicts, f, indent=2)
	os.replace(temporary_queries_file, queries_file)

	embeddings_file = os.path.join(query_path, "embeddings.npz")
	temporary_embeddings_file = embeddings_file + ".tmp"
	# Written through an open file, because np.savez adds ".npz" to a path that does not already end with it
	with open(temporary_embeddings_file, "wb") as f:
		np.savez(
			f,
			query_ids=np.array(query_ids),
			original=np.array(original_vectors, dtype=np.float32),
			rewritten=np.array(rewritten_vectors, dtype=np.float32),
			multi=np.array(multi_vectors, dtype=np.float32),
			hyde=np.array(hyde_vectors, dtype=np.float32),
		)
	os.replace(temporary_embeddings_file, embeddings_file)
