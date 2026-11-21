import json
import os

import numpy as np

from common import constants
from schema.query import Ans
from schema.query import Qus


def save_query_entry(query_file: str, query_type: str, question: str, result: Qus | list[Qus] | Ans, embedding_model: str, llm_model: str | None) -> None:
	"""
	Input: query_file, the file of this query type of this question; query_type; question, the question text; result,
	a Qus for original and rewritten, a list of Qus for multi, an Ans for hyde; embedding_model; llm_model, the query
	LLM model, None for original.
	Output: nothing.
	What it does: saves the result with its texts, tokens and vectors as one JSON file, through a temporary file.
	"""
	os.makedirs(os.path.dirname(query_file), exist_ok=True)

	entry = {}
	entry["query_type"] = query_type
	entry["question"] = question
	if query_type == constants.MULTI_QUERY:
		phrasing_dicts = []
		for phrasing in result:
			phrasing_dicts.append({"qus": phrasing.qus, "tokens": phrasing.tokens, "embeddings": phrasing.embeddings.tolist()})
		entry["multi_qus"] = phrasing_dicts
	elif query_type == constants.HYDE_QUERY:
		entry["ans"] = result.ans
		entry["embeddings"] = result.embeddings.tolist()
	else:
		entry["qus"] = result.qus
		entry["tokens"] = result.tokens
		entry["embeddings"] = result.embeddings.tolist()
	entry["embedding_model"] = embedding_model
	entry["llm_model"] = llm_model

	with open(query_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(entry, f)
	os.replace(query_file + ".tmp", query_file)


def load_query_entry(query_file: str) -> Qus | list[Qus] | Ans | None:
	"""
	Input: query_file, the file of one query type of one question.
	Output: its result, a Qus for original and rewritten, a list of Qus for multi, an Ans for hyde, or None if the
	file does not exist.
	What it does: reads the JSON file and turns its texts, tokens and vectors back into the result.
	"""
	if not os.path.exists(query_file):
		return None

	with open(query_file, encoding="utf-8") as f:
		entry = json.load(f)

	query_type = entry["query_type"]
	if query_type == constants.MULTI_QUERY:
		phrasings = []
		for phrasing_dict in entry["multi_qus"]:
			phrasings.append(Qus(
				qus=phrasing_dict["qus"],
				tokens=phrasing_dict["tokens"],
				embeddings=np.array(phrasing_dict["embeddings"], dtype=np.float32)
			))
		return phrasings

	if query_type == constants.HYDE_QUERY:
		return Ans(ans=entry["ans"], embeddings=np.array(entry["embeddings"], dtype=np.float32))

	return Qus(qus=entry["qus"], tokens=entry["tokens"], embeddings=np.array(entry["embeddings"], dtype=np.float32))
