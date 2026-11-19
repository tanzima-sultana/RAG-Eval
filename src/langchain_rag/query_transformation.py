import dataclasses
import json
import os
import random

import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import QnaArgs
from common.args_temp import QueryArgs
from llm_provider.llm_provider import LLMProvider
from schema.qna import Qna
from schema.query import Ans
from schema.query import Query
from schema.query import Qus

# The prompts the query LLM gets for each eval question, filled with str.format(). The query folder does not record
# them, so delete the saved queries after changing one

# Input: {qus}. Output: one rewritten question
REWRITE_PROMPT = """Rewrite the question below so it is clear and specific. Do not mention documents or searching.
Keep its meaning. Use the words a document answering it would likely use.
Return only the rewritten question, with no introduction, quotes or explanation.

Question: {qus}"""

# Input: {qus}, {n_multi_qus}. Output: n_multi_qus questions, one per line
MULTI_QUERY_PROMPT = """Write exactly {n_multi_qus} different phrasings of the question below.
You are only rephrasing the question, not answering it or searching for it, so never say you cannot find information.
Each phrasing must keep the meaning of the question but use different words.
Return only the {n_multi_qus} phrasings, one per line, with no numbering, bullets, introduction or explanation.

Question: {qus}"""

# Input: {qus}. Output: one hypothetical answer passage. It is only used to search, so it may hold invented facts
HYDE_PROMPT = """Write a short passage of 3 to 5 sentences that answers the question below, as it might appear in a document.
If you are not sure of the answer, still write a plausible passage, using the terms a real document on this topic would use.
Never say you do not know or lack information.
Return only the passage, with no introduction, title or explanation.

Question: {qus}"""


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


class QueryTransformation:
	def __init__(self, query_args: QueryArgs, ollama_args: OllamaArgs):
		"""
		Input: query_args, the query transformation settings; ollama_args, the Ollama settings, used only if the query
		LLM is an Ollama model.
		Output: nothing.
		What it does: stores both settings.
		"""
		self.query_args = query_args
		self.ollama_args = ollama_args

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a query setting is not valid. The query LLM is validated by LLMProvider.
		"""
		query_args = self.query_args

		if query_args.embedding_model.strip() == "":
			raise ValueError("embedding_model is empty. It is the chunking model_name, e.g. all-MiniLM-L6-v2.")

		if query_args.n_multi_qus < 1:
			raise ValueError(f"n_multi_qus ({query_args.n_multi_qus}) must be at least 1.")

		supported_query_types = [
			constants.ORIGINAL_QUERY,
			constants.REWRITTEN_QUERY,
			constants.MULTI_QUERY,
			constants.HYDE_QUERY
		]
		if len(query_args.query_types) == 0:
			raise ValueError("query_types is empty. Give one or more query types.")
		for query_type in query_args.query_types:
			if query_type not in supported_query_types:
				raise ValueError(
					f"query_type ({query_type}) is not supported. Use \"{constants.ORIGINAL_QUERY}\", "
					f"\"{constants.REWRITTEN_QUERY}\", \"{constants.MULTI_QUERY}\" or \"{constants.HYDE_QUERY}\"."
				)

	def compute_queries(self, qna_items: list[Qna], qna_args: QnaArgs, output_path: str) -> list[Query]:
		"""
		Input: qna_items, the eval questions; qna_args, the settings of their QnA set; output_path, the dataset's
		output folder.
		Output: one Query per eval question, holding only the query types in query_types.
		What it does: loads the saved queries, generates only the requested query types that are not saved, adds them
		to the saved queries and saves them. A saved query type is never generated again.
		"""
		query_args = self.query_args

		# 1. Validate the settings
		self.validate()

		# 2. Load the saved queries, or start one empty Query per eval question
		query_path = paths_temp.get_query_path(output_path, qna_args, query_args)
		print(f"Query folder : {query_path}")
		saved_queries = load_queries(query_path)
		if saved_queries is None:
			print(f"No saved queries for {len(qna_items)} questions")
			query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
			saved_queries = []
			for qna_item in qna_items:
				saved_queries.append(Query(
					id=qna_item.id,
					original_qus=None,
					rewritten_qus=None,
					multi_qus=None,
					hyde_ans=None,
					normalized=True,
					embedding_model=query_args.embedding_model,
					llm=query_llm,
					llm_model=query_llm_model
				))

		# The saved queries are made from this QnA set, so they must be in the same question order
		if len(saved_queries) != len(qna_items):
			raise ValueError(
				f"The saved queries in {query_path} hold {len(saved_queries)} questions, but the QnA set holds "
				f"{len(qna_items)}. Delete the folder and run again."
			)
		for qna_index in range(len(qna_items)):
			if saved_queries[qna_index].id != qna_items[qna_index].id:
				raise ValueError(
					f"The saved queries in {query_path} are not in the QnA order at position {qna_index}. "
					"Delete the folder and run again."
				)

		# 3. Find the requested query types that are not saved. Every query holds the same query types
		missing_query_types = []
		if constants.ORIGINAL_QUERY in query_args.query_types and saved_queries[0].original_qus is None:
			missing_query_types.append(constants.ORIGINAL_QUERY)
		if constants.REWRITTEN_QUERY in query_args.query_types and saved_queries[0].rewritten_qus is None:
			missing_query_types.append(constants.REWRITTEN_QUERY)
		if constants.MULTI_QUERY in query_args.query_types and saved_queries[0].multi_qus is None:
			missing_query_types.append(constants.MULTI_QUERY)
		if constants.HYDE_QUERY in query_args.query_types and saved_queries[0].hyde_ans is None:
			missing_query_types.append(constants.HYDE_QUERY)

		# 4. Generate only the missing query types, add them to the saved queries and save them
		if len(missing_query_types) == 0:
			print(f"Queries : {len(saved_queries)} (all requested query types from cache)")
		else:
			print(f"Generating query types {missing_query_types} for {len(qna_items)} questions")

			# Normalized like the chunk embeddings, so a query vector matches the chunk vectors
			device = "cuda" if torch.cuda.is_available() else "cpu"
			embedding_model = HuggingFaceEmbeddings(
				model_name=query_args.embedding_model,
				model_kwargs={"device": device},
				encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
			)

			# The LLM is built only when a missing type needs one. LLMProvider validates it, so a paid model needs
			# ALLOW_PAID_API=1
			query_llm_provider = None
			if missing_query_types != [constants.ORIGINAL_QUERY]:
				query_llm_provider = LLMProvider(query_args.query_llm_args, self.ollama_args)

			if constants.ORIGINAL_QUERY in missing_query_types:
				original_qus_list = self.original_queries(qna_items, embedding_model)
				for qna_index in range(len(qna_items)):
					saved_queries[qna_index].original_qus = original_qus_list[qna_index]
			if constants.REWRITTEN_QUERY in missing_query_types:
				rewritten_qus_list = self.rewritten_queries(qna_items, query_llm_provider, embedding_model)
				for qna_index in range(len(qna_items)):
					saved_queries[qna_index].rewritten_qus = rewritten_qus_list[qna_index]
			if constants.MULTI_QUERY in missing_query_types:
				multi_qus_list = self.multi_queries(qna_items, query_llm_provider, embedding_model)
				for qna_index in range(len(qna_items)):
					saved_queries[qna_index].multi_qus = multi_qus_list[qna_index]
			if constants.HYDE_QUERY in missing_query_types:
				hyde_ans_list = self.hyde_queries(qna_items, query_llm_provider, embedding_model)
				for qna_index in range(len(qna_items)):
					saved_queries[qna_index].hyde_ans = hyde_ans_list[qna_index]

			if query_llm_provider is not None:
				query_llm_provider.compute_cost()
			save_queries(query_path, saved_queries)

		# 5. Return copies holding only the requested query types, so the saved queries stay whole
		queries = []
		for saved_query in saved_queries:
			query = dataclasses.replace(saved_query)
			if constants.ORIGINAL_QUERY not in query_args.query_types:
				query.original_qus = None
			if constants.REWRITTEN_QUERY not in query_args.query_types:
				query.rewritten_qus = None
			if constants.MULTI_QUERY not in query_args.query_types:
				query.multi_qus = None
			if constants.HYDE_QUERY not in query_args.query_types:
				query.hyde_ans = None
			queries.append(query)

		return queries

	def original_queries(self, qna_items: list[Qna], embedding_model: HuggingFaceEmbeddings) -> list[Qus]:
		"""
		Input: qna_items, the eval questions; embedding_model.
		Output: one Qus per eval question, in the same order.
		What it does: embeds each eval question as it is and splits it into BM25 tokens. No LLM is used.
		"""
		original_qus_str_list = []
		for qna_item in qna_items:
			original_qus_str_list.append(qna_item.qus)

		original_qus_vectors = embedding_model.embed_documents(original_qus_str_list)

		# Tokens are split the same way as the BM25 chunks
		original_qus_list = []
		for qna_index in range(len(qna_items)):
			original_qus_list.append(Qus(
				qus=original_qus_str_list[qna_index],
				tokens=original_qus_str_list[qna_index].lower().split(),
				embeddings=np.array(original_qus_vectors[qna_index], dtype=np.float32)
			))
		return original_qus_list

	def rewritten_queries(self, qna_items: list[Qna], query_llm_provider: LLMProvider, embedding_model: HuggingFaceEmbeddings) -> list[Qus]:
		"""
		Input: qna_items, the eval questions; query_llm_provider, the query LLM; embedding_model.
		Output: one rewritten Qus per eval question, in the same order.
		What it does: asks the query LLM to rewrite each question, then embeds it and splits it into BM25 tokens.
		"""
		rewritten_qus_str_list = []
		for qna_index in range(len(qna_items)):
			qna_item = qna_items[qna_index]
			rewritten_qus_str = query_llm_provider.generate("", REWRITE_PROMPT.format(qus=qna_item.qus)).text.strip()
			rewritten_qus_str_list.append(rewritten_qus_str)

			if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(qna_items):
				print(f"Questions rewritten : {qna_index + 1} of {len(qna_items)}")

		rewritten_qus_vectors = embedding_model.embed_documents(rewritten_qus_str_list)

		# Tokens are split the same way as the BM25 chunks
		rewritten_qus_list = []
		for qna_index in range(len(qna_items)):
			rewritten_qus_list.append(Qus(
				qus=rewritten_qus_str_list[qna_index],
				tokens=rewritten_qus_str_list[qna_index].lower().split(),
				embeddings=np.array(rewritten_qus_vectors[qna_index], dtype=np.float32)
			))
		return rewritten_qus_list

	def multi_queries(self, qna_items: list[Qna], query_llm_provider: LLMProvider, embedding_model: HuggingFaceEmbeddings) -> list[list[Qus]]:
		"""
		Input: qna_items, the eval questions; query_llm_provider, the query LLM; embedding_model.
		Output: one list of n_multi_qus Qus per eval question, in the same order. Empty if the LLM did not return
		exactly n_multi_qus phrasings; these are counted and the run goes on.
		What it does: asks the query LLM for n_multi_qus phrasings of each question, then embeds them and splits them
		into BM25 tokens.
		"""
		n_multi_qus = self.query_args.n_multi_qus

		multi_qus_str_list = []
		multi_dont_know_count = 0
		for qna_index in range(len(qna_items)):
			qna_item = qna_items[qna_index]

			# The phrasings come one per line, sometimes with blank lines between them
			multi_reply = query_llm_provider.generate(
				"", MULTI_QUERY_PROMPT.format(qus=qna_item.qus, n_multi_qus=n_multi_qus)
			).text
			phrasings = []
			for reply_line in multi_reply.split("\n"):
				stripped_line = reply_line.strip()
				if stripped_line != "":
					phrasings.append(stripped_line)

			# Any other number of lines means the reply is not just the phrasings, so no phrasings are kept
			if len(phrasings) != n_multi_qus:
				print(
					f"Multi-query I don't know for {qna_item.id}: {len(phrasings)} lines instead of "
					f"{n_multi_qus}. Reply: {multi_reply!r}"
				)
				phrasings = []
				multi_dont_know_count = multi_dont_know_count + 1
			multi_qus_str_list.append(phrasings)

			if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(qna_items):
				print(f"Questions rephrased : {qna_index + 1} of {len(qna_items)}")
		print(f"Multi-query I don't know : {multi_dont_know_count} of {len(qna_items)}")

		# The phrasings are embedded as one flat list: n_multi_qus rows per question, none for an I don't know
		flat_multi_qus_str_list = []
		for phrasings in multi_qus_str_list:
			for phrasing in phrasings:
				flat_multi_qus_str_list.append(phrasing)
		flat_multi_qus_vectors = embedding_model.embed_documents(flat_multi_qus_str_list)

		# Tokens are split the same way as the BM25 chunks
		multi_qus_list = []
		# Position of the question's first phrasing in flat_multi_qus_vectors
		flat_multi_position = 0
		for phrasings in multi_qus_str_list:
			multi_qus = []
			for phrasing in phrasings:
				multi_qus.append(Qus(
					qus=phrasing,
					tokens=phrasing.lower().split(),
					embeddings=np.array(flat_multi_qus_vectors[flat_multi_position], dtype=np.float32)
				))
				flat_multi_position = flat_multi_position + 1
			multi_qus_list.append(multi_qus)
		return multi_qus_list

	def hyde_queries(self, qna_items: list[Qna], query_llm_provider: LLMProvider, embedding_model: HuggingFaceEmbeddings) -> list[Ans]:
		"""
		Input: qna_items, the eval questions; query_llm_provider, the query LLM; embedding_model.
		Output: one HyDE Ans per eval question, in the same order. A refusal is stored as I_DONT_KNOW; these are
		counted and the run goes on.
		What it does: asks the query LLM for a hypothetical answer passage of each question, then embeds it.
		"""
		hyde_ans_str_list = []
		hyde_dont_know_count = 0
		for qna_index in range(len(qna_items)):
			qna_item = qna_items[qna_index]

			# A refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it
			hyde_ans_str = query_llm_provider.generate("", HYDE_PROMPT.format(qus=qna_item.qus)).text.strip()
			hyde_ans_lower = hyde_ans_str.lower().replace("’", "'")
			if hyde_ans_lower.startswith("i don't know") or hyde_ans_lower.startswith("i do not know"):
				hyde_ans_str = constants.I_DONT_KNOW
				hyde_dont_know_count = hyde_dont_know_count + 1
			hyde_ans_str_list.append(hyde_ans_str)

			if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(qna_items):
				print(f"HyDE answers written : {qna_index + 1} of {len(qna_items)}")
		print(f"HyDE I don't know : {hyde_dont_know_count} of {len(qna_items)}")

		hyde_ans_vectors = embedding_model.embed_documents(hyde_ans_str_list)

		hyde_ans_list = []
		for qna_index in range(len(qna_items)):
			hyde_ans_list.append(Ans(
				ans=hyde_ans_str_list[qna_index],
				embeddings=np.array(hyde_ans_vectors[qna_index], dtype=np.float32)
			))
		return hyde_ans_list

	def split_queries(self, queries: list[Query]) -> tuple[list[Query], list[Query]]:
		"""
		Input: queries, one Query per eval question.
		Output: (val_queries, test_queries). Val is for retrieval and retrieval evaluation, test for answer generation
		and answer evaluation.
		What it does: with query_split true, shuffles a copy of queries with TUNING_SPLIT_SEED, so every run gets the
		same split, and takes the first val_queries_size as val and the next test_queries_size as test. With
		query_split false, val and test are both all the queries.
		"""
		query_args = self.query_args

		if not query_args.query_split:
			print(f"No query split : val and test are all {len(queries)} queries")
			return queries, queries

		val_queries_size = query_args.val_queries_size
		test_queries_size = query_args.test_queries_size
		if val_queries_size < 1 or test_queries_size < 1:
			raise ValueError(
				f"val_queries_size ({val_queries_size}) and test_queries_size ({test_queries_size}) must each be "
				"at least 1."
			)
		if val_queries_size + test_queries_size > len(queries):
			raise ValueError(
				f"val_queries_size ({val_queries_size}) plus test_queries_size ({test_queries_size}) must not be "
				f"more than the number of queries ({len(queries)})."
			)

		shuffled_queries = []
		for query in queries:
			shuffled_queries.append(query)
		random_generator = random.Random(constants.TUNING_SPLIT_SEED)
		random_generator.shuffle(shuffled_queries)

		val_queries = shuffled_queries[:val_queries_size]
		test_queries = shuffled_queries[val_queries_size:val_queries_size + test_queries_size]
		print(f"Val queries : {len(val_queries)}, test queries : {len(test_queries)}")
		return val_queries, test_queries
