import json
import os

import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import QnaArgs
from common.args_temp import QueryArgs
from common.constants import I_DONT_KNOW
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
	Input: query_path, the query folder; queries, the Query list.
	Output: nothing.
	What it does: saves the texts and tokens to queries.json and the vectors to embeddings.npz, each through a
	temporary file. The multi vectors of all queries are stored one after another.
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
		for multi_qus in query.multi_qus:
			multi_qus_dicts.append({"qus": multi_qus.qus, "tokens": multi_qus.tokens})
			multi_vectors.append(multi_qus.embeddings)

		query_dicts.append({
			"id": query.id,
			"original_qus": {"qus": query.original_qus.qus, "tokens": query.original_qus.tokens},
			"rewritten_qus": {"qus": query.rewritten_qus.qus, "tokens": query.rewritten_qus.tokens},
			"multi_qus": multi_qus_dicts,
			"hyde_ans": {"ans": query.hyde_ans.ans},
			"normalized": query.normalized,
			"embedding_model": query.embedding_model,
			"llm": query.llm,
			"llm_model": query.llm_model
		})
		query_ids.append(query.id)
		original_vectors.append(query.original_qus.embeddings)
		rewritten_vectors.append(query.rewritten_qus.embeddings)
		hyde_vectors.append(query.hyde_ans.embeddings)

	queries_file = os.path.join(query_path, "queries.json")
	with open(queries_file + ".tmp", "w", encoding="utf-8") as f:
		json.dump(query_dicts, f, indent=2)
	os.replace(queries_file + ".tmp", queries_file)

	# Written through an open file, because np.savez adds ".npz" to a path that does not already end with it
	embeddings_file = os.path.join(query_path, "embeddings.npz")
	with open(embeddings_file + ".tmp", "wb") as f:
		np.savez(
			f,
			query_ids=np.array(query_ids),
			original=np.array(original_vectors, dtype=np.float32),
			rewritten=np.array(rewritten_vectors, dtype=np.float32),
			multi=np.array(multi_vectors, dtype=np.float32),
			hyde=np.array(hyde_vectors, dtype=np.float32),
		)
	os.replace(embeddings_file + ".tmp", embeddings_file)


def load_queries(query_path: str) -> list[Query] | None:
	"""
	Input: query_path, the query folder.
	Output: the saved list of Query, or None if queries.json or embeddings.npz is missing.
	What it does: joins the texts and tokens of queries.json with the vectors of embeddings.npz. Raises ValueError if
	the two files are not in the same query order.
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
	# Position of the query's first phrasing in multi_vectors
	multi_position = 0
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
		for multi_qus_dict in query_dict["multi_qus"]:
			multi_qus.append(Qus(
				qus=multi_qus_dict["qus"],
				tokens=multi_qus_dict["tokens"],
				embeddings=multi_vectors[multi_position]
			))
			multi_position = multi_position + 1
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
		if self.query_args.embedding_model.strip() == "":
			raise ValueError("embedding_model is empty. It is the chunking model_name, e.g. all-MiniLM-L6-v2.")

		if self.query_args.n_multi_qus < 1:
			raise ValueError(f"n_multi_qus ({self.query_args.n_multi_qus}) must be at least 1.")

	def compute_queries(self, qna_items: list[Qna], qna_args: QnaArgs, output_path: str) -> list[Query]:
		"""
		Input: qna_items, the eval questions; qna_args, the settings of their QnA set; output_path, the dataset's
		output folder.
		Output: one Query per eval question.
		What it does: loads the saved queries. Only if there are none, it generates them and saves them.
		"""
		# 1. Validate the settings
		self.validate()

		# 2. Load the saved queries. No LLM is built, so a cached run needs no Ollama server or API key
		query_path = paths_temp.get_query_path(output_path, qna_args, self.query_args)
		print(f"Query folder : {query_path}")
		queries = load_queries(query_path)
		if queries is not None:
			print(f"Queries : {len(queries)} (all from cache)")
			return queries

		# 3. Generate and save the queries
		print(f"No saved queries, transforming {len(qna_items)} questions")
		queries = self.generate_queries(qna_items)
		save_queries(query_path, queries)
		return queries

	def generate_queries(self, qna_items: list[Qna]) -> list[Query]:
		"""
		Input: qna_items, the eval questions.
		Output: one Query per eval question, in the same order.
		What it does: asks the query LLM for a rewrite, n_multi_qus phrasings and a HyDE answer of each question,
		splits every question into BM25 tokens and embeds every text. A wrong number of phrasings gives no phrasings,
		and a HyDE refusal is stored as I_DONT_KNOW; both are counted and the run goes on.
		"""
		query_args = self.query_args
		n_multi_qus = query_args.n_multi_qus

		# 1. Get the query LLM. LLMProvider validates it, so a paid model needs ALLOW_PAID_API=1
		query_llm_provider = LLMProvider(query_args.query_llm_args, self.ollama_args)

		# 2. Transform each question with the LLM
		original_qus_list = []
		rewritten_qus_list = []
		multi_qus_list = []
		hyde_ans_list = []
		multi_dont_know_count = 0
		hyde_dont_know_count = 0
		for qna_index in range(len(qna_items)):
			qna_item = qna_items[qna_index]

			# Rewritten question
			rewritten_qus_str = query_llm_provider.generate("", REWRITE_PROMPT.format(qus=qna_item.qus)).text.strip()

			# Multi-query phrasings. They come one per line, sometimes with blank lines between them
			multi_reply = query_llm_provider.generate(
				"", MULTI_QUERY_PROMPT.format(qus=qna_item.qus, n_multi_qus=n_multi_qus)
			).text
			multi_qus_str_list = []
			for reply_line in multi_reply.split("\n"):
				stripped_line = reply_line.strip()
				if stripped_line != "":
					multi_qus_str_list.append(stripped_line)
			# Any other number of lines means the reply is not just the phrasings, so no phrasings are kept
			if len(multi_qus_str_list) != n_multi_qus:
				print(
					f"Multi-query I don't know for {qna_item.id}: {len(multi_qus_str_list)} lines instead of "
					f"{n_multi_qus}. Reply: {multi_reply!r}"
				)
				multi_qus_str_list = []
				multi_dont_know_count = multi_dont_know_count + 1

			# Hypothetical answer. A refusal is stored as exactly I_DONT_KNOW, so retrieval can skip it
			hyde_ans_str = query_llm_provider.generate("", HYDE_PROMPT.format(qus=qna_item.qus)).text.strip()
			hyde_ans_lower = hyde_ans_str.lower().replace("’", "'")
			if hyde_ans_lower.startswith("i don't know") or hyde_ans_lower.startswith("i do not know"):
				hyde_ans_str = I_DONT_KNOW
				hyde_dont_know_count = hyde_dont_know_count + 1

			original_qus_list.append(qna_item.qus)
			rewritten_qus_list.append(rewritten_qus_str)
			multi_qus_list.append(multi_qus_str_list)
			hyde_ans_list.append(hyde_ans_str)

			if (qna_index + 1) % 20 == 0 or qna_index + 1 == len(qna_items):
				print(f"Questions transformed : {qna_index + 1} of {len(qna_items)}")

		print(f"Multi-query I don't know : {multi_dont_know_count} of {len(qna_items)}")
		print(f"HyDE I don't know : {hyde_dont_know_count} of {len(qna_items)}")
		query_llm_provider.compute_cost()

		# 3. Embed every text with LangChain's HuggingFaceEmbeddings, normalized like the chunk embeddings.
		# The phrasings are embedded as one flat list: n_multi_qus rows per question, none for an I don't know
		flat_multi_qus_list = []
		for multi_qus_str_list in multi_qus_list:
			for multi_qus_str in multi_qus_str_list:
				flat_multi_qus_list.append(multi_qus_str)

		device = "cuda" if torch.cuda.is_available() else "cpu"
		embedding_model = HuggingFaceEmbeddings(
			model_name=query_args.embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)
		original_qus_vectors = embedding_model.embed_documents(original_qus_list)
		rewritten_qus_vectors = embedding_model.embed_documents(rewritten_qus_list)
		flat_multi_qus_vectors = embedding_model.embed_documents(flat_multi_qus_list)
		hyde_ans_vectors = embedding_model.embed_documents(hyde_ans_list)

		# 4. Build one Query per question. Tokens are split the same way as the BM25 chunks
		# The model string is "{provider}/{model}", e.g. ollama_chat/llama3.1:8b
		query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
		queries = []
		# Position of the question's first phrasing in flat_multi_qus_vectors
		flat_multi_position = 0
		for qna_index in range(len(qna_items)):
			original_qus = Qus(
				qus=original_qus_list[qna_index],
				tokens=original_qus_list[qna_index].lower().split(),
				embeddings=np.array(original_qus_vectors[qna_index], dtype=np.float32)
			)
			rewritten_qus = Qus(
				qus=rewritten_qus_list[qna_index],
				tokens=rewritten_qus_list[qna_index].lower().split(),
				embeddings=np.array(rewritten_qus_vectors[qna_index], dtype=np.float32)
			)
			multi_qus = []
			for multi_qus_str in multi_qus_list[qna_index]:
				multi_qus.append(Qus(
					qus=multi_qus_str,
					tokens=multi_qus_str.lower().split(),
					embeddings=np.array(flat_multi_qus_vectors[flat_multi_position], dtype=np.float32)
				))
				flat_multi_position = flat_multi_position + 1
			hyde_ans = Ans(
				ans=hyde_ans_list[qna_index],
				embeddings=np.array(hyde_ans_vectors[qna_index], dtype=np.float32)
			)

			queries.append(Query(
				id=qna_items[qna_index].id,
				original_qus=original_qus,
				rewritten_qus=rewritten_qus,
				multi_qus=multi_qus,
				hyde_ans=hyde_ans,
				normalized=True,
				embedding_model=query_args.embedding_model,
				llm=query_llm,
				llm_model=query_llm_model
			))

		return queries
