import os
import random

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common.args import OllamaArgs
from common.args import QueryArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query_transformation.cache import QueryLLMCache
from rag.serving.query_transformation.single_query import transform_single_query
from schema.query import Query


class QueryTransformation:
	def __init__(self, query_args: QueryArgs, query_path: str):
		"""
		Input: query_args, the query transformation settings; query_path, the folder the query LLM cache file
		(.langchain.db) is saved in, used as given.
		Output: nothing.
		What it does: stores both, validates the settings, then loads the embedding model once and builds the query LLM
		once, so every question reuses them. Original is always made, and query_types holds only the extra types
		(rewritten, multi and/or hyde), which all need the query LLM. So the query LLM is built only if query_types is
		not empty, otherwise it is None. The query LLM gets a LangChain SQLite LLM cache in query_path, so each of its
		replies is made once and read back on later runs.
		"""
		self.query_args = query_args
		self.query_path = query_path

		# 1. Validate the settings before anything is loaded
		self.validate()

		# 2. The embedding model, normalized like the chunk embeddings, so a query vector matches the chunk vectors
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = HuggingFaceEmbeddings(
			model_name=query_args.embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
		)

		# 3. The query LLM, only if there is an extra query type, since each of them needs it. LLMProvider validates it,
		# so a paid model needs ALLOW_PAID_API=1
		self.query_llm_provider = None
		if len(query_args.query_types) > 0:
			ollama_args = OllamaArgs(
				ollama_api_base=query_args.ollama_api_base,
				ollama_context_length=query_args.ollama_context_length
			)
			self.query_llm_provider = LLMProvider(query_args.query_llm_args, ollama_args)

			# 4. The LLM cache of the query LLM only, in query_path. LangChain's key leaves out max_tokens, the Ollama
			# context length and the Ollama URL, so they are added to it
			os.makedirs(query_path, exist_ok=True)
			settings_key = (
				f"|max_tokens={query_args.query_llm_args.max_tokens}"
				f"|num_ctx={query_args.ollama_context_length}"
				f"|api_base={query_args.ollama_api_base}"
			)
			self.query_llm_provider.chat_model.cache = QueryLLMCache(os.path.join(query_path, ".langchain.db"), settings_key)

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

		# Original is always made, so query_types holds only the extra types. Empty query_types means original only
		supported_query_types = [
			constants.REWRITTEN_QUERY,
			constants.MULTI_QUERY,
			constants.HYDE_QUERY
		]
		for query_type in query_args.query_types:
			if query_type == constants.ORIGINAL_QUERY:
				raise ValueError(
					f"query_type \"{constants.ORIGINAL_QUERY}\" must not be in query_types. It is always made. "
					f"List only the extra query types, or none for original only."
				)
			if query_type not in supported_query_types:
				raise ValueError(
					f"query_type ({query_type}) is not supported. Use \"{constants.REWRITTEN_QUERY}\", "
					f"\"{constants.MULTI_QUERY}\" or \"{constants.HYDE_QUERY}\"."
				)

	def query(self, question_id: str, question: str) -> Query:
		"""
		Input: question_id, the id of one question; question, its text.
		Output: one Query of this question, with its id, always its original query, and the extra query types in
		query_types. The other extra query types are None.
		What it does: makes original and each query type in query_types of the question with the embedding model and
		query LLM loaded in __init__. Each query LLM call reads the LLM cache first and calls the LLM only when the reply
		is not saved. Prints the question's multi-query and HyDE I don't know and the query LLM's running cost.
		"""
		query_args = self.query_args

		# 1. Validate the input
		if question_id.strip() == "":
			raise ValueError("question_id is empty. Give the id of the question.")
		if question.strip() == "":
			raise ValueError(f"question of {question_id} is empty. Give the question text.")

		# 2. Original is always made, then each extra query type
		query_types = [constants.ORIGINAL_QUERY]
		for query_type in query_args.query_types:
			query_types.append(query_type)

		query = transform_single_query(
			query_args, question_id, question, query_types, self.query_llm_provider, self.embedding_model
		)

		# 3. The question's I don't know, for the requested types. The cost is the running total of every call so far
		if constants.MULTI_QUERY in query_args.query_types:
			multi_dont_know_count = 0
			if len(query.multi_qus) == 0:
				multi_dont_know_count = 1
			print(f"Multi-query I don't know : {multi_dont_know_count}")
		if constants.HYDE_QUERY in query_args.query_types:
			hyde_dont_know_count = 0
			if query.hyde_ans.ans == constants.I_DONT_KNOW:
				hyde_dont_know_count = 1
			print(f"HyDE I don't know : {hyde_dont_know_count}")
		if self.query_llm_provider is not None:
			self.query_llm_provider.compute_cost()

		return query

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
