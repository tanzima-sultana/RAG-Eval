import os
import random

from common import constants
from common import paths
from common.args import OllamaArgs
from common.args import QueryArgs
from common.args import QueryBenchmarkArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query.cache import QueryLLMCache
from rag.serving.query.cache import load_query
from rag.serving.query.cache import save_query
from rag.serving.query.single_query import make_hyde_query
from rag.serving.query.single_query import make_multi_queries
from rag.serving.query.single_query import make_original_query
from rag.serving.query.single_query import make_rewritten_query
from schema.llm import LLMArgs
from schema.query import Query


class QueryTransformation:
	def __init__(self, query_args: QueryArgs, query_path: str, query_benchmark_args: QueryBenchmarkArgs | None = None):
		"""
		Input: query_args, the query settings, holding one query type and the query LLM settings; query_path, the one
		folder every query is saved in, used as given; query_benchmark_args, the benchmark's query settings, used only
		by split_queries, None when the layer is not run by the benchmark, as in the API.
		Output: nothing.
		What it does: stores them, validates the settings, creates the folder, then builds the query LLM once, so
		every question reuses it. Original needs no LLM, so the query LLM is built only for rewritten, multi and hyde,
		otherwise it is None. The query LLM gets a LangChain SQLite LLM cache in query_path, so each of its replies is
		made once and read back on later runs, which is also what makes multi repeat.
		"""
		self.query_args = query_args
		self.query_path = query_path
		self.query_benchmark_args = query_benchmark_args

		# 1. Validate the settings before anything is built
		self.validate()

		# 2. The folder the queries are saved in
		os.makedirs(query_path, exist_ok=True)

		# 3. The query LLM, only for the query types that need it. LLMProvider validates it, so a paid model needs
		# ALLOW_PAID_API=1
		self.query_llm_provider = None
		if query_args.query_type != constants.ORIGINAL_QUERY:
			llm_args = LLMArgs(
				llm_model=query_args.query_llm_args.llm_model,
				max_tokens=query_args.query_llm_args.max_tokens,
				timeout_seconds=query_args.query_llm_args.timeout_seconds
			)
			ollama_args = OllamaArgs(
				ollama_api_base=query_args.query_llm_args.ollama_api_base,
				ollama_context_length=query_args.query_llm_args.ollama_context_length
			)
			self.query_llm_provider = LLMProvider(llm_args, ollama_args)

			# 4. The LLM cache of the query LLM only, in query_path. LangChain's key leaves out max_tokens, the Ollama
			# context length and the Ollama URL, so they are added to it
			settings_key = (
				f"|max_tokens={query_args.query_llm_args.max_tokens}"
				f"|num_ctx={query_args.query_llm_args.ollama_context_length}"
				f"|api_base={query_args.query_llm_args.ollama_api_base}"
			)
			self.query_llm_provider.chat_model.cache = QueryLLMCache(os.path.join(query_path, ".langchain.db"), settings_key)

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a query setting is not valid. The query LLM is validated by LLMProvider.
		"""
		query_args = self.query_args

		if query_args.n_multi_qus < 1:
			raise ValueError(f"n_multi_qus ({query_args.n_multi_qus}) must be at least 1.")

		supported_query_types = [
			constants.ORIGINAL_QUERY,
			constants.REWRITTEN_QUERY,
			constants.MULTI_QUERY,
			constants.HYDE_QUERY
		]
		if query_args.query_type not in supported_query_types:
			raise ValueError(
				f"query_type ({query_args.query_type}) is not supported. Use \"{constants.ORIGINAL_QUERY}\", "
				f"\"{constants.REWRITTEN_QUERY}\", \"{constants.MULTI_QUERY}\" or \"{constants.HYDE_QUERY}\"."
			)

	def query(self, question_id: str, question: str) -> tuple[list[Query], list[str], list[str]]:
		"""
		Input: question_id, the id of one question; question, its text.
		Output: (list of Query, list of hashes, list of paths) for every query type. Original, rewritten and hyde
		give one item in each list. Multi gives one item per phrasing, and all three lists are empty if the LLM did
		not return the phrasings.
		What it does: checks the input, then calls the method of the query type in query_args, which loads the saved
		query or makes and saves it. Prints the query LLM's running cost when the query type uses the LLM.
		"""
		# 1. Validate the input
		if question_id.strip() == "":
			raise ValueError("question_id is empty. Give the id of the question.")
		if question.strip() == "":
			raise ValueError(f"question of {question_id} is empty. Give the question text.")

		# 2. The query type is decided once, here
		query_type = self.query_args.query_type
		if query_type == constants.ORIGINAL_QUERY:
			result = self.query_original(question_id, question)
		elif query_type == constants.REWRITTEN_QUERY:
			result = self.query_rewritten(question_id, question)
		elif query_type == constants.MULTI_QUERY:
			result = self.query_multi(question_id, question)
		else:
			result = self.query_hyde(question_id, question)

		# 3. The cost is the running total of every call so far
		if self.query_llm_provider is not None:
			self.query_llm_provider.compute_cost()

		return result

	def query_batch(self, question_ids: list[str], questions: list[str]) -> dict:
		"""
		Input: question_ids, the ids of the questions; questions, their texts, in the same order.
		Output: a dict keyed by question id, each value what query() returns for that question.
		What it does: calls query() once for each question. The query LLM is built once in __init__, not per question.
		"""
		if len(question_ids) != len(questions):
			raise ValueError(
				f"question_ids ({len(question_ids)}) and questions ({len(questions)}) must be the same length."
			)

		results = {}
		for index in range(len(question_ids)):
			question_id = question_ids[index]
			results[question_id] = self.query(question_id, questions[index])
		return results

	def query_original(self, question_id: str, question: str) -> tuple[list[Query], list[str], list[str]]:
		"""
		Input: question_id; question, its text.
		Output: ([Query], [hash], [path]) of the original query, one item in each list.
		What it does: hashes the question and the settings, loads the saved query if there is one, otherwise makes it
		without an LLM and saves it.
		"""
		query_hash = paths.get_query_hash(self.query_args, question)
		query_file_path = paths.get_query_file_path(self.query_path, query_hash)

		query = load_query(query_file_path)
		if query is None:
			query = make_original_query(self.query_args, question_id, question)
			save_query(query_file_path, query)
		return [query], [query_hash], [query_file_path]

	def query_rewritten(self, question_id: str, question: str) -> tuple[list[Query], list[str], list[str]]:
		"""
		Input: question_id; question, its text.
		Output: ([Query], [hash], [path]) of the rewritten query, one item in each list.
		What it does: hashes the question and the settings, loads the saved query if there is one, otherwise asks the
		query LLM to rewrite the question and saves the query.
		"""
		query_hash = paths.get_query_hash(self.query_args, question)
		query_file_path = paths.get_query_file_path(self.query_path, query_hash)

		query = load_query(query_file_path)
		if query is None:
			query = make_rewritten_query(self.query_args, question_id, question, self.query_llm_provider)
			save_query(query_file_path, query)
		return [query], [query_hash], [query_file_path]

	def query_hyde(self, question_id: str, question: str) -> tuple[list[Query], list[str], list[str]]:
		"""
		Input: question_id; question, its text.
		Output: ([Query], [hash], [path]) of the HyDE query, one item in each list.
		What it does: hashes the question and the settings, loads the saved query if there is one, otherwise asks the
		query LLM for a hypothetical answer and saves the query. Prints 1 if the LLM refused, otherwise 0.
		"""
		query_hash = paths.get_query_hash(self.query_args, question)
		query_file_path = paths.get_query_file_path(self.query_path, query_hash)

		query = load_query(query_file_path)
		if query is None:
			query = make_hyde_query(self.query_args, question_id, question, self.query_llm_provider)
			save_query(query_file_path, query)

		hyde_dont_know_count = 0
		if query.transformed_query == constants.I_DONT_KNOW:
			hyde_dont_know_count = 1
		print(f"HyDE I don't know : {hyde_dont_know_count}")
		return [query], [query_hash], [query_file_path]

	def query_multi(self, question_id: str, question: str) -> tuple[list[Query], list[str], list[str]]:
		"""
		Input: question_id; question, its text.
		Output: (list of Query, list of hashes, list of paths), one of each per phrasing. All three are empty if the
		LLM did not return exactly n_multi_qus phrasings.
		What it does: asks the query LLM for the phrasings every time, since the phrasings are not known before the
		LLM runs, then saves each phrasing as its own query, under the hash of its own text and the settings. Prints 1
		if no phrasings were kept, otherwise 0.
		"""
		queries = make_multi_queries(self.query_args, question_id, question, self.query_llm_provider)

		query_hashes = []
		query_file_paths = []
		for query in queries:
			query_hash = paths.get_query_hash(self.query_args, query.transformed_query)
			query_file_path = paths.get_query_file_path(self.query_path, query_hash)
			save_query(query_file_path, query)
			query_hashes.append(query_hash)
			query_file_paths.append(query_file_path)

		multi_dont_know_count = 0
		if len(queries) == 0:
			multi_dont_know_count = 1
		print(f"Multi-query I don't know : {multi_dont_know_count}")
		return queries, query_hashes, query_file_paths

	def split_queries(self, queries: list[Query]) -> tuple[list[Query], list[Query]]:
		"""
		Input: queries, one Query per eval question.
		Output: (val_queries, test_queries). Val is for retrieval and retrieval evaluation, test for answer generation
		and answer evaluation.
		What it does: with query_split true, shuffles a copy of queries with TUNING_SPLIT_SEED, so every run gets the
		same split, and takes the first val_queries_size as val and the next test_queries_size as test. With
		query_split false, val and test are both all the queries. It reads the split settings from the query benchmark
		args, so it raises ValueError if the layer was built without them.
		"""
		query_benchmark_args = self.query_benchmark_args
		if query_benchmark_args is None:
			raise ValueError("split_queries needs query_benchmark_args. Build QueryTransformation with them.")

		if not query_benchmark_args.query_split:
			print(f"No query split : val and test are all {len(queries)} queries")
			return queries, queries

		val_queries_size = query_benchmark_args.val_queries_size
		test_queries_size = query_benchmark_args.test_queries_size
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
