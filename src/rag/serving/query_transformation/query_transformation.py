import random

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths
from common.args import OllamaArgs
from common.args import QueryArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query_transformation.cache import load_query_entry
from rag.serving.query_transformation.cache import save_query_entry
from rag.serving.query_transformation.single_query import transform_single_query
from schema.query import Query


class QueryTransformation:
	def __init__(self, query_args: QueryArgs, query_path: str):
		"""
		Input: query_args, the query transformation settings; query_path, the folder the query files are saved in,
		used as given.
		Output: nothing.
		What it does: stores both.
		"""
		self.query_args = query_args
		self.query_path = query_path

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

	def transform_query(self, question_ids: list[str], questions: list[str]) -> list[Query]:
		"""
		Input: question_ids, one id per question; questions, the question texts, in the same order.
		Output: one Query per question, in the same order, with the question's id and only the query types in
		query_types. The other query types are None.
		What it does: looks up each query type of each question in its own saved file, named by the hash of what decides
		it. Only the query types not saved are made and saved, each as its own file. The embedding model, and the query
		LLM only if a missing query type needs it, are loaded once at the first miss.
		"""
		query_args = self.query_args

		# 1. Validate the settings and the inputs
		self.validate()
		if len(question_ids) != len(questions):
			raise ValueError(f"question_ids ({len(question_ids)}) and questions ({len(questions)}) must be the same length.")

		query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
		embedding_model = None
		query_llm_provider = None
		loaded_count = 0
		generated_count = 0
		multi_dont_know_count = 0
		hyde_dont_know_count = 0

		queries = []
		for question_index in range(len(questions)):
			question_id = question_ids[question_index]
			question = questions[question_index]

			# 2. Load the saved query types of this question, and list the missing ones
			results = {}
			missing_query_types = []
			for query_type in query_args.query_types:
				query_file = paths.get_query_file_path(self.query_path, query_args, query_type, question)
				result = load_query_entry(query_file)
				if result is None:
					missing_query_types.append(query_type)
				else:
					results[query_type] = result
					loaded_count = loaded_count + 1

			# 3. Make only the missing query types, and save each as its own file
			if len(missing_query_types) > 0:
				if embedding_model is None:
					# Normalized like the chunk embeddings, so a query vector matches the chunk vectors
					device = "cuda" if torch.cuda.is_available() else "cpu"
					embedding_model = HuggingFaceEmbeddings(
						model_name=query_args.embedding_model,
						model_kwargs={"device": device},
						encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
					)
				# The LLM is built only when a missing type needs one. LLMProvider validates it, so a paid model needs
				# ALLOW_PAID_API=1
				if query_llm_provider is None and missing_query_types != [constants.ORIGINAL_QUERY]:
					ollama_args = OllamaArgs(
						ollama_api_base=query_args.ollama_api_base,
						ollama_context_length=query_args.ollama_context_length
					)
					query_llm_provider = LLMProvider(query_args.query_llm_args, ollama_args)

				new_query = transform_single_query(
					query_args, question_id, question, missing_query_types, query_llm_provider, embedding_model
				)
				for query_type in missing_query_types:
					if query_type == constants.ORIGINAL_QUERY:
						result = new_query.original_qus
						result_llm_model = None
					elif query_type == constants.REWRITTEN_QUERY:
						result = new_query.rewritten_qus
						result_llm_model = query_args.query_llm_args.llm_model
					elif query_type == constants.MULTI_QUERY:
						result = new_query.multi_qus
						result_llm_model = query_args.query_llm_args.llm_model
						if len(result) == 0:
							multi_dont_know_count = multi_dont_know_count + 1
					else:
						result = new_query.hyde_ans
						result_llm_model = query_args.query_llm_args.llm_model
						if result.ans == constants.I_DONT_KNOW:
							hyde_dont_know_count = hyde_dont_know_count + 1

					query_file = paths.get_query_file_path(self.query_path, query_args, query_type, question)
					save_query_entry(query_file, query_type, question, result, query_args.embedding_model, result_llm_model)
					results[query_type] = result
					generated_count = generated_count + 1

			# 4. One Query of this question, holding only the requested query types
			queries.append(Query(
				id=question_id,
				original_qus=results.get(constants.ORIGINAL_QUERY),
				rewritten_qus=results.get(constants.REWRITTEN_QUERY),
				multi_qus=results.get(constants.MULTI_QUERY),
				hyde_ans=results.get(constants.HYDE_QUERY),
				normalized=True,
				embedding_model=query_args.embedding_model,
				llm=query_llm,
				llm_model=query_llm_model
			))

		# 5. What was loaded and what was made
		print(f"Query folder : {self.query_path}")
		print(f"Queries : {len(queries)}, query types loaded from cache : {loaded_count}, made : {generated_count}")
		if generated_count > 0:
			print(f"Multi-query I don't know : {multi_dont_know_count}")
			print(f"HyDE I don't know : {hyde_dont_know_count}")
		if query_llm_provider is not None:
			query_llm_provider.compute_cost()

		return queries

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
