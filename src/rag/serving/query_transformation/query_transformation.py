import dataclasses
import random

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import EvalDatasetArgs
from common.args_temp import QueryArgs
from llm_provider.llm_provider import LLMProvider
from rag.serving.query_transformation.cache import load_queries
from rag.serving.query_transformation.cache import save_queries
from rag.serving.query_transformation.single_query import transform_question
from schema.eval_sample import EvalSample
from schema.query import Query


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

	def compute_queries(self, eval_samples: list[EvalSample], eval_dataset_args: EvalDatasetArgs, output_path: str) -> list[Query]:
		"""
		Input: eval_samples, the eval questions; eval_dataset_args, the settings of their eval dataset; output_path, the dataset's
		output folder.
		Output: one Query per eval question, holding only the query types in query_types.
		What it does: loads the saved queries, generates only the requested query types that are not saved, adds them
		to the saved queries and saves them. A saved query type is never generated again.
		"""
		query_args = self.query_args

		# 1. Validate the settings
		self.validate()

		# 2. Load the saved queries, or start one empty Query per eval question
		query_path = paths_temp.get_query_path(output_path, eval_dataset_args, query_args)
		print(f"Query folder : {query_path}")
		saved_queries = load_queries(query_path)
		if saved_queries is None:
			print(f"No saved queries for {len(eval_samples)} questions")
			query_llm, query_llm_model = query_args.query_llm_args.llm_model.split("/", 1)
			saved_queries = []
			for eval_sample in eval_samples:
				saved_queries.append(Query(
					id=eval_sample.id,
					original_qus=None,
					rewritten_qus=None,
					multi_qus=None,
					hyde_ans=None,
					normalized=True,
					embedding_model=query_args.embedding_model,
					llm=query_llm,
					llm_model=query_llm_model
				))

		# The saved queries are made from this eval dataset, so they must be in the same question order
		if len(saved_queries) != len(eval_samples):
			raise ValueError(
				f"The saved queries in {query_path} hold {len(saved_queries)} questions, but the eval dataset holds "
				f"{len(eval_samples)}. Delete the folder and run again."
			)
		for sample_index in range(len(eval_samples)):
			if saved_queries[sample_index].id != eval_samples[sample_index].id:
				raise ValueError(
					f"The saved queries in {query_path} are not in the eval dataset order at position {sample_index}. "
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
			print(f"Generating query types {missing_query_types} for {len(eval_samples)} questions")

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

			# Each question is transformed on its own, the same way as one live question, and only its missing query
			# types are copied into its saved query
			multi_dont_know_count = 0
			hyde_dont_know_count = 0
			for sample_index in range(len(eval_samples)):
				eval_sample = eval_samples[sample_index]
				new_query = transform_question(
					query_args, eval_sample.id, eval_sample.qus, missing_query_types, query_llm_provider, embedding_model
				)

				saved_query = saved_queries[sample_index]
				if constants.ORIGINAL_QUERY in missing_query_types:
					saved_query.original_qus = new_query.original_qus
				if constants.REWRITTEN_QUERY in missing_query_types:
					saved_query.rewritten_qus = new_query.rewritten_qus
				if constants.MULTI_QUERY in missing_query_types:
					saved_query.multi_qus = new_query.multi_qus
					if len(new_query.multi_qus) == 0:
						multi_dont_know_count = multi_dont_know_count + 1
				if constants.HYDE_QUERY in missing_query_types:
					saved_query.hyde_ans = new_query.hyde_ans
					if new_query.hyde_ans.ans == constants.I_DONT_KNOW:
						hyde_dont_know_count = hyde_dont_know_count + 1

				if (sample_index + 1) % 20 == 0 or sample_index + 1 == len(eval_samples):
					print(f"Questions transformed : {sample_index + 1} of {len(eval_samples)}")

			if constants.MULTI_QUERY in missing_query_types:
				print(f"Multi-query I don't know : {multi_dont_know_count} of {len(eval_samples)}")
			if constants.HYDE_QUERY in missing_query_types:
				print(f"HyDE I don't know : {hyde_dont_know_count} of {len(eval_samples)}")

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
