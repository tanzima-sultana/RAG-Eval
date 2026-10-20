import random
import time

from common import constants
from common import paths
from common.args import parse_eval_args
from common.args import make_ingest_args
from common.args import make_chunking_args
from common.args import make_index_args
from common.args import make_eval_qna_args
from common.args import make_query_args
from common.args import make_retrieval_args
from common.args import make_llm_args
from common.args import make_generation_args
from common.args import make_answer_eval_args
from common.args import make_analysis_args
from stages.build.ingestion.ingestion import ingestion
from stages.build.chunking.cache import load_chunks
from stages.build.indexing import cache as index_cache
from stages.question.qna.qna import get_eval_qna
from stages.question.query.query import get_queries
from stages.evaluation.retrieval.retrieval import Retrieval
from stages.evaluation.context.context import get_context_list
from stages.evaluation.prompt.prompt import get_prompt_list
from stages.evaluation.prompt.prompt_template import SYSTEM_PROMPT
from stages.evaluation.prompt.prompt_template import USER_PROMPT_TEMPLATE
from stages.evaluation.generation.generation import get_generated_list
from l10_evaluation.retrieval_evaluation import RetrievalEvaluator
from l10_evaluation.answer_evaluation import get_answer_evaluations
from analysis.analysis import analyze_run
from stages.question.query.schema import Query
from stages.evaluation.retrieval.schema import Retrieved
from stages.evaluation.context.schema import Context
from stages.evaluation.prompt.schema import Prompt
from stages.evaluation.generation.schema import Generated
from l10_evaluation.schema import RetrievalEvaluation
from l10_evaluation.schema import AnswerEvaluation
from analysis.schema import AnalysisRun
from stages.question.qna.schema import EvalQnA

def load_eval_inputs(args, output_path: str):
	"""
	Input: args, the parsed eval arguments; output_path, the dataset's output folder.
	Output: the ingested documents, the eval QnA, the tuning queries and the held-out queries.
	What it does: loads the documents, eval QnA and queries from their caches and splits the queries.
	"""
	ingest_args = make_ingest_args(args)
	eval_qna_args = make_eval_qna_args(args)
	qna_llm_args = make_llm_args(args.qna_llm, args.qna_llm_model, args.qna_input_price, args.qna_output_price, args.qna_cache_write_price, args.qna_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)
	qna_kg_llm_args = make_llm_args(args.qna_kg_llm, args.qna_kg_llm_model, args.qna_kg_input_price, args.qna_kg_output_price, args.qna_kg_cache_write_price, args.qna_kg_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)
	query_args = make_query_args(args)
	query_llm_args = make_llm_args(args.query_llm, args.query_llm_model, args.query_input_price, args.query_output_price, args.query_cache_write_price, args.query_cache_read_price, args.llm_max_tokens, args.llm_timeout_seconds, args.llm_cache_ttl)

	# 1. Load the ingested documents
	# The same ingestion call as build.py; it returns the documents from the ingestion cache
	print("\n\nLoading documents")
	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nLoading documents time : ", e1)

	# 2. Load eval QnA
	# Reuses {output_path}/eval_qna_{n_sample_documents}_{n_sample_evals}_{qna_llm_model}_{eval_qna_model}.json if it exists
	print("\n\nLoading eval QnA")
	s2 = time.time()
	eval_qna = get_eval_qna(output_path, documents, eval_qna_args, qna_llm_args, qna_kg_llm_args)
	e2 = time.time() - s2
	print(f"Eval QnA loaded : {len(eval_qna)}")
	print("\nLoading eval QnA time : ", e2)

	# 3. Query
	print("\n\nLoading Query")
	s3 = time.time()
	queries = get_queries(output_path, eval_qna, eval_qna_args, query_args, query_llm_args)
	e3 = time.time() - s3
	print(f"Query loaded : {len(queries)}")
	print("\nLoading Query time : ", e3)

	# 4. Split the queries into tuning and held-out
	# Shuffled with a fixed seed, so every run gets the same split
	if len(queries) <= args.tuning_queries_size:
		raise ValueError(
			f"Only {len(queries)} queries, need more than {args.tuning_queries_size} to keep some held out"
		)
	shuffled_queries = []
	for query in queries:
		shuffled_queries.append(query)
	random_generator = random.Random(constants.TUNING_SPLIT_SEED)
	random_generator.shuffle(shuffled_queries)
	tuning_queries = shuffled_queries[:args.tuning_queries_size]
	held_out_queries = shuffled_queries[args.tuning_queries_size:]
	print(f"\n\nTuning queries : {len(tuning_queries)}, held-out queries : {len(held_out_queries)}")

	return documents, eval_qna, tuning_queries, held_out_queries

def run_retrieval_and_evaluation(args, output_path: str, doc_ids: list[str], queries: list[Query], retrieval_evaluator: RetrievalEvaluator) -> list[tuple[Retrieved, RetrievalEvaluation]]:
	"""
	Input: args, the parsed eval arguments; output_path; doc_ids, the ingested document ids; queries, the queries of this
	run, tuning or held-out; and retrieval_evaluator, which scores each Retrieved.
	Output: one (Retrieved, RetrievalEvaluation) per query and setup, plain and reranked.
	What it does: retrieves, reranks and scores, one chunk type and one index at a time.
	"""
	# Retrieval settings do not depend on the chunk type
	retrieval_args = make_retrieval_args(args)

	# Dense and BM25 outputs are computed for their own setups and for hybrid, which fuses them, but kept as setups
	# only when given themselves
	keep_dense = constants.DENSE_RETRIEVAL in retrieval_args.retrieval_types
	keep_bm25 = constants.BM25_RETRIEVAL in retrieval_args.retrieval_types
	keep_hybrid = constants.HYBRID_RETRIEVAL in retrieval_args.retrieval_types
	compute_dense = keep_dense or keep_hybrid
	compute_bm25 = keep_bm25 or keep_hybrid

	# List of (retrieved, retrieval_evaluation) tuples, one per Retrieved
	retrieval_evaluations = []
	# One chunk type at a time, so only its chunks and indexes are in memory
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		chunking_args = make_chunking_args(args, chunk_type)

		# 5. Load the chunks
		# Load-only; raises an error if a document has no chunk file
		print("\n\nLoading chunks")
		s5 = time.time()
		chunks = load_chunks(output_path, chunking_args, doc_ids)
		e5 = time.time() - s5
		print(f"Chunks loaded : {len(chunks)} from {len(doc_ids)} documents")
		print("\nLoading chunks time : ", e5)

		# One Retrieval per chunk type. index_type is left empty here; each Retrieval method sets its own
		retrieval = Retrieval(retrieval_args, chunking_args, make_index_args(args, ""), args.query_split, args.tuning_queries_size, output_path)

		# Kept for hybrid: dense outputs of each dense index, and the BM25 outputs
		retrieved_dense_outputs = {}
		retrieved_bm25_outputs = []

		# One index at a time, so only one index is in memory
		for index_type in args.index_types:
			# 6. Load the index
			# Raises an error if the index is missing or was built with a different dataset size
			print(f"\n\nLoading index : {index_type}")
			s6 = time.time()
			index_args = make_index_args(args, index_type)
			index = index_cache.load_index(output_path, chunking_args, index_args, args.dataset_size)
			e6 = time.time() - s6
			print("\nLoading index time : ", e6)

			# 7. Dense retrieval on this run's queries. Hybrid fuses the dense outputs, so they are also computed when
			# only hybrid is given, but then not evaluated or kept as a setup of their own
			if compute_dense:
				s7 = time.time()
				dense_outputs = retrieval.compute_dense_retrieval(queries, chunk_type, index_type, index)
				if len(dense_outputs) > 0:
					retrieved_dense_outputs[index_type] = dense_outputs
					print(f"\n\nDense retrieved outputs on {index_type} : {len(dense_outputs)}")
					print("\nDense retrieval time : ", time.time() - s7)

					if keep_dense:
						# 8. Retrieval evaluation of dense
						for retrieved in dense_outputs:
							retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

						# 9. Rerank
						if retrieval_args.reranking:
							reranked_dense_outputs = retrieval.get_reranked_dense_retrieval(queries, chunks, index_type, dense_outputs)

							# 10. Retrieval evaluation of reranked_dense
							for retrieved in reranked_dense_outputs:
								retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

			# 11. BM25 retrieval on this run's queries. Also computed for hybrid alone, like dense
			if compute_bm25:
				s9 = time.time()
				bm25_outputs = retrieval.compute_bm25_retrieval(queries, chunk_type, index_type, index)
				if len(bm25_outputs) > 0:
					retrieved_bm25_outputs = bm25_outputs
					print(f"\n\nBM25 retrieved outputs on {index_type} : {len(bm25_outputs)}")
					print("\nBM25 retrieval time : ", time.time() - s9)

					if keep_bm25:
						# 12. Retrieval evaluation of BM25
						for retrieved in bm25_outputs:
							retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

						# 13. Rerank
						if retrieval_args.reranking:
							reranked_bm25_outputs = retrieval.get_reranked_bm25_retrieval(queries, chunks, bm25_outputs)

							# 14. Retrieval evaluation of reranked_bm25
							for retrieved in reranked_bm25_outputs:
								retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

		# 15. Hybrid: each dense index's outputs fused with the BM25 outputs, only if both exist and hybrid is given
		if len(retrieved_dense_outputs) > 0 and len(retrieved_bm25_outputs) > 0 and keep_hybrid:

			for index_type in retrieved_dense_outputs:
				s11 = time.time()
				hybrid_outputs = retrieval.compute_hybrid_retrieval(chunk_type, index_type, retrieved_dense_outputs[index_type], retrieved_bm25_outputs)
				print(f"\n\nHybrid retrieved outputs on {index_type}+{constants.BM25_INDEX} : {len(hybrid_outputs)}")
				print("\nHybrid retrieval time : ", time.time() - s11)

				# 16. Retrieval evaluation of hybrid
				for retrieved in hybrid_outputs:
					retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

				# 17. Rerank
				if retrieval_args.reranking:
					reranked_hybrid_outputs = retrieval.get_reranked_hybrid_retrieval(queries, chunks, index_type, hybrid_outputs)

					# 18. Retrieval evaluation of reranked_hybrid
					for retrieved in reranked_hybrid_outputs:
						retrieval_evaluations.append((retrieved, retrieval_evaluator.compute_retrieval_evaluation(retrieved, chunks)))

	return retrieval_evaluations

def run_context(args, output_path: str, doc_ids: list[str], filtered_retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]]) -> tuple[list[Retrieved], list[RetrievalEvaluation], list[Context]]:
	"""
	Input: args, the parsed eval arguments; output_path; doc_ids, the ingested document ids; and the filtered
	(Retrieved, RetrievalEvaluation).
	Output: retrieved_list, retrieval_evaluation_list and context_list, in the same order, so retrieval_evaluation_list[i]
	and context_list[i] belong to retrieved_list[i].
	What it does: builds the context of each filtered Retrieved, one chunk type at a time.
	"""
	# 21. Context, one chunk type at a time, so only its chunks are in memory
	# retrieved_list, retrieval_evaluation_list and context_list are filled in the same loop, so they stay in the
	# same order
	print("\n\nContext")
	s21 = time.time()
	retrieved_list = []
	retrieval_evaluation_list = []
	context_list = []
	for chunk_type in args.chunk_types:

		# Load retrieved and its retrieval evaluation for a specific chunk type
		chunk_type_retrieved_list = []
		chunk_type_retrieval_evaluation_list = []
		for retrieved, retrieval_evaluation in filtered_retrieval_evaluations:
			if retrieved.chunk_type == chunk_type:
				chunk_type_retrieved_list.append(retrieved)
				chunk_type_retrieval_evaluation_list.append(retrieval_evaluation)
		if len(chunk_type_retrieved_list) == 0:
			continue
		
		# Load chunks and get context list for a specific chunk type
		chunking_args = make_chunking_args(args, chunk_type)
		chunks = load_chunks(output_path, chunking_args, doc_ids)
		chunk_type_context_list = get_context_list(chunk_type_retrieved_list, chunks)

		# Update global retrieved list, retrieval evaluation list and context list
		retrieved_list.extend(chunk_type_retrieved_list)
		retrieval_evaluation_list.extend(chunk_type_retrieval_evaluation_list)
		context_list.extend(chunk_type_context_list)
		print(f"Contexts built for {chunk_type} : {len(chunk_type_context_list)}")

	print(f"Context list : {len(context_list)}")
	print("\nContext time : ", time.time() - s21)

	# retrieved_list and retrieval_evaluation_list are returned too, because they are regrouped by chunk type: their
	# order can differ from filtered_retrieval_evaluations when the kept setups use different chunk types
	return retrieved_list, retrieval_evaluation_list, context_list

def run_prompt(context_list: list[Context], queries: list[Query]) -> list[Prompt]:
	"""
	Input: context_list, the contexts; queries, the queries of this run, tuning or held-out.
	Output: prompt_list, in the same order, so prompt_list[i] belongs to context_list[i] and retrieved_list[i].
	What it does: builds the system prompt and user prompt of each context, with its original question.
	"""
	# 22. Prompt
	print("\n\nPrompt")
	s22 = time.time()
	prompt_list = get_prompt_list(context_list, queries)

	print(f"Prompt list : {len(prompt_list)}")
	print("\nPrompt time : ", time.time() - s22)

	return prompt_list

def run_generation(args, output_path: str, prompt_list: list[Prompt]) -> list[Generated]:
	"""
	Input: args, the parsed eval arguments; output_path; prompt_list, the prompts.
	Output: generated_list, in the same order, so generated_list[i] belongs to prompt_list[i], context_list[i]
	and retrieved_list[i].
	What it does: gets the LLM answer of each prompt, from the cache or from the generation LLM.
	"""
	# 23. Generation
	print("\n\nGeneration")
	s23 = time.time()
	generation_args = make_generation_args(args)
	generated_list = get_generated_list(output_path, prompt_list, generation_args)

	print(f"Generated list : {len(generated_list)}")
	print("\nGeneration time : ", time.time() - s23)

	return generated_list

def run_answer_evaluation(args, output_path: str, eval_qna: list[EvalQnA], context_list: list[Context], generated_list: list[Generated]) -> list[AnswerEvaluation]:
	"""
	Input: args, the parsed eval arguments; output_path; eval_qna, the eval questions; context_list and generated_list,
	the contexts and generated answers.
	Output: answer_evaluations, in the same order, so answer_evaluations[i] belongs to generated_list[i], context_list[i]
	and retrieved_list[i].
	What it does: scores each answer's correctness, faithfulness and relevance, from the cache or the judge LLM.
	"""
	# 24. Answer evaluation
	print("\n\nAnswer evaluation")
	s24 = time.time()
	answer_eval_args = make_answer_eval_args(args)
	answer_evaluations = get_answer_evaluations(output_path, eval_qna, context_list, generated_list, answer_eval_args)

	print(f"Answer evaluations : {len(answer_evaluations)}")
	print("\nAnswer evaluation time : ", time.time() - s24)

	return answer_evaluations

def run_analysis(args, output_path: str, retrieval_evaluations: list[tuple[Retrieved, RetrievalEvaluation]], retrieved_list: list[Retrieved], retrieval_evaluation_list: list[RetrievalEvaluation], generated_list: list[Generated], answer_evaluations: list[AnswerEvaluation], eval_qna: list[EvalQnA]) -> AnalysisRun:
	"""
	Input: args, the parsed eval arguments; output_path; retrieval_evaluations, every (Retrieved, RetrievalEvaluation);
	retrieved_list, retrieval_evaluation_list, generated_list and answer_evaluations, in the same order; eval_qna,
	the eval questions.
	Output: the run's AnalysisRun.
	What it does: summarises, diagnoses and compares the run, and saves it with its report.
	"""
	# 25. Analysis
	print("\n\nAnalysis")
	s25 = time.time()
	analysis_args = make_analysis_args(args)
	# Every setting of the run, saved with it so later runs can show which settings changed. The prompts are not
	# settings in the config, so they are added, and a run with another prompt counts as a different experiment
	settings = dict(vars(args))
	settings["system_prompt"] = SYSTEM_PROMPT
	settings["user_prompt_template"] = USER_PROMPT_TEMPLATE
	analysis_run = analyze_run(output_path, analysis_args, settings, retrieval_evaluations, retrieved_list, retrieval_evaluation_list, generated_list, answer_evaluations, eval_qna)

	print("\nAnalysis time : ", time.time() - s25)

	return analysis_run

def run_eval(args) -> AnalysisRun:
	"""
	Input: args, the parsed eval arguments, holding the settings of the build to evaluate.
	Output: the run's AnalysisRun.
	What it does: loads the build, eval QnA and queries, then runs retrieval to l10 evaluation and the analysis on
	the query split chosen by query_split.
	"""
	ingest_args = make_ingest_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 1. Documents, eval QnA and queries, split into tuning and held-out
	documents, eval_qna, tuning_queries, held_out_queries = load_eval_inputs(args, output_path)

	# The tuning queries while tuning, the held-out queries for the one final check
	if args.query_split == "held_out":
		queries = held_out_queries
	else:
		queries = tuning_queries
	print(f"Query split : {args.query_split}, queries used : {len(queries)}")

	doc_ids = []
	for doc_id in documents:
		doc_ids.append(doc_id)

	# Finds each Retrieved's eval question by its id
	retrieval_evaluator = RetrievalEvaluator(eval_qna)

	# 2. Retrieval, reranking and retrieval evaluation
	# retrieval_evaluations is a list of (Retrieved, RetrievalEvaluation) tuples, one per eval question and setup
	retrieval_evaluations = run_retrieval_and_evaluation(args, output_path, doc_ids, queries, retrieval_evaluator)
	# Nothing retrieved means the retrieval and index settings do not fit together, e.g. dense without a dense index,
	# so the run stops here instead of going on with empty lists
	if len(retrieval_evaluations) == 0:
		raise ValueError(
			f"No retrieved outputs for retrieval_types {args.retrieval_types} on index_types {args.index_types} "
			f"and chunk_types {args.chunk_types}. Check that each retrieval type has an index it can search."
		)

	# 3. Keep the best n_retrieval_evaluation plain setups and the best n_retrieval_evaluation reranked setups
	# filtered_retrieval_evaluations is also a list of (Retrieved, RetrievalEvaluation) tuples
	filtered_retrieval_evaluations = retrieval_evaluator.get_filtered_retrieval_evaluations(args.n_retrieval_evaluation, retrieval_evaluations)

	# 4. Context
	# retrieved_list and retrieval_evaluation_list hold the filtered Retrieved and their RetrievalEvaluation regrouped by
	# chunk type, in the order of context_list
	retrieved_list, retrieval_evaluation_list, context_list = run_context(args, output_path, doc_ids, filtered_retrieval_evaluations)

	# 5. Prompt
	prompt_list = run_prompt(context_list, queries)

	# 6. Generation
	generated_list = run_generation(args, output_path, prompt_list)

	# 7. Answer evaluation
	answer_evaluations = run_answer_evaluation(args, output_path, eval_qna, context_list, generated_list)

	# 8. Analysis: summaries, diagnosis, worst answers and comparison with earlier runs, saved with the run's report
	analysis_run = run_analysis(args, output_path, retrieval_evaluations, retrieved_list, retrieval_evaluation_list, generated_list, answer_evaluations, eval_qna)

	return analysis_run

if __name__ == "__main__":
	print("\n ------------- RAG Eval -------- \n")
	start = time.time()

	args = parse_eval_args()
	print(args)

	run_eval(args)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
