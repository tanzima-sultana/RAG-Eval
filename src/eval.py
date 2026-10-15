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
from l1_ingestion.ingestion import ingestion
from l2_chunking.cache import load_chunks
from l4_indexing import cache as index_cache
from qna.qna import get_eval_qna
from l5_query.query import get_queries
from l6_retrieval.retrieval import Retrieval
from l7_context.context import get_context_list
from l8_prompt.prompt import get_prompt_list
from l9_generation.generation import get_generated_list
from l10_evaluation.summary import print_retrieval_summary
from l10_evaluation.retrieval_evaluation import RetrievalEvaluation
from l5_query.schema import Query
from l6_retrieval.schema import Retrieved
from l7_context.schema import Context
from l8_prompt.schema import Prompt
from l9_generation.schema import Generated
from l10_evaluation.schema import RetrievalQuality

def load_eval_inputs(args, output_path: str):
	"""
	Input: args, the parsed eval arguments; output_path, the dataset's output folder.
	Output: the l1 documents, the eval QnA, the tuning queries and the held-out queries.
	What it does: loads the documents, eval QnA and l5 queries from their caches and splits the queries.
	"""
	ingest_args = make_ingest_args(args)
	eval_qna_args = make_eval_qna_args(args)
	qna_llm_args = make_llm_args(args.qna_llm, args.qna_llm_model, args.qna_input_price, args.qna_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	qna_kg_llm_args = make_llm_args(args.qna_kg_llm, args.qna_kg_llm_model, args.qna_kg_input_price, args.qna_kg_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	query_args = make_query_args(args)
	query_llm_args = make_llm_args(args.query_llm, args.query_llm_model, args.query_input_price, args.query_output_price, args.llm_max_tokens, args.llm_timeout_seconds)

	# 1. Load the l1 documents
	# The same ingestion call as build.py; it returns the documents from the l1 cache
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

def run_retrieval(args, output_path: str, doc_ids: list[str], tuning_queries: list[Query], retrieval_evaluation: RetrievalEvaluation) -> list[tuple[Retrieved, RetrievalQuality]]:
	"""
	Input: args, the parsed eval arguments; output_path; doc_ids, the l1 document ids; tuning_queries; and
	retrieval_evaluation, which scores each Retrieved.
	Output: one (Retrieved, RetrievalQuality) per tuning query and setup, plain and reranked.
	What it does: retrieves, reranks and scores, one chunk type and one index at a time.
	"""
	# Retrieval settings do not depend on the chunk type
	retrieval_args = make_retrieval_args(args)

	# List of (retrieved, quality) tuples, one per Retrieved
	retrieval_qualities = []
	# One chunk type at a time, so only its chunks and indexes are in memory
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		chunking_args = make_chunking_args(args, chunk_type)

		# 5. Load the l2 chunks
		# Load-only; raises an error if a document has no chunk file
		print("\n\nLoading chunks")
		s5 = time.time()
		chunks = load_chunks(output_path, chunking_args, doc_ids)
		e5 = time.time() - s5
		print(f"Chunks loaded : {len(chunks)} from {len(doc_ids)} documents")
		print("\nLoading chunks time : ", e5)

		# One Retrieval per chunk type. index_type is left empty here; each Retrieval method sets its own
		retrieval = Retrieval(retrieval_args, chunking_args, make_index_args(args, ""), args.tuning_queries_size, output_path)

		# Kept for hybrid: dense outputs of each dense index, and the BM25 outputs
		retrieved_dense_outputs = {}
		retrieved_bm25_outputs = []

		# One index at a time, so only one index is in memory
		for index_type in args.index_types:
			# 6. Load the l4 index
			# Raises an error if the index is missing or was built with a different dataset size
			print(f"\n\nLoading index : {index_type}")
			s6 = time.time()
			index_args = make_index_args(args, index_type)
			index = index_cache.load_index(output_path, chunking_args, index_args, args.dataset_size)
			e6 = time.time() - s6
			print("\nLoading index time : ", e6)

			# 7. Dense retrieval on the tuning queries. 
			if constants.DENSE_RETRIEVAL in retrieval_args.retrieval_types:
				s7 = time.time()
				dense_outputs = retrieval.compute_dense_retrieval(tuning_queries, chunk_type, index_type, index)
				if len(dense_outputs) > 0:
					retrieved_dense_outputs[index_type] = dense_outputs
					print(f"\n\nDense retrieved outputs on {index_type} : {len(dense_outputs)}")
					print("\nDense retrieval time : ", time.time() - s7)

					# 8. Retrieval quality evaluation of dense
					for retrieved in dense_outputs:
						retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))
					
					# 9. Rerank
					if retrieval_args.reranking:
						reranked_dense_outputs = retrieval.get_reranked_dense_retrieval(tuning_queries, chunks, index_type, dense_outputs)

						# 10. Retrieval quality evaluation of reranked_dense
						for retrieved in reranked_dense_outputs:
							retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))

			# 11. BM25 retrieval on the tuning queries. 
			if constants.BM25_RETRIEVAL in retrieval_args.retrieval_types:
				s9 = time.time()
				bm25_outputs = retrieval.compute_bm25_retrieval(tuning_queries, chunk_type, index_type, index)
				if len(bm25_outputs) > 0:
					retrieved_bm25_outputs = bm25_outputs
					print(f"\n\nBM25 retrieved outputs on {index_type} : {len(bm25_outputs)}")
					print("\nBM25 retrieval time : ", time.time() - s9)

					# 12. Retrieval quality evaluation of BM25
					for retrieved in bm25_outputs:
						retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))

					# 13. Rerank
					if retrieval_args.reranking:
						reranked_bm25_outputs = retrieval.get_reranked_bm25_retrieval(tuning_queries, chunks, bm25_outputs)

						# 14. Retrieval quality evaluation of reranked_bm25
						for retrieved in reranked_bm25_outputs:
							retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))

		# 15. Hybrid: each dense index's outputs fused with the BM25 outputs, only if both exist and hybrid is given
		if len(retrieved_dense_outputs) > 0 and len(retrieved_bm25_outputs) > 0 and constants.HYBRID_RETRIEVAL in args.retrieval_types:

			for index_type in retrieved_dense_outputs:
				s11 = time.time()
				hybrid_outputs = retrieval.compute_hybrid_retrieval(chunk_type, index_type, retrieved_dense_outputs[index_type], retrieved_bm25_outputs)
				print(f"\n\nHybrid retrieved outputs on {index_type}+{constants.BM25_INDEX} : {len(hybrid_outputs)}")
				print("\nHybrid retrieval time : ", time.time() - s11)

				# 16. Retrieval quality evaluation of hybrid
				for retrieved in hybrid_outputs:
					retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))

				# 17. Rerank
				if retrieval_args.reranking:
					reranked_hybrid_outputs = retrieval.get_reranked_hybrid_retrieval(tuning_queries, chunks, index_type, hybrid_outputs)

					# 18. Retrieval quality evaluation of reranked_hybrid
					for retrieved in reranked_hybrid_outputs:
						retrieval_qualities.append((retrieved, retrieval_evaluation.compute_retrieval_quality(retrieved, chunks)))

	return retrieval_qualities

def run_context(args, output_path: str, doc_ids: list[str], filtered_retrieval_qualities: list[tuple[Retrieved, RetrievalQuality]]) -> tuple[list[Retrieved], list[Context]]:
	"""
	Input: args, the parsed eval arguments; output_path; doc_ids, the l1 document ids; and the filtered
	(Retrieved, RetrievalQuality).
	Output: retrieved_list and context_list, in the same order, so context_list[i] belongs to retrieved_list[i].
	What it does: builds the context of each filtered Retrieved, one chunk type at a time.
	"""
	# 21. Context, one chunk type at a time, so only its chunks are in memory
	# retrieved_list and context_list are filled in the same loop, so context_list[i] belongs to
	# retrieved_list[i]
	print("\n\nContext")
	s21 = time.time()
	retrieved_list = []
	context_list = []
	for chunk_type in args.chunk_types:

		# Load retrieved for a specific chunk type
		chunk_type_retrieved_list = []
		for retrieved, retrieval_quality in filtered_retrieval_qualities:
			if retrieved.chunk_type == chunk_type:
				chunk_type_retrieved_list.append(retrieved)
		if len(chunk_type_retrieved_list) == 0:
			continue
		
		# Load chunks and get context list for a specific chunk type
		chunking_args = make_chunking_args(args, chunk_type)
		chunks = load_chunks(output_path, chunking_args, doc_ids)
		chunk_type_context_list = get_context_list(chunk_type_retrieved_list, chunks)

		# Update global retrieved list and context list
		retrieved_list.extend(chunk_type_retrieved_list)
		context_list.extend(chunk_type_context_list)
		print(f"Contexts built for {chunk_type} : {len(chunk_type_context_list)}")

	print(f"Context list : {len(context_list)}")
	print("\nContext time : ", time.time() - s21)

	# retrieved_list is returned too, because it is regrouped by chunk type: its order can differ from
	# filtered_retrieval_qualities when the kept setups use different chunk types
	return retrieved_list, context_list

def run_prompt(context_list: list[Context], tuning_queries: list[Query]) -> list[Prompt]:
	"""
	Input: context_list, the l7 contexts; tuning_queries, the l5 tuning queries.
	Output: prompt_list, in the same order, so prompt_list[i] belongs to context_list[i] and retrieved_list[i].
	What it does: builds the system prompt and user prompt of each context, with its original question.
	"""
	# 22. Prompt
	print("\n\nPrompt")
	s22 = time.time()
	prompt_list = get_prompt_list(context_list, tuning_queries)

	print(f"Prompt list : {len(prompt_list)}")
	print("\nPrompt time : ", time.time() - s22)

	return prompt_list

def run_generation(args, output_path: str, prompt_list: list[Prompt]) -> list[Generated]:
	"""
	Input: args, the parsed eval arguments; output_path; prompt_list, the l8 prompts.
	Output: generated_list, in the same order, so generated_list[i] belongs to prompt_list[i], context_list[i]
	and retrieved_list[i].
	What it does: gets the LLM answer of each prompt, from the cache or from the generation LLM.
	"""
	# 23. Generation
	print("\n\nGeneration")
	s23 = time.time()
	generation_llm_args = make_llm_args(args.generation_llm, args.generation_llm_model, args.generation_input_price, args.generation_output_price, args.llm_max_tokens, args.llm_timeout_seconds)
	generated_list = get_generated_list(output_path, prompt_list, generation_llm_args)

	print(f"Generated list : {len(generated_list)}")
	print("\nGeneration time : ", time.time() - s23)

	return generated_list

if __name__ == "__main__":
	print("\n ------------- RAG Eval -------- \n")
	start = time.time()

	args = parse_eval_args()
	print(args)

	ingest_args = make_ingest_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 1. Documents, eval QnA and queries, split into tuning and held-out
	documents, eval_qna, tuning_queries, held_out_queries = load_eval_inputs(args, output_path)

	doc_ids = []
	for doc_id in documents:
		doc_ids.append(doc_id)

	# Finds each Retrieved's eval question by its id
	retrieval_evaluation = RetrievalEvaluation(eval_qna)

	# 2. Retrieval, reranking and retrieval quality
	retrieval_qualities = run_retrieval(args, output_path, doc_ids, tuning_queries, retrieval_evaluation)

	# 3. Retrieval quality of every setup, averaged over the tuning queries
	print_retrieval_summary(retrieval_qualities)

	# 4. Keep the best n_retrieval_quality plain setups and the best n_retrieval_quality reranked setups
	filtered_retrieval_qualities = retrieval_evaluation.get_filtered_retrieval_qualities(args.n_retrieval_quality, retrieval_qualities)
	print_retrieval_summary(filtered_retrieval_qualities)

	# 5. Context
	# retrieved_list holds the filtered Retrieved regrouped by chunk type, in the order of context_list.
	retrieved_list, context_list = run_context(args, output_path, doc_ids, filtered_retrieval_qualities)

	# 6. Prompt
	prompt_list = run_prompt(context_list, tuning_queries)

	# 7. Generation
	# Off until l9 is finished
	# generated_list = run_generation(args, output_path, prompt_list)

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
