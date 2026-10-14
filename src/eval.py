import argparse
import random
import time

from common import constants
from common import paths
from common.args import IngestArgs
from common.args import ChunkingArgs
from common.args import IndexArgs
from common.args import EvalQnAArgs
from common.args import QueryArgs
from common.args import RetrievalArgs
from l1_ingestion.ingestion import ingestion
from l2_chunking.cache import load_chunks
from l4_indexing import cache as index_cache
from qna.qna import get_eval_qna
from l5_query.query import get_queries
from l6_retrieval.retrieval import Retrieval
from l7_context.context import get_context_list
from l10_evaluation.summary import print_retrieval_summary
from l10_evaluation.retrieval_evaluation import RetrievalEvaluation

def parse_args():
	"""
	Input: none. Reads the command line arguments passed to eval.py.
	Output: an argparse.Namespace with one attribute per argument.
	What it does: defines the arguments that identify which build to
	evaluate and parses them. They must have the same values the build was
	run with, because the chunk, embedding and index folders and the
	Qdrant collection name are built from them. Defaults match build.py.
	"""
	parser = argparse.ArgumentParser(description="RAG-Eval evaluation parameters")

	parser.add_argument(
		"--dataset_size",
		type=int,
		required=False,
		default=0,
		help="Number of documents the build was made with")
	parser.add_argument(
		"--input_path",
		type=str,
		required=False,
		default="",
		help="Input dataset local path or Hugging Face dataset id used by the build")
	parser.add_argument(
		"--output_path",
		type=str,
		required=False,
		default="",
		help="Base output folder the build wrote to")
	parser.add_argument(
		"--download_workers",
		type=int,
		required=False,
		default=constants.DOWNLOAD_WORKERS,
		help="Number of parallel download workers")
	parser.add_argument(
		"--max_batch_rows",
		type=int,
		required=False,
		default=constants.MAX_BATCH_ROWS,
		help="Maximum number of source rows per batch")
	parser.add_argument(
		"--hugging_face_split",
		type=str,
		required=False,
		default=constants.HUGGING_FACE_SPLIT,
		help="Hugging Face dataset split to read")
	parser.add_argument(
		"--parse_workers",
		type=int,
		required=False,
		default=constants.PARSE_WORKERS,
		help="Number of parallel parse workers")
	parser.add_argument(
		"--parse_timeout_seconds",
		type=int,
		required=False,
		default=constants.PARSE_TIMEOUT_SECONDS,
		help="Per-document parse timeout in seconds")
	parser.add_argument(
		"--dedup_threshold",
		type=float,
		required=False,
		default=constants.DEDUP_THRESHOLD,
		help="Near-duplicate similarity threshold")
	parser.add_argument(
		"--pdf_parser",
		type=str,
		required=False,
		default=constants.PDF_PARSER,
		help="Library used to parse PDF files: pypdf or pypdfium2")
	parser.add_argument(
		"--framework",
		type=str,
		required=False,
		default="",
		help="Framework the build used: langchain, or empty for the built-in layers")
	parser.add_argument(
		"--model_name",
		type=str,
		required=False,
		default="",
		help="SentenceTransformer model the build used")
	parser.add_argument(
		"--chunk_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.FIXED_CHUNKING, constants.SENTENCE_CHUNKING, constants.SEMANTIC_CHUNKING],
		help="Chunking strategies the build made, loaded one after another: fixed, sentence and/or semantic")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=False,
		default=constants.MIN_CHUNK_SIZE,
		help="Minimum chunk size the build used")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=False,
		default=constants.MAX_CHUNK_SIZE,
		help="Maximum chunk size the build used")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		required=False,
		default=constants.FIXED_CHUNK_OVERLAP,
		help="Fixed chunk overlap the build used")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=False,
		default=constants.SEMANTIC_THRESHOLD,
		help="Semantic threshold the build used")
	parser.add_argument(
		"--index_types",
		type=str,
		nargs="*",
		required=False,
		default=[constants.FAISS_FLATIP_INDEX, constants.FAISS_IVF_INDEX, constants.FAISS_HNSW_INDEX, constants.BM25_INDEX, constants.QDRANT_INDEX],
		help="Indexes the build made, loaded one after another: any of faiss_flatip, faiss_ivf, faiss_hnsw, bm25 and qdrant, or none")
	parser.add_argument(
		"--ivf_nlist",
		type=int,
		required=False,
		default=constants.IVF_NLIST,
		help="Number of IVF clusters the build used")
	parser.add_argument(
		"--hnsw_m",
		type=int,
		required=False,
		default=constants.HNSW_M,
		help="HNSW neighbours per node the build used")
	parser.add_argument(
		"--hnsw_ef_construction",
		type=int,
		required=False,
		default=constants.HNSW_EF_CONSTRUCTION,
		help="HNSW build candidate list size the build used")
	parser.add_argument(
		"--bm25_k1",
		type=float,
		required=False,
		default=constants.BM25_K1,
		help="BM25 k1 the build used")
	parser.add_argument(
		"--bm25_b",
		type=float,
		required=False,
		default=constants.BM25_B,
		help="BM25 b the build used")
	parser.add_argument(
		"--bm25_tokenizer",
		type=str,
		required=False,
		default=constants.BM25_TOKENIZER,
		help="BM25 tokenizer the build used")
	parser.add_argument(
		"--n_sample_evals",
		type=int,
		required=False,
		default=constants.N_EVAL_SAMPLES,
		help="Number of eval QnA items to load or generate")
	parser.add_argument(
		"--eval_qna_llm",
		type=str,
		required=False,
		default=constants.EVAL_QNA_LLM,
		help="LLM provider RAGAS uses to generate the eval QnA: claude")
	parser.add_argument(
		"--eval_qna_model",
		type=str,
		required=False,
		default=constants.EVAL_QNA_MODEL,
		help="Local HuggingFace embedding model RAGAS uses to build its knowledge graph")
	parser.add_argument(
		"--n_sample_documents",
		type=int,
		required=False,
		default=constants.SAMPLE_DOCUMENTS,
		help="Number of documents randomly picked for RAGAS to generate the eval QnA from")
	parser.add_argument(
		"--query_llm",
		type=str,
		required=False,
		default=constants.QUERY_LLM,
		help="Local LLM provider that transforms the eval questions: ollama")
	parser.add_argument(
		"--query_llm_model",
		type=str,
		required=False,
		default=constants.QUERY_LLM_MODEL,
		help="Local model the query LLM provider runs, e.g. llama3.1:8b")
	parser.add_argument(
		"--n_multi_qus",
		type=int,
		required=False,
		default=constants.N_MULTI_QUS,
		help="Number of phrasings the query LLM writes for each eval question")
	parser.add_argument(
		"--retrieval_types",
		type=str,
		nargs="+",
		required=False,
		default=[constants.DENSE_RETRIEVAL, constants.BM25_RETRIEVAL, constants.HYBRID_RETRIEVAL],
		help="Retrieval types to run, one after another: dense, bm25 and/or hybrid")
	parser.add_argument(
		"--top_k",
		type=int,
		required=False,
		default=constants.TOP_K,
		help="Number of chunks each retriever returns for a query")
	parser.add_argument(
		"--reranking",
		type=str,
		required=False,
		choices=["true", "false"],
		default="false",
		help="true to also rerank the top_k chunks of every retrieval with the cross-encoder, false not to")
	parser.add_argument(
		"--rerank_top_k",
		type=int,
		required=False,
		default=constants.RERANK_TOP_K,
		help="Number of chunks kept after the cross-encoder reranks the top_k chunks")
	parser.add_argument(
		"--tuning_queries_size",
		type=int,
		required=True,
		help="Number of eval questions used for tuning; the rest are held out for one final check")
	parser.add_argument(
		"--n_retrieval_quality",
		type=int,
		required=False,
		default=constants.N_RETRIEVAL_QUALITY,
		help="Number of best plain setups and of best reranked setups whose retrieved outputs go on to the next layers")

	return parser.parse_args()


def make_ingest_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an IngestArgs object holding the same values.
	What it does: copies each parsed command line value into the matching
	IngestArgs field.
	"""
	ingest_args = IngestArgs(
		dataset_size=parsed_args.dataset_size,
		input_path=parsed_args.input_path,
		output_path=parsed_args.output_path,
		download_workers=parsed_args.download_workers,
		max_batch_rows=parsed_args.max_batch_rows,
		hugging_face_split=parsed_args.hugging_face_split,
		parse_workers=parsed_args.parse_workers,
		parse_timeout_seconds=parsed_args.parse_timeout_seconds,
		dedup_threshold=parsed_args.dedup_threshold,
		pdf_parser=parsed_args.pdf_parser
	)
	return ingest_args


def make_chunking_args(parsed_args, chunk_type: str):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunk_type, one of parsed_args.chunk_types.
	Output: a ChunkingArgs object holding the chunking values of the build for chunk_type.
	What it does: copies each parsed chunking value into the matching
	ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		framework=parsed_args.framework,
		model_name=parsed_args.model_name,
		chunk_type=chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=parsed_args.fixed_chunk_overlap,
		semantic_threshold=parsed_args.semantic_threshold
	)
	return chunking_args


def make_index_args(parsed_args, index_type: str):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); index_type, one of
	parsed_args.index_types.
	Output: an IndexArgs object holding the index values of the build for index_type.
	What it does: copies each parsed index value into the matching
	IndexArgs field. qdrant_batch_size only matters while building, so it
	is set to its default.
	"""
	index_args = IndexArgs(
		index_type=index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer,
		qdrant_batch_size=constants.QDRANT_BATCH_SIZE
	)
	return index_args


def make_eval_qna_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an EvalQnAArgs object holding the QnA generator settings.
	What it does: copies the parsed LLM generator, eval embedding
	model name, number of sample documents and number of eval QnA items into
	the matching EvalQnAArgs fields.
	"""
	eval_qna_args = EvalQnAArgs(
		eval_qna_llm=parsed_args.eval_qna_llm,
		eval_qna_model=parsed_args.eval_qna_model,
		n_sample_documents=parsed_args.n_sample_documents,
		n_sample_evals=parsed_args.n_sample_evals
	)
	return eval_qna_args


def make_query_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a QueryArgs object holding the query transformation settings.
	What it does: copies the build's embedding model (model_name), the parsed
	query LLM provider, its model and the number of multi-query phrasings into
	the matching QueryArgs fields.
	"""
	query_args = QueryArgs(
		embedding_model=parsed_args.model_name,
		query_llm=parsed_args.query_llm,
		query_llm_model=parsed_args.query_llm_model,
		n_multi_qus=parsed_args.n_multi_qus
	)
	return query_args


def make_retrieval_args(parsed_args):
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: a RetrievalArgs object holding the retrieval settings.
	What it does: copies the parsed framework, retrieval_types, top_k, reranking and rerank_top_k into the matching RetrievalArgs fields.
	"""
	retrieval_args = RetrievalArgs(
		framework=parsed_args.framework,
		retrieval_types=parsed_args.retrieval_types,
		top_k=parsed_args.top_k,
		reranking=parsed_args.reranking == "true",
		rerank_top_k=parsed_args.rerank_top_k
	)
	return retrieval_args

if __name__ == "__main__":
	print("\n ------------- RAG Eval -------- \n")
	start = time.time()

	args = parse_args()
	print(args)

	ingest_args = make_ingest_args(args)
	eval_qna_args = make_eval_qna_args(args)
	query_args = make_query_args(args)
	output_path = paths.get_output_path(ingest_args.input_path, ingest_args.output_path)

	# 1. Load the l1 documents
	# The same ingestion call as build.py; it returns the documents from the l1 cache
	print("\n\nLoading documents")
	s1 = time.time()
	documents = ingestion(ingest_args)
	e1 = time.time() - s1
	print("\nLoading documents time : ", e1)

	# 2. Load eval QnA
	# Reuses {output_path}/eval_qna_{n_sample_documents}_{n_sample_evals}_{eval_qna_llm}_{eval_qna_model}.json if it exists
	print("\n\nLoading eval QnA")
	s2 = time.time()
	eval_qna = get_eval_qna(output_path, documents, eval_qna_args)
	e2 = time.time() - s2
	print(f"Eval QnA loaded : {len(eval_qna)}")
	print("\nLoading eval QnA time : ", e2)

	# 3. Query
	print("\n\nLoading Query")
	s3 = time.time()
	queries = get_queries(output_path, eval_qna, eval_qna_args, query_args)
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

	doc_ids = []
	for doc_id in documents:
		doc_ids.append(doc_id)

	# Retrieval settings do not depend on the chunk type
	retrieval_args = make_retrieval_args(args)

	# Finds each Retrieved's eval question by its id
	retrieval_evaluation = RetrievalEvaluation(eval_qna)
	
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

	# 19. Retrieval quality of every setup, averaged over the tuning queries
	print_retrieval_summary(retrieval_qualities)

	# 20. Keep the best n_retrieval_quality plain setups and the best n_retrieval_quality reranked setups
	filtered_retrieval_qualities = retrieval_evaluation.get_filtered_retrieval_qualities(args.n_retrieval_quality, retrieval_qualities)
	print_retrieval_summary(filtered_retrieval_qualities)

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

	# ---------
	end = time.time()
	elapsed_time = end - start
	print("\n\nElapsed time : ", elapsed_time)
