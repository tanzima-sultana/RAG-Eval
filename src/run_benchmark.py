import time

from common import paths
from common.constants import HYBRID_RETRIEVAL
from common.args import parse_args
from common.args import make_ingestion_args
from common.args import make_chunking_args
from common.args import make_index_args
from common.args import make_vectordb_args
from common.args import make_eval_dataset_args
from common.args import make_ollama_args
from common.args import make_query_args
from common.args import make_retrieval_args
from rag.build.ingestion import load_saved_documents
from rag.build.chunking import Chunking
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from rag.build.manifest import find_chunk_entry
from rag.build.manifest import find_index_entry
from rag.build.manifest import get_index_settings
from rag.build.manifest import load_build_manifest
from evaluation.eval_dataset import EvalDataset
from rag.serving.query_transformation.query_transformation import QueryTransformation
from rag.serving.retrieval.retrieval import Retrieval
from evaluation.retrieval_evaluation import RetrievalEvaluation
from evaluation.setup_ranking import rank_setups
from evaluation.setup_ranking import save_best_setups


def run_benchmark(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: runs the benchmark: reads the build manifest, loads the documents and the saved eval dataset,
	transforms the queries, then per chunk type loads the saved chunks and indexes the manifest lists, retrieves the val
	queries and scores them. The config says what to benchmark, the manifest says where it is. It never builds: a
	missing or differently built output raises an error. It never generates the eval dataset.
	"""
	output_path = paths.get_output_path(args.output_path, args.dataset_name)

	# 1. The build manifest: what the build saved and where
	print("\n\nBuild manifest")
	manifest = load_build_manifest(output_path)
	print(f"Build manifest : built at {manifest.built_at}")

	# 2. Documents, only loaded from the manifest's documents folder
	print("\n\nDocuments")
	ingestion_args = make_ingestion_args(args)

	s1 = time.time()
	documents = load_saved_documents(manifest.documents_path, ingestion_args.dataset_size)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nDocuments time : ", e1)

	# 3. Eval dataset, only loaded. scripts/eval_dataset.sh generates it
	print("\n\nEval dataset")
	eval_dataset_args = make_eval_dataset_args(args)
	ollama_args = make_ollama_args(args)

	s2 = time.time()
	eval_samples = EvalDataset(eval_dataset_args, ollama_args).load_saved_eval_dataset(output_path)
	e2 = time.time() - s2
	print("\nEval dataset time : ", e2)

	# 4. Query transformation, once for the eval dataset
	print("\n\nQuery transformation")
	query_args = make_query_args(args)

	s3 = time.time()
	query_transformation = QueryTransformation(query_args, ollama_args)
	queries = query_transformation.compute_queries(eval_samples, eval_dataset_args, output_path)
	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)

	# 5. Query split: val for retrieval and retrieval evaluation, test for answer generation and answer evaluation
	print("\n\nQuery split")
	val_queries, test_queries = query_transformation.split_queries(queries)

	# Scores of every plain and reranked result, of every chunk type
	retrieval_evaluation = RetrievalEvaluation(eval_samples)
	retrieved_evals = []

	# 6. Chunks, only loaded, once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s5 = time.time()
		chunk_entry = find_chunk_entry(manifest, chunking_args)
		chunks = Chunking(chunking_args).load_saved_chunks(documents, chunk_entry.chunk_path)
		e5 = time.time() - s5
		print("\nChunking time : ", e5)

		# The plain dense and BM25 results of this chunk type, filled by the index and vector DB loops
		retrieval_args = make_retrieval_args(args, chunking_args)
		retrieval = Retrieval(retrieval_args, query_args, len(val_queries))
		retrieved_items = []

		# 7. Indexing and retrieval, once per index type, on the saved index
		for index_type in args.index_types:
			print(f"\n\nIndexing : {index_type}")
			index_args = make_index_args(args, chunking_args, index_type)

			s7 = time.time()
			index_entry = find_index_entry(manifest, chunk_type, index_type, get_index_settings(index_args))
			index = Indexing(index_args).load_saved_index(chunks, index_entry.location)
			e7 = time.time() - s7
			print("\nIndexing time : ", e7)

			print(f"\n\nRetrieval : {index_type}")
			s8 = time.time()
			index_retrieved_items = retrieval.compute_retrieval(index, index_type, val_queries, output_path)
			retrieved_items.extend(index_retrieved_items)
			e8 = time.time() - s8
			print("\nRetrieval time : ", e8)

		# 8. Vector DB and retrieval, once per vector DB type, on the saved collection
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type)

			s9 = time.time()
			vectordb_entry = find_index_entry(manifest, chunk_type, vectordb_type, {})
			vectordb = IndexingVectorDb(vectordb_args).load_saved_vectordb(chunks, vectordb_entry.location)
			e9 = time.time() - s9
			print("\nVector DB indexing time : ", e9)

			print(f"\n\nRetrieval : {vectordb_type}")
			s10 = time.time()
			vectordb_retrieved_items = retrieval.compute_vectordb_retrieval(vectordb, val_queries, output_path)
			retrieved_items.extend(vectordb_retrieved_items)
			e10 = time.time() - s10
			print("\nRetrieval time : ", e10)

		# 9. Hybrid retrieval, from the dense and BM25 results of this chunk type
		if HYBRID_RETRIEVAL in retrieval_args.retrieval_types:
			print("\n\nHybrid retrieval")
			s11 = time.time()
			hybrid_retrieved_items = retrieval.hybrid_retrieval(retrieved_items, output_path)
			retrieved_items.extend(hybrid_retrieved_items)
			e11 = time.time() - s11
			print("\nHybrid retrieval time : ", e11)

		# 10. Reranking of every plain result. The reranked results are added to the same list
		if retrieval_args.reranking:
			print("\n\nReranking")
			s12 = time.time()
			reranked_retrieved_items = retrieval.rerank_retrieval(retrieved_items, chunks, eval_samples, output_path)
			retrieved_items.extend(reranked_retrieved_items)
			e12 = time.time() - s12
			print(f"Reranked results : {len(reranked_retrieved_items)}")
			print("\nReranking time : ", e12)

		# 11. Retrieval evaluation of the plain and reranked results, while this chunk type's chunks are loaded
		print("\n\nRetrieval evaluation")
		s13 = time.time()
		chunk_type_retrieved_evals = retrieval_evaluation.compute_retrieved_evals(retrieved_items, chunks)
		retrieved_evals.extend(chunk_type_retrieved_evals)
		e13 = time.time() - s13
		print("\nRetrieval evaluation time : ", e13)

	print(f"\n\nRetrieved evals : {len(retrieved_evals)}")

	# 12. Rank every setup, plain and reranked together, and keep the best n_top_setups
	print("\n\nSetup ranking")
	best_setups = rank_setups(retrieved_evals, args.n_top_setups)
	print(f"Best setups : {len(best_setups)}")
	save_best_setups(best_setups, manifest, output_path)


if __name__ == "__main__":
	print("\n ------------- Benchmark -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the args and runs the benchmark.
	"""
	args = parse_args()
	run_benchmark(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
