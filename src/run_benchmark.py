import time

from common import paths_temp
from common.constants import HYBRID_RETRIEVAL
from common.args_temp import parse_args
from common.args_temp import make_ingestion_args
from common.args_temp import make_chunking_args
from common.args_temp import make_embedding_args
from common.args_temp import make_index_args
from common.args_temp import make_vectordb_args
from common.args_temp import make_eval_dataset_args
from common.args_temp import make_ollama_args
from common.args_temp import make_query_args
from common.args_temp import make_retrieval_args
from rag.build.ingestion import ingestion
from rag.build.chunking import Chunking
from rag.build.embedding import Embedding
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from evaluation.eval_dataset import EvalDataset
from rag.serving.query_transformation.query_transformation import QueryTransformation
from rag.serving.retrieval.retrieval import Retrieval
from evaluation.retrieval_evaluation import RetrievalEvaluation
from evaluation.setup_ranking import rank_setups


def run_benchmark(args) -> None:
	"""
	Input: args, the parsed args.
	Output: nothing.
	What it does: runs the benchmark: loads the documents and the saved eval dataset, transforms the queries,
	then per chunk type loads the build layers, retrieves the val queries and scores them. It never generates the eval
	dataset.
	"""
	# 1. Ingestion
	print("\n\nIngestion")
	ingestion_args = make_ingestion_args(args)

	s1 = time.time()
	documents = ingestion(ingestion_args)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nIngestion time : ", e1)

	output_path = paths_temp.get_output_path(args.output_path, args.dataset_name)

	# 2. Eval dataset, only loaded. scripts/eval_dataset.sh generates it
	print("\n\nEval dataset")
	eval_dataset_args = make_eval_dataset_args(args)
	ollama_args = make_ollama_args(args)

	s2 = time.time()
	eval_samples = EvalDataset(eval_dataset_args, ollama_args).load_saved_eval_dataset(output_path)
	e2 = time.time() - s2
	print("\nEval dataset time : ", e2)

	# 3. Query transformation, once for the eval dataset
	print("\n\nQuery transformation")
	query_args = make_query_args(args)

	s3 = time.time()
	query_transformation = QueryTransformation(query_args, ollama_args)
	queries = query_transformation.compute_queries(eval_samples, eval_dataset_args, output_path)
	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)

	# 4. Query split: val for retrieval and retrieval evaluation, test for answer generation and answer evaluation
	print("\n\nQuery split")
	val_queries, test_queries = query_transformation.split_queries(queries)

	# Scores of every plain and reranked result, of every chunk type
	retrieval_evaluation = RetrievalEvaluation(eval_samples)
	retrieved_evals = []

	# 5. Chunking, once per chunk type
	for chunk_type in args.chunk_types:
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_chunking_args(args, chunk_type)

		s5 = time.time()
		chunks = Chunking(chunking_args).compute_chunks(documents, output_path)
		e5 = time.time() - s5
		print("\nChunking time : ", e5)

		# 6. Embedding
		print("\n\nEmbedding")
		embedding_args = make_embedding_args(chunking_args)

		s6 = time.time()
		embeddings = Embedding(embedding_args).compute_embeddings(chunks, output_path)
		e6 = time.time() - s6
		print("\nEmbedding time : ", e6)

		# The plain dense and BM25 results of this chunk type, filled by the index and vector DB loops
		retrieval_args = make_retrieval_args(args, chunking_args)
		retrieval = Retrieval(retrieval_args, query_args, len(val_queries))
		retrieved_items = []

		# 7. Indexing and retrieval, once per index type, on the index just built or loaded
		for index_type in args.index_types:
			print(f"\n\nIndexing : {index_type}")
			index_args = make_index_args(args, chunking_args, index_type)

			s7 = time.time()
			index = Indexing(index_args).compute_index(chunks, embeddings, output_path)
			e7 = time.time() - s7
			print("\nIndexing time : ", e7)

			print(f"\n\nRetrieval : {index_type}")
			s8 = time.time()
			index_retrieved_items = retrieval.compute_retrieval(index, index_type, val_queries, output_path)
			retrieved_items.extend(index_retrieved_items)
			e8 = time.time() - s8
			print("\nRetrieval time : ", e8)

		# 8. Vector DB indexing and retrieval, once per vector DB type, on the collection just built or loaded
		for vectordb_type in args.vectordb_types:
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type)

			s9 = time.time()
			vectordb = IndexingVectorDb(vectordb_args).compute_vectordb(chunks, embeddings, output_path)
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
