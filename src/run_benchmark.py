import json
import time

from common import paths
from common.constants import HYBRID_RETRIEVAL
from common.constants import QDRANT_INDEX
from common.args import parse_benchmark_config
from common.args import make_benchmark_args
from common.args import make_manifest_chunking_args
from common.args import make_manifest_index_args
from common.args import make_ollama_args
from common.args import make_query_args
from common.args import make_retrieval_args
from common.args import make_vectordb_args
from evaluation.eval_dataset.cache import load_eval_dataset
from evaluation.retrieval_evaluation import RetrievalEvaluation
from evaluation.setup_ranking import rank_setups
from evaluation.setup_ranking import save_best_setups
from rag.build.chunking import Chunking
from rag.build.indexing import Indexing
from rag.build.indexing_vectordb import IndexingVectorDb
from rag.build.ingestion import load_documents
from rag.build.manifest import load_build_manifest_from_path
from rag.serving.query_transformation.query_transformation import QueryTransformation
from rag.serving.retrieval.retrieval import Retrieval
from schema.eval_dataset_metadata import EvalDatasetMetadata


def run_benchmark(args) -> None:
	"""
	Input: args, the parsed benchmark args.
	Output: nothing.
	What it does: runs the benchmark on one build and one eval dataset: reads the build manifest and the eval dataset
	metadata, checks they are of the same dataset, loads the documents and the eval set, transforms the queries, then
	for every chunk type and index the manifest lists loads the saved chunks and index, retrieves the val queries and
	scores them, and saves the best setups. It never builds and never generates the eval dataset.
	"""
	# 1. The build manifest and the eval dataset metadata. The only check is that they are of the same dataset
	print("\n\nBuild manifest and eval dataset metadata")
	manifest = load_build_manifest_from_path(args.manifest_path)
	print(f"Build manifest : {args.manifest_path}, built at {manifest.built_at}")

	with open(args.eval_dataset_metadata_path, encoding="utf-8") as f:
		eval_dataset_metadata = EvalDatasetMetadata(**json.load(f))
	print(f"Eval dataset metadata : {args.eval_dataset_metadata_path}")

	if manifest.dataset_name != eval_dataset_metadata.dataset_name:
		raise ValueError(
			f"The build manifest is of dataset {manifest.dataset_name}, but the eval dataset metadata is of dataset "
			f"{eval_dataset_metadata.dataset_name}."
		)

	output_path = paths.get_output_path(args.output_path, manifest.dataset_name)

	# 2. Documents, the ones the build saved for its input files
	print("\n\nDocuments")
	s1 = time.time()
	documents = load_documents(manifest.documents_path, manifest.input_hash)
	if documents is None:
		raise FileNotFoundError(
			f"No saved document list doc_ids_{manifest.input_hash}.json in {manifest.documents_path}. Run scripts/build.sh first."
		)
	e1 = time.time() - s1
	print(f"Documents : {len(documents)}")
	print("\nDocuments time : ", e1)

	# 3. Eval dataset, the one the metadata names
	print("\n\nEval dataset")
	s2 = time.time()
	eval_samples = load_eval_dataset(eval_dataset_metadata.eval_dataset_path)
	if eval_samples is None:
		raise FileNotFoundError(
			f"No saved eval dataset at {eval_dataset_metadata.eval_dataset_path}. Run scripts/eval_dataset.sh first."
		)
	e2 = time.time() - s2
	print(f"Eval dataset : {len(eval_samples)} from {eval_dataset_metadata.eval_dataset_path}")
	print("\nEval dataset time : ", e2)

	# 4. Query transformation, once for the eval dataset, with the build's embedding model
	print("\n\nQuery transformation")
	ollama_args = make_ollama_args(args)
	query_args = make_query_args(args, manifest.embedding_model, eval_dataset_metadata.eval_dataset_hash)

	s3 = time.time()
	query_transformation = QueryTransformation(query_args, ollama_args)
	queries = query_transformation.compute_queries(eval_samples, output_path)
	e3 = time.time() - s3
	print("\nQuery transformation time : ", e3)

	# 5. Query split: val for retrieval and retrieval evaluation, test for answer generation and answer evaluation
	print("\n\nQuery split")
	val_queries, test_queries = query_transformation.split_queries(queries)

	# Scores of every plain and reranked result, of every chunk type
	retrieval_evaluation = RetrievalEvaluation(eval_samples)
	retrieved_evals = []

	# 6. Every chunk type the build made, with the settings it made them with
	for chunk_entry in manifest.chunk_entries:
		chunk_type = chunk_entry.chunk_type
		print(f"\n\n========== Chunk type : {chunk_type} ==========")
		print("\n\nChunking")
		chunking_args = make_manifest_chunking_args(chunk_entry, manifest.embedding_model)

		s5 = time.time()
		chunks = Chunking(chunking_args).load_saved_chunks(documents, chunk_entry.chunk_path)
		e5 = time.time() - s5
		print("\nChunking time : ", e5)

		# The index settings of this chunk type, from every index the build made of it
		chunk_type_index_settings = {}
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type:
				continue
			for setting_name in index_entry.index_settings:
				chunk_type_index_settings[setting_name] = index_entry.index_settings[setting_name]

		# The plain dense and BM25 results of this chunk type, filled by the index and vector DB loops
		chunk_type_index_args = make_manifest_index_args(chunking_args, "", chunk_type_index_settings, manifest.input_hash)
		retrieval_args = make_retrieval_args(args, chunk_type_index_args)
		retrieval = Retrieval(retrieval_args, query_args, len(val_queries))
		retrieved_items = []

		# 7. Every FAISS and BM25 index the build made of this chunk type, and retrieval on it
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type or index_entry.index_type == QDRANT_INDEX:
				continue
			index_type = index_entry.index_type
			print(f"\n\nIndexing : {index_type}")
			index_args = make_manifest_index_args(chunking_args, index_type, chunk_type_index_settings, manifest.input_hash)

			s7 = time.time()
			index = Indexing(index_args).load_saved_index(chunks, index_entry.location)
			e7 = time.time() - s7
			print("\nIndexing time : ", e7)

			print(f"\n\nRetrieval : {index_type}")
			s8 = time.time()
			index_retrieved_items = retrieval.compute_retrieval(index, index_type, val_queries, output_path)
			retrieved_items.extend(index_retrieved_items)
			e8 = time.time() - s8
			print("\nRetrieval time : ", e8)

		# 8. Every Qdrant collection the build made of this chunk type, and retrieval on it
		for index_entry in manifest.index_entries:
			if index_entry.chunk_type != chunk_type or index_entry.index_type != QDRANT_INDEX:
				continue
			vectordb_type = index_entry.index_type
			print(f"\n\nVector DB indexing : {vectordb_type}")
			vectordb_args = make_vectordb_args(args, chunking_args, vectordb_type, manifest.input_hash)

			s9 = time.time()
			vectordb = IndexingVectorDb(vectordb_args).load_saved_vectordb(chunks, index_entry.location)
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

	# 12. Rank every setup, plain and reranked together, keep the best n_top_setups and save them, named by the build,
	# eval dataset and benchmark hashes
	print("\n\nSetup ranking")
	best_setups = rank_setups(retrieved_evals, args.n_top_setups)
	print(f"Best setups : {len(best_setups)}")
	benchmark_hash = paths.get_benchmark_hash(make_benchmark_args(args, query_args), ollama_args)
	save_best_setups(best_setups, manifest, output_path, eval_dataset_metadata.eval_dataset_hash, benchmark_hash)


if __name__ == "__main__":
	print("\n ------------- Benchmark -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the benchmark config and runs the benchmark.
	"""
	args = parse_benchmark_config()
	run_benchmark(args)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
